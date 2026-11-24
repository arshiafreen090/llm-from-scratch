"""
HTTP API for the model, with token streaming over Server-Sent Events (SSE).

Start it (from the project root):
    python -m serve.api --ckpt checkpoints/smoke/smoke-tiny-llama.pt --port 8000

Then:
    curl localhost:8000/health
    curl -X POST localhost:8000/generate -H "Content-Type: application/json" \
         -d '{"prompt": "Once upon a time", "max_new_tokens": 50}'
    # streaming (-N turns off curl's buffering so tokens appear as they arrive)
    curl -N -X POST localhost:8000/generate -H "Content-Type: application/json" \
         -d '{"prompt": "Once upon a time", "stream": true}'

Interactive docs (try requests from the browser): http://localhost:8000/docs

Architecture
------------
    HTTP request threads  --submit job-->  [ job queue ]  -->  ONE model worker thread
           ^                                                    running a BatchEngine
           +---------------- tokens (per-job queue) ---------------+

* One model worker thread owns the model: it loads it and runs every forward pass.
  PyTorch's CPU thread pool (OpenMP) slows down badly when model calls come from many
  different threads (measured 37 tok/s instead of 460+ on this laptop); one owning
  thread avoids that entirely.
* Continuous batching (inference/batching.py): up to --max_batch requests run TOGETHER,
  one token each per forward pass. A finished request leaves at once and a waiting one
  joins at the next step. --max_batch 1 turns it off (one request at a time).
* Backpressure: when all rows are busy, jobs wait in the queue; if too many are already
  waiting, new requests get HTTP 503 ("try again later") immediately, instead of piling
  up until everything times out.
* Cancellation: if a client disconnects (or times out), its job is marked cancelled and the
  worker drops it before the next step, freeing its row for someone else.

Request isolation (what one request can and can't affect in another)
--------------------------------------------------------------------
    per request, never shared    sampling parameters, random generator (own seed, returned
                                 in the response), token history, UTF-8 decoder, stop
                                 condition, output queue, cancel flag, result object
    own row in a shared tensor   KV cache: each running request owns one row ("slot"). The
                                 attention mask limits row b to its own positions 0..pos[b],
                                 so it never reads another request's keys or the leftovers
                                 of the slot's previous owner. Every model operation is
                                 per-row, so a request generates exactly what it would alone
                                 (tests/test_batching.py checks token-for-token equality).
    shared, read-only            model weights (eval mode, inference_mode: never modified)
    shared, written              the tokenizer's chunk cache: dict reads/writes are atomic
                                 under CPython's GIL, and a cached value is the same no matter
                                 which request computed it
    failure isolation            a failure in one request's prefill or sampling is reported
                                 to that request only (500 / error event). A failure inside
                                 the SHARED decode pass reaches every request in that pass:
                                 each gets an error, and the worker keeps serving new ones.
    latency (partly shared)      no more head-of-line blocking up to max_batch requests, but
                                 each step is slower with more rows, and a new prompt's
                                 prefill pauses the others for one pass (chunked prefill
                                 would smooth that out).
"""

import argparse
import json
import queue
import threading
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from inference.batching import BatchEngine, Sequence
from inference.generate import GenerationResult, Generator
from inference.quantize import MODES as QUANT_MODES
from inference.sampling import SamplingParams


class GenerateRequest(BaseModel):
    prompt: str = Field(..., max_length=4000, examples=["Once upon a time"])
    max_new_tokens: int = Field(200, ge=1, le=4096)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(None, ge=1)
    top_p: Optional[float] = Field(None, gt=0.0, le=1.0)
    min_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.0, ge=1.0, le=2.0)
    seed: Optional[int] = None
    stream: bool = False
    chat: bool = False  # true: `prompt` is an instruction, wrapped in the SFT chat template

    def sampling_params(self):
        return SamplingParams(max_new_tokens=self.max_new_tokens, temperature=self.temperature,
                              top_k=self.top_k, top_p=self.top_p, min_p=self.min_p,
                              repetition_penalty=self.repetition_penalty, seed=self.seed)


def result_payload(result: GenerationResult, queue_ms: float):
    return {
        "text": result.text,
        "finish_reason": result.finish_reason,
        "seed": result.seed,
        "usage": {"prompt_tokens": result.prompt_tokens, "completion_tokens": len(result.token_ids)},
        "latency": {"queue_ms": round(queue_ms, 1), "ttft_ms": round(result.ttft_ms, 1),
                    "decode_tokens_per_s": round(result.decode_tokens_per_s, 1),
                    "total_ms": round(result.total_ms, 1)},
    }


# =============================================================================
# The model worker
# =============================================================================

class Job:
    def __init__(self, prompt_ids, params: SamplingParams):
        self.prompt_ids, self.params = prompt_ids, params
        self.out = queue.Queue()            # ("token", text) ... then ("done", result) or ("error", msg)
        self.cancelled = threading.Event()  # set by the request side when the client is gone
        self.submitted = time.perf_counter()
        self.queue_ms = 0.0                 # time until it got a row and its prefill started


class ModelWorker:
    """A single thread that owns the model and runs a BatchEngine: up to max_batch jobs at once."""

    def __init__(self, load_generator, max_waiting=16, max_batch=8):
        self.jobs = queue.Queue()  # waiting for a free row
        self.max_waiting = max_waiting
        self.max_batch = max_batch
        self.running = 0
        self.generator = None
        self._load_error = None
        self._ready = threading.Event()
        threading.Thread(target=self._run, args=(load_generator,), daemon=True, name="model-worker").start()
        self._ready.wait()
        if self._load_error is not None:
            raise self._load_error

    @property
    def busy(self):
        return self.running > 0

    def _run(self, load_generator):
        try:
            self.generator = load_generator()  # loaded ON this thread, like every model call after it
            engine = BatchEngine(self.generator, self.max_batch)
        except Exception as e:
            self._load_error = e
        self._ready.set()
        if self._load_error is not None:
            return

        while True:
            # 1. take new jobs while there are free rows. Block only when there's nothing
            #    to run; otherwise just grab whatever has already arrived and keep stepping.
            while engine.free_slots() > 0:
                try:
                    job = self.jobs.get(block=not engine.has_work())
                except queue.Empty:
                    break
                if job.cancelled.is_set():
                    continue  # the client gave up while waiting: skip it
                try:
                    engine.add(Sequence(job.prompt_ids, job.params, tag=job))
                except Exception as e:
                    job.out.put(("error", str(e)))

            # 2. drop requests whose client is gone; their rows go to someone else
            for seq in engine.sequences():
                if seq.tag.cancelled.is_set():
                    engine.abort(seq)

            # 3. one step for every running request, then hand each its events
            events = engine.step()
            # update the count BEFORE clients hear "done", so a /health call made right
            # after a response never sees a stale busy=True
            self.running = len(engine.running)
            for seq, kind, value in events:
                job = seq.tag
                if seq.t_start is not None:
                    job.queue_ms = (seq.t_start - job.submitted) * 1000
                job.out.put((kind, value))

    def submit(self, prompt_ids, params) -> Job:
        if self.jobs.qsize() >= self.max_waiting:
            raise HTTPException(503, f"server busy: {self.jobs.qsize()} requests already waiting")
        job = Job(prompt_ids, params)
        self.jobs.put(job)
        return job

    def warm_up(self, n_tokens=32):
        """The first generation after loading runs ~3x slower (PyTorch sets up its kernels
        and its CPU thread pool). Run one at startup, through the queue like any request,
        so it happens on the worker thread and no real client pays for it."""
        job = self.submit(self.generator.encode_prompt("Once upon a time"),
                          SamplingParams(max_new_tokens=n_tokens, temperature=0))
        while job.out.get()[0] not in ("done", "error"):
            pass


# =============================================================================
# The app
# =============================================================================

def create_app(generator: Generator = None, load_generator=None, max_waiting: int = 16,
               timeout_s: float = 300.0, max_batch: int = 8, warm_up: bool = False) -> FastAPI:
    """Pass either a ready `generator` (tests) or a `load_generator` function (real server,
    so the model is created on the worker thread). max_batch=1 turns batching off.
    warm_up=True runs one short generation before the app is returned."""
    worker = ModelWorker(load_generator or (lambda: generator), max_waiting, max_batch)
    if warm_up:
        worker.warm_up()
    gen = worker.generator
    cfg = gen.model.cfg
    app = FastAPI(title="llm-from-scratch", description="A small LLaMA-style model trained from scratch.")

    def next_event(job):
        try:
            return job.out.get(timeout=timeout_s)
        except queue.Empty:
            job.cancelled.set()
            return ("error", f"timed out after {timeout_s:.0f}s")

    @app.get("/health")
    def health():
        return {"status": "ok", "busy": worker.busy, "running": worker.running,
                "max_batch": worker.max_batch, "waiting": worker.jobs.qsize()}

    @app.get("/info")
    def info():
        return {"params": gen.model.num_params(), "vocab_size": cfg.vocab_size,
                "context_length": cfg.block_size, "n_layer": cfg.n_layer, "n_embd": cfg.n_embd,
                "n_head": cfg.n_head, "n_kv_head": cfg.n_kv_head, "pos_emb": cfg.pos_emb,
                "norm": cfg.norm, "mlp": cfg.mlp, "device": str(gen.device),
                "quant": getattr(gen.model, "quant", None), "instruction_tuned": gen.is_sft}

    # Plain `def` endpoints run in FastAPI's thread pool, so waiting on a job here never
    # blocks the server (e.g. /health keeps answering while a generation runs).
    @app.post("/generate")
    def generate(req: GenerateRequest):
        prompt_ids = gen.encode_prompt(req.prompt, chat=req.chat)
        if len(prompt_ids) >= cfg.block_size:
            raise HTTPException(400, f"prompt is {len(prompt_ids)} tokens; the model's context is {cfg.block_size}")
        job = worker.submit(prompt_ids, req.sampling_params())  # 503 here if the queue is full

        if not req.stream:
            try:
                while True:
                    kind, value = next_event(job)
                    if kind == "done":
                        return result_payload(value, job.queue_ms)
                    if kind == "error":
                        raise HTTPException(500, value)
            finally:
                job.cancelled.set()  # no-op if finished; stops the worker if we bailed out

        def events():
            """SSE format: each event is a line 'data: <json>' followed by a blank line."""
            try:
                while True:
                    kind, value = next_event(job)
                    if kind == "token":
                        yield f"data: {json.dumps({'type': 'token', 'text': value})}\n\n"
                    elif kind == "done":
                        yield f"data: {json.dumps({'type': 'done', **result_payload(value, job.queue_ms)})}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    else:  # headers are already sent, so errors are reported as an event
                        yield f"data: {json.dumps({'type': 'error', 'message': value})}\n\n"
                        return
            finally:
                # also runs when the client disconnects mid-stream: tells the worker to stop
                job.cancelled.set()

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_waiting", type=int, default=16, help="queue length before returning 503")
    parser.add_argument("--max_batch", type=int, default=8,
                        help="requests decoded together per forward pass (1 = no batching)")
    parser.add_argument("--quant", default=None, choices=QUANT_MODES,
                        help="int8 weights. int8-dynamic is faster on CPU, but a request's output can then "
                             "depend on the others in its batch (see inference/quantize.py)")
    args = parser.parse_args()

    import uvicorn
    app = create_app(load_generator=lambda: Generator.from_checkpoint(args.ckpt, args.device, args.quant),
                     max_waiting=args.max_waiting, max_batch=args.max_batch, warm_up=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
