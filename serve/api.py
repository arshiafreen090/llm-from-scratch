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
           ^                                                       |
           +---------------- tokens (per-job queue) ---------------+

* One model worker thread owns the model: it loads it and runs every generation.
  Two reasons:
    1. Speed. PyTorch's CPU thread pool (OpenMP) slows down badly when model calls come
       from many different threads: measured 37 tok/s instead of 460+ on this laptop.
       One owning thread avoids that entirely.
    2. Running two generations at once on one CPU/GPU doesn't finish them faster; they just
       slow each other down. So jobs wait in a queue, and the response reports how long
       each one waited (queue_ms). Serving many users efficiently needs BATCHING, i.e.
       running several requests through the model together, which will live in this same
       worker. This server is the baseline that batching gets measured against.
* Backpressure: if too many jobs are already waiting, new requests get HTTP 503
  ("try again later") immediately, instead of piling up until everything times out.
* Cancellation: if a client disconnects (or times out), its job is marked cancelled and the
  worker stops generating for it at the next token.

Request isolation (what one request can and can't affect in another)
--------------------------------------------------------------------
    per request, never shared    KV cache (made fresh per generation), sampling parameters,
                                 random generator (own seed, returned in the response),
                                 output queue, cancel flag, result object
    shared, read-only            model weights (eval mode, inference_mode: never modified)
    shared, written              the tokenizer's chunk cache: dict reads/writes are atomic
                                 under CPython's GIL, and a cached value is the same no matter
                                 which request computed it
    failure isolation            an exception inside one generation is reported to that
                                 request only (500 / error event); the worker keeps serving
    NOT isolated yet             latency: jobs run one at a time, so a long generation delays
                                 everyone queued behind it (head-of-line blocking). Fixing
                                 that is the job of batching, which must keep every row above
                                 per-request: its own cache slot, sampling params, RNG and
                                 stop condition.
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

from inference.generate import GenerationResult, Generator
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
    def __init__(self, prompt: str, params: SamplingParams, chat: bool = False):
        self.prompt, self.params, self.chat = prompt, params, chat
        self.out = queue.Queue()            # ("token", text) ... then ("done", result) or ("error", msg)
        self.cancelled = threading.Event()  # set by the request side when the client is gone
        self.submitted = time.perf_counter()
        self.queue_ms = 0.0


class ModelWorker:
    """A single thread that owns the model and runs queued jobs one at a time."""

    def __init__(self, load_generator, max_waiting=16):
        self.jobs = queue.Queue()
        self.max_waiting = max_waiting
        self.busy = False
        self.generator = None
        self._load_error = None
        self._ready = threading.Event()
        threading.Thread(target=self._run, args=(load_generator,), daemon=True, name="model-worker").start()
        self._ready.wait()
        if self._load_error is not None:
            raise self._load_error

    def _run(self, load_generator):
        try:
            self.generator = load_generator()  # loaded ON this thread, like every model call after it
        except Exception as e:
            self._load_error = e
        self._ready.set()
        if self._load_error is not None:
            return

        while True:
            job = self.jobs.get()
            if job.cancelled.is_set():
                continue  # the client gave up while waiting: skip it
            job.queue_ms = (time.perf_counter() - job.submitted) * 1000
            self.busy = True
            result = GenerationResult(text="")
            try:
                for piece in self.generator.stream(job.prompt, job.params, result, chat=job.chat):
                    if job.cancelled.is_set():
                        break
                    job.out.put(("token", piece))
                outcome = ("done", result)
            except Exception as e:
                outcome = ("error", str(e))
            # mark the worker free BEFORE the client hears "done", so a /health call made
            # right after a response never sees a stale busy=True
            self.busy = False
            job.out.put(outcome)

    def submit(self, prompt, params, chat=False) -> Job:
        if self.jobs.qsize() >= self.max_waiting:
            raise HTTPException(503, f"server busy: {self.jobs.qsize()} requests already waiting")
        job = Job(prompt, params, chat)
        self.jobs.put(job)
        return job


# =============================================================================
# The app
# =============================================================================

def create_app(generator: Generator = None, load_generator=None, max_waiting: int = 16,
               timeout_s: float = 300.0) -> FastAPI:
    """Pass either a ready `generator` (tests) or a `load_generator` function (real server,
    so the model is created on the worker thread)."""
    worker = ModelWorker(load_generator or (lambda: generator), max_waiting)
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
        return {"status": "ok", "busy": worker.busy, "waiting": worker.jobs.qsize()}

    @app.get("/info")
    def info():
        return {"params": gen.model.num_params(), "vocab_size": cfg.vocab_size,
                "context_length": cfg.block_size, "n_layer": cfg.n_layer, "n_embd": cfg.n_embd,
                "n_head": cfg.n_head, "n_kv_head": cfg.n_kv_head, "pos_emb": cfg.pos_emb,
                "norm": cfg.norm, "mlp": cfg.mlp, "device": str(gen.device),
                "instruction_tuned": gen.is_sft}

    # Plain `def` endpoints run in FastAPI's thread pool, so waiting on a job here never
    # blocks the server (e.g. /health keeps answering while a generation runs).
    @app.post("/generate")
    def generate(req: GenerateRequest):
        prompt_len = len(gen.encode_prompt(req.prompt, chat=req.chat))
        if prompt_len >= cfg.block_size:
            raise HTTPException(400, f"prompt is {prompt_len} tokens; the model's context is {cfg.block_size}")
        job = worker.submit(req.prompt, req.sampling_params(), req.chat)  # 503 here if the queue is full

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
    args = parser.parse_args()

    import uvicorn
    app = create_app(load_generator=lambda: Generator.from_checkpoint(args.ckpt, args.device),
                     max_waiting=args.max_waiting)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
