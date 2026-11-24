"""
OpenAI-compatible endpoints, so any OpenAI client can talk to this model:

    GET  /v1/models
    POST /v1/chat/completions        (stream: true for Server-Sent Events, token by token)

From Python, with the official library (pip install openai):

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-tl-...")   # any string if keys are off
    stream = client.chat.completions.create(
        model="tinyllama", stream=True,
        messages=[{"role": "user", "content": "Write a story. Use the words: rain, ball, happy."}])
    for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="", flush=True)

WHAT "CHAT" MEANS FOR THIS MODEL
    It was instruction-tuned on single instructions, not conversations. So the reply is to
    the LAST user message only; earlier messages and system prompts are accepted (clients
    send them) but not used. A model that was never instruction-tuned continues the last
    user message as a story instead.

Extras beyond the OpenAI fields: top_k, min_p, repetition_penalty, and in responses an
"x_tinyllama" object with the seed (send it back to reproduce a reply) and timings.
"""

import json
import queue
import time
import uuid
from typing import List, Literal, Optional

from fastapi import Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from inference.sampling import SamplingParams
from serve.auth import APIError

DEFAULT_MAX_TOKENS = 300


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "developer"]
    content: str = Field(..., max_length=4000)


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Message] = Field(..., min_length=1, max_length=100)
    max_tokens: Optional[int] = Field(None, ge=1, le=4096)
    max_completion_tokens: Optional[int] = Field(None, ge=1, le=4096)  # the newer OpenAI name
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(0.9, gt=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=1)
    min_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.0, ge=1.0, le=2.0)
    seed: Optional[int] = None
    stream: bool = False

    def sampling_params(self):
        max_new = self.max_completion_tokens or self.max_tokens or DEFAULT_MAX_TOKENS
        return SamplingParams(max_new_tokens=max_new, temperature=self.temperature,
                              top_p=self.top_p if self.top_p is not None and self.top_p < 1 else None,
                              top_k=self.top_k, min_p=self.min_p,
                              repetition_penalty=self.repetition_penalty, seed=self.seed)

    def last_user_message(self):
        for m in reversed(self.messages):
            if m.role == "user":
                return m.content
        raise APIError(400, "messages must contain at least one message with role 'user'.")


def finish_reason(result):
    return "stop" if result.finish_reason == "stop" else "length"  # "context" is also a length limit


def extras(result, queue_ms):
    return {"seed": result.seed, "queue_ms": round(queue_ms, 1), "ttft_ms": round(result.ttft_ms, 1),
            "decode_tokens_per_s": round(result.decode_tokens_per_s, 1)}


def usage(result):
    return {"prompt_tokens": result.prompt_tokens, "completion_tokens": len(result.token_ids),
            "total_tokens": result.prompt_tokens + len(result.token_ids)}


def add_openai_routes(app, worker, guard, model_name, timeout_s):
    gen = worker.generator
    cfg = gen.model.cfg

    def next_event(job):
        try:
            return job.out.get(timeout=timeout_s)
        except queue.Empty:
            job.cancelled.set()
            return ("error", f"timed out after {timeout_s:.0f}s")

    @app.get("/v1/models", tags=["OpenAI-compatible"])
    def list_models(client: str = Depends(guard)):
        return {"object": "list", "data": [{"id": model_name, "object": "model", "created": 0,
                                            "owned_by": "llm-from-scratch"}]}

    @app.post("/v1/chat/completions", tags=["OpenAI-compatible"])
    def chat_completions(req: ChatCompletionRequest, client: str = Depends(guard)):
        text = req.last_user_message()
        prompt_ids = gen.encode_prompt(text, chat=gen.is_sft)
        if len(prompt_ids) >= cfg.block_size:
            raise APIError(400, f"The message is {len(prompt_ids)} tokens; the model's context is "
                                f"{cfg.block_size}.", code="context_length_exceeded")
        try:
            job = worker.submit(prompt_ids, req.sampling_params())
        except Exception:
            raise APIError(503, "The server is busy. Try again in a moment.", type_="server_error",
                           code="overloaded", headers={"Retry-After": "2"})
        reply_id, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())

        if not req.stream:
            try:
                while True:
                    kind, value = next_event(job)
                    if kind == "done":
                        return {"id": reply_id, "object": "chat.completion", "created": created,
                                "model": model_name,
                                "choices": [{"index": 0, "finish_reason": finish_reason(value),
                                             "message": {"role": "assistant", "content": value.text}}],
                                "usage": usage(value), "x_tinyllama": extras(value, job.queue_ms)}
                    if kind == "error":
                        raise APIError(500, value, type_="server_error")
            finally:
                job.cancelled.set()

        def chunk(delta, finish=None, **more):
            body = {"id": reply_id, "object": "chat.completion.chunk", "created": created, "model": model_name,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **more}
            return f"data: {json.dumps(body)}\n\n"

        def events():
            try:
                yield chunk({"role": "assistant", "content": ""})
                while True:
                    kind, value = next_event(job)
                    if kind == "token":
                        yield chunk({"content": value})
                    elif kind == "done":
                        yield chunk({}, finish_reason(value), usage=usage(value),
                                    x_tinyllama=extras(value, job.queue_ms))
                        yield "data: [DONE]\n\n"
                        return
                    else:  # headers are already sent, so an error has to travel as an event
                        yield f"data: {json.dumps({'error': {'message': value, 'type': 'server_error'}})}\n\n"
                        return
            finally:
                job.cancelled.set()  # also when the client disconnects: frees the row

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
