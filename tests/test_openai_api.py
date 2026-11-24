"""
Tests for the OpenAI-compatible API, API keys, rate limits and the web UI routes
(serve/openai_compat.py, serve/auth.py).
    python -m pytest tests/test_openai_api.py -v
"""

import json
import os

import pytest
import torch
from fastapi.testclient import TestClient

from inference.generate import Generator
from inference.sampling import SamplingParams
from model.config import get_config
from model.model import TinyLLaMA
from serve.api import create_app
from serve.auth import Guard, RateLimiter, hash_key, load_key_hashes, new_key
from tokenizer.tokenizer import Tokenizer

TOKENIZER = os.path.join(os.path.dirname(__file__), "..", "tokenizer", "tinystories_4k.json")
INSTRUCTION = "Write a story. Use the words: rain, ball, happy."


@pytest.fixture(scope="module")
def generator():
    torch.manual_seed(0)
    model = TinyLLaMA(get_config("tiny", "llama", block_size=128))
    return Generator(model, Tokenizer.load(TOKENIZER), is_sft=True)


@pytest.fixture(scope="module")
def client(generator):
    return TestClient(create_app(generator, model_name="tiny-test"))


def chat(client, content=INSTRUCTION, headers=None, **extra):
    body = {"model": "tiny-test", "messages": [{"role": "user", "content": content}],
            "max_tokens": 20, "seed": 5, **extra}
    return client.post("/v1/chat/completions", json=body, headers=headers or {})


def sse_events(response):
    events = []
    for line in response.iter_lines():
        if line.startswith("data: "):
            data = line[6:]
            events.append(data if data == "[DONE]" else json.loads(data))
    return events


# ---- the OpenAI format ----------------------------------------------------------------------

def test_models_lists_the_model(client):
    body = client.get("/v1/models").json()
    assert body["object"] == "list" and body["data"][0]["id"] == "tiny-test"


def test_chat_completion_matches_generating_directly(client, generator):
    body = chat(client).json()
    assert body["object"] == "chat.completion" and body["model"] == "tiny-test"
    choice = body["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert choice["finish_reason"] in ("stop", "length")
    direct = generator.generate(INSTRUCTION, SamplingParams(max_new_tokens=20, temperature=0.7, top_p=0.9, seed=5),
                                chat=True)
    assert choice["message"]["content"] == direct.text
    assert body["usage"]["completion_tokens"] == len(direct.token_ids)
    assert body["x_tinyllama"]["seed"] == 5


def test_stream_gives_the_same_text_in_chunks(client):
    full = chat(client).json()["choices"][0]["message"]["content"]
    with client.stream("POST", "/v1/chat/completions",
                       json={"messages": [{"role": "user", "content": INSTRUCTION}], "max_tokens": 20,
                             "seed": 5, "stream": True}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        events = sse_events(r)
    assert events[-1] == "[DONE]"
    chunks = events[:-1]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == full
    last = chunks[-1]
    assert last["choices"][0]["finish_reason"] in ("stop", "length") and "usage" in last and "x_tinyllama" in last


def test_only_the_last_user_message_is_answered(client):
    alone = chat(client).json()["choices"][0]["message"]["content"]
    with_history = client.post("/v1/chat/completions", json={
        "max_tokens": 20, "seed": 5,
        "messages": [{"role": "system", "content": "You are helpful."},
                     {"role": "user", "content": "Something else entirely."},
                     {"role": "assistant", "content": "Once upon a time..."},
                     {"role": "user", "content": INSTRUCTION}]}).json()
    assert with_history["choices"][0]["message"]["content"] == alone


def test_errors_use_the_openai_shape(client):
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "system", "content": "hi"}]})
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"
    r = chat(client, content="word " * 300)  # longer than the 128-token context
    assert r.status_code == 400 and r.json()["error"]["code"] == "context_length_exceeded"
    assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 422


# ---- API keys and rate limits ---------------------------------------------------------------

def test_api_keys_are_enforced_when_configured(generator):
    key = new_key()
    client = TestClient(create_app(generator, guard=Guard({hash_key(key)})))
    r = chat(client)
    assert r.status_code == 401 and r.json()["error"]["code"] == "missing_api_key"
    r = chat(client, headers={"Authorization": "Bearer sk-tl-wrong"})
    assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_api_key"
    assert chat(client, headers={"Authorization": f"Bearer {key}"}).status_code == 200
    # the legacy endpoint is protected too; health, info and the page stay open
    assert client.post("/generate", json={"prompt": "Tom"}).status_code == 401
    info = client.get("/info").json()
    assert info["api_key_required"] is True and client.get("/health").status_code == 200


def test_rate_limit_returns_429_with_retry_after(generator):
    client = TestClient(create_app(generator, guard=Guard(limiter=RateLimiter(per_minute=60, burst=2))))
    assert chat(client).status_code == 200
    assert chat(client).status_code == 200
    r = chat(client)
    assert r.status_code == 429 and r.json()["error"]["code"] == "rate_limit_exceeded"
    assert int(r.headers["Retry-After"]) >= 1


def test_token_bucket_refills_over_time():
    limiter = RateLimiter(per_minute=60, burst=2)  # 1 token per second
    assert limiter.check("a", now=0) == 0
    assert limiter.check("a", now=0) == 0
    assert limiter.check("a", now=0) == 1          # empty: wait 1 s
    assert limiter.check("b", now=0) == 0          # other clients have their own bucket
    assert limiter.check("a", now=1.0) == 0        # refilled


def test_keys_load_from_env_and_file(tmp_path):
    keys_file = tmp_path / "keys.txt"
    keys_file.write_text("# comment\nsk-tl-file\n\n")
    hashes = load_key_hashes(str(keys_file), env={"TINYLLAMA_API_KEYS": "sk-tl-a, sk-tl-b"})
    assert hashes == {hash_key("sk-tl-a"), hash_key("sk-tl-b"), hash_key("sk-tl-file")}
    assert load_key_hashes(env={}) == set()
    assert new_key().startswith("sk-tl-") and new_key() != new_key()


# ---- the web UI ---------------------------------------------------------------------------

def test_web_ui_is_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "<title>TinyLLaMA</title>" in page.text
    for asset in ("app.js", "style.css"):
        assert client.get(f"/static/{asset}").status_code == 200
