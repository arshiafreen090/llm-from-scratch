"""
Tests for the HTTP API (no real server needed: FastAPI's TestClient calls the app directly).
    python -m pytest tests/test_api.py -v
"""

import json
import os
import threading

import pytest
import torch
from fastapi.testclient import TestClient

from inference.generate import Generator
from model.config import get_config
from model.model import TinyLLaMA
from serve.api import create_app
from tokenizer.tokenizer import Tokenizer

TOKENIZER = os.path.join(os.path.dirname(__file__), "..", "tokenizer", "tinystories_4k.json")


@pytest.fixture(scope="module")
def generator():
    torch.manual_seed(0)
    return Generator(TinyLLaMA(get_config("tiny", "llama", block_size=128)), Tokenizer.load(TOKENIZER))


@pytest.fixture(scope="module")
def client(generator):
    return TestClient(create_app(generator))


def read_sse(response):
    """Parse a Server-Sent Events body into a list of JSON events (and the final [DONE])."""
    events = []
    for line in response.iter_lines():
        if line.startswith("data: "):
            data = line[len("data: "):]
            events.append(data if data == "[DONE]" else json.loads(data))
    return events


def test_health_and_info(client):
    assert client.get("/health").json()["status"] == "ok"
    info = client.get("/info").json()
    assert info["context_length"] == 128 and info["pos_emb"] == "rope"


def test_generate(client):
    r = client.post("/generate", json={"prompt": "Once upon a time", "max_new_tokens": 20, "temperature": 0})
    assert r.status_code == 200
    body = r.json()
    assert body["usage"]["completion_tokens"] <= 20
    assert body["finish_reason"] in ("length", "stop")
    assert body["latency"]["ttft_ms"] > 0


def test_stream_matches_non_stream(client):
    req = {"prompt": "Lily", "max_new_tokens": 25, "temperature": 0.9, "seed": 5}
    full = client.post("/generate", json=req).json()
    with client.stream("POST", "/generate", json={**req, "stream": True}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        events = read_sse(r)
    tokens = [e["text"] for e in events if isinstance(e, dict) and e["type"] == "token"]
    done = next(e for e in events if isinstance(e, dict) and e["type"] == "done")
    assert events[-1] == "[DONE]"
    assert "".join(tokens) == done["text"] == full["text"]


def test_invalid_requests_are_rejected(client):
    assert client.post("/generate", json={"prompt": "hi", "temperature": 5}).status_code == 422
    assert client.post("/generate", json={"temperature": 0.5}).status_code == 422  # no prompt
    long_prompt = "word " * 300  # more tokens than the 128-token context
    r = client.post("/generate", json={"prompt": long_prompt})
    assert r.status_code == 400 and "context" in r.json()["detail"]


def test_model_is_freed_after_client_stops_reading(client):
    """A client that disconnects mid-stream must not leave the model locked."""
    with client.stream("POST", "/generate", json={"prompt": "Tom", "stream": True, "max_new_tokens": 100}) as r:
        for line in r.iter_lines():
            if line.startswith("data: "):
                break  # read one event, then hang up
    r = client.post("/generate", json={"prompt": "Tom", "max_new_tokens": 5})
    assert r.status_code == 200
    assert client.get("/health").json()["busy"] is False


def test_full_queue_returns_503(generator):
    client = TestClient(create_app(generator, max_waiting=0))
    assert client.post("/generate", json={"prompt": "hi", "max_new_tokens": 2}).status_code == 503
    assert client.post("/generate", json={"prompt": "hi", "stream": True}).status_code == 503


def test_unseeded_requests_report_a_seed_that_reproduces_them(client):
    req = {"prompt": "Once upon a time", "max_new_tokens": 20, "temperature": 1.0}
    first = client.post("/generate", json=req).json()
    again = client.post("/generate", json={**req, "seed": first["seed"]}).json()
    assert again["text"] == first["text"]


def test_requests_do_not_touch_global_randomness(generator):
    """Sampling must use the request's own generator, never PyTorch's global RNG, so one
    request (or any other code) can't change what another request generates."""
    from inference.sampling import SamplingParams
    state = torch.get_rng_state()
    generator.generate("Tom", SamplingParams(temperature=1.0, max_new_tokens=10))
    assert torch.equal(state, torch.get_rng_state())


class FailingGenerator(Generator):
    """Crashes on prompts containing "boom", works normally otherwise."""

    def stream(self, prompt, params=None, result=None):
        if "boom" in prompt:
            raise RuntimeError("simulated failure inside generation")
        yield from super().stream(prompt, params, result)


def test_one_failing_request_does_not_affect_others(generator):
    client = TestClient(create_app(FailingGenerator(generator.model, generator.tok)))
    bad = client.post("/generate", json={"prompt": "boom", "max_new_tokens": 5})
    assert bad.status_code == 500 and "simulated failure" in bad.json()["detail"]

    with client.stream("POST", "/generate", json={"prompt": "boom", "stream": True}) as r:
        events = read_sse(r)
    assert events[-1]["type"] == "error"

    good = client.post("/generate", json={"prompt": "Lily", "max_new_tokens": 5})
    assert good.status_code == 200 and good.json()["usage"]["completion_tokens"] > 0
    assert client.get("/health").json()["busy"] is False


def test_concurrent_requests_queue_up(client):
    """Two requests at once: one runs, the other waits for the model, and reports that wait."""
    results = []

    def call():
        r = client.post("/generate", json={"prompt": "Once", "max_new_tokens": 60, "temperature": 0})
        results.append(r.json())

    threads = [threading.Thread(target=call) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 2
    waits = sorted(r["latency"]["queue_ms"] for r in results)
    assert waits[1] > 0.5 * min(r["latency"]["total_ms"] for r in results)
