"""
Tests for continuous batching (inference/batching.py and the per-row positions in the model).
    python -m pytest tests/test_batching.py -v

The property that matters: putting a request in a batch must not change what it generates.
"""

import os

import pytest
import torch

from inference.batching import BatchEngine, Sequence
from inference.generate import Generator
from inference.sampling import SamplingParams
from model.config import get_config
from model.model import TinyLLaMA
from tokenizer.tokenizer import Tokenizer

TOKENIZER = os.path.join(os.path.dirname(__file__), "..", "tokenizer", "tinystories_4k.json")


@pytest.fixture(scope="module")
def tok():
    return Tokenizer.load(TOKENIZER)


def make_model(arch, block_size=64):
    torch.manual_seed(0)
    return TinyLLaMA(get_config("tiny", arch, block_size=block_size)).eval()


# ---- the model: one decode pass, every row at its own position --------------------------

@pytest.mark.parametrize("arch", ["llama", "gpt2"])  # RoPE + GQA, and learned positions + MHA
@torch.no_grad()
def test_per_row_positions_match_full_forward(arch):
    """Prefill 3 prompts of different lengths into rows of one cache, then decode them
    together. Each row's logits must match running that row's whole sequence without a cache."""
    model = make_model(arch)
    lengths = [5, 11, 17]
    seqs = [torch.randint(0, model.cfg.vocab_size, (1, n)) for n in lengths]
    caches = model.make_kv_caches(4)
    for c in caches:  # garbage in every row: stale data must be masked out, never read
        c.k.normal_()
        c.v.normal_()
    for row, s in enumerate(seqs):
        model(s, kv_caches=[c.rows(row, row + 1) for c in caches], start_pos=0)

    for _ in range(6):
        new = torch.randint(0, model.cfg.vocab_size, (3, 1))
        positions = torch.tensor([s.size(1) for s in seqs])
        batched, _ = model(new, kv_caches=[c.rows(0, 3) for c in caches], start_pos=positions)
        for row in range(3):
            seqs[row] = torch.cat([seqs[row], new[row:row + 1]], dim=1)
            reference, _ = model(seqs[row])  # no cache, whole sequence
            err = (batched[row] - reference[0]).abs().max() / reference.abs().max()
            assert err < 1e-5, f"row {row}: relative error {err:.2e}"


def test_copy_row_moves_a_request():
    model = make_model("llama")
    caches = model.make_kv_caches(3)
    c = caches[0]
    c.k[2].normal_()
    c.copy_row(2, 0, 10)
    assert torch.equal(c.k[0, :, :10], c.k[2, :, :10])


# ---- the engine: same outputs as generating alone ---------------------------------------

REQUESTS = [  # (prompt, params): different lengths, sampling methods and finish times
    ("Once upon a time", SamplingParams(max_new_tokens=30, temperature=0)),
    ("Lily had a red ball", SamplingParams(max_new_tokens=6, temperature=0.9, top_p=0.9, seed=1)),
    ("Tom", SamplingParams(max_new_tokens=40, temperature=1.0, top_k=20, seed=2)),
    ("The sun was hot", SamplingParams(max_new_tokens=3, temperature=0.7, repetition_penalty=1.3, seed=3)),
    ("A cat", SamplingParams(max_new_tokens=25, temperature=1.2, min_p=0.05, seed=4)),
    ("word " * 50, SamplingParams(max_new_tokens=100, temperature=0.8, seed=5)),  # hits the context limit
    ("Ben and Mia", SamplingParams(max_new_tokens=15, temperature=1.0, seed=6, stop_at_special=False)),
]


def run_engine(gen, requests, max_batch):
    engine = BatchEngine(gen, max_batch=max_batch)
    seqs = [Sequence(gen.encode_prompt(p), params) for p, params in requests]
    for s in seqs:
        engine.add(s)
    streamed = {id(s): [] for s in seqs}
    done = {}
    while engine.has_work():
        for seq, kind, value in engine.step():
            assert kind != "error", value
            if kind == "token":
                streamed[id(seq)].append(value)
            else:
                done[id(seq)] = value
    return [(done[id(s)], "".join(streamed[id(s)])) for s in seqs]


@pytest.mark.parametrize("arch", ["llama", "gpt2"])
@pytest.mark.parametrize("max_batch", [1, 3, 8])  # 3 < 7 requests: rows get freed and reused
def test_engine_matches_generating_alone(tok, arch, max_batch):
    gen = Generator(make_model(arch), tok)
    batched = run_engine(gen, REQUESTS, max_batch)
    for (prompt, params), (result, streamed) in zip(REQUESTS, batched):
        alone = gen.generate(prompt, params)
        assert result.token_ids == alone.token_ids, prompt
        assert result.text == alone.text == streamed
        assert result.finish_reason == alone.finish_reason
        if params.seed is not None:
            assert result.seed == params.seed
    assert batched[5][0].finish_reason == "context"


def test_unseeded_requests_report_a_seed_that_reproduces_them(tok):
    gen = Generator(make_model("llama"), tok)
    params = SamplingParams(max_new_tokens=20, temperature=1.0)
    (first, _), = run_engine(gen, [("Tom", params)], 4)
    again = gen.generate("Tom", SamplingParams(max_new_tokens=20, temperature=1.0, seed=first.seed))
    assert again.token_ids == first.token_ids


def test_engine_does_not_touch_global_randomness(tok):
    gen = Generator(make_model("llama"), tok)
    state = torch.get_rng_state()
    run_engine(gen, REQUESTS[:3], 3)
    assert torch.equal(state, torch.get_rng_state())


def test_abort_frees_the_row(tok):
    gen = Generator(make_model("llama"), tok)
    engine = BatchEngine(gen, max_batch=2)
    a = Sequence(gen.encode_prompt("Tom"), SamplingParams(max_new_tokens=50, temperature=0))
    b = Sequence(gen.encode_prompt("Lily"), SamplingParams(max_new_tokens=50, temperature=0))
    c = Sequence(gen.encode_prompt("Ben"), SamplingParams(max_new_tokens=5, temperature=0))
    for s in (a, b, c):
        engine.add(s)
    engine.step()
    assert engine.running == [a, b] and list(engine.waiting) == [c]
    engine.abort(a)  # b moves into row 0
    assert engine.running == [b]
    events = engine.step()
    assert engine.running == [b, c]
    assert all(seq is not a for seq, _, _ in events)


class FlakyModel(TinyLLaMA):
    """Fails on a chosen prompt (prefill) or on the next decode pass, on demand."""
    bad_prompt = None
    fail_next_decode = False

    def forward(self, idx, *args, **kwargs):
        if idx.size(1) > 1 and idx[0].tolist() == self.bad_prompt:
            raise RuntimeError("simulated prefill failure")
        if idx.size(1) == 1 and self.fail_next_decode:
            self.fail_next_decode = False
            raise RuntimeError("simulated decode failure")
        return super().forward(idx, *args, **kwargs)


def test_failures_are_contained(tok):
    torch.manual_seed(0)
    model = FlakyModel(get_config("tiny", "llama", block_size=64)).eval()
    gen = Generator(model, tok)
    model.bad_prompt = gen.encode_prompt("boom")
    engine = BatchEngine(gen, max_batch=4)

    # a prefill failure belongs to ONE request: the others in the batch carry on
    ok = Sequence(gen.encode_prompt("Tom"), SamplingParams(max_new_tokens=10, temperature=0))
    bad = Sequence(model.bad_prompt, SamplingParams(max_new_tokens=10))
    engine.add(ok)
    engine.add(bad)
    events = engine.step()
    assert [(s, k) for s, k, _ in events if k == "error"] == [(bad, "error")]
    assert engine.running == [ok]

    # a failure in the SHARED decode pass reaches everyone in it, and the engine recovers
    other = Sequence(gen.encode_prompt("Lily"), SamplingParams(max_new_tokens=10, temperature=0))
    engine.add(other)
    model.fail_next_decode = True
    events = engine.step()
    assert {s for s, k, _ in events if k == "error"} == {ok, other}
    assert not engine.has_work()

    fresh = Sequence(gen.encode_prompt("Ben"), SamplingParams(max_new_tokens=5, temperature=0))
    engine.add(fresh)
    kinds = []
    while engine.has_work():
        kinds += [k for _, k, _ in engine.step()]
    assert kinds[-1] == "done" and "error" not in kinds
