"""
Tests for sampling and generation.
    python -m pytest tests/test_inference.py -v
"""

import math
import os

import pytest
import torch

from inference.generate import GenerationResult, Generator
from inference.sampling import (
    SamplingParams, apply_min_p, apply_repetition_penalty, apply_top_k, apply_top_p, sample_next,
)
from model.config import get_config
from model.model import TinyLLaMA
from tokenizer.tokenizer import Tokenizer

TOKENIZER = os.path.join(os.path.dirname(__file__), "..", "tokenizer", "tinystories_4k.json")


def logits_from_probs(probs):
    return torch.log(torch.tensor([probs]))


def kept(logits):
    """Indices of tokens that can still be sampled."""
    return torch.isfinite(logits[0]).nonzero().flatten().tolist()


# ---- individual sampling methods ----------------------------------------------------

def test_top_k_keeps_k_best():
    logits = logits_from_probs([0.1, 0.4, 0.2, 0.3])
    assert sorted(kept(apply_top_k(logits, 2))) == [1, 3]


def test_top_p_worked_example():
    """The example from the docstring: probs [0.5, 0.3, 0.15, 0.05], p = 0.7 keeps 2 tokens."""
    logits = logits_from_probs([0.15, 0.5, 0.05, 0.3])  # shuffled, to check the un-sorting
    assert sorted(kept(apply_top_p(logits, 0.7))) == [1, 3]


def test_top_p_always_keeps_the_best_token():
    logits = logits_from_probs([0.97, 0.01, 0.01, 0.01])
    assert kept(apply_top_p(logits, 0.1)) == [0]


def test_top_p_adapts_to_confidence():
    confident = logits_from_probs([0.92, 0.04, 0.02, 0.02])
    unsure = logits_from_probs([0.28, 0.26, 0.24, 0.22])
    assert len(kept(apply_top_p(confident, 0.9))) == 1
    assert len(kept(apply_top_p(unsure, 0.9))) == 4


def test_min_p():
    # best token has p = 0.5, so min_p = 0.3 keeps tokens with p >= 0.15
    logits = logits_from_probs([0.5, 0.3, 0.15, 0.05])
    assert kept(apply_min_p(logits, 0.3)) == [0, 1, 2]


def test_repetition_penalty_lowers_seen_tokens_whatever_their_sign():
    logits = torch.tensor([[2.0, -2.0, 1.0]])
    history = torch.tensor([[0, 1]])
    out = apply_repetition_penalty(logits, history, 2.0)
    assert out[0, 0] == 1.0    # positive: divided
    assert out[0, 1] == -4.0   # negative: multiplied (dividing would have RAISED it)
    assert out[0, 2] == 1.0    # not seen: untouched


def test_greedy_is_argmax():
    logits = torch.randn(3, 50)
    nxt = sample_next(logits, torch.empty(3, 0, dtype=torch.long), SamplingParams(temperature=0))
    assert torch.equal(nxt.flatten(), logits.argmax(-1))


def test_low_temperature_approaches_greedy():
    logits = torch.tensor([[1.0, 1.5, 0.5]])
    g = torch.Generator().manual_seed(0)
    picks = {int(sample_next(logits, torch.empty(1, 0, dtype=torch.long), SamplingParams(temperature=0.01), g))
             for _ in range(50)}
    assert picks == {1}


# ---- generator --------------------------------------------------------------------

@pytest.fixture(scope="module")
def gen():
    torch.manual_seed(0)
    model = TinyLLaMA(get_config("tiny", "llama", block_size=128))
    return Generator(model, Tokenizer.load(TOKENIZER))


def test_greedy_matches_model_generate(gen):
    """Our engine and the simple model.generate() must agree token for token."""
    params = SamplingParams(temperature=0, max_new_tokens=30, stop_at_eot=False)
    result = gen.generate("Once upon a time", params)
    prompt = torch.tensor([gen.encode_prompt("Once upon a time")])
    reference = gen.model.generate(prompt, 30, temperature=0)[0, prompt.size(1):].tolist()
    assert result.token_ids == reference


def test_seed_makes_sampling_reproducible(gen):
    params = SamplingParams(temperature=1.0, top_p=0.95, max_new_tokens=20, seed=7, stop_at_eot=False)
    assert gen.generate("Tom", params).token_ids == gen.generate("Tom", params).token_ids


def test_stream_pieces_add_up_to_full_text(gen):
    params = SamplingParams(temperature=1.0, max_new_tokens=40, seed=3, stop_at_eot=False)
    result = GenerationResult(text="")
    pieces = list(gen.stream("Lily", params, result))
    assert "".join(pieces) == result.text == gen.tok.decode(result.token_ids)


def test_stream_never_emits_half_a_character(gen):
    """Force the model's output to be the bytes of an emoji, one byte token at a time."""
    emoji_bytes = list("🦙".encode("utf-8"))  # 4 byte-level tokens (ids 0..255)

    class FakeStream(Generator):
        def stream_ids(self, prompt_ids, params, result):
            for b in emoji_bytes:
                result.token_ids.append(b)
                yield b

    fake = FakeStream(gen.model, gen.tok)
    pieces = list(fake.stream("x"))
    assert pieces == ["🦙"]  # one complete character, no broken pieces before it


def test_stops_at_context_limit(gen):
    prompt_len = len(gen.encode_prompt("Once upon a time"))
    params = SamplingParams(temperature=0, max_new_tokens=1000, stop_at_eot=False)
    result = gen.generate("Once upon a time", params)
    assert len(result.token_ids) == 128 - prompt_len
    assert result.finish_reason == "context"


def test_latency_stats_are_filled(gen):
    result = gen.generate("Once", SamplingParams(temperature=0, max_new_tokens=10, stop_at_eot=False))
    assert result.finish_reason == "length"
    assert result.ttft_ms > 0 and result.decode_tokens_per_s > 0 and result.prompt_tokens == 2
