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
    params = SamplingParams(temperature=0, max_new_tokens=30, stop_at_special=False)
    result = gen.generate("Once upon a time", params)
    prompt = torch.tensor([gen.encode_prompt("Once upon a time")])
    reference = gen.model.generate(prompt, 30, temperature=0)[0, prompt.size(1):].tolist()
    assert result.token_ids == reference


def test_seed_makes_sampling_reproducible(gen):
    params = SamplingParams(temperature=1.0, top_p=0.95, max_new_tokens=20, seed=7, stop_at_special=False)
    assert gen.generate("Tom", params).token_ids == gen.generate("Tom", params).token_ids


def test_stream_pieces_add_up_to_full_text(gen):
    params = SamplingParams(temperature=1.0, max_new_tokens=40, seed=3, stop_at_special=False)
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
    params = SamplingParams(temperature=0, max_new_tokens=1000, stop_at_special=False)
    result = gen.generate("Once upon a time", params)
    assert len(result.token_ids) == 128 - prompt_len
    assert result.finish_reason == "context"


def test_latency_stats_are_filled(gen):
    result = gen.generate("Once", SamplingParams(temperature=0, max_new_tokens=10, stop_at_special=False))
    assert result.finish_reason == "length"
    assert result.ttft_ms > 0 and result.decode_tokens_per_s > 0 and result.prompt_tokens == 2


# ---- tokenizer-specific decoding ------------------------------------------------------

class ScriptedModel(torch.nn.Module):
    """Stands in for the model: emits a fixed sequence of tokens, one per forward call."""

    def __init__(self, cfg, script):
        super().__init__()
        self.cfg, self.script, self.calls = cfg, list(script), 0
        self.dummy = torch.nn.Parameter(torch.zeros(1))  # so .parameters() has a device

    def make_kv_caches(self, batch_size):
        return None

    def forward(self, idx, targets=None, kv_caches=None, start_pos=0):
        token = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        logits = torch.full((idx.size(0), 1, self.cfg.vocab_size), -1e9)
        logits[..., token] = 0.0
        return logits, None


def scripted_generator(gen, script):
    return Generator(ScriptedModel(gen.model.cfg, script), gen.tok)


@pytest.mark.parametrize("special", ["<|user|>", "<|assistant|>", "<|endoftext|>"])
def test_any_special_token_ends_the_reply(gen, special):
    """After instruction tuning, <|user|> means "the assistant is done". It must stop the
    reply, and a control token must never reach the user as visible text."""
    script = gen.tok.encode(" The end.") + [gen.tok.special_tokens[special]] + gen.tok.encode(" Hello")
    result = scripted_generator(gen, script).generate("x", SamplingParams(temperature=0, max_new_tokens=20))
    assert result.text == " The end."
    assert result.finish_reason == "stop"


def test_special_tokens_are_never_streamed_even_when_not_stopping(gen):
    script = gen.tok.encode(" A") + [gen.tok.special_tokens["<|user|>"]] + gen.tok.encode(" B")
    params = SamplingParams(temperature=0, max_new_tokens=3, stop_at_special=False)
    result = scripted_generator(gen, script).generate("x", params)
    assert "<|" not in result.text
    assert result.text.startswith(" A")


def test_streaming_equals_bulk_decoding_for_any_bytes(gen):
    """Randomized check: for arbitrary token sequences, including byte tokens that form
    INVALID UTF-8, the streamed pieces must join to exactly what decode() gives in one go.
    (A naive streamer that decodes each token separately fails this.)"""
    rng = torch.Generator().manual_seed(0)
    ordinary = gen.tok.vocab_size - len(gen.tok.special_tokens)

    class FakeStream(Generator):
        def __init__(self, base, ids):
            super().__init__(base.model, base.tok)
            self.ids = ids

        def stream_ids(self, prompt_ids, params, result):
            for t in self.ids:
                result.token_ids.append(t)
                yield t

    for _ in range(300):
        n = int(torch.randint(1, 30, (1,), generator=rng))
        # half the time only raw byte tokens (0..255), which often form invalid UTF-8
        high = 256 if torch.rand(1, generator=rng) < 0.5 else ordinary
        ids = torch.randint(0, high, (n,), generator=rng).tolist()
        pieces = list(FakeStream(gen, ids).stream("x"))
        assert "".join(pieces) == gen.tok.decode(ids)
