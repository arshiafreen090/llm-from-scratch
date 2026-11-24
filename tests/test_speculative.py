"""
Tests for speculative decoding (inference/speculative.py).
    python -m pytest tests/test_speculative.py -v

The property that matters: speculative decoding changes the speed, never the output.
    greedy:   token-for-token identical to plain greedy decoding
    sampling: every token distributed exactly as the target model's distribution
"""

import os

import pytest
import torch

from inference.generate import Generator
from inference.sampling import SamplingParams
from inference.speculative import ModelDrafter, NgramDrafter, SpeculativeGenerator, verify
from model.config import get_config
from model.model import TinyLLaMA
from tokenizer.tokenizer import Tokenizer

TOKENIZER = os.path.join(os.path.dirname(__file__), "..", "tokenizer", "tinystories_4k.json")


@pytest.fixture(scope="module")
def tok():
    return Tokenizer.load(TOKENIZER)


def make_model(seed, size="tiny", block_size=64):
    torch.manual_seed(seed)
    return TinyLLaMA(get_config(size, "llama", block_size=block_size)).eval()


# ---- the accept / resample rule, on small made-up distributions ----------------------------

def random_dist(n, rng):
    return torch.softmax(torch.randn(n, generator=rng) * 2, dim=0)


@pytest.mark.parametrize("q_kind", ["model", "one-hot"])
def test_first_token_follows_the_target_distribution(q_kind):
    """Whatever the drafter proposes, the first token out of a round is distributed as p_1."""
    rng = torch.Generator().manual_seed(0)
    V, k, n = 6, 3, 100_000  # sampling error of a frequency: at most 0.0016 (1 std)
    p = torch.stack([random_dist(V, rng) for _ in range(k + 1)])
    q = torch.stack([random_dist(V, rng) for _ in range(k)])  # deliberately different from p
    counts = torch.zeros(V)
    for _ in range(n):
        if q_kind == "model":
            draft = [int(torch.multinomial(q[i], 1, generator=rng)) for i in range(k)]
            q_used = q
        else:  # a fixed guess (like the n-gram drafter): all of q's mass on it
            draft = [2, 4, 1]
            q_used = torch.nn.functional.one_hot(torch.tensor(draft), V).float()
        new, _ = verify(draft, q_used, p, rng)
        counts[new[0]] += 1
    # 5 std of slack; a wrong rule (e.g. accepting every guess) is off by ~0.1 or more
    assert (counts / n - p[0]).abs().max() < 0.008


def test_greedy_rule_accepts_exactly_the_argmax():
    rng = torch.Generator().manual_seed(0)
    one_hot = lambda ids: torch.nn.functional.one_hot(torch.tensor(ids), 5).float()  # noqa: E731
    p = one_hot([3, 1, 4, 0])            # the target's greedy choices
    assert verify([3, 1, 4], one_hot([3, 1, 4]), p, rng) == ([3, 1, 4, 0], 3)  # all right + bonus
    assert verify([3, 2, 4], one_hot([3, 2, 4]), p, rng) == ([3, 1], 1)        # 2nd wrong -> target's
    assert verify([0, 1, 4], one_hot([0, 1, 4]), p, rng) == ([3], 0)


def test_a_perfect_drafter_is_always_accepted():
    rng = torch.Generator().manual_seed(0)
    p = torch.stack([random_dist(8, rng) for _ in range(4)])
    for _ in range(200):
        draft = [int(torch.multinomial(p[i], 1, generator=rng)) for i in range(3)]
        new, accepted = verify(draft, p[:3], p, rng)
        assert accepted == 3 and new[:3] == draft


# ---- the whole generator ---------------------------------------------------------------------

def plain_and_speculative(tok, drafter, k):
    target = make_model(0)
    return Generator(target, tok), SpeculativeGenerator(target, tok, drafter, k)


GREEDY_CASES = [
    ("Once upon a time", SamplingParams(max_new_tokens=40, temperature=0)),
    ("Lily had a red ball", SamplingParams(max_new_tokens=30, temperature=0, repetition_penalty=1.3)),
    ("Tom", SamplingParams(max_new_tokens=25, temperature=0, stop_at_special=False)),
    ("word " * 40, SamplingParams(max_new_tokens=100, temperature=0)),  # runs into the 64-token context
]


@pytest.mark.parametrize("drafter_kind", ["other model", "same model", "ngram"])
@pytest.mark.parametrize("k", [1, 3, 6])
def test_greedy_output_is_identical_to_plain_greedy(tok, drafter_kind, k):
    drafter = {"other model": lambda: ModelDrafter(make_model(1)),
               "same model": lambda: ModelDrafter(make_model(0)),
               "ngram": lambda: NgramDrafter()}[drafter_kind]()
    plain, spec = plain_and_speculative(tok, drafter, k)
    for prompt, params in GREEDY_CASES:
        a, b = plain.generate(prompt, params), spec.generate(prompt, params)
        assert b.token_ids == a.token_ids, prompt
        assert b.text == a.text and b.finish_reason == a.finish_reason
        assert spec.stats.tokens <= params.max_new_tokens
        if drafter_kind == "same model":
            assert spec.stats.acceptance == 1.0  # the draft IS the target


def test_sampling_with_the_target_as_drafter_accepts_everything(tok):
    """p == q, so every guess passes, at any temperature."""
    plain, spec = plain_and_speculative(tok, ModelDrafter(make_model(0)), 4)
    spec.generate("Once upon a time", SamplingParams(max_new_tokens=40, temperature=0.9, top_p=0.9, seed=3))
    assert spec.stats.acceptance == 1.0
    assert spec.stats.tokens_per_round > 4


def test_sampling_is_reproducible_with_a_seed(tok):
    _, spec = plain_and_speculative(tok, ModelDrafter(make_model(1)), 4)
    params = SamplingParams(max_new_tokens=40, temperature=1.0, seed=11)
    a, b = spec.generate("Tom", params), spec.generate("Tom", params)
    assert a.token_ids == b.token_ids and a.seed == 11


def test_ngram_drafter_proposes_what_followed_last_time():
    d = NgramDrafter(max_n=3, min_n=2)
    d.vocab_size = 100
    tokens = [5, 6, 7, 8, 9, 1, 2, 5, 6, 7]
    guess, q = d.propose(tokens, 3, SamplingParams(), None)
    assert guess == [8, 9, 1] and q.argmax(-1).tolist() == [8, 9, 1]
    assert d.propose([1, 2, 3, 4], 3, SamplingParams(), None) == ([], None)  # nothing repeats
