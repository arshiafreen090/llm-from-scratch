"""
Tests for the BPE tokenizer. Run from the project root:
    python -m pytest tests/ -v
"""

import regex as re
import pytest

from tokenizer.tokenizer import (
    GPT4_SPLIT_PATTERN, Tokenizer, best_pair, get_pair_counts, merge,
)

TRAIN_TEXT = (
    "Once upon a time, there was a little girl named Lily. She loved to play outside. "
    "One day, Lily saw a big red ball. She said, \"I want to play with the ball!\" "
    "Her mom smiled and said, \"Let's go to the park.\" They played all day and were happy. "
) * 20


@pytest.fixture(scope="module")
def tok():
    t = Tokenizer()
    t.train(TRAIN_TEXT, vocab_size=256 + 3 + 100, verbose=False)
    return t


# ---- building blocks ----------------------------------------------------------

def test_get_pair_counts():
    assert get_pair_counts([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}


def test_merge():
    assert merge([1, 2, 3, 1, 2], (1, 2), 99) == [99, 3, 99]
    assert merge([1, 1, 1], (1, 1), 99) == [99, 1]  # left to right, no overlaps


# ---- training ---------------------------------------------------------------

def test_wikipedia_example():
    """The classic example: aaabdaaabac -> aa=Z, ab=Y, ZY=X -> X d X a c"""
    t = Tokenizer(special_tokens=[])
    t.train("aaabdaaabac", vocab_size=256 + 3, verbose=False)
    a, b = ord("a"), ord("b")
    assert list(t.merges) == [(a, a), (a, b), (256, 257)]
    assert t.encode("aaabdaaabac") == [258, ord("d"), 258, a, ord("c")]


def naive_train(text, num_merges):
    """Textbook BPE: recount every pair in every chunk after each merge. Slow but obviously correct."""
    words = [list(c.encode("utf-8")) for c in re.findall(GPT4_SPLIT_PATTERN, text)]
    merges = []
    for step in range(num_merges):
        counts = {}
        for w in words:
            get_pair_counts(w, counts)
        if not counts:
            break
        pair = best_pair(counts)
        merges.append(pair)
        words = [merge(w, pair, 256 + step) for w in words]
    return merges


def test_fast_training_matches_naive():
    """Our fast trainer (dedup + incremental counts) must learn exactly the same merges."""
    t = Tokenizer(special_tokens=[])
    t.train(TRAIN_TEXT, vocab_size=256 + 80, verbose=False)
    assert list(t.merges) == naive_train(TRAIN_TEXT, 80)


def test_vocab_size(tok):
    assert tok.vocab_size == 256 + 3 + 100
    assert len(tok.merges) == 100


# ---- encode / decode ----------------------------------------------------------

@pytest.mark.parametrize("text", [
    "",
    "Once upon a time, there was a little girl named Lily.",
    "Words never seen in training: xylophone, quantum, Kubernetes!",
    "Numbers 1234567 and   multiple   spaces\n\nand newlines\ttabs",
    "Unicode: नमस्ते दुनिया, 你好, emoji 🦙🚀, accents café",
    "def f(x):\n    return x ** 2  # code",
])
def test_round_trip(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_compresses(tok):
    """After training, text from the training distribution needs fewer tokens than bytes."""
    text = "Once upon a time, there was a little girl named Lily."
    assert len(tok.encode(text)) < len(text.encode("utf-8")) / 2


def test_special_tokens(tok):
    text = "The end.<|endoftext|>Once upon a time"
    ids = tok.encode(text)
    assert ids.count(tok.eot_id) == 1          # one special id, not 13 bytes
    assert tok.decode(ids) == text
    # with allow_special=False it's just ordinary text
    assert tok.eot_id not in tok.encode(text, allow_special=False)


def test_special_ids_come_after_merges(tok):
    assert tok.special_tokens == {"<|endoftext|>": 356, "<|user|>": 357, "<|assistant|>": 358}


def test_decode_invalid_utf8_does_not_crash(tok):
    # 0xF0 alone is the start of a 4-byte character with the rest missing
    assert tok.decode([0xF0]) == "�"


# ---- save / load ------------------------------------------------------------

def test_save_load(tok, tmp_path):
    path = str(tmp_path / "tok.json")
    tok.save(path)
    loaded = Tokenizer.load(path)
    assert loaded.merges == tok.merges
    assert loaded.special_tokens == tok.special_tokens
    text = "Lily saw a big red ball.<|endoftext|>🦙"
    assert loaded.encode(text) == tok.encode(text)
    assert (tmp_path / "tok.vocab.txt").exists()
