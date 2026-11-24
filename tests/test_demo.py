"""
Tests for the Gradio demo's logic (serve/demo.py). The UI itself isn't launched.
    python -m pytest tests/test_demo.py -v
"""

import os
import time

import pytest
import torch

pytest.importorskip("gradio")

from inference.generate import Generator  # noqa: E402
from inference.sampling import SamplingParams  # noqa: E402
from model.config import get_config  # noqa: E402
from model.model import TinyLLaMA  # noqa: E402
from serve.api import ModelWorker  # noqa: E402
from serve.demo import (COMPLETION, EXAMPLES, INSTRUCTION, build_demo, required_words,  # noqa: E402
                        stream_story, word_report)
from tokenizer.tokenizer import Tokenizer  # noqa: E402

TOKENIZER = os.path.join(os.path.dirname(__file__), "..", "tokenizer", "tinystories_4k.json")


def load_tiny_sft_model():
    torch.manual_seed(0)
    model = TinyLLaMA(get_config("tiny", "llama", block_size=64))
    return Generator(model, Tokenizer.load(TOKENIZER), is_sft=True)


@pytest.fixture(scope="module")
def worker():
    return ModelWorker(load_tiny_sft_model, max_waiting=4, max_batch=4)


@pytest.mark.parametrize("instruction, words", [
    ("Write a story. Use the words: rain, ball, happy.", ["rain", "ball", "happy"]),
    ("Write a story for children. Include these words: meet, waffle, new.", ["meet", "waffle", "new"]),
    ("Write a short story. The story must use the words call, leak, foolish. Also, include some dialogue.",
     ["call", "leak", "foolish"]),
    ("Use the words: sun, tree and bird", ["sun", "tree", "bird"]),
    ("Write a story about a cat.", []),
])
def test_required_words(instruction, words):
    assert required_words(instruction) == words


def test_every_example_asks_for_words():
    assert all(len(required_words(e)) == 3 for e in EXAMPLES)


def test_word_report_marks_used_and_missing_words():
    report = word_report("Use the words: rain, ball, happy.", "The rain fell. She was happy.")
    assert "2/3 used" in report and "✅ rain" in report and "❌ ball" in report and "✅ happy" in report
    assert word_report("Write a story.", "anything") == ""


def test_stream_story_grows_and_ends_with_stats(worker):
    instruction = "Use the words: rain, ball, happy."
    updates = list(stream_story(worker, instruction, INSTRUCTION, 0.8, 0.9, 30, 7))
    texts = [u[0] for u in updates]
    assert all(texts[i] == texts[i + 1][:len(texts[i])] for i in range(len(texts) - 1))  # only ever appends
    story, words, stats = updates[-1]
    assert "seed 7" in stats and "finished:" in stats
    assert words.startswith("**Required words:")
    # the same story as generating directly with the same seed
    alone = worker.generator.generate(instruction, SamplingParams(max_new_tokens=30, temperature=0.8,
                                                                  top_p=0.9, seed=7), chat=True)
    assert story == alone.text


def test_completion_mode_keeps_the_users_start(worker):
    story, words, _ = list(stream_story(worker, "Once upon a time", COMPLETION, 0.0, 1.0, 10, -1))[-1]
    assert story.startswith("Once upon a time") and words == ""


def test_empty_instruction_and_too_long_prompt_give_a_message(worker):
    assert "Type an instruction" in list(stream_story(worker, "  ", INSTRUCTION, 0.7, 0.9, 20, -1))[-1][2]
    too_long = "word " * 200  # far past the 64-token context
    assert "context" in list(stream_story(worker, too_long, COMPLETION, 0.7, 0.9, 20, -1))[-1][2]


def test_stopping_early_frees_the_slot(worker):
    """The Stop button closes the stream; the worker must drop the job."""
    stream = stream_story(worker, "Once upon a time", COMPLETION, 1.0, 1.0, 60, 1)
    next(stream)
    stream.close()
    for _ in range(200):
        if worker.running == 0 and worker.jobs.empty():
            break
        time.sleep(0.01)
    assert worker.running == 0 and worker.jobs.empty()


def test_build_demo_without_launching(worker):
    import gradio as gr
    assert isinstance(build_demo(worker, "test.pt"), gr.Blocks)
