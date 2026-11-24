"""
Tests for the text metrics in experiments/sampling_sweep.py.
    python -m pytest tests/test_sampling_sweep.py -v
"""

import pytest

from experiments.sampling_sweep import diversity, repetition


def test_repetition_counts_repeated_4grams():
    assert repetition("the dog ran to the park and sat down") == 0.0
    # "the dog was happy" appears twice: 2 of the 5 four-word windows are repeats
    assert repetition("the dog was happy the dog was happy") == pytest.approx(1 - 4 / 5)
    assert repetition("too short") == 0.0


def test_diversity_is_low_when_every_story_is_the_same():
    same = ["once upon a time there was a cat"] * 4
    different = ["once upon a time there was a cat", "a big dog ran to the red ball",
                 "lily and tom played in the sun", "the bird sang a happy song today"]
    assert diversity(same) == pytest.approx(0.25)  # each 3-gram distinct once, seen 4 times
    assert diversity(different) == 1.0
    assert diversity([]) == 0.0
