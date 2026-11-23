"""
Tests for the training loop and the experiment controls.
    python -m pytest tests/test_train.py -v
"""

import copy
import json
import math
import os
import random

import numpy as np
import pytest
import torch

from experiments.compare import check_controls, describe_changes, group_by_model
from tokenizer.tokenizer import Tokenizer
from train.pretrain import TrainBatches, TrainConfig, fixed_val_starts, train, windows

TOKENIZER = os.path.join(os.path.dirname(__file__), "..", "tokenizer", "tinystories_4k.json")


def make_stories(n, seed):
    rng = random.Random(seed)
    names, things = ["Lily", "Tom", "Max", "Sue"], ["ball", "cat", "kite", "cake", "tree"]
    stories = []
    for _ in range(n):
        a, b = rng.choice(names), rng.choice(things)
        stories.append(f"Once upon a time, {a} had a {b}. {a} loved the {b} very much. "
                       f"One day, the {b} was gone. {a} was sad, but then {a} found it. The end.")
    return stories


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    """A small dataset in the same format data/prepare_pretrain.py writes."""
    d = tmp_path_factory.mktemp("data")
    tok = Tokenizer.load(TOKENIZER)
    for name, n, seed in [("train", 3000, 0), ("val", 300, 1)]:
        ids = []
        for s in make_stories(n, seed):
            ids += tok.encode_ordinary(s) + [tok.eot_id]
        np.array(ids, dtype=np.uint16).tofile(d / f"{name}.bin")
    (d / "meta.json").write_text(json.dumps({"tokenizer": TOKENIZER, "vocab_size": tok.vocab_size}))
    return str(d)


def tiny_cfg(data_dir, tmp_path, **kw):
    base = dict(data_dir=data_dir, out_dir=str(tmp_path / "results"), ckpt_dir=str(tmp_path / "ckpt"),
                size="tiny", block_size=64, batch_size=8, grad_accum=1, token_budget=512 * 80,
                lr=3e-3, eval_every=40, eval_windows=20, final_eval_windows=40, eval_batch_size=8)
    base.update(kw)
    return TrainConfig(**base)


# ---- controls --------------------------------------------------------------------

def test_batches_do_not_depend_on_global_randomness(data_dir):
    """Building a different model uses the global RNG differently. The data order must not change."""
    path = os.path.join(data_dir, "train.bin")
    a = TrainBatches(path, 64, 8, seed=42)
    torch.manual_seed(0)
    first_a = [a.next()[0] for _ in range(3)]

    b = TrainBatches(path, 64, 8, seed=42)
    torch.randn(1000)            # simulate a model whose init consumed more random numbers
    np.random.rand(1000)
    first_b = [b.next()[0] for _ in range(3)]
    for xa, xb in zip(first_a, first_b):
        assert torch.equal(xa, xb)


def test_targets_are_inputs_shifted_by_one(data_dir):
    data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    x, y = windows(data, np.array([0, 100]), 64)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_val_windows_are_fixed_and_dont_overlap():
    a = fixed_val_starts(100_000, 512, 50)
    assert np.array_equal(a, fixed_val_starts(100_000, 512, 50))  # deterministic
    assert np.all(np.diff(a) >= 512)                              # no overlap
    assert a[-1] + 512 + 1 <= 100_000                              # stays inside the data


# ---- end-to-end training ---------------------------------------------------------------

@pytest.fixture(scope="module")
def finished_run(data_dir, tmp_path_factory):
    torch.manual_seed(0)
    return train(tiny_cfg(data_dir, tmp_path_factory.mktemp("run"), run_name="base"))


def test_training_reduces_loss(finished_run):
    log_path = os.path.join(finished_run["train_config"]["out_dir"], "base", "log.jsonl")
    evals = [json.loads(l) for l in open(log_path) if "val_loss" in l]
    assert evals[0]["val_loss"] > 7.5                       # starts near ln(4096) = 8.3
    assert finished_run["final_val_loss"] < 0.5 * evals[0]["val_loss"]


def test_run_record_is_complete(finished_run):
    for key in ["params", "tokens_trained", "tokenizer", "dtype", "gpu", "git_commit", "final_val_bpb", "model_config"]:
        assert key in finished_run
    assert finished_run["tokens_trained"] == 80 * 512
    assert os.path.exists(os.path.join(finished_run["train_config"]["ckpt_dir"], "base.pt"))


def test_bpb_matches_loss(finished_run, data_dir):
    """bpb = loss (nats/token) / ln 2 / (bytes per token). Check the two metrics agree."""
    tok = Tokenizer.load(TOKENIZER)
    val = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    starts = fixed_val_starts(len(val), 64, 40)
    _, y = windows(val, starts, 64)
    special = set(tok.special_tokens.values())
    n_bytes = sum(0 if int(t) in special else len(tok.vocab[int(t)]) for t in y.view(-1))
    expected = finished_run["final_val_loss"] / math.log(2) / (n_bytes / y.numel())
    assert finished_run["final_val_bpb"] == pytest.approx(expected, rel=1e-6)


# ---- the comparison script catches uncontrolled experiments --------------------------

def variant(run, **model_changes):
    r = copy.deepcopy(run)
    r["model_config"].update(model_changes)
    return r


def test_compare_accepts_a_controlled_ablation(finished_run):
    runs = [finished_run, variant(finished_run, pos_emb="learned")]
    assert check_controls(runs) == []
    groups = group_by_model(runs)
    assert len(groups) == 2
    assert describe_changes(groups[1][0]["model_config"], groups[0][0]["model_config"]) == {"pos_emb": "learned"}


def test_compare_flags_a_different_training_budget(finished_run):
    other = copy.deepcopy(finished_run)
    other["train_config"]["lr"] = 1e-4
    other["tokens_trained"] = 999
    problems = check_controls([finished_run, other])
    assert any("'lr'" in p for p in problems)
    assert any("tokens_trained" in p for p in problems)


def test_compare_flags_a_different_tokenizer(finished_run):
    other = copy.deepcopy(finished_run)
    other["tokenizer"] = "tokenizer/tinystories_1k.json"
    assert any("tokenizer" in p for p in check_controls([finished_run, other]))


def test_seeds_are_grouped_together(finished_run):
    other_seed = copy.deepcopy(finished_run)
    other_seed["train_config"]["seed"] = 2
    assert len(group_by_model([finished_run, other_seed])) == 1


# ---- precision choice -----------------------------------------------------------------

@pytest.mark.parametrize("capability,expected", [
    ((7, 5), "float16"),    # T4: bf16 only emulated -> fp16 on the tensor cores
    ((6, 0), "float16"),    # P100
    ((8, 0), "bfloat16"),   # A100
    ((8, 9), "bfloat16"),   # RTX 40xx / L4
])
def test_auto_dtype_uses_bf16_only_where_native(monkeypatch, capability, expected):
    from train import pretrain
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: capability)
    # recent PyTorch says True on a T4 too (emulation): the choice must not depend on it
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)
    assert pretrain.resolve_device_dtype(TrainConfig(device="cuda")) == ("cuda", expected)
    assert pretrain.resolve_device_dtype(TrainConfig(device="cuda", dtype="bfloat16"))[1] == "bfloat16"


def test_auto_dtype_on_cpu_is_float32():
    assert TrainConfig().dtype == "auto"
    from train.pretrain import resolve_device_dtype
    assert resolve_device_dtype(TrainConfig(device="cpu")) == ("cpu", "float32")


def test_runner_refuses_to_mix_precisions(tmp_path):
    from experiments.run_ablations import dtype_conflicts
    for name, dtype in [("small-gpt2-s1", "bfloat16"), ("small-llama-s1", "bfloat16")]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "final.json").write_text(json.dumps({"dtype": dtype}))
    (tmp_path / "unfinished-run").mkdir()  # no final.json: ignored
    assert dtype_conflicts(str(tmp_path), "bfloat16") == []
    assert dtype_conflicts(str(tmp_path), "float16") == [("small-gpt2-s1", "bfloat16"), ("small-llama-s1", "bfloat16")]
    assert dtype_conflicts(str(tmp_path / "missing"), "float16") == []
