"""
Tests for INT8 quantization (inference/quantize.py).
    python -m pytest tests/test_quantize.py -v
"""

import dataclasses
import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from inference.batching import BatchEngine, Sequence
from inference.generate import Generator
from inference.quantize import (Int8Embedding, Int8Linear, dequantize_tensor, int8_engine, quantize_checkpoint,
                                quantize_model, quantize_tensor, weight_bytes)
from inference.sampling import SamplingParams
from model.config import get_config
from model.model import TinyLLaMA
from tokenizer.tokenizer import Tokenizer

TOKENIZER = os.path.join(os.path.dirname(__file__), "..", "tokenizer", "tinystories_4k.json")


def has_int8_kernel():
    try:
        int8_engine()
        return True
    except RuntimeError:
        return False


needs_kernel = pytest.mark.skipif(not has_int8_kernel(), reason="no int8 CPU engine in this PyTorch build")


@pytest.fixture(scope="module")
def tok():
    return Tokenizer.load(TOKENIZER)


def make_model(arch="llama", block_size=64):
    torch.manual_seed(0)
    return TinyLLaMA(get_config("tiny", arch, block_size=block_size)).eval()


def quantized_copy(model, mode="int8"):
    copy = TinyLLaMA(model.cfg).eval()
    copy.load_state_dict(model.state_dict())
    return quantize_model(copy, mode)


# ---- the quantizer ----------------------------------------------------------------------

def test_rounding_error_is_at_most_half_a_step():
    torch.manual_seed(0)
    w = torch.randn(64, 48) * torch.logspace(-3, 0, 64)[:, None]  # rows with very different ranges
    w[5] = 0                                                       # and an all-zero row
    q, scale = quantize_tensor(w)
    assert q.dtype == torch.int8 and q.abs().max() <= 127
    err = (dequantize_tensor(q, scale) - w).abs()
    assert (err <= scale[:, None] / 2 + 1e-7).all()
    assert (q[5] == 0).all()
    assert (q.abs().amax(dim=1)[torch.arange(64) != 5] == 127).all()  # every row uses the full range


def test_per_row_scales_beat_one_shared_scale():
    """The reason for per-channel scales: a row of small weights next to a row of large
    ones. One shared scale leaves the small row only a few levels."""
    torch.manual_seed(0)
    w = torch.cat([torch.randn(1, 256) * 1.0, torch.randn(1, 256) * 0.01])
    q, scale = quantize_tensor(w)
    per_row_err = (dequantize_tensor(q, scale) - w)[1].abs().mean()
    shared = w.abs().max() / 127
    shared_err = ((w / shared).round() * shared - w)[1].abs().mean()
    assert per_row_err < shared_err / 20


def test_int8_linear_is_a_linear_with_the_rounded_weight():
    torch.manual_seed(0)
    lin = nn.Linear(48, 32, bias=True)
    qlin = Int8Linear.from_linear(lin)
    x = torch.randn(3, 5, 48)
    expected = F.linear(x, dequantize_tensor(qlin.qweight, qlin.scale), lin.bias)
    assert torch.allclose(qlin(x), expected, atol=1e-5)


# ---- the quantized model ----------------------------------------------------------------

@pytest.mark.parametrize("arch", ["llama", "gpt2"])
@torch.no_grad()
def test_quantized_model_stays_close_to_float(arch):
    model = make_model(arch)
    qmodel = quantized_copy(model)
    x = torch.randint(0, model.cfg.vocab_size, (4, 48))
    ref, _ = model(x, x)
    out, _ = qmodel(x, x)
    assert (out - ref).abs().max() / ref.abs().max() < 0.03
    assert (out.argmax(-1) == ref.argmax(-1)).float().mean() > 0.95
    # everything that was a Linear or the token embedding is int8 now
    assert not any(isinstance(m, (nn.Linear, nn.Embedding)) and m is not qmodel.pos_emb for m in qmodel.modules())


def test_tied_weights_stay_tied_and_the_model_is_4x_smaller():
    model = make_model("llama")
    float_bytes = weight_bytes(model)
    qmodel = quantized_copy(model)
    assert isinstance(qmodel.tok_emb, Int8Embedding)
    assert qmodel.tok_emb.qweight is qmodel.lm_head.qweight  # one int8 matrix for both
    assert float_bytes / weight_bytes(qmodel) > 3.8
    assert qmodel.num_params() == model.num_params()


def test_kv_cache_stays_float():
    qmodel = quantized_copy(make_model("llama"))
    assert qmodel.make_kv_caches(1)[0].k.dtype == torch.float32


@torch.no_grad()
def test_int8_checkpoint_round_trip(tmp_path):
    model = make_model("llama")
    src, dst = tmp_path / "m.pt", tmp_path / "m-int8.pt"
    torch.save({"model": model.state_dict(), "model_config": dataclasses.asdict(model.cfg),
                "tokenizer": TOKENIZER}, src)
    quantize_checkpoint(str(src), str(dst))
    assert os.path.getsize(dst) < os.path.getsize(src) / 3

    x = torch.randint(0, model.cfg.vocab_size, (2, 32))
    from_file = Generator.from_checkpoint(str(dst), "cpu").model    # saved as int8
    on_load = Generator.from_checkpoint(str(src), "cpu", "int8").model  # quantized while loading
    assert from_file.quant == on_load.quant == "int8"
    assert torch.equal(from_file(x, x)[0], on_load(x, x)[0])
    with pytest.raises(ValueError, match="already quantized"):
        quantize_checkpoint(str(dst), str(tmp_path / "again.pt"))


def test_batched_int8_requests_match_running_alone(tok):
    """Weight-only int8 keeps rows independent, so batching still changes nothing."""
    gen = Generator(quantized_copy(make_model("llama")), tok)
    requests = [("Once upon a time", SamplingParams(max_new_tokens=25, temperature=0)),
                ("Lily had a red ball", SamplingParams(max_new_tokens=8, temperature=0.9, top_p=0.9, seed=1)),
                ("Tom", SamplingParams(max_new_tokens=30, temperature=1.0, top_k=20, seed=2))]
    engine = BatchEngine(gen, max_batch=3)
    seqs = [Sequence(gen.encode_prompt(p), params) for p, params in requests]
    for s in seqs:
        engine.add(s)
    while engine.has_work():
        assert all(kind != "error" for _, kind, _ in engine.step())
    for (prompt, params), seq in zip(requests, seqs):
        assert seq.result.token_ids == gen.generate(prompt, params).token_ids


# ---- int8-dynamic (PyTorch's int8 kernel) -----------------------------------------------

@needs_kernel
@torch.no_grad()
def test_dynamic_mode_is_close_and_keeps_the_int8_weights():
    model = make_model("llama")
    int8 = quantized_copy(model, "int8")
    dyn = quantized_copy(model, "int8-dynamic")
    assert int8.state_dict().keys() == dyn.state_dict().keys()
    x = torch.randint(0, model.cfg.vocab_size, (2, 40))
    ref, _ = model(x, x)
    out, _ = dyn(x, x)
    assert (out - ref).abs().max() / ref.abs().max() < 0.1      # activations are rounded too
    assert (out.argmax(-1) == ref.argmax(-1)).float().mean() > 0.9


@needs_kernel
def test_int8_model_converts_to_dynamic_but_not_back():
    qmodel = quantized_copy(make_model("llama"), "int8")
    quantize_model(qmodel, "int8-dynamic")
    assert qmodel.quant == "int8-dynamic"
    with pytest.raises(ValueError):
        quantize_model(qmodel, "int8")


def test_dynamic_mode_refuses_a_gpu(tmp_path):
    with pytest.raises(ValueError, match="CPU only"):
        Generator.from_checkpoint(str(tmp_path / "unused.pt"), "cuda", "int8-dynamic")


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown quantization mode"):
        quantize_model(make_model("llama"), "int4")
