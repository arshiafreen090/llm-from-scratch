"""
Tests for the ONNX export used by the in-browser demo (inference/export_onnx.py).
    python -m pytest tests/test_export_onnx.py -v
Needs onnx + onnxruntime (pip install onnx onnxruntime onnxscript); skipped without them.
"""

import dataclasses
import os

import numpy as np
import pytest
import torch

pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

from inference.export_onnx import OnnxDecoder, check_against_pytorch, empty_past, export  # noqa: E402
from inference.generate import Generator  # noqa: E402
from model.config import get_config  # noqa: E402
from model.model import TinyLLaMA  # noqa: E402

TOKENIZER = os.path.join(os.path.dirname(__file__), "..", "tokenizer", "tinystories_4k.json")


def save_tiny_checkpoint(path, quantized):
    torch.manual_seed(0)
    model = TinyLLaMA(get_config("tiny", "llama", block_size=64))
    torch.save({"model": model.state_dict(), "model_config": dataclasses.asdict(model.cfg),
                "tokenizer": TOKENIZER, "sft": True}, path)
    if quantized:
        from inference.quantize import quantize_checkpoint
        quantize_checkpoint(str(path), str(path))  # overwrite with the int8 version
    return path


@torch.no_grad()
def test_decoder_with_cache_matches_the_model():
    """OnnxDecoder (cache as inputs/outputs) gives the same logits as the model, step by step."""
    torch.manual_seed(0)
    model = TinyLLaMA(get_config("tiny", "llama", block_size=64)).eval()
    decoder = OnnxDecoder(model)
    seq = torch.randint(0, model.cfg.vocab_size, (1, 12))
    logits, k, v = decoder(seq[:, :8], empty_past(model.cfg), empty_past(model.cfg))   # prefill 8
    for t in range(8, 12):                                                              # then 1 by 1
        ref, _ = model(seq[:, :t])
        assert torch.allclose(logits, ref[:, -1], atol=1e-5)
        logits, k, v = decoder(seq[:, t:t + 1], k, v)
    assert k.shape == (model.cfg.n_layer, 1, model.cfg.n_kv_head, 12, model.cfg.head_dim)


@pytest.mark.parametrize("quantized", [False, True])
def test_exported_file_generates_the_same_tokens(tmp_path, quantized):
    ckpt = save_tiny_checkpoint(tmp_path / "tiny.pt", quantized)
    out = str(tmp_path / "tiny.onnx")
    gen = export(str(ckpt), out)
    max_diff, same = check_against_pytorch(gen, out, n_tokens=20)
    assert same and max_diff < 1e-4
    if quantized:  # the weights stay int8 in the file: about 1 byte per parameter
        assert os.path.getsize(out) < gen.model.num_params() * 1.5


def test_onnx_runs_without_a_past(tmp_path):
    """Prefill is the same graph with a zero-length cache."""
    import onnxruntime as ort

    ckpt = save_tiny_checkpoint(tmp_path / "tiny.pt", quantized=True)
    out = str(tmp_path / "tiny.onnx")
    gen = export(str(ckpt), out)
    session = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    empty = empty_past(gen.model.cfg).numpy()
    logits, k, v = session.run(None, {"input_ids": np.array([[5, 6, 7]], dtype=np.int64),
                                      "past_key": empty, "past_value": empty})
    assert logits.shape == (1, gen.model.cfg.vocab_size) and k.shape[3] == 3
