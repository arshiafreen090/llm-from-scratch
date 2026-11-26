"""
Export a checkpoint to ONNX, so the model can run in a web browser (ONNX Runtime Web).

    python -m inference.export_onnx --ckpt checkpoints/sft-small-llama-s1-int8.pt
    # -> checkpoints/sft-small-llama-s1-int8.onnx, checked against PyTorch before it's written

ONNX is a portable file format for a model's computation graph plus its weights. Runtimes
for it exist for C++, the browser (WebAssembly / WebGPU), phones, and so on. The browser
demo (web/) loads the file written here.

ONE STEP OF THE EXPORTED GRAPH
    inputs   input_ids    (1, T)                       new tokens (the whole prompt, or 1 token)
             past_key     (n_layer, 1, n_kv_head, P, head_dim)  keys of the P earlier tokens
             past_value   (same)                                values of the P earlier tokens
    outputs  logits       (1, vocab)                   scores for the token after the last one
             present_key  (n_layer, 1, n_kv_head, P+T, head_dim)  past + the new tokens' keys
             present_value

The model's own KV cache (model/layers.py) writes into a preallocated tensor in place, which
a graph format can't express. Here the cache is an input and an output instead: the caller
feeds present_* back as past_* on the next step. Prefill is the same graph with P = 0.

The weights stay int8 if the checkpoint is int8: the graph stores int8 tensors and converts
them to float inside each matmul (Cast, MatMul, Mul by the per-channel scale), so the file is
about as small as the .pt one.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from inference.generate import Generator
from model.layers import apply_rope


class OnnxDecoder(nn.Module):
    """The same weights and math as TinyLLaMA.forward, with the KV cache as plain tensors."""

    def __init__(self, model):
        super().__init__()
        assert model.cfg.pos_emb == "rope", "browser export supports the LLaMA architecture (RoPE)"
        self.model = model.eval()

    def forward(self, input_ids, past_key, past_value):
        m, cfg = self.model, self.model.cfg
        T = input_ids.shape[1]
        P = past_key.shape[3]
        positions = torch.arange(T) + P                                   # (T,)
        cos, sin = m.rope_cos[positions], m.rope_sin[positions]          # (T, hd/2)
        repeats = cfg.n_head // cfg.n_kv_head

        x = m.tok_emb(input_ids)                                          # (1, T, C)
        keys, values = [], []
        for i, block in enumerate(m.blocks):
            attn = block.attn
            h = block.norm1(x)
            q = attn.q_proj(h).view(1, T, cfg.n_head, cfg.head_dim).transpose(1, 2)
            k = attn.k_proj(h).view(1, T, cfg.n_kv_head, cfg.head_dim).transpose(1, 2)
            v = attn.v_proj(h).view(1, T, cfg.n_kv_head, cfg.head_dim).transpose(1, 2)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            k = torch.cat([past_key[i], k], dim=2)                       # (1, nkv, P+T, hd)
            v = torch.cat([past_value[i], v], dim=2)
            keys.append(k)
            values.append(v)

            # GQA: each kv head serves `repeats` query heads (expand + reshape exports more
            # simply than repeat_interleave, and gives the same order)
            S = k.shape[2]
            k = k[:, :, None].expand(1, cfg.n_kv_head, repeats, S, cfg.head_dim).reshape(1, cfg.n_head, S, cfg.head_dim)
            v = v[:, :, None].expand(1, cfg.n_kv_head, repeats, S, cfg.head_dim).reshape(1, cfg.n_head, S, cfg.head_dim)

            # causal: new token t (at position P+t) sees keys 0 .. P+t
            visible = torch.arange(S)[None, :] <= positions[:, None]     # (T, S)
            scores = (q @ k.transpose(-2, -1)) / (cfg.head_dim ** 0.5)
            scores = scores.masked_fill(~visible, float("-inf"))
            y = F.softmax(scores, dim=-1) @ v                             # (1, nh, T, hd)
            y = y.transpose(1, 2).reshape(1, T, cfg.n_head * cfg.head_dim)
            x = x + attn.o_proj(y)
            x = x + block.mlp(block.norm2(x))

        x = m.norm_f(x)
        logits = m.lm_head(x[:, -1, :])                                   # (1, vocab): only the last
        return logits, torch.stack(keys), torch.stack(values)


def empty_past(cfg):
    return torch.zeros(cfg.n_layer, 1, cfg.n_kv_head, 0, cfg.head_dim)


@torch.inference_mode()
def check_against_pytorch(gen, onnx_path, n_tokens=60):
    """Greedy-generate with ONNX Runtime and with the PyTorch Generator; the tokens must match.
    Also reports the largest logit difference on the prompt."""
    import onnxruntime as ort

    from inference.sampling import SamplingParams

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    cfg = gen.model.cfg
    prompt = gen.encode_prompt("Write a story. Use the words: rain, ball, happy.", chat=gen.is_sft)
    expected = gen.generate("Write a story. Use the words: rain, ball, happy.",
                            SamplingParams(max_new_tokens=n_tokens, temperature=0, stop_at_special=False),
                            chat=gen.is_sft).token_ids

    ref_logits, _ = gen.model(torch.tensor([prompt]))
    past = empty_past(cfg).numpy()
    logits, pk, pv = session.run(None, {"input_ids": np.array([prompt], dtype=np.int64),
                                        "past_key": past, "past_value": past})
    max_diff = float(np.abs(logits[0] - ref_logits[0, -1].numpy()).max())

    got = []
    for _ in range(n_tokens):
        token = int(logits[0].argmax())
        got.append(token)
        logits, pk, pv = session.run(None, {"input_ids": np.array([[token]], dtype=np.int64),
                                            "past_key": pk, "past_value": pv})
    visible = [t for t in got if t not in gen.special_ids]
    return max_diff, visible[:len(expected)] == expected[:len(visible)]


def export(ckpt, out):
    gen = Generator.from_checkpoint(ckpt, "cpu")
    decoder = OnnxDecoder(gen.model)
    cfg = gen.model.cfg
    example = (torch.tensor([[1, 2, 3]]), empty_past(cfg), empty_past(cfg))
    torch.onnx.export(
        # do_constant_folding=False: otherwise the exporter pre-computes int8 -> float32 for
        # every weight (a "constant" expression) and the file grows 4x (46 MB instead of 11)
        decoder, example, out, dynamo=False, opset_version=17, do_constant_folding=False,
        input_names=["input_ids", "past_key", "past_value"],
        output_names=["logits", "present_key", "present_value"],
        dynamic_axes={"input_ids": {1: "new_tokens"}, "past_key": {3: "past_len"}, "past_value": {3: "past_len"},
                      "present_key": {3: "total_len"}, "present_value": {3: "total_len"}},
    )
    return gen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default=None, help="default: the checkpoint path with .onnx")
    args = parser.parse_args()
    out = args.out or args.ckpt.removesuffix(".pt") + ".onnx"

    gen = export(args.ckpt, out)
    max_diff, same = check_against_pytorch(gen, out)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    print(f"check vs PyTorch: max logit difference {max_diff:.2e}, greedy tokens identical: {same}")
    if not same:
        raise SystemExit("ONNX output differs from PyTorch: not safe to use")

    # what the browser needs to know to drive the graph (web/engine.js reads this)
    cfg = gen.model.cfg
    meta = {"format": "tinyllama-onnx-v1", "model": os.path.basename(out).removesuffix(".onnx"),
            "file": os.path.basename(out), "bytes": os.path.getsize(out),
            "n_layer": cfg.n_layer, "n_kv_head": cfg.n_kv_head, "head_dim": cfg.head_dim,
            "block_size": cfg.block_size, "vocab_size": cfg.vocab_size, "params": gen.model.num_params(),
            "quant": getattr(gen.model, "quant", None), "instruction_tuned": gen.is_sft}
    with open(out.removesuffix(".onnx") + ".onnx.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("wrote", out.removesuffix(".onnx") + ".onnx.json")


if __name__ == "__main__":
    main()
