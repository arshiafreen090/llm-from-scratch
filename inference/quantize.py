"""
INT8 weight quantization: store each weight in 1 byte instead of 4.

    python -m inference.quantize --ckpt checkpoints/small-llama-s1.pt                  # writes ...-int8.pt
    python -m inference.generate --ckpt checkpoints/small-llama-s1-int8.pt --prompt "Once upon a time"
    python -m inference.generate --ckpt checkpoints/small-llama-s1.pt --quant int8-dynamic --prompt "..."

In code:
    quantize_model(model)                    # "int8": weight-only
    quantize_model(model, "int8-dynamic")    # int8 weights AND activations, PyTorch's int8 kernel

HOW A WEIGHT BECOMES ONE BYTE (symmetric, per output channel)
    For every row w of a weight matrix (the weights of one output channel):
        scale = max(|w|) / 127                # one float per row
        q     = round(w / scale)              # integers in [-127, 127] -> int8
    and back:  w ~= q * scale. The rounding error is at most scale / 2 per weight.

    Why one scale PER ROW instead of one for the whole matrix? Rows have very different
    ranges. With one shared scale, a single large weight anywhere stretches the grid for
    every row, and rows of small weights get only a few distinct levels. Per row, each
    row uses the full -127..127 range. The cost is one float per row: ~0.1% extra.

    Why symmetric (no zero point)? Trained weights are centred on zero, so a zero point
    buys almost nothing, and without it the math stays a plain multiply.

THE TWO MODES
    "int8" (weight-only, W8A32)
        Weights are stored as int8 and converted back to float inside each matmul:
            y = x @ (q * s)^T = (x @ q^T) * s      (s per output column: scale AFTER the matmul)
        Activations stay in float, so the only error is the weight rounding: quality is
        essentially unchanged. Every row of a batch is computed independently, so batched
        requests still produce exactly what they'd produce alone. Works on CPU and GPU.
        Speed is about the same as float32: the matmul itself still runs in float.

    "int8-dynamic" (W8A8)
        The same int8 weights, handed to PyTorch's int8 matmul kernel (qnnpack on ARM,
        fbgemm on x86). Before each matmul the activations are quantized to int8 too,
        with a scale measured on the fly ("dynamic"): int8 x int8 products, summed in int32.
        Faster on CPU, but two costs:
          - more error: activations have outliers, and one scale for the whole input
            tensor gives the other values a coarser grid.
          - that scale is shared by every row of a batch, so in continuous batching a
            request's output can depend on its batch-mates. Use "int8" when that matters.
        CPU only.

WHAT IS QUANTIZED
    Every nn.Linear (attention and MLP projections, lm_head) and the token embedding:
    99.9% of the weights. Norm weights (a few thousand floats) and the GPT-2 position table
    stay in float. With tied weights the embedding and lm_head share ONE int8 matrix: an
    embedding row is a vocab entry, and so is an lm_head output channel, so the per-row
    scales are the same for both.
"""

import argparse
import os
import platform

import torch
import torch.nn as nn
import torch.nn.functional as F

MODES = ["int8", "int8-dynamic"]


def quantize_tensor(w):
    """(out, in) float weight -> (int8 weight, float scale per row). Symmetric, per row."""
    w = w.detach().float()
    scale = w.abs().amax(dim=1) / 127
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)  # all-zero row: any scale works
    q = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale


def dequantize_tensor(q, scale):
    return q.to(scale.dtype) * scale[:, None]


class Int8Linear(nn.Module):
    """Linear layer with an int8 weight and one scale per output channel (weight-only)."""

    def __init__(self, qweight, scale, bias=None):
        super().__init__()
        self.in_features, self.out_features = qweight.size(1), qweight.size(0)
        # buffers, not parameters: they're saved and moved with .to(), but never trained
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale)
        self.register_buffer("bias", bias.detach().clone() if bias is not None else None)

    @classmethod
    def from_linear(cls, linear):
        q, s = quantize_tensor(linear.weight)
        return cls(q, s.to(linear.weight.dtype), linear.bias)

    def forward(self, x):
        # int8 -> x's float type is exact (every integer -127..127 is representable);
        # the scale is applied to the output, one multiply per output value
        y = F.linear(x, self.qweight.to(x.dtype)) * self.scale.to(x.dtype)
        return y + self.bias.to(x.dtype) if self.bias is not None else y

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, int8 per-channel"


class Int8Embedding(nn.Module):
    """Token embedding with int8 rows: looks up the int8 row and its scale, multiplies."""

    def __init__(self, qweight, scale):
        super().__init__()
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale)

    def forward(self, idx):
        return self.qweight[idx].to(self.scale.dtype) * self.scale[idx].unsqueeze(-1)

    def extra_repr(self):
        return f"num={self.qweight.size(0)}, dim={self.qweight.size(1)}, int8 per-row"


def int8_engine():
    """PyTorch's int8 CPU kernels come in per-architecture "engines"; the default it picks
    isn't always right (on ARM it can pick the x86 one, which gives garbage)."""
    arm = platform.machine().lower() in ("aarch64", "arm64")
    wanted = ["qnnpack"] if arm else ["fbgemm", "x86"]
    for engine in wanted:
        if engine in torch.backends.quantized.supported_engines:
            return engine
    raise RuntimeError(f"no int8 CPU engine for {platform.machine()} in this PyTorch build "
                       f"(available: {torch.backends.quantized.supported_engines}); use --quant int8")


class DynamicInt8Linear(nn.Module):
    """The same int8 weight and scales, run through PyTorch's int8 kernel.

    The kernel wants the weight as a "quantized tensor" (values + scales + zero points)
    packed into its own layout. We build that from OUR int8 weights: the quantization math
    is ours, the fast int8 x int8 matmul is PyTorch's (like llama.cpp ships its own kernels)."""

    def __init__(self, int8_linear: Int8Linear):
        super().__init__()
        self.engine = int8_engine()
        torch.backends.quantized.engine = self.engine  # a process-wide setting
        self.in_features, self.out_features = int8_linear.in_features, int8_linear.out_features
        q, s = int8_linear.qweight.cpu(), int8_linear.scale.cpu()
        # kept so state_dict() looks like an Int8Linear (the packed copy below is runtime-only)
        self.register_buffer("qweight", q)
        self.register_buffer("scale", s)
        qtensor = torch._make_per_channel_quantized_tensor(q, s.double(), torch.zeros_like(s, dtype=torch.long), 0)
        bias = int8_linear.bias.float().cpu() if int8_linear.bias is not None else None
        self._packed = torch.ops.quantized.linear_prepack(qtensor, bias)
        # fbgemm keeps activations to 7 bits so its int16 intermediate sums can't overflow
        self._reduce_range = self.engine in ("fbgemm", "x86")

    def forward(self, x):
        shape = x.shape
        y = torch.ops.quantized.linear_dynamic(x.reshape(-1, shape[-1]).float().contiguous(), self._packed,
                                               self._reduce_range)
        return y.view(*shape[:-1], self.out_features).to(x.dtype)

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, int8 weights + activations ({self.engine})"


def quantize_model(model, mode="int8"):
    """Replace the model's Linear layers and token embedding in place. Returns the model."""
    if mode not in MODES:
        raise ValueError(f"unknown quantization mode {mode!r}, choose from {MODES}")
    current = getattr(model, "quant", None)
    if current == mode:
        return model
    if current == "int8-dynamic":
        raise ValueError("the model is already int8-dynamic; load it again to get weight-only int8")

    if current is None:
        # 1. the token embedding, shared with lm_head when the weights are tied
        tied = model.lm_head.weight is model.tok_emb.weight
        q, s = quantize_tensor(model.tok_emb.weight)
        s = s.to(model.tok_emb.weight.dtype)
        model.tok_emb = Int8Embedding(q, s)
        model.lm_head = Int8Linear(q, s) if tied else Int8Linear.from_linear(model.lm_head)

        # 2. every other Linear (attention and MLP projections)
        for module in list(model.modules()):
            for name, child in list(module.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(module, name, Int8Linear.from_linear(child))

    # 3. optionally swap in the int8 x int8 kernel (shares nothing with the float path)
    if mode == "int8-dynamic":
        for module in list(model.modules()):
            for name, child in list(module.named_children()):
                if isinstance(child, Int8Linear):
                    setattr(module, name, DynamicInt8Linear(child))

    model.quant = mode
    return model


def weight_bytes(model):
    """Bytes of everything the model stores (state_dict), counting shared tensors once."""
    seen, total = set(), 0
    for t in model.state_dict().values():
        key = (t.untyped_storage().data_ptr(), t.storage_offset(), t.numel())
        if key not in seen:
            seen.add(key)
            total += t.numel() * t.element_size()
    return total


def quantize_checkpoint(src, dst):
    """Load a float checkpoint, quantize it (weight-only int8), save it. The int8 file works
    for both modes: --quant int8-dynamic at load time repacks the same weights."""
    from model.config import ModelConfig
    from model.model import TinyLLaMA

    ckpt = torch.load(src, map_location="cpu")
    if ckpt.get("quant"):
        raise ValueError(f"{src} is already quantized ({ckpt['quant']})")
    model = TinyLLaMA(ModelConfig(**ckpt["model_config"]))
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()})
    float_bytes = weight_bytes(model)
    quantize_model(model, "int8")
    torch.save({**ckpt, "model": model.state_dict(), "quant": "int8", "quantized_from": src}, dst)
    return float_bytes, weight_bytes(model)


def main():
    parser = argparse.ArgumentParser(description="Quantize a float checkpoint to int8 (weight-only).")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default=None, help="default: <ckpt>-int8.pt")
    args = parser.parse_args()
    out = args.out or args.ckpt.removesuffix(".pt") + "-int8.pt"
    float_bytes, int8_bytes = quantize_checkpoint(args.ckpt, out)
    print(f"weights: {float_bytes / 1e6:.1f} MB -> {int8_bytes / 1e6:.1f} MB ({float_bytes / int8_bytes:.2f}x smaller)")
    print(f"files:   {os.path.getsize(args.ckpt) / 1e6:.1f} MB -> {os.path.getsize(out) / 1e6:.1f} MB  ({out})")


if __name__ == "__main__":
    main()
