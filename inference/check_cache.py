"""
Check that KV-cached decoding gives the same logits as a full uncached forward pass,
for a REAL checkpoint, on a real device, in a real precision.

    python -m inference.check_cache --ckpt checkpoints/small-llama-s1.pt                  # fp32
    python -m inference.check_cache --ckpt checkpoints/small-llama-s1.pt --dtype bfloat16
    python -m inference.check_cache --ckpt checkpoints/small-llama-s1.pt --device cuda --dtype float16

Why this exists next to the unit tests: tests/test_model.py proves the cache logic is right
with random weights in float32. In fp16/bf16 the cached and uncached paths do arithmetic in
a different order, so tiny rounding differences are EXPECTED. What matters is whether they're
big enough to change which token gets picked. This script measures that on the actual model.

Reports, over every position of the sequence:
    max relative logit error   |cached - uncached| / |uncached|, worst case
    argmax agreement           % of positions where greedy decoding would pick the same token
    max KL divergence          how different the two probability distributions are (0 = identical)
"""

import argparse
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from inference.generate import Generator

SAMPLE_TEXT = ("Once upon a time, there was a little girl named Lily. She loved to play outside in the sun. "
               "One day, she saw a big red ball in the park. She ran to the ball and kicked it very hard. ")


@torch.inference_mode()
def compare(model, idx, ctx):
    T = idx.size(1)
    with ctx:
        full, _ = model(idx, idx)  # passing targets returns logits for every position
        caches = model.make_kv_caches(batch_size=idx.size(0))
        steps = []
        for t in range(T):
            step, _ = model(idx[:, t:t + 1], kv_caches=caches, start_pos=t)
            steps.append(step[:, 0])
    full = full.float()
    cached = torch.stack(steps, dim=1).float()
    rel = ((cached - full).abs().amax(-1) / full.abs().amax(-1))           # per position
    agree = (cached.argmax(-1) == full.argmax(-1)).float()
    kl = F.kl_div(F.log_softmax(cached, -1), F.log_softmax(full, -1), log_target=True, reduction="none").sum(-1)
    # how close the top two tokens were in the reference: a flip between near-ties is rounding,
    # a flip with a big gap would mean the cached path is actually wrong
    top2 = F.softmax(full, -1).topk(2, dim=-1).values
    gap = top2[..., 0] - top2[..., 1]
    return rel, agree, kl, gap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--val_bin", default=None, help="use real validation text, e.g. data/tinystories/val.bin")
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    gen = Generator.from_checkpoint(args.ckpt, args.device)
    model, device = gen.model, gen.device
    T = model.cfg.block_size  # the whole context window

    if args.val_bin:
        data = np.memmap(args.val_bin, dtype=np.uint16, mode="r")
        starts = np.linspace(0, len(data) - T - 1, args.batch).astype(int)
        idx = torch.from_numpy(np.stack([data[s:s + T] for s in starts]).astype(np.int64))
    else:
        ids = gen.tok.encode(SAMPLE_TEXT * 50)[:T]
        idx = torch.tensor([ids] * args.batch)
    idx = idx.to(device)

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    ctx = nullcontext() if dtype == torch.float32 else torch.autocast(device_type=device.type, dtype=dtype)
    rel, agree, kl, gap = compare(model, idx, ctx)

    print(f"checkpoint {args.ckpt} on {device} in {args.dtype}: {idx.size(0)} sequences x {T} positions")
    print(f"  max relative logit error : {rel.max().item():.2e}   (median {rel.median().item():.2e})")
    print(f"  argmax agreement         : {agree.mean().item() * 100:.2f}%")
    print(f"  max KL divergence        : {kl.max().item():.2e}")
    if agree.mean().item() < 1.0:
        flips = gap[agree == 0]
        print(f"  {len(flips)} flips; top-1 vs top-2 probability gap at those positions: "
              f"max {flips.max().item():.4f}, typical gap elsewhere {gap.median().item():.4f}")
        print("  (flips only between near-tied tokens = expected rounding; a flip with a large gap = a bug)")


if __name__ == "__main__":
    main()
