"""
Float32 vs int8 vs int8-dynamic on one checkpoint: size, quality and speed.

    python -m inference.bench_quant --ckpt checkpoints/small-llama-s1.pt
    python -m inference.bench_quant --ckpt checkpoints/small-llama-s1.pt --rounds 10 --out experiments/bench/quant.json

Quality, on the same 200 validation windows as every training eval:
    val loss / bpb       the number the ablations were judged on
    top-1 agreement      share of positions where the quantized model's most likely next token
                         is the float model's most likely next token
    greedy match         greedy generation from a few prompts: how many tokens the quantized
                         model produces before it first differs from the float one

Speed (CPU timings swing with whatever else the laptop is doing, so the modes take turns,
round after round, and each number is a median over rounds):
    decode tok/s         one request, greedy, 200 new tokens
    prefill ms           one forward pass over a 256-token prompt (sets the time to first token)
    batched tok/s        16 requests decoded together (continuous batching, max_batch 16)
"""

import argparse
import json
import os
import statistics
import tempfile
import time
from contextlib import nullcontext

import numpy as np
import torch

from inference.bench_batching import make_requests, run as run_batched
from inference.generate import Generator
from inference.quantize import int8_engine, quantize_checkpoint, weight_bytes
from inference.sampling import SamplingParams
from train.pretrain import evaluate, fixed_val_starts, windows

GREEDY_PROMPTS = ["Once upon a time", "Lily had a red ball. She", "Tom and his dog went to the park"]


@torch.inference_mode()
def top1_agreement(model, ref_model, val, starts, block_size, batch_size=10):
    same = total = 0
    for i in range(0, len(starts), batch_size):
        x, y = windows(val, starts[i:i + batch_size], block_size)
        same += (model(x, y)[0].argmax(-1) == ref_model(x, y)[0].argmax(-1)).sum().item()
        total += y.numel()
    return same / total


def greedy_match(gen, ref_gen, new_tokens):
    """Tokens generated before the first difference from the float model, averaged over prompts."""
    p = SamplingParams(temperature=0, max_new_tokens=new_tokens, stop_at_special=False)
    matched = []
    for prompt in GREEDY_PROMPTS:
        a, b = gen.generate(prompt, p).token_ids, ref_gen.generate(prompt, p).token_ids
        n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        matched.append(n)
    return statistics.mean(matched)


@torch.inference_mode()
def prefill_ms(gen, prompt_ids):
    x = torch.tensor([prompt_ids])
    caches = gen.model.make_kv_caches(1)
    t = time.perf_counter()
    gen.model(x, kv_caches=caches, start_pos=0)
    return (time.perf_counter() - t) * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="a float checkpoint")
    parser.add_argument("--val_bin", default="data/tinystories/val.bin")
    parser.add_argument("--windows", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--new_tokens", type=int, default=200)
    parser.add_argument("--out", default=None, help="save the results as JSON")
    args = parser.parse_args()

    modes = [None, "int8"]
    try:
        int8_engine()
        modes.append("int8-dynamic")
    except RuntimeError as e:
        print(f"skipping int8-dynamic: {e}")
    gens = {m: Generator.from_checkpoint(args.ckpt, "cpu", m) for m in modes}
    name = {None: "float32", "int8": "int8", "int8-dynamic": "int8-dynamic"}
    ref = gens[None]
    cfg = ref.model.cfg
    print(f"{args.ckpt}: {ref.model.num_params() / 1e6:.2f}M params, CPU, {torch.get_num_threads()} threads\n")

    # ---- size --------------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        int8_file = os.path.join(tmp, "int8.pt")
        quantize_checkpoint(args.ckpt, int8_file)
        file_mb = {None: os.path.getsize(args.ckpt) / 1e6, "int8": os.path.getsize(int8_file) / 1e6}
    file_mb["int8-dynamic"] = file_mb["int8"]  # same file, repacked at load time

    # ---- quality -----------------------------------------------------------------------
    val = np.memmap(args.val_bin, dtype=np.uint16, mode="r")
    starts = fixed_val_starts(len(val), cfg.block_size, args.windows)
    tok = ref.tok
    token_bytes = torch.tensor([0 if i in tok.special_tokens.values() else len(tok.vocab[i])
                                for i in range(tok.vocab_size)])
    rows = {}
    for m, gen in gens.items():
        loss, bpb = evaluate(gen.model, val, starts, cfg.block_size, 10, token_bytes, "cpu", nullcontext())
        gen.model.eval()  # evaluate() leaves the model in train mode
        rows[m] = {"mode": name[m], "weights_mb": weight_bytes(gen.model) / 1e6, "file_mb": file_mb[m],
                   "val_loss": loss, "val_bpb": bpb,
                   "top1_agreement": 1.0 if m is None else top1_agreement(gen.model, ref.model, val, starts,
                                                                          cfg.block_size),
                   "greedy_match": args.new_tokens if m is None else greedy_match(gen, ref, args.new_tokens)}
        print(f"  {name[m]:<13} val_loss {loss:.4f}  top-1 agreement {rows[m]['top1_agreement']:.2%}  "
              f"greedy match {rows[m]['greedy_match']:.0f}/{args.new_tokens} tokens", flush=True)

    # ---- speed: modes take turns every round ---------------------------------------------
    prompt_256 = [int(t) for t in val[:256]]
    decode = {m: [] for m in modes}
    prefill = {m: [] for m in modes}
    batched = {m: [] for m in modes}
    requests = make_requests(16, 32, 160)
    p = SamplingParams(temperature=0, max_new_tokens=args.new_tokens, stop_at_special=False)
    for gen in gens.values():  # warm-up
        gen.generate("Once upon a time", SamplingParams(temperature=0, max_new_tokens=20))
        run_batched(gen, requests[:4], 4)
    print()
    for r in range(args.rounds):
        for m, gen in gens.items():
            decode[m].append(gen.generate(GREEDY_PROMPTS[r % len(GREEDY_PROMPTS)], p).decode_tokens_per_s)
            prefill[m].append(statistics.median(prefill_ms(gen, prompt_256) for _ in range(3)))
            batched[m].append(run_batched(gen, requests, 16)["throughput"])
        print(f"  round {r + 1}/{args.rounds}: " + "  ".join(
            f"{name[m]} {decode[m][-1]:.0f} tok/s" for m in modes), flush=True)
    for m in modes:
        rows[m].update(decode_tok_s=statistics.median(decode[m]), prefill_ms=statistics.median(prefill[m]),
                       batched_tok_s=statistics.median(batched[m]))

    # ---- table -------------------------------------------------------------------------
    base = rows[None]
    print(f"\n| weights | file | val loss | top-1 agreement | greedy match ({args.new_tokens} tok) "
          f"| decode tok/s | prefill 256 tok | batched tok/s (16 req) |")
    print("|---|---|---|---|---|---|---|---|")
    for m in modes:
        r = rows[m]
        print(f"| {r['mode']} | {r['file_mb']:.1f} MB | {r['val_loss']:.4f} ({r['val_loss'] - base['val_loss']:+.4f}) "
              f"| {r['top1_agreement']:.2%} | {r['greedy_match']:.0f} "
              f"| {r['decode_tok_s']:.0f} ({r['decode_tok_s'] / base['decode_tok_s']:.2f}x) "
              f"| {r['prefill_ms']:.0f} ms | {r['batched_tok_s']:.0f} ({r['batched_tok_s'] / base['batched_tok_s']:.2f}x) |")
    print(f"\nmedians over {args.rounds} rounds; {len(starts)} val windows of {cfg.block_size} tokens")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"ckpt": args.ckpt, "threads": torch.get_num_threads(), "rounds": args.rounds,
                       "results": list(rows.values())}, f, indent=2)
        print("saved", args.out)


if __name__ == "__main__":
    main()
