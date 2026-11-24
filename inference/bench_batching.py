"""
Benchmark: continuous batching vs one request at a time.

The same requests (same prompts, lengths and seeds) are served with different max_batch
values. max_batch=1 is the baseline: no batching, requests queue up. Reported per setting:

    throughput    generated tokens per second, all requests together (what the server can handle)
    TTFT          submit -> first token, INCLUDING time spent waiting for a free row
                  (what a user feels before anything appears)
    latency       submit -> last token, for the whole reply
    step          time of one decode pass with a full batch (cost per step grows with rows)

    python -m inference.bench_batching                                   # random-weight "small" model
    python -m inference.bench_batching --ckpt checkpoints/small-llama-s1.pt
    python -m inference.bench_batching --rate 8                          # Poisson arrivals, 8 req/s

Untrained weights are fine here: speed depends on the model's shape, not on what it learned.
Every request generates exactly its max_new_tokens (stop tokens ignored), so each setting
does the same total work.
"""

import argparse
import json
import random
import statistics
import time

import torch

from inference.batching import BatchEngine, Sequence
from inference.generate import Generator
from inference.quantize import MODES as QUANT_MODES, quantize_model
from inference.sampling import SamplingParams
from model.config import get_config
from model.model import TinyLLaMA
from tokenizer.tokenizer import Tokenizer

PROMPTS = ["Once upon a time", "Lily had a red ball. She", "Tom and his dog went to the park",
           "The sun was hot and the", "A little bird", "Ben wanted to build a big tower"]


def make_requests(n, min_new, max_new, seed=0):
    rng = random.Random(seed)
    return [(rng.choice(PROMPTS),
             SamplingParams(max_new_tokens=rng.randint(min_new, max_new), temperature=0.8,
                            top_p=0.95, seed=i, stop_at_special=False))
            for i in range(n)]


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def run(gen, requests, max_batch, rate=None, seed=0):
    """Serve `requests` arriving all at once (rate=None) or as a Poisson process."""
    rng = random.Random(seed)
    arrivals, t = [], 0.0
    for _ in requests:
        arrivals.append(t)
        if rate:
            t += rng.expovariate(rate)

    engine = BatchEngine(gen, max_batch)
    seqs = [Sequence(gen.encode_prompt(p), params) for p, params in requests]
    first, last = {}, {}
    full_steps = []  # decode-pass times with every row busy
    t0 = time.perf_counter()
    nxt = 0
    while nxt < len(seqs) or engine.has_work():
        now = time.perf_counter() - t0
        while nxt < len(seqs) and arrivals[nxt] <= now:
            engine.add(seqs[nxt])
            nxt += 1
        if not engine.has_work():
            time.sleep(max(0.0, arrivals[nxt] - now))
            continue
        full = len(engine.running) == max_batch  # no free row: this step has no prefill in it
        ts = time.perf_counter()
        events = engine.step()
        if full:
            full_steps.append(time.perf_counter() - ts)
        tn = time.perf_counter() - t0
        for seq, kind, value in events:
            assert kind != "error", value
            if kind == "token":
                first.setdefault(id(seq), tn)
            elif kind == "done":
                first.setdefault(id(seq), tn)
                last[id(seq)] = tn
    wall = time.perf_counter() - t0

    ttft = [(first[id(s)] - a) * 1000 for s, a in zip(seqs, arrivals)]
    latency = [(last[id(s)] - a) * 1000 for s, a in zip(seqs, arrivals)]
    tokens = sum(len(s.result.token_ids) for s in seqs)
    return {
        "max_batch": max_batch, "requests": len(seqs), "tokens": tokens, "wall_s": wall,
        "throughput": tokens / wall,
        "ttft_mean": statistics.mean(ttft), "ttft_p95": percentile(ttft, 0.95),
        "latency_p50": percentile(latency, 0.5), "latency_p95": percentile(latency, 0.95),
        "step_ms": 1000 * statistics.median(full_steps) if full_steps else float("nan"),
        "texts": [s.result.text for s in seqs],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None, help="checkpoint (default: random weights)")
    parser.add_argument("--size", default="small", help="model size when no --ckpt is given")
    parser.add_argument("--tokenizer", default="tokenizer/tinystories_4k.json")
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--min_new", type=int, default=32)
    parser.add_argument("--max_new", type=int, default=160)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--rate", type=float, default=None, help="Poisson arrivals per second (default: all at once)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quant", default=None, choices=QUANT_MODES, help="quantize the weights first")
    parser.add_argument("--out", default=None, help="save the results as JSON")
    args = parser.parse_args()

    if args.ckpt:
        gen = Generator.from_checkpoint(args.ckpt, args.device, args.quant)
    else:
        device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        torch.manual_seed(0)
        model = TinyLLaMA(get_config(args.size, "llama"))
        if args.quant:
            quantize_model(model, args.quant)
        gen = Generator(model.to(device), Tokenizer.load(args.tokenizer))
    cfg = gen.model.cfg
    print(f"model: {gen.model.num_params() / 1e6:.1f}M params, {cfg.n_layer} layers, context {cfg.block_size}, "
          f"weights {getattr(gen.model, 'quant', None) or 'float'}, device {gen.device}, "
          f"torch threads {torch.get_num_threads()}")
    requests = make_requests(args.requests, args.min_new, args.max_new)
    arrival = f"Poisson, {args.rate:g} req/s" if args.rate else "all at once"
    print(f"{len(requests)} requests ({arrival}), {args.min_new}-{args.max_new} new tokens each\n")

    run(gen, requests[:4], max(args.batches))  # warm-up (allocations, kernel selection)

    rows = []
    for b in args.batches:
        rows.append(run(gen, requests, b, args.rate))
        r = rows[-1]
        print(f"  max_batch {b:>3}: {r['throughput']:7.1f} tok/s   wall {r['wall_s']:6.1f} s", flush=True)

    base = rows[0]
    print("\n| max_batch | throughput (tok/s) | speedup | TTFT mean / p95 (ms) | latency p50 / p95 (ms) | full step (ms) |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['max_batch']} | {r['throughput']:.0f} | {r['throughput'] / base['throughput']:.2f}x "
              f"| {r['ttft_mean']:.0f} / {r['ttft_p95']:.0f} | {r['latency_p50']:.0f} / {r['latency_p95']:.0f} "
              f"| {r['step_ms']:.1f} |")

    same = all(r["texts"] == base["texts"] for r in rows)
    print(f"\nidentical outputs in every setting: {same}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "results": [{k: v for k, v in r.items() if k != "texts"} for r in rows]},
                      f, indent=2)
        print("saved", args.out)


if __name__ == "__main__":
    main()
