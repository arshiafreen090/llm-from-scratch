"""
Which sampling settings write the best stories? Greedy vs temperature vs top-k / top-p / min-p.

    python -m experiments.sampling_sweep --ckpt checkpoints/sft-small-llama-s1.pt --n 200

Every setting answers the SAME instructions with the SAME seeds, and is scored on:

    word inclusion   share of the required words used (instruction following, higher = better)
    repetition       share of a story's 4-word sequences that already appeared earlier in that
                     story ("the dog was happy. the dog was happy."), lower = better
    diversity        distinct 3-word sequences / all 3-word sequences, across all stories of the
                     setting. Low = every story sounds the same. Higher = more varied
    judge ppl        perplexity of the story under a SEPARATE model (the pretrained mini-LLaMA,
                     never instruction-tuned): how natural the text looks to a model of
                     TinyStories. Lower = more fluent. Very low often means bland and repetitive;
                     high means incoherent
    stopped          share of stories that ended on their own (vs hitting the token limit)

The trade-off it shows: low temperature -> fluent but repetitive and samey; high temperature
-> varied but incoherent. Truncation (top-k / top-p / min-p) cuts off the unlikely tail of the
distribution, which is where incoherent tokens come from, so it allows a higher temperature
(more variety) before the text falls apart.

Generation runs through the BatchEngine (continuous batching): ~5x faster than one at a
time, with exactly the same outputs (tests/test_batching.py).
"""

import argparse
import json
import math
import os
import statistics

import torch

from experiments.eval_instruct import load_examples, word_present
from inference.batching import BatchEngine, Sequence
from inference.generate import Generator
from inference.sampling import SamplingParams

# (label, settings). Ordered from "no randomness" to "a lot of randomness".
SETTINGS = [
    ("greedy", dict(temperature=0)),
    ("T=0.5", dict(temperature=0.5)),
    ("T=0.7", dict(temperature=0.7)),
    ("T=0.7, top-p 0.9", dict(temperature=0.7, top_p=0.9)),
    ("T=1.0 (pure sampling)", dict(temperature=1.0)),
    ("T=1.0, top-k 50", dict(temperature=1.0, top_k=50)),
    ("T=1.0, top-p 0.9", dict(temperature=1.0, top_p=0.9)),
    ("T=1.0, min-p 0.1", dict(temperature=1.0, min_p=0.1)),
    ("T=1.0, top-k 50, top-p 0.9", dict(temperature=1.0, top_k=50, top_p=0.9)),
    ("T=1.3, top-p 0.9", dict(temperature=1.3, top_p=0.9)),
    ("T=1.3 (pure sampling)", dict(temperature=1.3)),
]


def words(text):
    return text.lower().split()


def repetition(text, n=4):
    """Share of n-grams that already occurred earlier in the same text."""
    w = words(text)
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def diversity(texts, n=3):
    """Distinct n-grams / all n-grams, pooled over every text."""
    grams = [tuple(w[i:i + n]) for t in texts for w in [words(t)] for i in range(len(w) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 0.0


@torch.inference_mode()
def judge_ppl(judge: Generator, texts):
    """Perplexity of each story under the judge model, as a fresh story (after <|endoftext|>)."""
    nll, count = 0.0, 0
    block = judge.model.cfg.block_size
    for t in texts:
        ids = [judge.tok.eot_id] + judge.tok.encode_ordinary(t)
        ids = ids[:block + 1]
        if len(ids) < 2:
            continue
        x, y = torch.tensor([ids[:-1]]), torch.tensor([ids[1:]])
        logits, _ = judge.model(x, y)
        nll += torch.nn.functional.cross_entropy(logits[0], y[0], reduction="sum").item()
        count += y.numel()
    return math.exp(nll / count)


def generate_all(gen, examples, params, max_batch):
    engine = BatchEngine(gen, max_batch)
    seqs = []
    for i, ex in enumerate(examples):
        p = SamplingParams(**{**params.__dict__, "seed": i})  # same seed per instruction in every setting
        seqs.append(Sequence(gen.encode_prompt(ex["instruction"], chat=True), p))
        engine.add(seqs[-1])
    while engine.has_work():
        for _, kind, value in engine.step():
            assert kind != "error", value
    return [s.result for s in seqs]


def plot(rows, path):
    """Diversity vs fluency: one point per setting. Up-left is the goal (varied AND fluent)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.subplots_adjust(right=0.66)  # room for the numbered key on the right
    key = []
    for i, r in enumerate(rows):
        family = ("greedy" if r["temperature"] == 0 else "temperature only"
                  if not any(r.get(k) for k in ("top_k", "top_p", "min_p")) else "with truncation")
        style = {"greedy": ("s", "tab:gray"), "temperature only": ("o", "tab:blue"),
                 "with truncation": ("^", "tab:orange")}[family]
        ax.scatter(r["diversity"], r["judge_ppl"], marker=style[0], color=style[1], s=60,
                   label=family if family not in ax.get_legend_handles_labels()[1] else None)
        # points are numbered; the key on the right says which setting each number is
        # (names next to the points overlap where settings land close together)
        ax.annotate(str(i + 1), (r["diversity"], r["judge_ppl"]), textcoords="offset points",
                    xytext=(5, 4), fontsize=9, fontweight="bold")
        key.append(f"{i + 1:>2}. {r['setting']}  ({r['word_inclusion']:.0%} words)")
    ax.set_yscale("log")
    ax.set_xlabel("diversity: distinct 3-grams across stories (higher = more varied)")
    ax.set_ylabel("judge perplexity (lower = more fluent), log scale")
    ax.set_title("Sampling settings: variety vs fluency")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.text(0.68, 0.88, "setting  (required words used)\n\n" + "\n".join(key),
             va="top", fontsize=9, family="monospace")
    fig.savefig(path, dpi=130)
    print("saved", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot_only", action="store_true", help="redraw the plot from --out without generating")
    parser.add_argument("--plot", default="experiments/sampling_sweep.png")
    parser.add_argument("--ckpt", default="checkpoints/sft-small-llama-s1.pt")
    parser.add_argument("--judge", default="checkpoints/small-llama-s1.pt", help="pretrained model used to score fluency")
    parser.add_argument("--examples", default="data/tinystories_sft/val_examples.jsonl")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--max_batch", type=int, default=16)
    parser.add_argument("--out", default="experiments/sampling_sweep.json")
    args = parser.parse_args()
    if args.plot_only:
        with open(args.out, encoding="utf-8") as f:
            plot(json.load(f)["results"], args.plot)
        return

    gen = Generator.from_checkpoint(args.ckpt, "cpu")
    judge = Generator.from_checkpoint(args.judge, "cpu")
    examples = load_examples(args.examples, args.n)
    print(f"{len(examples)} instructions, up to {args.max_new_tokens} tokens, model {args.ckpt}, judge {args.judge}\n")

    rows = []
    for label, settings in SETTINGS:
        params = SamplingParams(max_new_tokens=args.max_new_tokens, **settings)
        results = generate_all(gen, examples, params, args.max_batch)
        texts = [r.text for r in results]
        hits = [[word_present(w, t) for w in ex["words"]] for ex, t in zip(examples, texts)]
        row = {
            "setting": label, **settings,
            "word_inclusion": statistics.mean(sum(h) / len(h) for h in hits),
            "all_words": statistics.mean(all(h) for h in hits),
            "repetition": statistics.mean(repetition(t) for t in texts),
            "diversity": diversity(texts),
            "judge_ppl": judge_ppl(judge, texts),
            "stopped": statistics.mean(r.finish_reason == "stop" for r in results),
            "mean_tokens": statistics.mean(len(r.token_ids) for r in results),
            "example": texts[0],
        }
        rows.append(row)
        print(f"  {label:<28} words {row['word_inclusion']:6.1%}  repetition {row['repetition']:5.1%}  "
              f"diversity {row['diversity']:.3f}  judge ppl {row['judge_ppl']:5.2f}  stopped {row['stopped']:4.0%}",
              flush=True)

    print("\n| setting | required words used | all 3 words | repetition ↓ | diversity ↑ | judge perplexity | stopped |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['setting']} | {r['word_inclusion']:.1%} | {r['all_words']:.1%} | {r['repetition']:.1%} "
              f"| {r['diversity']:.3f} | {r['judge_ppl']:.2f} | {r['stopped']:.0%} |")
    print(f"\ninstruction 1: {examples[0]['instruction']}")
    for r in rows:
        print(f"\n--- {r['setting']} ---\n{r['example'][:400]}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "results": rows}, f, indent=2, ensure_ascii=False)
        print("\nsaved", args.out)
    if args.plot:
        plot(rows, args.plot)


if __name__ == "__main__":
    main()
