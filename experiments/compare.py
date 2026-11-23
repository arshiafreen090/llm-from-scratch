"""
Build the ablation table from finished runs, and refuse to quietly compare runs that differ
in anything other than the model.

    python -m experiments.compare                                    # every run in experiments/results
    python -m experiments.compare --baseline small-gpt2-s1           # deltas against this run's group
    python -m experiments.compare --out experiments/ablations.md     # also save the table

What it does:
  1. CHECKS that every run used the same data, tokenizer, token budget, optimizer settings,
     data order, eval windows and precision. Any mismatch is printed as an error, because
     then a difference in loss can't be pinned on the architecture.
  2. GROUPS runs that differ only in their seed, and reports mean +- std over seeds.
  3. LISTS, for each group, which model settings differ from the baseline. A group that
     changes more than one setting at once is flagged.
"""

import argparse
import glob
import json
import os
import statistics

# must be identical across runs for the comparison to measure only the model
CONTROLLED_TRAIN = ["data_dir", "token_budget", "batch_size", "grad_accum", "block_size", "dropout",
                    "lr", "min_lr", "warmup_frac", "weight_decay", "beta1", "beta2", "grad_clip",
                    "data_seed", "compile", "final_eval_windows"]
CONTROLLED_RECORD = ["tokenizer", "tokens_trained", "train_tokens_available", "final_val_windows", "dtype"]


def load_runs(paths):
    runs = []
    for p in paths:
        path = os.path.join(p, "final.json") if os.path.isdir(p) else p
        if os.path.exists(path):
            with open(path) as f:
                runs.append(json.load(f))
    return runs


def check_controls(runs):
    problems = []
    for key in CONTROLLED_TRAIN:
        values = {json.dumps(r["train_config"][key]) for r in runs}
        if len(values) > 1:
            problems.append(f"train setting '{key}' differs between runs: {sorted(values)}")
    for key in CONTROLLED_RECORD:
        values = {json.dumps(r[key]) for r in runs}
        if len(values) > 1:
            problems.append(f"'{key}' differs between runs: {sorted(values)}")
    gpus = {r["gpu"] for r in runs}
    if len(gpus) > 1:
        problems.append(f"runs used different hardware {sorted(gpus)}: loss is still comparable, "
                        f"tokens/s is NOT")
    return problems


def group_by_model(runs):
    """Runs with the same model config (differing only in seed) form one group."""
    groups = {}
    for r in runs:
        key = json.dumps(r["model_config"], sort_keys=True)
        groups.setdefault(key, []).append(r)
    return list(groups.values())


def mean_std(values):
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}"


def describe_changes(cfg, base_cfg):
    return {k: v for k, v in cfg.items() if base_cfg.get(k) != v}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*", help="run directories (default: all in experiments/results)")
    parser.add_argument("--baseline", default=None, help="run name whose group is the baseline")
    parser.add_argument("--out", default=None, help="write the markdown table here too")
    args = parser.parse_args()

    runs = load_runs(args.runs or sorted(glob.glob("experiments/results/*")))
    if not runs:
        print("no finished runs found (a run is finished once it has final.json)")
        return

    problems = check_controls(runs)
    for p in problems:
        print(f"⚠ NOT CONTROLLED: {p}")
    if not problems:
        print("✓ all runs share data, tokenizer, budget, optimizer, data order, eval windows and dtype")

    groups = group_by_model(runs)
    base_group = groups[0]
    if args.baseline:
        base_group = next(g for g in groups if any(r["run_name"] == args.baseline for r in g))
    base_cfg = base_group[0]["model_config"]
    base_loss = statistics.mean(r["final_val_loss"] for r in base_group)

    lines = [
        "| changes vs baseline | params | seeds | val loss | val bpb | Δ loss | tokens/s |",
        "|---|---|---|---|---|---|---|",
    ]
    for g in groups:
        changes = describe_changes(g[0]["model_config"], base_cfg)
        label = "baseline" if not changes else ", ".join(f"{k}={v}" for k, v in changes.items())
        if len(changes) > 1:
            label += f" ⚠ {len(changes)} changes"
        losses = [r["final_val_loss"] for r in g]
        bpbs = [r["final_val_bpb"] for r in g]
        delta = statistics.mean(losses) - base_loss
        speed = statistics.mean(r["tokens_per_s"] for r in g)
        lines.append(f"| {label} | {g[0]['params'] / 1e6:.2f}M | {len(g)} | {mean_std(losses)} | "
                     f"{mean_std(bpbs)} | {delta:+.4f} | {speed:,.0f} |")

    ref = runs[0]
    footer = (f"\nAll runs: {ref['tokens_trained'] / 1e6:.0f}M tokens, block {ref['train_config']['block_size']}, "
              f"lr {ref['train_config']['lr']}, data_seed {ref['train_config']['data_seed']}, "
              f"{ref['final_val_windows']} val windows, {ref['dtype']}, tokenizer {ref['tokenizer']}.")
    if any(len(g) == 1 for g in groups):
        footer += ("\nGroups with 1 seed have no error bar: a small Δ loss there may just be seed noise.")
    table = "\n".join(lines) + "\n" + footer
    print("\n" + table)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(table + "\n")


if __name__ == "__main__":
    main()
