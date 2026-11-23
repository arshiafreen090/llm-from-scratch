"""
Plot validation-loss curves for every finished run.

    python -m experiments.plot                              # -> experiments/loss_curves.png
    python -m experiments.plot --out experiments/my.png

One color per ablation; seeds of the same ablation share the color (solid, dashed, ...),
so you can see seed noise next to the differences between ablations.
The right panel zooms into the second half of training, where the curves separate.
"""

import argparse
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")  # no screen needed (Kaggle, WSL)
import matplotlib.pyplot as plt


def load(run_dir):
    evals = []
    with open(os.path.join(run_dir, "log.jsonl")) as f:
        for line in f:
            entry = json.loads(line)
            if "val_loss" in entry:
                evals.append(entry)
    return evals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="experiments/results")
    parser.add_argument("--out", default="experiments/loss_curves.png")
    parser.add_argument("--metric", default="val_loss", choices=["val_loss", "val_bpb"])
    args = parser.parse_args()

    run_dirs = sorted(d for d in glob.glob(os.path.join(args.results, "*"))
                      if os.path.exists(os.path.join(d, "final.json")))
    if not run_dirs:
        print("no finished runs found")
        return

    # group by ablation = run name without the "-s<seed>" suffix
    groups = {}
    for d in run_dirs:
        name = os.path.basename(d)
        groups.setdefault(re.sub(r"-s\d+$", "", name), []).append(d)

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 5))
    styles = ["-", "--", ":", "-."]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    zoom_values = []
    for i, (label, dirs) in enumerate(sorted(groups.items())):
        for j, d in enumerate(sorted(dirs)):
            evals = load(d)
            x = [e["tokens"] / 1e6 for e in evals]
            y = [e[args.metric] for e in evals]
            kw = dict(color=colors[i % len(colors)], linestyle=styles[j % len(styles)],
                      marker="o", markersize=3, label=label if j == 0 else None)
            ax_full.plot(x, y, **kw)
            half = len(x) // 2
            ax_zoom.plot(x[half:], y[half:], **kw)
            zoom_values += y[half:]

    for ax, title in [(ax_full, "full training"), (ax_zoom, "second half (zoomed)")]:
        ax.set_xlabel("training tokens (millions)")
        ax.set_ylabel(args.metric.replace("_", " "))
        ax.set_title(title)
        ax.grid(alpha=0.3)
    if zoom_values:
        lo, hi = min(zoom_values), max(zoom_values)
        pad = (hi - lo) * 0.05 or 0.01
        ax_zoom.set_ylim(lo - pad, hi + pad)
    ax_full.legend(fontsize=8)
    fig.suptitle("Validation loss by ablation (line style = seed)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
