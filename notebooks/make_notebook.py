"""Generates notebooks/kaggle_train.ipynb (valid notebook JSON). Usage: python make_notebook.py <out_path>"""
import json
import sys

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


md("""
# llm-from-scratch: ablation runs on Kaggle

Trains the GPT-2 baseline, each single LLaMA upgrade, and the full mini-LLaMA under identical
conditions (same data, data order, token budget, optimizer, eval windows and precision).
The rules are in `experiments/PROTOCOL.md`.

## Before running
1. **Settings → Accelerator → GPU T4 x2** (two GPUs = two runs at a time)
2. **Settings → Internet → On** (needs a phone-verified Kaggle account)
3. Run it with **Save Version → Save & Run All (Commit)**. That keeps running after you close
   the browser (up to 12 hours), and everything in `/kaggle/working` is saved with the version.

## Seeds and sessions
This reproduces the README's results: 6 configurations x 2 seeds, then instruction tuning.
About 7.5 hours on two T4s, which fits in one 12-hour session. To split it, set `SEEDS = "1"`
in one session and `SEEDS = "2"` in the next, with the first one's output added as an input.

If a session times out, add the timed-out version as an input
(**Add Input → Your Work → this notebook**): the "restore" cell copies its finished runs
back, and the runner skips them.
""")

code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")

md("## 1. Get the code")
code("""
import os
REPO = "https://github.com/arshiafreen090/llm-from-scratch.git"
WORK = "/kaggle/working/llm-from-scratch"

if not os.path.exists(WORK):
    !git clone -q {REPO} {WORK}
os.chdir(WORK)
!git pull -q
!git log --oneline -1
!pip install -q regex
""")

md("""
## 2. Restore finished runs from a previous session (skip on the first session)
""")
code("""
import glob, shutil
for src in glob.glob("/kaggle/input/*/llm-from-scratch/experiments/results/*"):
    dst = os.path.join("experiments/results", os.path.basename(src))
    if os.path.exists(os.path.join(src, "final.json")) and not os.path.exists(dst):
        shutil.copytree(src, dst)
        print("restored", os.path.basename(src))
for src in glob.glob("/kaggle/input/*/llm-from-scratch/checkpoints/*.pt"):
    os.makedirs("checkpoints", exist_ok=True)
    if not os.path.exists(os.path.join("checkpoints", os.path.basename(src))):
        shutil.copy(src, "checkpoints/")
""")

md("""
## 3. Tokenize TinyStories
Written to `/tmp` (local disk, not saved with the notebook) so the saved output stays small.
Takes a few minutes: downloading ~1 GB, then encoding 2.1M stories with our BPE tokenizer.
""")
code("""
DATA = "/tmp/tinystories"
if not os.path.exists(f"{DATA}/train.bin"):
    !python -m data.prepare_pretrain --out_dir {DATA} --workers 4
!cat {DATA}/meta.json
""")

md("## 4. Sanity check: the test suite (runs on CPU, ~1 min)")
code("""
!python -m pytest tests/ -q
""")

md("""
## 5. Benchmark: how long will the full plan take?
Two short runs (baseline and mini-LLaMA, ~2.6M tokens each) measure training speed on
this GPU, then print an estimate for the 100M-token runs. Measured twice: in float16
(the T4's tensor cores) and in bfloat16 (only emulated on a T4), to see the difference.
""")
code("""
!python -m experiments.run_ablations --benchmark --dtype float16 --data_dir {DATA}
!python -m experiments.run_ablations --benchmark --dtype bfloat16 --data_dir {DATA}
""")

md("""
## 6. The ablation runs
6 configurations x the seeds below, one run per GPU at a time. Finished runs are skipped.

**Precision must match across seeds.** The README's runs were trained in bfloat16, so this uses
bfloat16 too, to reproduce them exactly, even though float16 is faster on a T4. A brand-new
comparison could use `DTYPE = "auto"` (float16 on a T4, about 2x faster). The runner refuses to
mix precisions in one results folder.

Progress lines are printed as runs start and finish. Each run's full log is in
`experiments/results/<run>/stdout.log`.
""")
code("""
SEEDS = "1 2"
DTYPE = "bfloat16"
!python -m experiments.run_ablations --seeds {SEEDS} --dtype {DTYPE} --data_dir {DATA}
""")

md("## 7. Results")
code("""
!python -m experiments.compare --baseline small-gpt2-s1 --out experiments/ablations.md
!python -m experiments.plot
from IPython.display import Image
Image("experiments/loss_curves.png")
""")

md("""
## 8. Instruction tuning (Part 2)
Fine-tunes a pretrained mini-LLaMA (`small-llama-s1`, or `small-llama-s2` if seed 1's
checkpoint isn't here) on 100k TinyStories-Instruct examples, then measures instruction
following (share of required words used) against the pretrained model's chance level.
Precision is auto: float16 on a T4.
""")
code("""
SFT_DATA = "/tmp/tinystories_sft"
BASE = next((p for p in ["checkpoints/small-llama-s1.pt", "checkpoints/small-llama-s2.pt"] if os.path.exists(p)), None)
if BASE:
    SFT_CKPT = "checkpoints/sft-" + os.path.basename(BASE)
    print("fine-tuning", BASE)
    if not os.path.exists(f"{SFT_DATA}/train.bin"):
        !python -m data.prepare_sft --out_dir {SFT_DATA}
    !python -m train.sft --init_ckpt {BASE} --data_dir {SFT_DATA}
    !python -m experiments.eval_instruct --ckpts {BASE} {SFT_CKPT} --examples {SFT_DATA}/val_examples.jsonl --n 200
else:
    print("no pretrained mini-LLaMA checkpoint yet: run the ablations first")
""")

md("""
## 9. Package for download
Download `results.zip` from the **Output** tab. Checkpoints stay in `checkpoints/`
(they're in the saved output too, ~45 MB each).
""")
code("""
!zip -q -r /kaggle/working/results.zip experiments/results experiments/results_sft experiments/ablations.md experiments/loss_curves.png
!ls -lh /kaggle/working/results.zip checkpoints/
""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
print("wrote", sys.argv[1], len(cells), "cells")
