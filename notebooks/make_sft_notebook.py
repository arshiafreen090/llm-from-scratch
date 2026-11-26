"""Generates notebooks/kaggle_sft.ipynb: instruction tuning v2 only (no pretraining).
Usage: python notebooks/make_sft_notebook.py notebooks/kaggle_sft.ipynb"""
import json
import sys

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


md("""
# TinyLLaMA: instruction tuning v2 (varied phrasings)

Re-runs only the SFT stage, starting from the pretrained mini-LLaMA. Version 1 was trained on
100k examples whose instructions came from 5 x 3 x 3 fixed templates, so it followed those
sentences and ignored requests worded differently. Version 2 trains on **300k examples phrased
many ways** (lowercase, "about <topic>", "use words a and b", ...), and is evaluated on
phrasings that were **held out of training**.

About **40 minutes** on one T4.

## Before running
1. **Settings → Accelerator → GPU T4 x2** (one GPU is used)
2. **Settings → Internet → On**
3. Run cell by cell, or **Save Version → Save & Run All (Commit)**.
""")

code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")

md("## 1. Code and the pretrained model")
code("""
import os
REPO = "https://github.com/arshiafreen090/llm-from-scratch.git"
WORK = "/kaggle/working/llm-from-scratch"
HF = "https://huggingface.co/afeening/tinyllama-from-scratch/resolve/main"

if not os.path.exists(WORK):
    !git clone -q {REPO} {WORK}
os.chdir(WORK)
!git pull -q && git log --oneline -1
!pip install -q regex datasets

os.makedirs("checkpoints", exist_ok=True)
# the SFT starting point (float32), and the v1 SFT model to compare against
!curl -sSL -o checkpoints/small-llama-s1.pt {HF}/small-llama-s1.pt
!curl -sSL -o checkpoints/sft-small-llama-s1-int8.pt {HF}/sft-small-llama-s1-int8.pt
!echo "6a7ee935df977b5bedf78c40610a87b230f3d7f09b79b080cd5c6bc79e9f96d9  checkpoints/small-llama-s1.pt" | sha256sum -c -
!ls -la checkpoints/
""")

md("""
## 2. Build the SFT data (about 5 minutes)
300k training examples with varied instruction phrasings; 5k validation examples, half of them
in phrasings held out of training. Written to `/tmp` (not saved with the notebook).
""")
code("""
SFT_DATA = "/tmp/tinystories_sft_v2"
!python -m data.prepare_sft --out_dir {SFT_DATA} --max_train 300000 --max_val 5000
""")

md("""
## 3. Fine-tune (about 25 minutes)
One pass over the 300k examples (9,375 steps of 32), the same compute as 3 epochs of v1's
100k, but on three times as many different stories and phrasings. The settings are v1's
(lr 1e-4, loss on response tokens only).
""")
code("""
!python -m train.sft --init_ckpt checkpoints/small-llama-s1.pt --data_dir {SFT_DATA} \\
    --run_name sft-small-llama-s1-v2 --word_eval_n 200
""")

md("""
## 4. v1 vs v2 on the same 500 validation instructions
Same examples, same seeds. The **unseen** rows are phrasings neither model trained on.
""")
code("""
!python -m experiments.eval_instruct \\
    --ckpts checkpoints/sft-small-llama-s1-int8.pt checkpoints/sft-small-llama-s1-v2.pt \\
    --examples {SFT_DATA}/val_examples.jsonl --n 500
""")

md("## 5. The request that started this, 3 samples per model")
code("""
PROMPT = "write a story about fish which is dancing in river. use words octopus and jellyfish"
for ckpt in ["checkpoints/sft-small-llama-s1-int8.pt", "checkpoints/sft-small-llama-s1-v2.pt"]:
    print("=" * 30, ckpt)
    !python -m inference.generate --ckpt {ckpt} --chat --prompt "{PROMPT}" --seed 0 --num_samples 3 \\
        --max_new_tokens 150 --temperature 0.7 --top_p 0.9
""")

md("""
## 6. Package for download
Click **↻** next to `/kaggle/working` in the Output panel, then download
`sft-small-llama-s1-v2-int8.pt` (11 MB) and `results_sft_v2.zip`.
""")
code("""
!python -m inference.quantize --ckpt checkpoints/sft-small-llama-s1-v2.pt
!cp checkpoints/sft-small-llama-s1-v2-int8.pt checkpoints/sft-small-llama-s1-v2.pt /kaggle/working/
!cp {SFT_DATA}/val_examples.jsonl experiments/results_sft/sft-small-llama-s1-v2/
!zip -q -r /kaggle/working/results_sft_v2.zip experiments/results_sft/sft-small-llama-s1-v2
!ls -lh /kaggle/working/
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
