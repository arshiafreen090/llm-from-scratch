# TinyLLaMA from scratch

A small LLaMA-style language model built from scratch: pretrained, instruction-tuned, and served through a fast inference engine.

> 🚧 Work in progress

## The three parts

| Part | What | Status |
|---|---|---|
| 1. Pretrain | LLaMA architecture (RoPE, RMSNorm, SwiGLU, GQA) + custom BPE tokenizer, trained on TinyStories | ⏳ |
| 2. Instruction-tune | SFT so the model follows commands ("write a story about X using words Y, Z") | ⏳ |
| 3. Serve fast | KV cache, sampling, INT8 quantization, speculative decoding, streaming FastAPI + Gradio demo | ⏳ |

## Project layout

```
tokenizer/     BPE tokenizer, trained from scratch
data/          dataset download + tokenization scripts (outputs .bin files)
model/         config, layers (RoPE, RMSNorm, SwiGLU, GQA), full model
train/         pretraining and SFT training loops
inference/     generation (KV cache + sampling), quantization, speculative decoding
serve/         FastAPI streaming server, Gradio demo
experiments/   ablations, benchmarks, plots
checkpoints/   saved model weights (not in git)
notebooks/     Kaggle notebooks used for GPU training
```

## Workflow

- **Laptop (CPU):** write code, quick debug runs, inference, benchmarks, serving
- **Kaggle (free GPU):** real training runs
- **GitHub:** moves code between the two

## Setup

Inside WSL (Ubuntu):

```bash
cd /mnt/c/Users/<you>/Desktop/tinyllama
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only build
pip install -r requirements.txt
```

On Kaggle, PyTorch with GPU support is already installed.

## Results

_Ablation and benchmark tables will go here._
