# Experiment protocol

An ablation should measure **one change**. Everything else is held fixed, otherwise a
difference in loss could come from any of several changes made at the same time.

## Held fixed across every run in a comparison

| What | How it's enforced |
|---|---|
| Training data | Same `data/tinystories/train.bin`, same tokenizer (`tokenizer/tinystories_4k.json`) |
| Data order | The batch sampler has its own generator seeded by `--data_seed`, separate from model init. Every run sees identical batches in identical order. |
| Training budget | Set in **tokens** (`--token_budget`), with the same batch size, grad accumulation and block size |
| Optimizer | Same lr, schedule (warmup + cosine), weight decay, betas, grad clip |
| Evaluation | Fixed, evenly spaced validation windows (no random sampling). The reported number uses `final_eval_windows` windows. |
| Precision | Same dtype and the same `--compile` setting |

`experiments/compare.py` checks all of these from each run's saved record and prints
`⚠ NOT CONTROLLED` for anything that differs.

## What varies

- **One model switch at a time**, starting from the GPT-2 baseline:
  `--pos_emb rope`, `--norm rmsnorm`, `--mlp swiglu`, `--n_kv_head 2`, then all of them together.
- **The model init seed** (`--seed`), to measure noise. Key comparisons use at least 2–3 seeds.

## Things to keep in mind when reading results

- **Parameter counts differ** between some variants (learned position embeddings add parameters,
  GQA removes them). The table always shows params; a change that removes capacity and loses
  a little loss may still be worth it.
- **GQA is an efficiency trade-off**, not a quality upgrade: report it as loss vs KV-cache size.
- **Seed noise:** a Δ loss smaller than the spread between seeds is not a result.
- **A shared learning rate** can favor one architecture. Either sweep lr per variant on a short
  budget or state that the lr was shared.
- **Tokenizer comparisons** (1k vs 4k vocab) use `val_bpb` (bits per byte), never `val_loss`:
  per-token loss depends on what a token is, bytes don't.
- **Speed (tokens/s)** is only comparable on the same GPU, same dtype, same compile setting,
  on an otherwise idle machine. Kaggle gives either a T4 or a P100; check the `gpu` field.

## Planned ablation runs (small model, 100M tokens each)

| Run | Flags |
|---|---|
| baseline | `--arch gpt2` |
| + RoPE | `--arch gpt2 --pos_emb rope` |
| + RMSNorm | `--arch gpt2 --norm rmsnorm` |
| + SwiGLU | `--arch gpt2 --mlp swiglu` |
| + GQA | `--arch gpt2 --n_kv_head 2` |
| all (mini-LLaMA) | `--arch llama` |

Each with `--seed 1` and `--seed 2` (`--data_seed` stays at its default).
