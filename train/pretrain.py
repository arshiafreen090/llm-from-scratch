"""
Pretraining loop, built for controlled experiments (see experiments/PROTOCOL.md).

Run from the project root:
    # quick CPU smoke test on the laptop
    python -m train.pretrain --size tiny --token_budget 200000 --batch_size 8 --block_size 128

    # an ablation run on Kaggle: baseline + one upgrade
    python -m train.pretrain --size small --arch gpt2 --pos_emb rope --seed 1

What makes runs comparable (the same for every run unless you deliberately change it):
    * data      every run reads identical training batches in identical order: the batch
                sampler has its OWN seeded random generator, so changing the model
                (which changes how many random numbers weight init uses) can't change the data
    * budget    training length is set in TOKENS, not steps
    * eval      a fixed set of validation windows (same offsets every time, no random sampling)
    * metric    val_loss (nats/token) AND val_bpb (bits per byte, comparable across tokenizers)
    * record    every run saves its full config, parameter count, seed, GPU, dtype and
                code version, so experiments/compare.py can check two runs really are comparable
"""

import argparse
import dataclasses
import json
import math
import os
import subprocess
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from model.config import get_config
from model.model import TinyLLaMA
from tokenizer.tokenizer import Tokenizer


@dataclass
class TrainConfig:
    # ---- run ----
    run_name: str = ""                   # default: built from the settings, e.g. small-llama-s1337
    out_dir: str = "experiments/results"
    ckpt_dir: str = "checkpoints"

    # ---- data ----
    data_dir: str = "data/tinystories"

    # ---- model: a size + architecture preset, plus optional single-switch overrides ----
    size: str = "small"                  # tiny | small | base
    arch: str = "llama"                  # gpt2 | llama
    pos_emb: Optional[str] = None        # override: rope | learned
    norm: Optional[str] = None           # override: rmsnorm | layernorm
    mlp: Optional[str] = None            # override: swiglu | gelu
    n_kv_head: Optional[int] = None      # override
    tie_weights: Optional[bool] = None   # override
    block_size: int = 512
    dropout: float = 0.0

    # ---- budget: the same for every run in a comparison ----
    token_budget: int = 100_000_000      # total training tokens
    batch_size: int = 32                 # sequences per micro-batch
    grad_accum: int = 4                  # micro-batches per optimizer step
    # tokens per step = batch_size * grad_accum * block_size = 65,536 by default

    # ---- optimizer ----
    lr: float = 1e-3
    min_lr: float = 1e-4                 # cosine decays down to this
    warmup_frac: float = 0.02            # fraction of steps spent warming up
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # ---- evaluation ----
    eval_every: int = 100                # steps
    eval_windows: int = 200              # fixed val windows during training
    final_eval_windows: int = 2000       # more windows for the final, reported number
    eval_batch_size: int = 16

    # ---- reproducibility / system ----
    seed: int = 1337                     # model init (and dropout)
    data_seed: int = 1337                # training batch order. Keep it fixed across a comparison
    device: str = "auto"                 # auto | cpu | cuda
    dtype: str = "auto"                  # auto | float32 | bfloat16 | float16
    compile: bool = False                # torch.compile. Keep the same for every run you compare

    def model_overrides(self):
        keys = ["pos_emb", "norm", "mlp", "n_kv_head", "tie_weights"]
        return {k: getattr(self, k) for k in keys if getattr(self, k) is not None}

    def default_run_name(self):
        parts = [self.size, self.arch] + [f"{k}={v}" for k, v in self.model_overrides().items()]
        return "-".join(parts) + f"-s{self.seed}"


# =============================================================================
# Data
# =============================================================================

class TrainBatches:
    """Random training windows from train.bin, drawn with a private seeded generator.
    Same data_seed + batch_size + block_size  =>  exactly the same batches, whatever the model."""

    def __init__(self, path, block_size, batch_size, seed):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size, self.batch_size = block_size, batch_size
        self.rng = np.random.default_rng(seed)

    def next(self):
        starts = self.rng.integers(0, len(self.data) - self.block_size - 1, size=self.batch_size)
        return windows(self.data, starts, self.block_size)


def windows(data, starts, block_size):
    """x = tokens[s : s+T], y = tokens[s+1 : s+T+1]  (y is x shifted by one: next-token targets)"""
    x = np.stack([data[s:s + block_size] for s in starts]).astype(np.int64)
    y = np.stack([data[s + 1:s + 1 + block_size] for s in starts]).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


def fixed_val_starts(n_tokens, block_size, n_windows):
    """Evenly spaced window offsets across the whole val set. Deterministic: the same
    windows every evaluation and every run, so the only thing that changes is the model."""
    last = n_tokens - block_size - 1
    n_windows = min(n_windows, last // block_size)  # don't overlap windows
    return np.linspace(0, last, n_windows).astype(np.int64)


@torch.no_grad()
def evaluate(model, val_data, starts, block_size, batch_size, token_bytes, device, ctx):
    """Returns val_loss (mean nats per token) and val_bpb (bits per byte).

    bits per byte = total loss in bits / total bytes of text predicted.
    Per-token loss depends on what a "token" is, so it can't compare a 1k-vocab model with a
    4k-vocab model. Bytes are the same whatever the tokenizer, so bpb can."""
    model.eval()
    total_nats, total_tokens, total_bytes = 0.0, 0, 0
    for i in range(0, len(starts), batch_size):
        x, y = windows(val_data, starts[i:i + batch_size], block_size)
        x, y = x.to(device), y.to(device)
        with ctx:
            logits, _ = model(x, y)
        nats = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1), reduction="sum")
        total_nats += nats.item()
        total_tokens += y.numel()
        total_bytes += int(token_bytes[y.view(-1).cpu()].sum())
    model.train()
    return total_nats / total_tokens, total_nats / math.log(2) / total_bytes


# =============================================================================
# Setup helpers
# =============================================================================

def has_native_bf16():
    """True only on GPUs with bfloat16 tensor cores: Ampere (A100, RTX 30xx) and newer,
    i.e. compute capability 8.0+.

    Don't use torch.cuda.is_bf16_supported() for this: in recent PyTorch it also returns
    True when bf16 only works through EMULATION, as on Kaggle's T4 (compute capability 7.5).
    Emulated bf16 runs, but skips the tensor cores, so it's much slower than fp16 there."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


def resolve_device_dtype(cfg):
    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = cfg.dtype
    if dtype == "auto":
        if device == "cuda":
            # bf16 where the hardware has it; otherwise fp16 (T4, P100), which needs the
            # GradScaler below but runs on the tensor cores
            dtype = "bfloat16" if has_native_bf16() else "float16"
        else:
            dtype = "float32"
    return device, dtype


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        return (out.stdout.strip() or "none") + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def lr_at(step, total_steps, cfg):
    """Linear warmup, then cosine decay to min_lr."""
    warmup = max(1, int(cfg.warmup_frac * total_steps))
    if step < warmup:
        return cfg.lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return cfg.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (cfg.lr - cfg.min_lr)


# =============================================================================
# Training
# =============================================================================

def train(cfg: TrainConfig):
    cfg.run_name = cfg.run_name or cfg.default_run_name()
    run_dir = os.path.join(cfg.out_dir, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    device, dtype = resolve_device_dtype(cfg)
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    # autocast runs matmuls in low precision, keeps sensitive ops (softmax, norms) in fp32
    ctx = nullcontext() if dtype == "float32" else torch.autocast(device_type=device, dtype=ptdtype)
    # fp16 has a small range: tiny gradients underflow to 0. GradScaler multiplies the loss
    # up before backward and divides the gradients back down. bf16 doesn't need it.
    scaler = torch.amp.GradScaler(device, enabled=(dtype == "float16"))

    # ---- data + tokenizer ----
    with open(os.path.join(cfg.data_dir, "meta.json")) as f:
        meta = json.load(f)
    tok = Tokenizer.load(meta["tokenizer"])
    # bytes of text behind each token id (for bits-per-byte); special tokens count as 0 bytes
    token_bytes = torch.tensor([0 if i in tok.special_tokens.values() else len(tok.vocab[i])
                                for i in range(tok.vocab_size)])
    train_batches = TrainBatches(os.path.join(cfg.data_dir, "train.bin"), cfg.block_size, cfg.batch_size, cfg.data_seed)
    val_data = np.memmap(os.path.join(cfg.data_dir, "val.bin"), dtype=np.uint16, mode="r")
    val_starts = fixed_val_starts(len(val_data), cfg.block_size, cfg.eval_windows)
    final_starts = fixed_val_starts(len(val_data), cfg.block_size, cfg.final_eval_windows)

    # ---- model ----
    torch.manual_seed(cfg.seed)  # model init only; the data has its own generator
    model_cfg = get_config(cfg.size, cfg.arch, vocab_size=meta["vocab_size"], block_size=cfg.block_size,
                           dropout=cfg.dropout, **cfg.model_overrides())
    model = TinyLLaMA(model_cfg).to(device)
    n_params = model.num_params()
    n_embedding = model.tok_emb.weight.numel() + (model.pos_emb.weight.numel() if model.pos_emb is not None else 0)
    optimizer = model.configure_optimizer(cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2), device)
    train_model = torch.compile(model) if cfg.compile else model

    tokens_per_step = cfg.batch_size * cfg.grad_accum * cfg.block_size
    total_steps = max(1, cfg.token_budget // tokens_per_step)

    # ---- record everything needed to judge whether two runs are comparable ----
    record = {
        "run_name": cfg.run_name,
        "train_config": dataclasses.asdict(cfg),
        "model_config": dataclasses.asdict(model_cfg),
        "params": n_params,
        "params_non_embedding": n_params - n_embedding,
        "tokens_per_step": tokens_per_step,
        "total_steps": total_steps,
        "tokens_trained": total_steps * tokens_per_step,
        "tokenizer": meta["tokenizer"],
        "data_dir": cfg.data_dir,
        "train_tokens_available": int(len(train_batches.data)),
        "val_windows": len(val_starts),
        "final_val_windows": len(final_starts),
        "device": device,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "dtype": dtype,
        "torch": torch.__version__,
        "git_commit": git_commit(),
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(record, f, indent=2)

    print(f"run {cfg.run_name}: {n_params / 1e6:.2f}M params ({(n_params - n_embedding) / 1e6:.2f}M non-embedding), "
          f"{device}/{dtype}, {total_steps} steps x {tokens_per_step:,} tokens = {total_steps * tokens_per_step / 1e6:.1f}M tokens")
    if total_steps * tokens_per_step > len(train_batches.data):
        print(f"  note: budget is larger than train.bin ({len(train_batches.data) / 1e6:.0f}M tokens); "
              f"some text will be seen more than once")

    # ---- loop ----
    log_path = os.path.join(run_dir, "log.jsonl")
    log_file = open(log_path, "w")
    ckpt_path = os.path.join(cfg.ckpt_dir, f"{cfg.run_name}.pt")
    train_time, best_val = 0.0, float("inf")

    def save_checkpoint(step, val_loss):
        torch.save({"model": model.state_dict(), "model_config": dataclasses.asdict(model_cfg),
                    "train_config": dataclasses.asdict(cfg), "step": step, "val_loss": val_loss,
                    "tokenizer": meta["tokenizer"]}, ckpt_path)

    x, y = train_batches.next()
    for step in range(total_steps + 1):
        # ---- evaluate (on the fixed windows) ----
        if step % cfg.eval_every == 0 or step == total_steps:
            val_loss, val_bpb = evaluate(model, val_data, val_starts, cfg.block_size, cfg.eval_batch_size,
                                         token_bytes, device, ctx)
            entry = {"step": step, "tokens": step * tokens_per_step, "val_loss": val_loss, "val_bpb": val_bpb,
                     "train_time_s": train_time}
            log_file.write(json.dumps(entry) + "\n")
            log_file.flush()
            print(f"  step {step:5d}  tokens {step * tokens_per_step / 1e6:7.1f}M  val_loss {val_loss:.4f}  val_bpb {val_bpb:.4f}")
            if step > 0 and val_loss < best_val:
                best_val = val_loss
                save_checkpoint(step, val_loss)  # Kaggle sessions can die: keep the best so far
        if step == total_steps:
            break

        # ---- one optimizer step ----
        t0 = time.time()
        lr = lr_at(step, total_steps, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        for _ in range(cfg.grad_accum):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with ctx:
                _, loss = train_model(x, y)
                loss = loss / cfg.grad_accum  # average over micro-batches = one big batch
            x, y = train_batches.next()       # fetch the next batch while the GPU works
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if device == "cuda":
            torch.cuda.synchronize()  # GPU work is async; wait so the timing is real
        dt = time.time() - t0
        train_time += dt

        if step % 10 == 0:
            train_loss = loss.item() * cfg.grad_accum
            entry = {"step": step, "tokens": step * tokens_per_step, "train_loss": train_loss, "lr": lr,
                     "grad_norm": float(grad_norm), "tokens_per_s": tokens_per_step / dt}
            log_file.write(json.dumps(entry) + "\n")
            if step % cfg.eval_every != 0:
                print(f"  step {step:5d}  loss {train_loss:.4f}  lr {lr:.2e}  "
                      f"grad_norm {float(grad_norm):.2f}  {tokens_per_step / dt:,.0f} tok/s")
    log_file.close()

    # ---- final evaluation: more windows, the number that goes in the results table ----
    final_loss, final_bpb = evaluate(model, val_data, final_starts, cfg.block_size, cfg.eval_batch_size,
                                     token_bytes, device, ctx)
    save_checkpoint(total_steps, final_loss)
    summary = {**record, "final_val_loss": final_loss, "final_val_bpb": final_bpb,
               "train_time_s": train_time, "tokens_per_s": total_steps * tokens_per_step / max(train_time, 1e-9)}
    with open(os.path.join(run_dir, "final.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"done: final val_loss {final_loss:.4f}, val_bpb {final_bpb:.4f} "
          f"({len(final_starts)} windows), {train_time / 60:.1f} min training. Saved {run_dir}/ and {ckpt_path}")
    return summary


# =============================================================================
# Command line: every TrainConfig field becomes a --flag
# =============================================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in dataclasses.fields(TrainConfig):
        default = f.default
        if f.type in (bool, Optional[bool]) or isinstance(default, bool):
            parser.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"), default=default)
        elif f.type == Optional[int]:
            parser.add_argument(f"--{f.name}", type=int, default=default)
        elif f.type == Optional[str]:
            parser.add_argument(f"--{f.name}", type=str, default=default)
        else:
            parser.add_argument(f"--{f.name}", type=type(default), default=default)
    return TrainConfig(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    train(parse_args())
