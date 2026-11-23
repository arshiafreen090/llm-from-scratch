"""
Supervised fine-tuning (SFT): teach a pretrained model to follow instructions.

    python -m train.sft --init_ckpt checkpoints/small-llama-s1.pt
    # quick CPU smoke test
    python -m train.sft --init_ckpt checkpoints/smoke/smoke-tiny-llama.pt --max_steps 30 --batch_size 8

How this differs from pretraining (train/pretrain.py):
  * Starts from PRETRAINED weights, with a ~10x smaller learning rate: the model already knows
    the language; we only teach it a new behaviour, without wiping out what it knows.
  * Examples are separate documents, not windows cut from one long stream. A batch is padded
    to its longest example.
  * LOSS MASKING: the loss is computed only on the response (and its closing <|endoftext|>).
    Prompt and padding positions get target -1, which cross_entropy ignores (ignore_index=-1).
    Without this, the model would also learn to GENERATE instructions, and part of the
    training signal would go into predicting the user's words instead of its own answer.
  * At the end it measures instruction following (experiments/eval_instruct.py), not just loss.
"""

import argparse
import dataclasses
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from experiments.eval_instruct import evaluate as eval_instructions
from experiments.eval_instruct import load_examples
from inference.generate import Generator
from inference.sampling import SamplingParams
from model.config import ModelConfig
from model.model import TinyLLaMA
from tokenizer.tokenizer import Tokenizer
from train.pretrain import git_commit, lr_at, resolve_device_dtype


@dataclass
class SFTConfig:
    init_ckpt: str = ""                  # pretrained checkpoint to start from (required)
    run_name: str = ""                   # default: sft-<init checkpoint name>
    data_dir: str = "data/tinystories_sft"
    out_dir: str = "experiments/results_sft"
    ckpt_dir: str = "checkpoints"

    epochs: float = 1.0
    max_steps: int = 0                   # > 0 overrides epochs (smoke tests)
    batch_size: int = 32
    grad_accum: int = 1

    lr: float = 1e-4                     # pretraining used 1e-3
    min_lr: float = 1e-5
    warmup_frac: float = 0.03
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    dropout: float = 0.0

    eval_every: int = 200
    eval_examples: int = 1000            # fixed: the first N val examples, every time
    word_eval_n: int = 100               # instruction-following eval at the end (0 = skip)

    seed: int = 1337
    data_seed: int = 1337
    device: str = "auto"
    dtype: str = "auto"
    compile: bool = False


class SFTData:
    """Variable-length examples stored back to back, with a response flag per token."""

    def __init__(self, data_dir, split):
        self.ids = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")
        self.is_target = np.memmap(os.path.join(data_dir, f"{split}_target.bin"), dtype=np.uint8, mode="r")
        self.offsets = np.load(os.path.join(data_dir, f"{split}_offsets.npy"))

    def __len__(self):
        return len(self.offsets) - 1

    def batch(self, indices, pad_id):
        """x: inputs (B, T), y: targets (B, T) with -1 wherever no loss should be computed.
        T = longest example in THIS batch, so short batches stay cheap."""
        spans = [(self.offsets[i], self.offsets[i + 1]) for i in indices]
        T = max(e - s for s, e in spans) - 1
        x = np.full((len(spans), T), pad_id, dtype=np.int64)
        y = np.full((len(spans), T), -1, dtype=np.int64)
        for row, (s, e) in enumerate(spans):
            ids = self.ids[s:e].astype(np.int64)
            target = ids[1:].copy()                  # position i predicts token i+1 ...
            target[self.is_target[s + 1:e] == 0] = -1  # ... but only if token i+1 is response
            x[row, :len(ids) - 1] = ids[:-1]
            y[row, :len(ids) - 1] = target
        # padding sits at the END of each row: causal attention means real tokens never look
        # at it, and its targets are -1, so it affects nothing
        return torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def masked_val_loss(model, data, n, batch_size, pad_id, device, ctx):
    model.eval()
    total, count = 0.0, 0
    indices = list(range(min(n, len(data))))
    for i in range(0, len(indices), batch_size):
        x, y = data.batch(indices[i:i + batch_size], pad_id)
        x, y = x.to(device), y.to(device)
        with ctx:
            logits, _ = model(x, y)
        total += F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1),
                                 ignore_index=-1, reduction="sum").item()
        count += (y != -1).sum().item()
    model.train()
    return total / count


def load_pretrained(path, device, dropout):
    ckpt = torch.load(path, map_location=device)
    cfg = dataclasses.replace(ModelConfig(**ckpt["model_config"]), dropout=dropout)
    model = TinyLLaMA(cfg)
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()})
    return model.to(device), ckpt


def train(cfg: SFTConfig):
    assert cfg.init_ckpt, "--init_ckpt is required: SFT starts from a pretrained model"
    cfg.run_name = cfg.run_name or "sft-" + os.path.splitext(os.path.basename(cfg.init_ckpt))[0]
    run_dir = os.path.join(cfg.out_dir, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    device, dtype = resolve_device_dtype(cfg)
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if dtype == "float32" else torch.autocast(device_type=device, dtype=ptdtype)
    scaler = torch.amp.GradScaler(device, enabled=(dtype == "float16"))

    with open(os.path.join(cfg.data_dir, "meta.json")) as f:
        meta = json.load(f)
    tok = Tokenizer.load(meta["tokenizer"])
    train_data, val_data = SFTData(cfg.data_dir, "train"), SFTData(cfg.data_dir, "val")
    pad_id = tok.eot_id  # any id works: padded positions have target -1

    torch.manual_seed(cfg.seed)
    model, init = load_pretrained(cfg.init_ckpt, device, cfg.dropout)
    # the data and the model must agree on tokenizer and context length
    assert init["tokenizer"] == meta["tokenizer"], f"tokenizer mismatch: {init['tokenizer']} vs {meta['tokenizer']}"
    assert meta["block_size"] <= model.cfg.block_size, "SFT examples are longer than the model's context"

    optimizer = model.configure_optimizer(cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2), device)
    train_model = torch.compile(model) if cfg.compile else model

    per_step = cfg.batch_size * cfg.grad_accum
    total_steps = cfg.max_steps or math.ceil(cfg.epochs * len(train_data) / per_step)
    rng = np.random.default_rng(cfg.data_seed)  # own generator: same example order for every run
    order, cursor = rng.permutation(len(train_data)), 0

    def next_indices():
        nonlocal order, cursor
        if cursor + cfg.batch_size > len(order):  # next epoch: reshuffle
            order, cursor = rng.permutation(len(train_data)), 0
        idx = order[cursor:cursor + cfg.batch_size]
        cursor += cfg.batch_size
        return idx

    sample_instruction = load_examples(os.path.join(cfg.data_dir, "val_examples.jsonl"), 1)[0]["instruction"]
    record = {"run_name": cfg.run_name, "train_config": dataclasses.asdict(cfg),
              "model_config": dataclasses.asdict(model.cfg), "params": model.num_params(),
              "init_ckpt": cfg.init_ckpt, "init_val_loss": init.get("val_loss"),
              "train_examples": len(train_data), "total_steps": total_steps, "tokenizer": meta["tokenizer"],
              "device": device, "gpu": torch.cuda.get_device_name() if device == "cuda" else "cpu",
              "dtype": dtype, "torch": torch.__version__, "git_commit": git_commit()}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(record, f, indent=2)
    print(f"SFT {cfg.run_name}: from {cfg.init_ckpt}, {len(train_data):,} examples, "
          f"{total_steps} steps x {per_step} examples, lr {cfg.lr}, {device}/{dtype}")

    ckpt_path = os.path.join(cfg.ckpt_dir, f"{cfg.run_name}.pt")

    def save(step, val_loss):
        torch.save({"model": model.state_dict(), "model_config": dataclasses.asdict(model.cfg),
                    "train_config": dataclasses.asdict(cfg), "sft": True, "init_ckpt": cfg.init_ckpt,
                    "step": step, "val_loss": val_loss, "tokenizer": meta["tokenizer"]}, ckpt_path)

    def sample():
        gen = Generator(model, tok, is_sft=True)
        out = gen.generate(sample_instruction, SamplingParams(temperature=0, max_new_tokens=120), chat=True)
        model.train()  # Generator switched the model to eval mode
        return out.text

    log = open(os.path.join(run_dir, "log.jsonl"), "w")
    best, train_time = float("inf"), 0.0
    for step in range(total_steps + 1):
        if step % cfg.eval_every == 0 or step == total_steps:
            val_loss = masked_val_loss(model, val_data, cfg.eval_examples, 16, pad_id, device, ctx)
            log.write(json.dumps({"step": step, "val_loss": val_loss}) + "\n")
            log.flush()
            print(f"  step {step:5d}  val_loss (response tokens) {val_loss:.4f}")
            if step % (cfg.eval_every * 5) == 0 or step == total_steps:
                print(f"    sample: {sample()[:200]!r}")
            if step > 0 and val_loss < best:
                best = val_loss
                save(step, val_loss)
        if step == total_steps:
            break

        t0 = time.time()
        lr = lr_at(step, total_steps, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        for _ in range(cfg.grad_accum):
            x, y = train_data.batch(next_indices(), pad_id)
            x, y = x.to(device), y.to(device)
            with ctx:
                _, loss = train_model(x, y)
                loss = loss / cfg.grad_accum
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if device == "cuda":
            torch.cuda.synchronize()
        train_time += time.time() - t0
        if step % 10 == 0:
            entry = {"step": step, "train_loss": loss.item() * cfg.grad_accum, "lr": lr, "grad_norm": float(grad_norm)}
            log.write(json.dumps(entry) + "\n")
            if step % cfg.eval_every != 0:
                print(f"  step {step:5d}  loss {entry['train_loss']:.4f}  lr {lr:.2e}  grad_norm {entry['grad_norm']:.2f}")
    log.close()

    # keep the best checkpoint (lowest val loss) as the result
    best_ckpt = torch.load(ckpt_path, map_location=device) if os.path.exists(ckpt_path) else None
    if best_ckpt is not None:
        model.load_state_dict(best_ckpt["model"])
    summary = {**record, "best_val_loss": best, "train_time_s": train_time}

    if cfg.word_eval_n > 0:
        examples = load_examples(os.path.join(cfg.data_dir, "val_examples.jsonl"), cfg.word_eval_n)
        params = SamplingParams(max_new_tokens=300, temperature=0.7, top_p=0.9, seed=0)
        summary["instruction_eval"] = eval_instructions(Generator(model, tok, is_sft=True), examples, params, "chat")
        print(f"instruction following on {cfg.word_eval_n} val examples: {summary['instruction_eval']}")

    with open(os.path.join(run_dir, "final.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"done: best val_loss {best:.4f}, {train_time / 60:.1f} min. Saved {ckpt_path} and {run_dir}/")
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in dataclasses.fields(SFTConfig):
        if isinstance(f.default, bool):
            parser.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"), default=f.default)
        else:
            parser.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    return SFTConfig(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    train(parse_args())
