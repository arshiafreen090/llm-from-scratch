"""
Run the ablation plan from experiments/PROTOCOL.md, one run per GPU at a time.

    python -m experiments.run_ablations --benchmark          # short runs: measure speed, estimate total time
    python -m experiments.run_ablations --seeds 1 2          # the full plan
    python -m experiments.run_ablations --only baseline llama --seeds 1
    python -m experiments.run_ablations --dry_run            # just print what would run

Resumable: runs that already have a final.json are skipped. If a Kaggle session times out,
start it again and it continues where it stopped (a run that was cut off restarts from zero).

Every run gets the SAME settings except its ablation switch and seed. That's what
experiments/compare.py checks afterwards.
"""

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time

import torch

from train.pretrain import TrainConfig

# baseline + one change at a time, then everything together (see PROTOCOL.md)
ABLATIONS = {
    "baseline": dict(arch="gpt2"),
    "rope":     dict(arch="gpt2", pos_emb="rope"),
    "rmsnorm":  dict(arch="gpt2", norm="rmsnorm"),
    "swiglu":   dict(arch="gpt2", mlp="swiglu"),
    "gqa":      dict(arch="gpt2", n_kv_head=2),
    "llama":    dict(arch="llama"),   # all four changes: the mini-LLaMA
}


def plan(names, seeds, common):
    """List of (run_name, args). Ordered seed-first, so every ablation gets one finished
    seed before any second seeds start (useful if the session dies halfway)."""
    runs = []
    for seed in seeds:
        for name in names:
            args = {**common, **ABLATIONS[name], "seed": seed}
            runs.append((TrainConfig(**args).default_run_name(), args))
    return runs


def is_finished(out_dir, run_name):
    return os.path.exists(os.path.join(out_dir, run_name, "final.json"))


def run_all(runs, out_dir, gpu_ids):
    """One worker thread per GPU. Each pulls the next run off a shared queue and trains it
    in a separate process that can only see its own GPU (CUDA_VISIBLE_DEVICES)."""
    todo = queue.Queue()
    for r in runs:
        todo.put(r)
    lock = threading.Lock()

    def say(msg):
        with lock:
            print(time.strftime("%H:%M:%S"), msg, flush=True)

    def worker(gpu):
        while True:
            try:
                name, args = todo.get_nowait()
            except queue.Empty:
                return
            env = os.environ.copy()
            if gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            os.makedirs(os.path.join(out_dir, name), exist_ok=True)
            log_path = os.path.join(out_dir, name, "stdout.log")
            say(f"[gpu {gpu}] start  {name}   (log: {log_path})")
            t0 = time.time()
            with open(log_path, "w") as log:
                cmd = [sys.executable, "-m", "train.pretrain"] + [f"--{k}={v}" for k, v in args.items()]
                code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
            minutes = (time.time() - t0) / 60
            if code == 0:
                with open(os.path.join(out_dir, name, "final.json")) as f:
                    final = json.load(f)
                say(f"[gpu {gpu}] done   {name}   val_loss {final['final_val_loss']:.4f}  "
                    f"{final['tokens_per_s']:,.0f} tok/s  {minutes:.1f} min")
            else:
                with open(log_path) as f:
                    tail = "".join(f.readlines()[-8:])
                say(f"[gpu {gpu}] FAILED {name} after {minutes:.1f} min. Last lines of the log:\n{tail}")

    threads = [threading.Thread(target=worker, args=(g,)) for g in gpu_ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=list(ABLATIONS), choices=list(ABLATIONS))
    parser.add_argument("--seeds", nargs="*", type=int, default=[1, 2])
    parser.add_argument("--size", default="small")
    parser.add_argument("--token_budget", type=int, default=100_000_000)
    parser.add_argument("--out_dir", default="experiments/results")
    parser.add_argument("--data_dir", default="data/tinystories")
    parser.add_argument("--benchmark", action="store_true", help="short runs to measure speed")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    n_gpus = torch.cuda.device_count()
    gpu_ids = list(range(n_gpus)) if n_gpus else [None]
    hw = ", ".join(torch.cuda.get_device_name(i) for i in range(n_gpus)) if n_gpus else "CPU only"
    print(f"hardware: {hw}  ->  {len(gpu_ids)} run(s) at a time")

    common = dict(size=args.size, token_budget=args.token_budget, out_dir=args.out_dir, data_dir=args.data_dir)

    if args.benchmark:
        # ~2.6M tokens each, with almost no evaluation: just measures training speed
        bench = dict(common, token_budget=65536 * 40, eval_every=10**9, eval_windows=20,
                     final_eval_windows=50, out_dir="experiments/bench", ckpt_dir="checkpoints/bench")
        runs = plan(["baseline", "llama"], [0], bench)
        run_all(runs, bench["out_dir"], gpu_ids)
        speeds = []
        for name, _ in runs:
            path = os.path.join(bench["out_dir"], name, "final.json")
            if os.path.exists(path):
                with open(path) as f:
                    speeds.append(json.load(f)["tokens_per_s"])
        if not speeds:
            print("benchmark runs failed, see the logs above")
            return
        speed = sum(speeds) / len(speeds)
        n_runs = len(ABLATIONS) * len(args.seeds)
        hours_per_run = args.token_budget / speed / 3600
        total = n_runs * hours_per_run / len(gpu_ids)
        print(f"\naverage speed {speed:,.0f} tokens/s per run")
        print(f"one {args.token_budget / 1e6:.0f}M-token run: ~{hours_per_run * 60:.0f} min (+ a few min of evaluation)")
        print(f"full plan ({n_runs} runs on {len(gpu_ids)} GPU(s) in parallel): ~{total:.1f} hours")
        if total > 11:
            print("that's more than one 12-hour Kaggle session: run --seeds 1 now and --seeds 2 in a second session")
        return

    runs = plan(args.only, args.seeds, common)
    todo = [r for r in runs if not is_finished(args.out_dir, r[0])]
    for name, run_args in runs:
        status = "done, skipping" if (name, run_args) not in todo else "to run"
        print(f"  {name:<40} {status}")
    if args.dry_run or not todo:
        return
    run_all(todo, args.out_dir, gpu_ids)
    print("\nall runs finished. Next: python -m experiments.compare && python -m experiments.plot")


if __name__ == "__main__":
    main()
