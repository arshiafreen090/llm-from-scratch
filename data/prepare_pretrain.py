"""
Turn TinyStories into token files for pretraining.

Output, in data/tinystories/:
    train.bin   every story's token ids in one long uint16 array:  story <eot> story <eot> ...
    val.bin     the same for the validation split
    meta.json   vocab size, special token ids, token counts

Run from the project root, inside the venv:
    python -m data.prepare_pretrain                       # full dataset (~2.1M stories)
    python -m data.prepare_pretrain --max_stories 20000   # quick test

Why one long array instead of a list of stories?
    The training loop picks a random offset and reads `block_size` tokens from there
    (exactly like nanoGPT's get_batch). One flat array on disk can be memory-mapped,
    so training reads only the bytes it needs and never loads 1 GB into RAM.

Why uint16?
    Our vocab has 4,096 ids, and uint16 holds 0..65,535. That's 2 bytes per token
    instead of 8 for int64, so the files are 4x smaller.
"""

import argparse
import json
import os
import time
from multiprocessing import get_context

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from tokenizer.tokenizer import Tokenizer

DATASET = "roneneldan/TinyStories"
BATCH_SIZE = 1024  # stories handed to a worker at a time

# -----------------------------------------------------------------------------
# Worker processes. Encoding is pure Python, so it's CPU-bound; we use one process
# per core, and each one loads its own copy of the tokenizer once at startup.

_tok = None


def _init_worker(tokenizer_path):
    global _tok
    _tok = Tokenizer.load(tokenizer_path)


def _encode_batch(stories):
    """Encode a batch of stories. Each story is followed by <|endoftext|>, so the model
    learns where a story ends and that a new one starts after that token."""
    ids, lengths = [], []
    for story in stories:
        story = story.strip()
        if not story:
            continue
        # encode_ordinary: if a story happens to contain the text "<|endoftext|>",
        # it stays plain text instead of turning into the real special token
        story_ids = _tok.encode_ordinary(story)
        story_ids.append(_tok.eot_id)
        ids.extend(story_ids)
        lengths.append(len(story_ids))
    return np.array(ids, dtype=np.uint16), lengths


# -----------------------------------------------------------------------------

def process_split(split, out_path, tokenizer_path, workers, max_stories=None):
    """Encode one split in parallel and append the ids to out_path.

    The first run downloads the dataset (~1 GB of parquet files) into the Hugging Face
    cache (~/.cache/huggingface); later runs read it from disk."""
    ds = load_dataset(DATASET, split=split)
    if max_stories:
        ds = ds.select(range(min(max_stories, len(ds))))
    # read the text column in batches, so 2M stories never sit in RAM at once
    batches = (batch["text"] for batch in ds.iter(batch_size=BATCH_SIZE))

    n_tokens, lengths = 0, []
    # "spawn" starts each worker as a fresh Python process. The default on Linux ("fork")
    # copies the parent mid-flight, including any open network/file threads, which can
    # deadlock. Spawn is a little slower to start but always safe.
    ctx = get_context("spawn")
    with open(out_path, "wb") as f, ctx.Pool(workers, _init_worker, (tokenizer_path,)) as pool:
        # imap (not imap_unordered) keeps the stories in their original order
        results = pool.imap(_encode_batch, batches)
        with tqdm(total=len(ds), unit=" stories", desc=split) as bar:
            for arr, lens in results:
                arr.tofile(f)  # append raw bytes; never holds the whole split in memory
                n_tokens += len(arr)
                lengths.extend(lens)
                bar.update(len(lens))
    return n_tokens, lengths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="tokenizer/tinystories_4k.json")
    parser.add_argument("--out_dir", default="data/tinystories")
    parser.add_argument("--max_stories", type=int, default=None, help="limit stories per split (for testing)")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    tok = Tokenizer.load(args.tokenizer)
    assert tok.vocab_size <= 2**16, "uint16 can't hold ids this large"
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"tokenizer: {args.tokenizer} (vocab {tok.vocab_size}), {args.workers} workers")

    stats = {}
    for split, name in [("validation", "val"), ("train", "train")]:
        t0 = time.time()
        path = os.path.join(args.out_dir, f"{name}.bin")
        n_tokens, lengths = process_split(split, path, args.tokenizer, args.workers, args.max_stories)
        lengths = np.array(lengths)
        stats[name] = {"tokens": int(n_tokens), "stories": int(len(lengths))}
        p50, p90, p99, pmax = np.percentile(lengths, [50, 90, 99, 100]).astype(int)
        print(f"{name}: {n_tokens:,} tokens from {len(lengths):,} stories in {time.time() - t0:.0f}s "
              f"-> {os.path.getsize(path) / 1e6:.0f} MB")
        print(f"  tokens per story: median {p50}, 90% {p90}, 99% {p99}, max {pmax}")

    meta = {
        "tokenizer": args.tokenizer,
        "vocab_size": tok.vocab_size,
        "eot_id": tok.eot_id,
        "special_tokens": tok.special_tokens,
        "dtype": "uint16",
        **{f"{k}_{name}": v for name, s in stats.items() for k, v in s.items()},
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ---- sanity check: read the files back like the training loop will ----------
    for name in ["val", "train"]:
        data = np.memmap(os.path.join(args.out_dir, f"{name}.bin"), dtype=np.uint16, mode="r")
        sample = data[: min(len(data), 5_000_000)]
        assert sample.max() < tok.vocab_size, f"{name}.bin contains an id outside the vocab!"
    head = np.array(data[:10_000])  # data is train.bin here
    first_eot = int(np.argmax(head == tok.eot_id))
    print("\nfirst training story, decoded back from train.bin:")
    print("  " + tok.decode(head[: first_eot + 1].tolist())[:400])
    print(f"\nsaved {args.out_dir}/train.bin, val.bin, meta.json")


if __name__ == "__main__":
    main()
