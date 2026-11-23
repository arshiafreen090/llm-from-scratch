"""
Train our BPE tokenizer on a sample of TinyStories, then report how well it compresses.

Run from the project root, inside the venv:
    python -m tokenizer.train_bpe                                   # 4096 tokens, 20 MB sample
    python -m tokenizer.train_bpe --vocab_size 1024 --out tokenizer/tinystories_1k.json

Why only a sample? BPE learns which pairs are frequent, and a 20 MB sample already
shows those patterns clearly. Training on all ~2 GB would take much longer for
almost the same merges.
"""

import argparse
import time

import tiktoken
from datasets import load_dataset

from tokenizer.tokenizer import Tokenizer


def load_stories(split, max_mb):
    """Stream stories from Hugging Face until we have `max_mb` megabytes.
    streaming=True downloads only what we read, not the whole dataset."""
    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    stories, size = [], 0
    for row in ds:
        story = row["text"].strip()
        if not story:
            continue
        stories.append(story)
        size += len(story.encode("utf-8"))
        if size >= max_mb * 1e6:
            break
    return stories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=4096)
    parser.add_argument("--train_mb", type=float, default=20, help="MB of training text")
    parser.add_argument("--val_mb", type=float, default=2, help="MB of held-out text for evaluation")
    parser.add_argument("--out", default="tokenizer/tinystories_4k.json")
    args = parser.parse_args()

    # ---- 1. data --------------------------------------------------------------
    print(f"downloading {args.train_mb} MB of TinyStories (train split)...")
    train_text = "\n".join(load_stories("train", args.train_mb))

    # ---- 2. train -------------------------------------------------------------
    tok = Tokenizer()
    t0 = time.time()
    tok.train(train_text, vocab_size=args.vocab_size)
    print(f"trained in {time.time() - t0:.1f}s, vocab size = {tok.vocab_size}")

    tok.save(args.out)
    print(f"saved {args.out}  (+ readable token list in {args.out[:-5]}.vocab.txt)")

    # ---- 3. evaluate on stories the tokenizer never saw ------------------------
    print(f"\nevaluating on {args.val_mb} MB of the validation split...")
    val_stories = load_stories("validation", args.val_mb)
    val_text = "<|endoftext|>".join(val_stories)

    t0 = time.time()
    ids = tok.encode(val_text)
    encode_time = time.time() - t0
    n_bytes = len(val_text.encode("utf-8"))

    # the most important check: encode then decode must give back the exact text
    assert tok.decode(ids) == val_text, "round trip failed!"

    gpt2 = tiktoken.get_encoding("gpt2")
    gpt2_ids = gpt2.encode(val_text, allowed_special={"<|endoftext|>"})

    print(f"  round trip decode(encode(text)) == text: OK")
    print(f"  encoding speed: {n_bytes / 1e6 / encode_time:.2f} MB/s")
    print(f"  {'tokenizer':<22}{'vocab':>8}{'tokens':>12}{'bytes/token':>14}")
    print(f"  {'ours':<22}{tok.vocab_size:>8,}{len(ids):>12,}{n_bytes / len(ids):>14.2f}")
    print(f"  {'GPT-2 (tiktoken)':<22}{gpt2.n_vocab:>8,}{len(gpt2_ids):>12,}{n_bytes / len(gpt2_ids):>14.2f}")
    print("  (higher bytes/token = better compression = shorter sequences for the model)")

    # ---- 4. show what tokens look like ----------------------------------------
    example = "Once upon a time, there was a little girl named Lily. She loved to play outside.<|endoftext|>"
    example_ids = tok.encode(example)
    print(f"\nexample ({len(example_ids)} tokens):")
    print("  " + " | ".join(tok.token_strings(example_ids)))


if __name__ == "__main__":
    main()
