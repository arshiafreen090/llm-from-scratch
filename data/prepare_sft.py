"""
Build the instruction-tuning (SFT) dataset from TinyStories-Instruct.

Each raw record has a story plus annotations, in random order, sometimes AFTER the story:
    Features: Dialogue, BadEnding        Words: quit, oak, gloomy
    Summary: Sara and Ben were ...       Random sentence: She hoped the apples would fall.
    Story:
    <the story, one paragraph per row>
    <|endoftext|>

We turn every record into an (instruction, response) pair:
    instruction: "Write a short story. Use the words: quit, oak, gloomy. Also, include some dialogue."
    response:    the story

and tokenize it with the shared chat template (tokenizer/chat.py), keeping a per-token flag
that says which tokens are the response (the only ones the loss is computed on).

Run from the project root:
    python -m data.prepare_sft                          # 100k train examples, 5k val
    python -m data.prepare_sft --max_train 2000         # quick test

Output, in data/tinystories_sft/:
    {train,val}.bin           all examples' token ids back to back (uint16)
    {train,val}_target.bin    1 where the token is part of the response, else 0 (uint8)
    {train,val}_offsets.npy   where each example starts (and the end of the last one)
    val_examples.jsonl        readable val examples with their required words (for evaluation)
    meta.json
"""

import argparse
import json
import os
import random
from collections import Counter

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from tokenizer.chat import example_ids
from tokenizer.tokenizer import Tokenizer

DATASET = "roneneldan/TinyStoriesInstruct"
FIELDS = {"Features:": "features", "Words:": "words", "Summary:": "summary", "Random sentence:": "sentence"}

# how each story feature is phrased in an instruction
FEATURE_TEXT = {
    "Dialogue": "include some dialogue",
    "BadEnding": "give it a sad ending",
    "MoralValue": "end with a lesson or moral",
    "Foreshadowing": "use some foreshadowing",
    "Twist": "include a surprising twist",
    "Conflict": "include a conflict",
}
OPENERS = ["Write a short story.", "Write a story for children.", "Tell me a story.",
           "Please write a simple story.", "Write a story."]
WORD_TEMPLATES = ["Use the words: {w}.", "Include these words: {w}.", "The story must use the words {w}."]
SUMMARY_TEMPLATES = ["The story should be about this: {s}", "Plot: {s}", "Here is what happens: {s}"]


# -----------------------------------------------------------------------------
# Parsing

def parse_records(lines):
    """Turn the stream of dataset rows into records: dict(story=..., words=[...], ...)."""
    record, story = {}, None
    for line in lines:
        if line.strip() == "<|endoftext|>":
            if story:
                record["story"] = "\n\n".join(story)  # paragraphs separated like the pretraining data
                yield record
            record, story = {}, None
            continue

        field = next((key for prefix, key in FIELDS.items() if line.startswith(prefix)), None)
        if field is not None:
            # a field line, before OR after the story (e.g. a Summary written after it)
            value = line.split(":", 1)[1].strip()
            if field in ("features", "words"):
                record[field] = [v.strip() for v in value.split(",") if v.strip()]
            else:
                record[field] = value
        elif line.startswith("Story:"):
            story = []
            rest = line[len("Story:"):].strip()
            if rest:
                story.append(rest)
        elif story is not None and line.strip():
            story.append(line.strip())


def build_instruction(record, rng):
    """Phrase the record's annotations as a natural instruction. Phrasing and order are
    randomized so the model learns the MEANING of the request, not one fixed sentence."""
    parts = []
    if record.get("words"):  # always included: it's what we measure instruction-following with
        parts.append(rng.choice(WORD_TEMPLATES).format(w=", ".join(record["words"])))
    if record.get("summary") and rng.random() < 0.5:
        parts.append(rng.choice(SUMMARY_TEMPLATES).format(s=record["summary"]))
    features = [FEATURE_TEXT[f] for f in record.get("features", []) if f in FEATURE_TEXT]
    if features:
        parts.append("Also, " + ", ".join(features) + ".")
    if record.get("sentence") and rng.random() < 0.5:
        parts.append(f'Include the sentence: "{record["sentence"]}"')
    rng.shuffle(parts)
    return " ".join([rng.choice(OPENERS)] + parts)


# -----------------------------------------------------------------------------

def build_split(split, max_examples, tok, block_size, seed, out_dir, name, n_readable=0):
    ds = load_dataset(DATASET, split=split, streaming=True)
    rng = random.Random(seed)
    ids_all, target_all, offsets = [], [], [0]
    readable = []
    stats = Counter()
    with tqdm(total=max_examples, desc=name, unit=" examples") as bar:
        for record in parse_records(row["text"] for row in ds):
            stats["records"] += 1
            for f in record.get("features", []):
                stats[f"feature:{f}"] += 1
            instruction = build_instruction(record, rng)
            ids, is_target = example_ids(tok, instruction, record["story"])
            if len(ids) > block_size + 1:  # +1: inputs and targets are the sequence shifted by one
                stats["dropped_too_long"] += 1
                continue
            ids_all.extend(ids)
            target_all.extend(is_target)
            offsets.append(len(ids_all))
            if len(readable) < n_readable and record.get("words"):
                readable.append({"instruction": instruction, "words": record["words"],
                                 "features": record.get("features", []), "response": record["story"]})
            bar.update(1)
            if len(offsets) - 1 >= max_examples:
                break

    np.array(ids_all, dtype=np.uint16).tofile(os.path.join(out_dir, f"{name}.bin"))
    np.array(target_all, dtype=np.uint8).tofile(os.path.join(out_dir, f"{name}_target.bin"))
    np.save(os.path.join(out_dir, f"{name}_offsets.npy"), np.array(offsets, dtype=np.int64))
    if readable:
        with open(os.path.join(out_dir, f"{name}_examples.jsonl"), "w", encoding="utf-8") as f:
            for r in readable:
                f.write(json.dumps(r) + "\n")
    n = len(offsets) - 1
    lengths = np.diff(offsets)
    print(f"{name}: {n:,} examples, {len(ids_all):,} tokens ({int(sum(target_all)):,} response tokens), "
          f"length median {int(np.median(lengths))} / max {lengths.max()}, "
          f"dropped {stats['dropped_too_long']} longer than {block_size + 1} tokens")
    return n, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="tokenizer/tinystories_4k.json")
    parser.add_argument("--out_dir", default="data/tinystories_sft")
    parser.add_argument("--max_train", type=int, default=100_000)
    parser.add_argument("--max_val", type=int, default=5_000)
    parser.add_argument("--block_size", type=int, default=512, help="must match the model's context")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tok = Tokenizer.load(args.tokenizer)
    os.makedirs(args.out_dir, exist_ok=True)
    n_val, _ = build_split("validation", args.max_val, tok, args.block_size, args.seed + 1, args.out_dir, "val",
                           n_readable=1000)
    n_train, stats = build_split("train", args.max_train, tok, args.block_size, args.seed, args.out_dir, "train")

    features = {k.split(":", 1)[1]: v for k, v in stats.items() if k.startswith("feature:")}
    unknown = sorted(set(features) - set(FEATURE_TEXT))
    if unknown:
        print(f"note: features not phrased in instructions (ignored): {unknown}")
    meta = {"tokenizer": args.tokenizer, "vocab_size": tok.vocab_size, "block_size": args.block_size,
            "train_examples": n_train, "val_examples": n_val, "feature_counts": features, "seed": args.seed}
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # show one example exactly as the model will see it
    ex = json.loads(open(os.path.join(args.out_dir, "val_examples.jsonl")).readline())
    print("\nexample instruction:", ex["instruction"])
    print("example response    :", ex["response"][:200].replace("\n", " "), "...")


if __name__ == "__main__":
    main()
