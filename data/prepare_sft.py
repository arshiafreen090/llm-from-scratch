"""
Build the instruction-tuning (SFT) dataset from TinyStories-Instruct.

Each raw record has a story plus annotations, in random order, sometimes AFTER the story:
    Features: Dialogue, BadEnding        Words: quit, oak, gloomy
    Summary: Sara and Ben were ...       Random sentence: She hoped the apples would fall.
    Story:
    <the story, one paragraph per row>
    <|endoftext|>

We turn every record into an (instruction, response) pair:
    instruction: "write a story about Sara and Ben picking apples. use the words quit, oak and gloomy"
    response:    the story

phrased in many different ways (see "How the annotations are phrased" below), and tokenize it
with the shared chat template (tokenizer/chat.py), keeping a per-token flag that says which
tokens are the response (the only ones the loss is computed on).

Run from the project root:
    python -m data.prepare_sft                          # 100k train examples, 5k val
    python -m data.prepare_sft --max_train 300000       # what the v2 model was trained on
    python -m data.prepare_sft --max_train 2000         # quick test

Output, in data/tinystories_sft/:
    {train,val}.bin           all examples' token ids back to back (uint16)
    {train,val}_target.bin    1 where the token is part of the response, else 0 (uint8)
    {train,val}_offsets.npy   where each example starts (and the end of the last one)
    val_examples.jsonl        readable val examples for evaluation: instruction, the words asked
                              for, the plot summary if it was asked for ("topic"), and whether
                              the phrasing was "seen" in training or held out ("unseen")
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
FEATURE_JOINERS = ["Also, {f}.", "Make sure to {f}.", "And {f}.", "{F}, please.", "Oh, and {f}."]

# -----------------------------------------------------------------------------
# How the annotations are phrased
#
# Version 1 used 5 openers x 3 word templates x 3 plot templates, always in tidy English. The
# model learned THOSE sentences rather than the request: asked "write a story about fish which
# is dancing in river. use words octopus and jellyfish", it ignored the fish and the river.
# Version 2 phrases the same annotations many ways, including how people type: lowercase, no
# punctuation, the topic fused into the opener ("write a story about ..."), "use words a and b",
# the words in any order, sometimes only some of them. A few phrasings are HELD OUT of
# training and used only in the validation set, to measure whether the model follows requests
# it has never seen worded that way (eval_instruct reports "seen" and "unseen" separately).
#
# Slots: {w} the word list, {s} the plot summary as given ("Lily rides a bike."),
#        {t} the summary as a topic phrase, without its final period ("Lily rides a bike"),
#        {x} the required sentence, {f}/{F} a feature phrase ("include some dialogue").

TRAIN_POOLS = {
    "openers": ["Write a short story.", "Write a story for children.", "Tell me a story.",
                "Please write a simple story.", "Write a story.", "Can you write a story?",
                "I want a story.", "Make up a story.", "Write me a bedtime story.", "Story time!",
                "Give me a short story.", "Write a little story for a kid."],
    # opener and topic in one sentence
    "fused": ["Write a story about {t}.", "Tell me a story about {t}.", "Write a short story where {t}.",
              "I want a story about {t}.", "Can you write a story in which {t}?", "A story about {t}.",
              "Make up a story about {t}.", "Write a children's story about {t}."],
    "topic": ["The story should be about this: {s}", "Plot: {s}", "Here is what happens: {s}",
              "It's about {t}.", "Story idea: {s}", "Make it about {t}.", "Topic: {s}", "In the story, {t}."],
    "words": ["Use the words: {w}.", "Include these words: {w}.", "The story must use the words {w}.",
              "use the words {w}", "Use words {w}.", "with the words {w}", "It should have the words {w} in it.",
              "Make sure to use {w}.", "Words to include: {w}.", "words: {w}", "Try to use {w}.",
              "Include the words {w}."],
    "sentence": ['Include the sentence: "{x}"', 'Use this sentence somewhere: "{x}"',
                 'The story must contain the sentence "{x}"', 'Put the sentence "{x}" in it.'],
}
# validation only: none of these appear in any training example
HELD_OUT_POOLS = {
    "openers": ["Could you tell me a short story?", "Please make a story for me."],
    "fused": ["Could you write a story about {t}?", "Please tell me a story where {t}."],
    "topic": ["Here's the idea: {s}", "The story is about {t}."],
    "words": ["Please put the words {w} somewhere in it.", "Required words: {w}.", "Using {w}."],
    "sentence": ['Somewhere in the story, use the sentence "{x}"'],
}
# kept for reading old code and tests; the pools above replace them
OPENERS, WORD_TEMPLATES, SUMMARY_TEMPLATES = TRAIN_POOLS["openers"], TRAIN_POOLS["words"], TRAIN_POOLS["topic"]


def join_words(words, rng):
    """'a, b, c' / 'a, b and c' / 'a, b, and c' / 'a and b': the ways people list words."""
    if len(words) == 1:
        return words[0]
    style = rng.random()
    if style < 0.5:
        return ", ".join(words)
    if style < 0.8 or len(words) == 2:
        return ", ".join(words[:-1]) + " and " + words[-1]
    if style < 0.9:
        return ", ".join(words[:-1]) + ", and " + words[-1]   # Oxford comma, 3+ words only
    return ", ".join(words)


def typing_noise(text, rng):
    """People type in lowercase and skip the final punctuation; the model should cope."""
    if rng.random() < 0.25:
        text = text.lower()
    if rng.random() < 0.15:
        text = text.rstrip(".!?")
    return text


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


def build_instruction(record, rng, phrasing="train"):
    """Phrase the record's annotations as an instruction a person might type.

    Returns (text, info). info says what was actually asked for, since not every annotation
    goes into every instruction: info["words"] (the words asked for, possibly a subset),
    info["topic"] (the plot summary if it was included, else None), info["phrasing"]
    ("seen": built from the training pools, "unseen": from the held-out pools)."""
    pools = TRAIN_POOLS if phrasing == "train" else HELD_OUT_POOLS
    summary = record.get("summary")
    topic = summary.strip().rstrip(".!") if summary else None   # "Lily rides a bike." -> "Lily rides a bike"

    words = list(record.get("words", []))
    if len(words) >= 2 and rng.random() < 0.2:
        words.pop(rng.randrange(len(words)))   # people often ask for fewer words than the data lists
    if rng.random() < 0.3:
        rng.shuffle(words)

    # the topic goes in 70% of the time: following it is the main thing people notice
    use_topic = bool(summary) and rng.random() < 0.7
    fused = use_topic and rng.random() < 0.5    # "Write a story about <topic>." as the opener

    parts = []
    if words:
        parts.append(rng.choice(pools["words"]).format(w=join_words(words, rng)))
    if use_topic and not fused:
        parts.append(rng.choice(pools["topic"]).format(s=summary, t=topic))
    if record.get("sentence") and rng.random() < 0.4:
        parts.append(rng.choice(pools["sentence"]).format(x=record["sentence"]))
    rng.shuffle(parts)
    # features last: "Also, include some dialogue" only reads right after something else
    for f in record.get("features", []):
        if f in FEATURE_TEXT:
            phrase = FEATURE_TEXT[f]
            parts.append(rng.choice(FEATURE_JOINERS).format(f=phrase, F=phrase[0].upper() + phrase[1:]))

    if fused:
        opener = rng.choice(pools["fused"]).format(t=topic)
    else:
        opener = rng.choice(pools["openers"]) if rng.random() < 0.9 else ""   # sometimes no opener at all
    text = typing_noise(" ".join([opener] + parts).strip(), rng)
    return text, {"words": words, "topic": summary if use_topic else None,
                  "phrasing": "seen" if phrasing == "train" else "unseen"}


# -----------------------------------------------------------------------------

def build_split(split, max_examples, tok, block_size, seed, out_dir, name, n_readable=0, held_out_frac=0.0):
    """held_out_frac: share of examples phrased with the held-out pools (validation only)."""
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
            phrasing = "unseen" if rng.random() < held_out_frac else "train"
            instruction, info = build_instruction(record, rng, phrasing)
            ids, is_target = example_ids(tok, instruction, record["story"])
            if len(ids) > block_size + 1:  # +1: inputs and targets are the sequence shifted by one
                stats["dropped_too_long"] += 1
                continue
            ids_all.extend(ids)
            target_all.extend(is_target)
            offsets.append(len(ids_all))
            stats[f"phrasing:{info['phrasing']}"] += 1
            if len(readable) < n_readable and info["words"]:
                readable.append({"instruction": instruction, "words": info["words"], "topic": info["topic"],
                                 "phrasing": info["phrasing"], "features": record.get("features", []),
                                 "response": record["story"]})
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
    # half of the validation set uses phrasings the model never trains on
    n_val, val_stats = build_split("validation", args.max_val, tok, args.block_size, args.seed + 1, args.out_dir,
                                   "val", n_readable=1000, held_out_frac=0.5)
    n_train, stats = build_split("train", args.max_train, tok, args.block_size, args.seed, args.out_dir, "train")

    features = {k.split(":", 1)[1]: v for k, v in stats.items() if k.startswith("feature:")}
    unknown = sorted(set(features) - set(FEATURE_TEXT))
    if unknown:
        print(f"note: features not phrased in instructions (ignored): {unknown}")
    meta = {"tokenizer": args.tokenizer, "vocab_size": tok.vocab_size, "block_size": args.block_size,
            "train_examples": n_train, "val_examples": n_val, "feature_counts": features, "seed": args.seed,
            "phrasing_version": 2, "val_unseen_phrasing": val_stats["phrasing:unseen"]}
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # show a few instructions exactly as the model will see them
    with open(os.path.join(args.out_dir, "val_examples.jsonl"), encoding="utf-8") as f:
        examples = [json.loads(line) for _, line in zip(range(6), f)]
    print()
    for ex in examples:
        print(f"[{ex['phrasing']:>6}] {ex['instruction']}")
    print("\nexample response:", examples[0]["response"][:200].replace("\n", " "), "...")


if __name__ == "__main__":
    main()
