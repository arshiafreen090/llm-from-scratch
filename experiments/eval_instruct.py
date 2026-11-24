"""
Measure instruction following: do generated stories actually use the required words?

    python -m experiments.eval_instruct --ckpts checkpoints/small-llama-s1.pt checkpoints/sft-small-llama.pt

For every validation example ("... Use the words: quit, oak, gloomy ..."), the model writes a
story and we check which required words appear in it.

    word inclusion   average fraction of required words used        (main metric)
    all words        fraction of stories that use ALL required words
    stopped          fraction of stories that ended with <|endoftext|> instead of running
                     out of length: did the model learn to finish its answer?

Baseline: a pretrained (non-SFT) model has never seen the chat format, so it's evaluated in
"plain" mode: it just writes a story with no instruction. Its word inclusion is the CHANCE
level, i.e. how often those words show up anyway. SFT has to beat that to mean anything.
Both modes use the same examples, sampling settings and seeds.
"""

import argparse
import json
import re
import statistics

from inference.generate import Generator
from inference.quantize import MODES as QUANT_MODES
from inference.sampling import SamplingParams


def word_present(word, text):
    """Whole-word match that also accepts simple inflections: quit -> quits, quitting is NOT
    matched (doubled consonant), jump -> jumped/jumping/jumps are."""
    pattern = rf"\b{re.escape(word.lower())}(s|es|d|ed|ing)?\b"
    return re.search(pattern, text.lower()) is not None


def evaluate(generator, examples, params, mode):
    inclusion, all_words, stopped, lengths = [], [], [], []
    for i, ex in enumerate(examples):
        p = SamplingParams(**{**params.__dict__, "seed": (params.seed or 0) + i})  # same seed per example in every mode
        if mode == "chat":
            result = generator.generate(ex["instruction"], p, chat=True)
        else:
            result = generator.generate("", p)  # no instruction: a story from scratch
        hits = [word_present(w, result.text) for w in ex["words"]]
        inclusion.append(sum(hits) / len(hits))
        all_words.append(all(hits))
        stopped.append(result.finish_reason == "stop")
        lengths.append(len(result.token_ids))
    return {"mode": mode, "n": len(examples),
            "word_inclusion": statistics.mean(inclusion),
            "all_words": statistics.mean(all_words),
            "stopped": statistics.mean(stopped),
            "mean_tokens": statistics.mean(lengths)}


def load_examples(path, n):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for _, line in zip(range(n), f)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts", nargs="+", required=True)
    parser.add_argument("--examples", default="data/tinystories_sft/val_examples.jsonl")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--mode", default="auto", choices=["auto", "chat", "plain"],
                        help="auto: chat for SFT checkpoints, plain (chance baseline) otherwise")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quant", default=None, choices=QUANT_MODES, help="quantize every checkpoint when loading")
    args = parser.parse_args()

    examples = load_examples(args.examples, args.n)
    params = SamplingParams(max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                            top_p=args.top_p, seed=args.seed)
    rows = []
    for path in args.ckpts:
        gen = Generator.from_checkpoint(path, args.device, args.quant)
        if getattr(gen.model, "quant", None):
            path = f"{path} ({gen.model.quant})"
        mode = args.mode if args.mode != "auto" else ("chat" if gen.is_sft else "plain")
        r = evaluate(gen, examples, params, mode)
        rows.append((path, r))
        print(f"{path}: {r}")

    print(f"\n{len(examples)} validation instructions, temperature {args.temperature}, top_p {args.top_p}, "
          f"max {args.max_new_tokens} tokens\n")
    print("| checkpoint | mode | word inclusion | all words | stopped | mean tokens |")
    print("|---|---|---|---|---|---|")
    for path, r in rows:
        print(f"| {path} | {r['mode']} | {r['word_inclusion']:.1%} | {r['all_words']:.1%} | "
              f"{r['stopped']:.1%} | {r['mean_tokens']:.0f} |")


if __name__ == "__main__":
    main()
