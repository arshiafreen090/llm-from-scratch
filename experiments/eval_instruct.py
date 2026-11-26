"""
Measure instruction following: do generated stories use the required words and follow the plot?

    python -m experiments.eval_instruct --ckpts checkpoints/small-llama-s1.pt checkpoints/sft-small-llama-s1.pt
    python -m experiments.eval_instruct --ckpts v1.pt v2.pt --examples data/tinystories_sft/val_examples.jsonl --n 500

For every validation example the model writes a story, and we check:
    word inclusion   average share of the required words that appear            (main metric)
    all words        share of stories that use ALL the required words
    plot words       for instructions that gave a plot ("about Lily riding a bike"): the share of
                     the plot's content words (lily, riding, bike) that appear in the story. A
                     rough proxy for "is the story about what was asked"
    stopped          share of stories that ended with <|endoftext|> instead of running out of
                     length: did the model learn to finish its answer?

The validation set (data/prepare_sft.py) phrases half of its instructions with templates that
were held out of training. Results are reported for all examples, then for "seen" and "unseen"
phrasings separately: the unseen rows show whether the model follows the REQUEST or only the
sentences it was trained on.

Baseline: a pretrained (non-SFT) model has never seen the chat format, so it's evaluated in
"plain" mode: it just writes a story with no instruction. Its word inclusion is the CHANCE
level, i.e. how often those words show up anyway. SFT has to beat that to mean anything.
Both modes use the same examples, sampling settings and seeds.

Stories are generated with continuous batching (inference/batching.py): the same text as
generating one at a time, several times faster.
"""

import argparse
import json
import re
import statistics

from inference.batching import BatchEngine, Sequence
from inference.generate import Generator
from inference.quantize import MODES as QUANT_MODES
from inference.sampling import SamplingParams

# words that say nothing about what a story is about
STOPWORDS = set("""
a an the and or but so because then when while with without about into onto from of to in on at
by for as is are was were be been being has have had do does did not no very his her their its
he she it they them him we you i me my our your this that these those there here who which what
where how why one two some any all each both more most other such only own same than too can will
just should would could get gets got goes went go come came day little big new old
""".split())


def word_present(word, text):
    """Whole-word match that also accepts simple inflections: quit -> quits, quitting is NOT
    matched (doubled consonant), jump -> jumped/jumping/jumps are."""
    pattern = rf"\b{re.escape(word.lower())}(s|es|d|ed|ing)?\b"
    return re.search(pattern, text.lower()) is not None


def plot_words(topic):
    """Content words of a plot summary: what a story about it should mention."""
    return [w for w in re.findall(r"[a-z']+", topic.lower()) if len(w) >= 3 and w not in STOPWORDS]


def generate_all(generator, examples, params, mode, max_batch=16):
    """One story per example, with the same seed per example whatever the mode or model."""
    engine = BatchEngine(generator, max_batch)
    seqs = []
    for i, ex in enumerate(examples):
        p = SamplingParams(**{**params.__dict__, "seed": (params.seed or 0) + i})
        if mode == "chat":
            prompt = generator.encode_prompt(ex["instruction"], chat=True)
        else:
            prompt = generator.encode_prompt("")  # no instruction: a story from scratch
        seqs.append(Sequence(prompt, p))
        engine.add(seqs[-1])
    while engine.has_work():
        for _, kind, value in engine.step():
            assert kind != "error", value
    return [s.result for s in seqs]


def score(examples, results):
    rows = []
    for ex, r in zip(examples, results):
        hits = [word_present(w, r.text) for w in ex["words"]]
        plot = plot_words(ex["topic"]) if ex.get("topic") else []
        rows.append({
            "word_inclusion": sum(hits) / len(hits),
            "all_words": float(all(hits)),
            "plot_words": sum(word_present(w, r.text) for w in plot) / len(plot) if plot else None,
            "stopped": float(r.finish_reason == "stop"),
            "tokens": len(r.token_ids),
            "phrasing": ex.get("phrasing", "seen"),
        })
    return rows


def summarize(rows, mode):
    plot = [r["plot_words"] for r in rows if r["plot_words"] is not None]
    return {"mode": mode, "n": len(rows),
            "word_inclusion": statistics.mean(r["word_inclusion"] for r in rows),
            "all_words": statistics.mean(r["all_words"] for r in rows),
            "plot_words": statistics.mean(plot) if plot else None, "n_with_plot": len(plot),
            "stopped": statistics.mean(r["stopped"] for r in rows),
            "mean_tokens": statistics.mean(r["tokens"] for r in rows)}


def evaluate(generator, examples, params, mode, max_batch=16):
    """Scores over all examples, plus "by_phrasing" when the examples mix seen and unseen phrasings."""
    rows = score(examples, generate_all(generator, examples, params, mode, max_batch))
    summary = summarize(rows, mode)
    phrasings = sorted({r["phrasing"] for r in rows})
    if len(phrasings) > 1:
        summary["by_phrasing"] = {p: summarize([r for r in rows if r["phrasing"] == p], mode) for p in phrasings}
    return summary


def load_examples(path, n):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for _, line in zip(range(n), f)]


def pct(x):
    return "—" if x is None else f"{x:.1%}"


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
    parser.add_argument("--max_batch", type=int, default=16, help="stories generated together")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quant", default=None, choices=QUANT_MODES, help="quantize every checkpoint when loading")
    args = parser.parse_args()

    examples = load_examples(args.examples, args.n)
    params = SamplingParams(max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                            top_p=args.top_p, seed=args.seed)
    rows = []
    for path in args.ckpts:
        gen = Generator.from_checkpoint(path, args.device, args.quant)
        name = path + (f" ({gen.model.quant})" if getattr(gen.model, "quant", None) and not path.endswith("-int8.pt") else "")
        mode = args.mode if args.mode != "auto" else ("chat" if gen.is_sft else "plain")
        r = evaluate(gen, examples, params, mode, args.max_batch)
        print(f"{name}: {json.dumps({k: v for k, v in r.items() if k != 'by_phrasing'})}")
        rows.append((name, "all", r))
        for phrasing, sub in r.get("by_phrasing", {}).items():
            rows.append((name, phrasing, sub))

    print(f"\n{len(examples)} validation instructions, temperature {args.temperature}, top_p {args.top_p}, "
          f"max {args.max_new_tokens} tokens. 'unseen' = phrasings held out of SFT training.\n")
    print("| checkpoint | phrasing | n | mode | word inclusion | all words | plot words | stopped | mean tokens |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, phrasing, r in rows:
        print(f"| {name} | {phrasing} | {r['n']} | {r['mode']} | {pct(r['word_inclusion'])} | {pct(r['all_words'])} "
              f"| {pct(r['plot_words'])} | {pct(r['stopped'])} | {r['mean_tokens']:.0f} |")


if __name__ == "__main__":
    main()
