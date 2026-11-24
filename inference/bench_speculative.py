"""
Plain decoding vs speculative decoding (n-gram drafter, draft model with several k).

    python -m inference.bench_speculative --ckpt checkpoints/small-llama-s1.pt --draft checkpoints/draft-llama.pt
    python -m inference.bench_speculative ... --temperatures 0 0.7 --ks 2 4 6 --rounds 3 --out experiments/bench/spec.json

For each setting it reports:
    tokens per target pass   how many tokens one forward pass of the big model produced
    guesses accepted         share of drafted tokens the target agreed with
    decode tok/s             end-to-end speed, median over every generation (all prompts, all rounds)
The settings take turns within each round, so a slow moment on the laptop hits all of them.
With temperature 0 it also checks that every speculative output equals the plain one.
"""

import argparse
import json
import os
import statistics

from inference.bench_batching import PROMPTS
from inference.generate import Generator
from inference.sampling import SamplingParams
from inference.speculative import ModelDrafter, NgramDrafter, SpeculativeGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="the target model")
    parser.add_argument("--draft", required=True, help="the draft model checkpoint")
    parser.add_argument("--ks", type=int, nargs="+", default=[2, 4, 6], help="draft lengths to try")
    parser.add_argument("--ngram_k", type=int, default=4)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.0, 0.7])
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--new_tokens", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--chat", action="store_true", help="prompts are instructions (for an SFT target)")
    parser.add_argument("--out", default=None, help="save the results as JSON")
    args = parser.parse_args()

    plain = Generator.from_checkpoint(args.ckpt, "cpu")
    draft = Generator.from_checkpoint(args.draft, "cpu")
    assert draft.tok.vocab == plain.tok.vocab, "the draft must use the target's tokenizer"
    print(f"target {plain.model.num_params() / 1e6:.1f}M params, draft {draft.model.num_params() / 1e6:.1f}M params, CPU")

    settings = {"plain": plain,
                f"n-gram, k={args.ngram_k}": SpeculativeGenerator(plain.model, plain.tok, NgramDrafter(), args.ngram_k)}
    for k in args.ks:
        settings[f"draft model, k={k}"] = SpeculativeGenerator(plain.model, plain.tok, ModelDrafter(draft.model), k)
    prompts = PROMPTS if not args.chat else [
        "Write a story. Use the words: rain, ball, happy.", "Write a story for children. Include these words: meet, waffle, new.",
        "Please write a simple story with a happy ending. Include these words: dog, moon, brave."]

    rows = []
    for temperature in args.temperatures:
        params = SamplingParams(max_new_tokens=args.new_tokens, temperature=temperature,
                                top_p=args.top_p if temperature > 0 else None, stop_at_special=False)
        speeds = {name: [] for name in settings}
        totals = {name: [0, 0, 0, 0] for name in settings}  # rounds, drafted, accepted, tokens
        mismatches = 0
        for gen in settings.values():  # warm-up
            gen.generate(prompts[0], SamplingParams(max_new_tokens=20, temperature=temperature), chat=args.chat)
        for r in range(args.rounds):
            for i, prompt in enumerate(prompts):
                p = SamplingParams(**{**params.__dict__, "seed": 1000 * r + i})
                reference = None
                for name, gen in settings.items():
                    result = gen.generate(prompt, p, chat=args.chat)
                    speeds[name].append(result.decode_tokens_per_s)
                    if name == "plain":
                        reference = result.token_ids
                    else:
                        s = gen.stats
                        for j, v in enumerate((s.rounds, s.drafted, s.accepted, s.tokens)):
                            totals[name][j] += v
                        if temperature == 0 and result.token_ids != reference:
                            mismatches += 1
            print(f"  temperature {temperature}: round {r + 1}/{args.rounds} done", flush=True)

        base = statistics.median(speeds["plain"])
        for name in settings:
            n_rounds, drafted, accepted, tokens = totals[name]
            speed = statistics.median(speeds[name])
            rows.append({"setting": name, "temperature": temperature, "decode_tok_s": speed, "speedup": speed / base,
                         "tokens_per_pass": tokens / n_rounds if n_rounds else 1.0,
                         "acceptance": accepted / drafted if drafted else None})
        if temperature == 0:
            print(f"  greedy outputs identical to plain decoding: {mismatches == 0} ({mismatches} mismatches)")

    print("\n| decoding | temperature | tokens per target pass | guesses accepted | decode tok/s | speedup |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        acc = f"{r['acceptance']:.0%}" if r["acceptance"] is not None else "—"
        print(f"| {r['setting']} | {r['temperature']:g} | {r['tokens_per_pass']:.2f} | {acc} "
              f"| {r['decode_tok_s']:.0f} | {r['speedup']:.2f}x |")
    n_gen = args.rounds * len(prompts)
    print(f"\nmedians over {n_gen} generations of {args.new_tokens} tokens per setting")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "results": rows}, f, indent=2)
        print("saved", args.out)


if __name__ == "__main__":
    main()
