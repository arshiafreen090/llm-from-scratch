"""
Speculative decoding: a cheap drafter guesses the next k tokens, the real model checks all
of them in ONE forward pass, and every guess it agrees with is a token we didn't have to
generate one at a time.

    python -m inference.speculative --ckpt checkpoints/small-llama-s1.pt --draft checkpoints/draft-llama.pt
    python -m inference.speculative --ckpt checkpoints/small-llama-s1.pt --draft ngram --temperature 0

In code:
    gen = SpeculativeGenerator.from_checkpoints("checkpoints/small-llama-s1.pt", "checkpoints/draft-llama.pt")
    result = gen.generate("Once upon a time", SamplingParams(temperature=0.7))   # same API as Generator
    print(gen.stats)                                                              # acceptance, tokens/round

WHY IT CAN BE FASTER
    Generating one token = one forward pass, and on a CPU a small model's pass is dominated by
    fixed per-layer overhead, not arithmetic: feeding 5 tokens costs ~3.8 ms vs ~3.5 ms for 1
    (measured on the mini-LLaMA). So checking 4 guesses costs about as much as making 1 token.

ONE ROUND (the target model's KV cache holds every token except the newest)
    1. the drafter proposes x1..xk, each drawn from its distribution q_i
    2. the target runs on [newest token, x1..xk] in one pass -> distributions p_1..p_k+1
    3. for i = 1..k: accept x_i with probability min(1, p_i(x_i) / q_i(x_i)).
       At the first rejection, draw a replacement from max(0, p_i - q_i) (normalized) and stop.
    4. if all k were accepted, draw one bonus token from p_k+1
    So every round makes between 1 and k+1 tokens, for one target pass.

WHY THE OUTPUT IS STILL THE TARGET MODEL'S (Leviathan et al. 2023, Chen et al. 2023)
    Accept-or-resample is built so that each produced token is distributed EXACTLY as p_i:
    where q over-proposes a token (q > p) it gets rejected just often enough, and the
    residual max(0, p - q) puts back exactly the mass q under-proposed. A bad drafter only
    makes it slower, never different. With temperature 0, p and q are one-hot, so a guess
    is accepted iff it's the target's argmax: greedy output is token-for-token identical to
    plain greedy decoding. With sampling, the text follows the same distribution but is not
    the same text as Generator gives for a seed (the random numbers are used differently).

    Repetition penalty, top-k/p and min-p apply to both p and q (see token_probs), with each
    position's history including the guesses before it: a guess is only checked if all the
    guesses before it were accepted, and then that history is the real one.

TWO DRAFTERS
    ModelDrafter   a 2.5M-parameter model (size "draft": 2 layers x 256) trained on the same
                   data. ~0.3x the target's cost per step. Train it (~40 min on a laptop CPU):
                       python -m train.pretrain --size draft --token_budget 24000000 --batch_size 32 \\
                           --grad_accum 1 --out_dir experiments/results_draft --run_name draft-llama
    NgramDrafter   no model: if the last few tokens appeared earlier in the text, propose what
                   followed them then ("prompt lookup decoding"). Free, but only right when
                   the text repeats itself, and as a one-hot guess it's accepted with
                   probability p(x), so it helps most with greedy decoding.
"""

import argparse
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from inference.generate import GenerationResult, Generator
from inference.sampling import SamplingParams, random_seed, token_probs


def draw(dist, rng):
    """One token id from a probability vector (vocab,)."""
    return int(torch.multinomial(dist, 1, generator=rng))


def verify(draft_ids, q, p, rng):
    """Steps 3-4 of a round, on plain tensors.
        draft_ids  k proposed tokens
        q          (k, vocab)   the drafter's distributions they were drawn from
        p          (k+1, vocab) the target's distributions at the same positions (+1 bonus)
    Returns (new tokens, number of draft tokens accepted). Always at least one new token."""
    for i, x in enumerate(draft_ids):
        px, qx = float(p[i, x]), float(q[i, x])
        if px >= qx or float(torch.rand((), generator=rng)) < px / qx:
            continue  # accepted
        residual = (p[i] - q[i]).clamp(min=0)
        total = residual.sum()
        replacement = draw(residual / total if total > 0 else p[i], rng)
        return draft_ids[:i] + [replacement], i
    return draft_ids + [draw(p[len(draft_ids)], rng)], len(draft_ids)


# =============================================================================
# Drafters
# =============================================================================

class ModelDrafter:
    """Drafts with a small model that keeps its own KV cache in step with the accepted text."""

    def __init__(self, model):
        self.model = model.eval()

    def reset(self):
        self.caches = self.model.make_kv_caches(1)
        self.n_cached = 0   # tokens[:n_cached] are in the draft's cache and are final
        self.n_fed = 0      # draft tokens of the current round that were fed into the cache

    def propose(self, tokens, k, params, rng):
        # catch up on what the target accepted since last round (always at least 1 token)
        new = torch.tensor([tokens[self.n_cached:]])
        logits, _ = self.model(new, kv_caches=self.caches, start_pos=self.n_cached)
        self.n_cached = len(tokens)
        ids, qs, history = [], [], list(tokens)
        for i in range(k):
            q = token_probs(logits[:, -1], torch.tensor([history]), params)[0]
            x = draw(q, rng)
            ids.append(x)
            qs.append(q)
            history.append(x)
            if i < k - 1:  # the last guess never needs to go through the draft model
                logits, _ = self.model(torch.tensor([[x]]), kv_caches=self.caches, start_pos=len(history) - 1)
        self.n_fed = max(k - 1, 0)
        return ids, (torch.stack(qs) if qs else None)

    def accepted(self, n_before, n_accepted):
        """The first n_accepted guesses were kept: the cache entries for those stay valid,
        anything after them is overwritten next round (positions are simply reused)."""
        self.n_cached = n_before + min(n_accepted, self.n_fed)


class NgramDrafter:
    """Proposes the continuation of the most recent earlier occurrence of the last n tokens."""

    def __init__(self, max_n=3, min_n=2):
        self.max_n, self.min_n = max_n, min_n

    def reset(self):
        pass

    def propose(self, tokens, k, params, rng):
        for n in range(self.max_n, self.min_n - 1, -1):
            if len(tokens) <= n:
                continue
            suffix = tokens[-n:]
            for start in range(len(tokens) - n - 1, -1, -1):  # most recent match first
                if tokens[start:start + n] == suffix:
                    guess = tokens[start + n:start + n + k]
                    # a deterministic guess is a distribution with all its mass on that token
                    return guess, F.one_hot(torch.tensor(guess), self.vocab_size).float()
        return [], None

    def accepted(self, n_before, n_accepted):
        pass


# =============================================================================
# The generator
# =============================================================================

@dataclass
class SpecStats:
    rounds: int = 0
    drafted: int = 0
    accepted: int = 0
    tokens: int = 0

    @property
    def acceptance(self):
        return self.accepted / self.drafted if self.drafted else 0.0

    @property
    def tokens_per_round(self):
        return self.tokens / self.rounds if self.rounds else 0.0

    def __str__(self):
        return (f"{self.rounds} rounds, {self.tokens_per_round:.2f} tokens per target pass, "
                f"{self.acceptance:.0%} of {self.drafted} guesses accepted")


class SpeculativeGenerator(Generator):
    """A Generator whose decoding loop is speculative. stream() / generate() / encode_prompt()
    are inherited, so it's a drop-in replacement (single requests; no batching)."""

    # k=2 measured fastest with the 2.5M draft model (README: "Speculative decoding")
    def __init__(self, model, tokenizer, drafter, k=2, is_sft=False):
        super().__init__(model, tokenizer, is_sft)
        self.drafter, self.k = drafter, k
        if isinstance(drafter, NgramDrafter):
            drafter.vocab_size = model.cfg.vocab_size
        self.stats = SpecStats()

    @classmethod
    def from_checkpoints(cls, target_path, draft="ngram", k=2, device="cpu"):
        target = Generator.from_checkpoint(target_path, device)
        if draft == "ngram":
            drafter = NgramDrafter()
        else:
            draft_model = Generator.from_checkpoint(draft, device)
            if draft_model.tok.vocab != target.tok.vocab:
                raise ValueError("the draft model must use the same tokenizer as the target")
            drafter = ModelDrafter(draft_model.model)
        return cls(target.model, target.tok, drafter, k, target.is_sft)

    @torch.inference_mode()
    def stream_ids(self, prompt_ids, params: SamplingParams, result: GenerationResult):
        cfg = self.model.cfg
        if len(prompt_ids) >= cfg.block_size:
            raise ValueError(f"prompt is {len(prompt_ids)} tokens, the model's context is {cfg.block_size}")
        result.prompt_tokens = len(prompt_ids)
        max_new = min(params.max_new_tokens, cfg.block_size - len(prompt_ids))
        seed = params.seed if params.seed is not None else random_seed()
        result.seed = seed
        rng = torch.Generator().manual_seed(seed)
        self.stats = stats = SpecStats()

        t_start = time.perf_counter()
        tokens = list(prompt_ids)
        caches = self.model.make_kv_caches(1)
        if len(tokens) > 1:  # the target's cache holds every token except the newest
            self.model(torch.tensor([tokens[:-1]]), kv_caches=caches, start_pos=0)
        self.drafter.reset()

        result.finish_reason = "context" if max_new < params.max_new_tokens else "length"
        generated, t_first = 0, None
        while generated < max_new:
            n = len(tokens)
            # never draft past the token budget or the context window
            k = min(self.k, max_new - generated - 1, cfg.block_size - n)
            draft_ids, q = self.drafter.propose(tokens, k, params, rng) if k > 0 else ([], None)

            # one target pass over [newest token, guesses...]: p_1 .. p_len+1
            x = torch.tensor([[tokens[-1]] + draft_ids])
            logits, _ = self.model(x, kv_caches=caches, start_pos=n - 1, all_logits=True)
            p = torch.cat([token_probs(logits[:, i], torch.tensor([tokens + draft_ids[:i]]), params)
                           for i in range(len(draft_ids) + 1)])
            new, n_accepted = verify(draft_ids, q, p, rng)
            self.drafter.accepted(n, n_accepted)
            stats.rounds += 1
            stats.drafted += len(draft_ids)
            stats.accepted += n_accepted

            for token in new:
                tokens.append(token)
                generated += 1
                stats.tokens += 1
                if t_first is None:
                    t_first = time.perf_counter()
                    result.ttft_ms = (t_first - t_start) * 1000
                if token in self.special_ids:
                    if params.stop_at_special:
                        result.finish_reason = "stop"
                        break
                    continue  # kept in the context, never shown as text
                result.token_ids.append(token)
                yield token
            if result.finish_reason == "stop":
                break

        t_end = time.perf_counter()
        result.total_ms = (t_end - t_start) * 1000
        if t_first is not None and len(result.token_ids) > 1:
            result.decode_tokens_per_s = (len(result.token_ids) - 1) / (t_end - t_first)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="the target model")
    parser.add_argument("--draft", default="ngram", help="a draft checkpoint, or 'ngram'")
    parser.add_argument("--k", type=int, default=2, help="tokens drafted per round")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    gen = SpeculativeGenerator.from_checkpoints(args.ckpt, args.draft, args.k)
    params = SamplingParams(max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                            top_p=args.top_p, seed=args.seed)
    result = GenerationResult(text="")
    print(f"[instruction] {args.prompt}\n[reply] " if args.chat else args.prompt, end="", flush=True)
    for piece in gen.stream(args.prompt, params, result, chat=args.chat):
        print(piece, end="", flush=True)
    print(f"\n[{result.stats()}]\n[speculative: {gen.stats}]")


if __name__ == "__main__":
    main()
