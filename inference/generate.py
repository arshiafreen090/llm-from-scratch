"""
Text generation: prompt in, text out, with streaming and latency metrics.

    python -m inference.generate --ckpt checkpoints/smoke/smoke-tiny-llama.pt --prompt "Once upon a time"
    python -m inference.generate --ckpt ... --prompt "..." --temperature 0.7 --top_p 0.9 --repetition_penalty 1.2
    python -m inference.generate --ckpt ... --prompt "..." --temperature 0   # greedy, deterministic

In code:
    gen = Generator.from_checkpoint("checkpoints/small-llama-s1.pt")
    for piece in gen.stream("Once upon a time", SamplingParams(top_p=0.9)):
        print(piece, end="", flush=True)
    result = gen.generate("Once upon a time")   # .text, .ttft_ms, .tokens_per_s, ...

The two latency numbers that matter when serving an LLM:
    TTFT (time to first token)  how long the user waits before anything appears.
                                Dominated by "prefill": one forward pass over the whole prompt.
    tokens/s (decode speed)     how fast text appears after that. One forward pass per token,
                                each reading the KV cache.
"""

import argparse
import codecs
import time
from dataclasses import dataclass, field
from typing import List

import torch

from inference.sampling import SamplingParams, sample_next
from model.config import ModelConfig
from model.model import TinyLLaMA
from tokenizer.tokenizer import Tokenizer


@dataclass
class GenerationResult:
    text: str
    token_ids: List[int] = field(default_factory=list)
    finish_reason: str = ""      # "stop" (<|endoftext|>), "length" (max_new_tokens), "context" (block_size full)
    prompt_tokens: int = 0
    ttft_ms: float = 0.0         # time to first token
    decode_tokens_per_s: float = 0.0  # speed after the first token
    total_ms: float = 0.0

    def stats(self):
        return (f"prompt {self.prompt_tokens} tokens | generated {len(self.token_ids)} | "
                f"TTFT {self.ttft_ms:.1f} ms | decode {self.decode_tokens_per_s:.1f} tok/s | "
                f"total {self.total_ms:.0f} ms | finish: {self.finish_reason}")


class Generator:

    def __init__(self, model: TinyLLaMA, tokenizer: Tokenizer):
        self.model = model.eval()
        self.tok = tokenizer
        self.device = next(model.parameters()).device

    @classmethod
    def from_checkpoint(cls, path, device="auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(path, map_location=device)
        model = TinyLLaMA(ModelConfig(**ckpt["model_config"]))
        # torch.compile saves weights as "_orig_mod.<name>"; strip that if present
        state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
        model.load_state_dict(state)
        return cls(model.to(device), Tokenizer.load(ckpt["tokenizer"]))

    def encode_prompt(self, prompt, add_bos=True):
        """Training data looks like: ...story <eot> story <eot>..., so a story always starts
        right after <|endoftext|>. Putting it in front of the prompt tells the model
        "a new story starts here", the same situation it saw in training."""
        ids = self.tok.encode(prompt)
        return ([self.tok.eot_id] if add_bos else []) + ids

    @torch.inference_mode()
    def stream_ids(self, prompt_ids, params: SamplingParams, result: GenerationResult):
        """Core loop. Yields new token ids one at a time and fills in `result` as it goes."""
        cfg = self.model.cfg
        if len(prompt_ids) >= cfg.block_size:
            raise ValueError(f"prompt is {len(prompt_ids)} tokens, the model's context is {cfg.block_size}")
        result.prompt_tokens = len(prompt_ids)
        max_new = min(params.max_new_tokens, cfg.block_size - len(prompt_ids))

        rng = None
        if params.seed is not None:
            rng = torch.Generator(device=self.device).manual_seed(params.seed)

        t_start = time.perf_counter()
        history = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        caches = self.model.make_kv_caches(batch_size=1)

        # prefill: the whole prompt in one forward pass, filling the KV cache
        logits, _ = self.model(history, kv_caches=caches, start_pos=0)
        pos = history.size(1)

        result.finish_reason = "context" if max_new < params.max_new_tokens else "length"
        t_first = None
        for i in range(max_new):
            next_id = sample_next(logits[:, -1, :], history, params, rng)
            if t_first is None:
                t_first = time.perf_counter()
                result.ttft_ms = (t_first - t_start) * 1000
            token = int(next_id)
            if params.stop_at_eot and token == self.tok.eot_id:
                result.finish_reason = "stop"
                break
            result.token_ids.append(token)
            yield token

            history = torch.cat([history, next_id], dim=1)
            if i < max_new - 1:
                # decode: only the new token goes through the model; the cache has the rest
                logits, _ = self.model(next_id, kv_caches=caches, start_pos=pos)
                pos += 1

        t_end = time.perf_counter()
        result.total_ms = (t_end - t_start) * 1000
        if t_first is not None and len(result.token_ids) > 1:
            result.decode_tokens_per_s = (len(result.token_ids) - 1) / (t_end - t_first)

    def stream(self, prompt, params: SamplingParams = None, result: GenerationResult = None):
        """Yields text pieces as they're generated.

        Why not just decode each token on its own? One character can be split across
        tokens: an emoji is 4 bytes, and a byte-level tokenizer may give it 2 tokens.
        Decoding the first token alone would print a broken character. The incremental
        UTF-8 decoder holds incomplete bytes back until the character is complete."""
        params = params or SamplingParams()
        result = result if result is not None else GenerationResult(text="")
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        pieces = []
        for token in self.stream_ids(self.encode_prompt(prompt), params, result):
            piece = decoder.decode(self.tok.vocab[token])
            if piece:
                pieces.append(piece)
                yield piece
        tail = decoder.decode(b"", final=True)
        if tail:
            pieces.append(tail)
            yield tail
        result.text = "".join(pieces)

    def generate(self, prompt, params: SamplingParams = None) -> GenerationResult:
        result = GenerationResult(text="")
        for _ in self.stream(prompt, params, result):
            pass
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--min_p", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    gen = Generator.from_checkpoint(args.ckpt, args.device)
    params = SamplingParams(max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k,
                            top_p=args.top_p, min_p=args.min_p, repetition_penalty=args.repetition_penalty,
                            seed=args.seed)
    for i in range(args.num_samples):
        result = GenerationResult(text="")
        print(args.prompt, end="", flush=True)
        for piece in gen.stream(args.prompt, params, result):
            print(piece, end="", flush=True)
        print(f"\n[{result.stats()}]\n")
        if params.seed is not None:
            params.seed += 1  # different but reproducible samples


if __name__ == "__main__":
    main()
