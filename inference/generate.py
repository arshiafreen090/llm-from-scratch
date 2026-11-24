"""
Text generation: prompt in, text out, with streaming and latency metrics.

    python -m inference.generate --ckpt checkpoints/smoke/smoke-tiny-llama.pt --prompt "Once upon a time"
    python -m inference.generate --ckpt ... --prompt "..." --temperature 0.7 --top_p 0.9 --repetition_penalty 1.2
    python -m inference.generate --ckpt ... --prompt "..." --temperature 0   # greedy, deterministic
    python -m inference.generate --ckpt ... --prompt "..." --quant int8      # int8 weights (inference/quantize.py)

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
import os
import time
from dataclasses import dataclass, field
from typing import List

import torch

from inference.quantize import MODES as QUANT_MODES, quantize_model
from inference.sampling import SamplingParams, sample_next
from model.config import ModelConfig
from model.model import TinyLLaMA
from tokenizer.chat import prompt_ids
from tokenizer.tokenizer import Tokenizer


@dataclass
class GenerationResult:
    text: str
    token_ids: List[int] = field(default_factory=list)
    finish_reason: str = ""      # "stop" (<|endoftext|>), "length" (max_new_tokens), "context" (block_size full)
    prompt_tokens: int = 0
    seed: int = 0                # the seed actually used; send it back to reproduce this output
    ttft_ms: float = 0.0         # time to first token
    decode_tokens_per_s: float = 0.0  # speed after the first token
    total_ms: float = 0.0

    def stats(self):
        return (f"prompt {self.prompt_tokens} tokens | generated {len(self.token_ids)} | "
                f"TTFT {self.ttft_ms:.1f} ms | decode {self.decode_tokens_per_s:.1f} tok/s | "
                f"total {self.total_ms:.0f} ms | finish: {self.finish_reason}")


class Generator:

    def __init__(self, model: TinyLLaMA, tokenizer: Tokenizer, is_sft: bool = False):
        self.model = model.eval()
        self.tok = tokenizer
        self.device = next(model.parameters()).device
        self.special_ids = set(tokenizer.special_tokens.values())
        self.is_sft = is_sft  # instruction-tuned: expects the chat template

    @classmethod
    def from_checkpoint(cls, path, device="auto", quant=None):
        """quant: None (as saved), "int8" or "int8-dynamic" (see inference/quantize.py).
        A float checkpoint is quantized after loading; an int8 checkpoint (made by
        `python -m inference.quantize`) loads as int8 by itself."""
        if device == "auto":
            device = "cpu" if quant == "int8-dynamic" or not torch.cuda.is_available() else "cuda"
        if quant == "int8-dynamic" and device != "cpu":
            raise ValueError("int8-dynamic runs on CPU only; use --quant int8 on a GPU")
        ckpt = torch.load(path, map_location="cpu")
        model = TinyLLaMA(ModelConfig(**ckpt["model_config"]))
        if ckpt.get("quant"):
            quantize_model(model, ckpt["quant"])  # give the model the int8 layout the weights were saved in
        # torch.compile saves weights as "_orig_mod.<name>"; strip that if present
        state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
        model.load_state_dict(state)
        if quant:
            quantize_model(model, quant)
        return cls(model.to(device), Tokenizer.load(ckpt["tokenizer"]), is_sft=ckpt.get("sft", False))

    def encode_prompt(self, prompt, chat=False):
        """chat=False (completion): the model continues `prompt`. Training data looks like
            ...story <eot> story <eot>..., so a story always starts right after <|endoftext|>;
            putting it in front tells the model "a new story starts here".
        chat=True (instruction): `prompt` is an instruction, wrapped in the SAME chat template
            the SFT data used (tokenizer/chat.py), ending in <|assistant|> so the model replies.

        User text is encoded with encode_ordinary in both modes: typing "<|endoftext|>" or
        "<|assistant|>" gives plain text, never the real control tokens."""
        if chat:
            return prompt_ids(self.tok, prompt)
        return [self.tok.eot_id] + self.tok.encode_ordinary(prompt)

    @torch.inference_mode()
    def stream_ids(self, prompt_ids, params: SamplingParams, result: GenerationResult):
        """Core loop. Yields new token ids one at a time and fills in `result` as it goes."""
        cfg = self.model.cfg
        if len(prompt_ids) >= cfg.block_size:
            raise ValueError(f"prompt is {len(prompt_ids)} tokens, the model's context is {cfg.block_size}")
        result.prompt_tokens = len(prompt_ids)
        max_new = min(params.max_new_tokens, cfg.block_size - len(prompt_ids))

        # Every request gets its OWN random generator. Using PyTorch's global one would make a
        # request's output depend on whatever other requests ran before it (and on any other
        # code that draws random numbers). Without a seed we draw a fresh one and report it,
        # so any response can be reproduced exactly by sending the same seed back.
        seed = params.seed if params.seed is not None else int.from_bytes(os.urandom(8), "little") >> 1
        result.seed = seed
        rng = torch.Generator(device=self.device).manual_seed(seed)

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
            if token in self.special_ids:
                if params.stop_at_special:
                    result.finish_reason = "stop"
                    break
                # not stopping: still never stream a control token as text
                history = torch.cat([history, next_id], dim=1)
                logits, _ = self.model(next_id, kv_caches=caches, start_pos=pos)
                pos += 1
                continue
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

    def stream(self, prompt, params: SamplingParams = None, result: GenerationResult = None, chat=False):
        """Yields text pieces as they're generated.

        Why not just decode each token on its own? One character can be split across
        tokens: an emoji is 4 bytes, and a byte-level tokenizer may give it 2 tokens.
        Decoding the first token alone would print a broken character. The incremental
        UTF-8 decoder holds incomplete bytes back until the character is complete."""
        params = params or SamplingParams()
        result = result if result is not None else GenerationResult(text="")
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        pieces = []
        for token in self.stream_ids(self.encode_prompt(prompt, chat), params, result):
            piece = decoder.decode(self.tok.vocab[token])
            if piece:
                pieces.append(piece)
                yield piece
        tail = decoder.decode(b"", final=True)
        if tail:
            pieces.append(tail)
            yield tail
        result.text = "".join(pieces)

    def generate(self, prompt, params: SamplingParams = None, chat=False) -> GenerationResult:
        result = GenerationResult(text="")
        for _ in self.stream(prompt, params, result, chat=chat):
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
    parser.add_argument("--quant", default=None, choices=QUANT_MODES, help="quantize the weights when loading")
    parser.add_argument("--chat", action="store_true",
                        help="treat --prompt as an instruction (for instruction-tuned checkpoints)")
    args = parser.parse_args()

    gen = Generator.from_checkpoint(args.ckpt, args.device, args.quant)
    if args.chat and not gen.is_sft:
        print("warning: this checkpoint was not instruction-tuned; it has never seen the chat format\n")
    params = SamplingParams(max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k,
                            top_p=args.top_p, min_p=args.min_p, repetition_penalty=args.repetition_penalty,
                            seed=args.seed)
    for i in range(args.num_samples):
        result = GenerationResult(text="")
        print(f"[instruction] {args.prompt}\n[reply] " if args.chat else args.prompt, end="", flush=True)
        for piece in gen.stream(args.prompt, params, result, chat=args.chat):
            print(piece, end="", flush=True)
        print(f"\n[{result.stats()}]\n")
        if params.seed is not None:
            params.seed += 1  # different but reproducible samples


if __name__ == "__main__":
    main()
