"""
Sampling: how to pick the next token from the model's scores (logits).

The model outputs one score per vocabulary token. Every method below reshapes those
scores before we draw a token. They're applied in this order (the same order as
Hugging Face transformers):

    1. repetition penalty   make tokens we've already produced less likely
    2. temperature          sharpen (< 1) or flatten (> 1) the whole distribution
    3. top-k                keep only the k highest-scoring tokens
    4. top-p (nucleus)      keep the smallest set of tokens whose probabilities add up to p
    5. min-p                keep tokens at least min_p times as likely as the best one
    6. sample               draw one token from what's left (or argmax if temperature = 0)

"Removing" a token means setting its logit to -inf, which becomes probability 0 after softmax.
All functions take logits of shape (B, vocab) and return the same shape.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class SamplingParams:
    max_new_tokens: int = 200
    temperature: float = 0.8           # 0 = greedy (always the single most likely token)
    top_k: Optional[int] = None        # e.g. 50
    top_p: Optional[float] = None      # e.g. 0.9
    min_p: Optional[float] = None      # e.g. 0.05
    repetition_penalty: float = 1.0    # 1.0 = off. 1.1-1.3 discourages loops
    seed: Optional[int] = None         # set for reproducible samples
    stop_at_eot: bool = True           # stop when the model produces <|endoftext|>


def apply_repetition_penalty(logits, history, penalty):
    """Penalize every token that already appears in `history` (B, T).

    Dividing a positive logit by penalty > 1 lowers it; but dividing a NEGATIVE logit
    would RAISE it (-2 / 1.2 = -1.67). So negative logits get multiplied instead.
    (This is the CTRL paper's formulation, used by Hugging Face.)"""
    if penalty == 1.0 or history.numel() == 0:
        return logits
    seen = torch.gather(logits, 1, history)                          # scores of seen tokens
    seen = torch.where(seen > 0, seen / penalty, seen * penalty)
    return logits.scatter(1, history, seen)


def apply_top_k(logits, k):
    """Keep the k largest logits per row."""
    k = min(k, logits.size(-1))
    kth_largest = torch.topk(logits, k, dim=-1).values[:, [-1]]
    return logits.masked_fill(logits < kth_largest, float("-inf"))


def apply_top_p(logits, p):
    """Nucleus sampling: keep the smallest set of top tokens whose probabilities sum to >= p.

    Top-k always keeps k tokens, whether the model is sure or not. Top-p adapts:
    when the model is confident, one or two tokens already cover 90%; when it's unsure,
    many tokens are kept.

    Example, p = 0.7, probabilities sorted: [0.50, 0.30, 0.15, 0.05]
        cumulative sum:                     [0.50, 0.80, 0.95, 1.00]
        keep while (cumsum - own prob) < p: [ yes,  yes,   no,   no]   -> 2 tokens kept
    """
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cumsum = probs.cumsum(dim=-1)
    remove = (cumsum - probs) >= p          # the mass BEFORE this token already reached p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    # undo the sort: put every logit back at its original vocab position
    return torch.empty_like(logits).scatter(1, sorted_idx, sorted_logits)


def apply_min_p(logits, min_p):
    """Keep tokens whose probability is at least min_p * (probability of the best token).
    Like top-p it adapts to confidence, but it's one comparison instead of a sort."""
    probs = F.softmax(logits, dim=-1)
    threshold = min_p * probs.max(dim=-1, keepdim=True).values
    return logits.masked_fill(probs < threshold, float("-inf"))


def sample_next(logits, history, params: SamplingParams, generator=None):
    """logits: (B, vocab) for the last position; history: (B, T) ids so far. Returns (B, 1)."""
    logits = logits.float()
    logits = apply_repetition_penalty(logits, history, params.repetition_penalty)

    if params.temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / params.temperature
    if params.top_k is not None:
        logits = apply_top_k(logits, params.top_k)
    if params.top_p is not None:
        logits = apply_top_p(logits, params.top_p)
    if params.min_p is not None:
        logits = apply_min_p(logits, params.min_p)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)
