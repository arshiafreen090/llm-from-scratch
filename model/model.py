"""
The full language model: token embedding -> N transformer blocks -> final norm -> logits.

    from model.config import get_config
    from model.model import TinyLLaMA

    model = TinyLLaMA(get_config("small", "llama"))
    logits, loss = model(idx, targets)       # training
    out = model.generate(idx, 100)           # generation (uses the KV cache by default)
"""

import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import ModelConfig
from model.layers import Block, KVCache, make_norm, precompute_rope


class TinyLLaMA(nn.Module):

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        # GPT-2 baseline only: a learned vector per position, added to the token embedding
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd) if cfg.pos_emb == "learned" else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = make_norm(cfg)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # weight tying: the matrix that turns token ids into vectors is reused to turn the
        # final vectors back into token scores. Saves vocab_size * n_embd parameters
        # (1.6M here, a big share of a small model) and usually helps.
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        # RoPE: angle tables computed once. Buffers move with .to(device) but are not
        # learned and not saved in checkpoints (persistent=False), since they're recomputable.
        if cfg.pos_emb == "rope":
            cos, sin = precompute_rope(cfg.head_dim, cfg.block_size, cfg.rope_theta)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

        # initialization (GPT-2 recipe): small normal weights, zero biases...
        self.apply(self._init_weights)
        # ...and the layers that write INTO the residual stream are scaled down by
        # 1/sqrt(2 * n_layer). Each block adds 2 such outputs to x; without this, the
        # residual stream's variance grows with depth.
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        # parameters() yields a tied weight only once, so it isn't double-counted
        return sum(p.numel() for p in self.parameters())

    # -------------------------------------------------------------------------

    def forward(self, idx, targets=None, kv_caches=None, start_pos=0):
        """
        idx:       (B, T) token ids
        targets:   (B, T) next-token ids, or None. -1 = "don't compute loss here"
                   (used in Part 2 to skip the loss on prompt tokens)
        kv_caches: one KVCache per layer, or None
        start_pos: position of idx[:, 0] in the full sequence (> 0 when using a cache)

        Returns (logits, loss). With targets: logits for every position.
        Without: logits for the last position only (that's all generation needs).
        """
        B, T = idx.shape
        assert start_pos + T <= self.cfg.block_size, (
            f"sequence length {start_pos + T} exceeds block_size {self.cfg.block_size}")

        x = self.tok_emb(idx)  # (B, T, C)
        if self.pos_emb is not None:
            pos = torch.arange(start_pos, start_pos + T, device=idx.device)
            x = x + self.pos_emb(pos)
        x = self.drop(x)

        rope = (self.rope_cos, self.rope_sin) if self.cfg.pos_emb == "rope" else None
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x = block(x, rope, cache, start_pos)
        x = self.norm_f(x)

        if targets is not None:
            logits = self.lm_head(x)  # (B, T, vocab)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        # inference: skip the (T-1) positions nobody will look at
        return self.lm_head(x[:, [-1], :]), None

    # -------------------------------------------------------------------------

    def make_kv_caches(self, batch_size, device=None, dtype=None):
        """One empty KV cache per layer, sized for the full context length."""
        ref = self.tok_emb.weight
        return [KVCache(batch_size, self.cfg.block_size, self.cfg.n_kv_head, self.cfg.head_dim,
                        device or ref.device, dtype or ref.dtype)
                for _ in range(self.cfg.n_layer)]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, use_cache=True, stop_token=None):
        """Simple autoregressive generation: predict a token, append it, repeat.
        (inference/generate.py adds top-p, repetition penalty, streaming and batching.)

        use_cache=False recomputes the whole sequence every step, like nanoGPT does.
        use_cache=True  feeds only the newest token and reuses the cached keys/values.
        Both give identical results; the cache is just much faster.
        """
        self.eval()
        # this simple version stops at the context limit instead of sliding a window
        max_new_tokens = min(max_new_tokens, self.cfg.block_size - idx.size(1))
        caches = self.make_kv_caches(idx.size(0)) if use_cache else None

        pos = 0
        for _ in range(max_new_tokens):
            if use_cache:
                # first step: the whole prompt ("prefill"), afterwards: only the newest token
                step_input = idx[:, pos:]
                logits, _ = self(step_input, kv_caches=caches, start_pos=pos)
                pos = idx.size(1)
            else:
                logits, _ = self(idx)

            logits = logits[:, -1, :]
            if temperature == 0:  # greedy: always take the most likely token
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

            idx = torch.cat([idx, next_id], dim=1)
            if stop_token is not None and (next_id == stop_token).all():
                break
        return idx

    # -------------------------------------------------------------------------

    def configure_optimizer(self, weight_decay, learning_rate, betas, device_type):
        """AdamW with weight decay only on matrices (Linear weights, embeddings).
        Norm weights and biases are 1D and are not decayed: shrinking them towards
        zero would just fight the normalization."""
        params = [p for p in self.parameters() if p.requires_grad]
        groups = [
            {"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
            {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
        ]
        use_fused = device_type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **({"fused": True} if use_fused else {}))
