"""
The building blocks of the model. Each LLaMA upgrade sits next to the GPT-2 version it replaces.

    RMSNorm          replaces  LayerNorm
    RoPE             replaces  learned position embeddings (lives in model.py)
    Attention (GQA)  replaces  multi-head attention (GQA with n_kv_head = n_head IS MHA)
    SwiGLU           replaces  GELU MLP
    KVCache          inference speedup: remember past keys/values instead of recomputing them

Shape names used in comments:
    B = batch size, T = sequence length, C = n_embd,
    nh = query heads, nkv = key/value heads, hd = head_dim
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Normalization
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square norm:  x / sqrt(mean(x^2) + eps) * weight

    LayerNorm does two things: subtract the mean, then divide by the std.
    RMSNorm drops the mean subtraction (and the bias) and only rescales.
    The re-scaling turned out to be the part that matters, so RMSNorm trains
    just as well while doing less work.
    """

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # compute in float32: squaring small fp16/bf16 values loses precision
        x_f = x.float()
        x_f = x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x_f.type_as(x) * self.weight


def make_norm(cfg):
    if cfg.norm == "rmsnorm":
        return RMSNorm(cfg.n_embd, cfg.norm_eps)
    return nn.LayerNorm(cfg.n_embd, eps=cfg.norm_eps, bias=cfg.bias)


# =============================================================================
# RoPE: Rotary Position Embedding
# =============================================================================
#
# GPT-2 ADDS a learned vector to each token embedding to say "I'm at position 7".
# RoPE instead ROTATES the query and key vectors by an angle that depends on position.
#
# Split a head's vector into pairs of numbers (x0,x1), (x2,x3), ... and treat each pair
# as a 2D point. At position m, pair i is rotated by angle  m * theta_i,  where
#     theta_i = base^(-2i / head_dim)        (base = 10000)
# Early pairs spin fast, late pairs spin slowly, like the hands of a clock.
#
# Why this is clever: rotating q by angle a and k by angle b, then taking q . k,
# gives a result that depends only on (a - b). So attention scores depend on the
# RELATIVE distance between tokens (m - n), not on absolute positions.
# tests/test_model.py checks exactly this property.
#
# Bonus: there are no learned position parameters at all.

def precompute_rope(head_dim, max_len, theta=10000.0):
    """cos/sin tables of shape (max_len, head_dim // 2): one angle per (position, pair)."""
    pair_freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))  # (hd/2,)
    positions = torch.arange(max_len).float()                                           # (max_len,)
    angles = torch.outer(positions, pair_freqs)                                         # (max_len, hd/2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotate every (even, odd) pair of x.   x: (B, heads, T, hd),  cos/sin: (T, hd/2)

    2D rotation of a point (x1, x2) by angle a:
        x1' = x1 * cos(a) - x2 * sin(a)
        x2' = x1 * sin(a) + x2 * cos(a)
    """
    x1 = x[..., 0::2].float()  # even dims: first number of each pair
    x2 = x[..., 1::2].float()  # odd dims: second number of each pair
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    # interleave back: [out1_0, out2_0, out1_1, out2_1, ...]
    return torch.stack((out1, out2), dim=-1).flatten(-2).type_as(x)


# =============================================================================
# KV cache
# =============================================================================
#
# Generating token t+1 needs attention over the keys/values of tokens 0..t.
# Without a cache, every step re-runs the model over the WHOLE sequence: O(T^2) work.
# With a cache, each step only computes k/v for the ONE new token and appends them.
#
# The cache stores only the n_kv_head heads, so GQA with 2 kv heads instead of 6
# makes the cache 3x smaller. Cache memory is a big limit on serving many users at once.

class KVCache:

    def __init__(self, batch_size, max_len, n_kv_head, head_dim, device, dtype):
        shape = (batch_size, n_kv_head, max_len, head_dim)
        # allocated once up front, so no tensors are re-created during generation
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)

    def update(self, k, v, start_pos):
        """Write the new k, v at positions start_pos.. and return everything cached so far."""
        T = k.size(2)
        self.k[:, :, start_pos:start_pos + T] = k
        self.v[:, :, start_pos:start_pos + T] = v
        return self.k[:, :, :start_pos + T], self.v[:, :, :start_pos + T]


# =============================================================================
# Attention (MHA / GQA / MQA in one class)
# =============================================================================
#
# Multi-head attention (MHA): every query head has its own key head and value head.
# Grouped-query attention (GQA): query heads are split into groups, and each group
#     SHARES one key/value head. With nh=6, nkv=2: heads 0-2 share kv head 0,
#     heads 3-5 share kv head 1.
# Multi-query attention (MQA): all query heads share a single k/v head (nkv=1).
#
# GQA gives fewer parameters and a smaller KV cache, with almost no quality loss.
# LLaMA 2 70B, LLaMA 3 and Mistral all use it.

class Attention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.head_dim
        self.use_rope = cfg.pos_emb == "rope"
        self.dropout = cfg.dropout

        # separate projections (nanoGPT fuses q,k,v into one Linear; with GQA they have
        # different sizes, so separate is clearer)
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_head * cfg.head_dim, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * cfg.head_dim, bias=cfg.bias)  # smaller with GQA
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * cfg.head_dim, bias=cfg.bias)  # smaller with GQA
        self.o_proj = nn.Linear(cfg.n_head * cfg.head_dim, cfg.n_embd, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, rope=None, kv_cache=None, start_pos=0):
        B, T, _ = x.shape

        # project, then split into heads: (B, T, heads*hd) -> (B, heads, T, hd)
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # RoPE rotates q and k only (v carries content, not position).
        # With a cache, the new tokens sit at positions start_pos .. start_pos+T-1.
        if self.use_rope:
            cos, sin = rope
            cos, sin = cos[start_pos:start_pos + T], sin[start_pos:start_pos + T]
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        if kv_cache is not None:
            k, v = kv_cache.update(k, v, start_pos)  # now covers positions 0 .. start_pos+T-1
        T_k = k.size(2)

        # GQA: repeat each kv head so every query head has a partner.
        # (B, nkv, T_k, hd) -> (B, nh, T_k, hd)
        if self.n_kv_head != self.n_head:
            repeats = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        # causal masking: a token may attend to itself and earlier tokens only
        if T == T_k:
            mask, is_causal = None, True   # training, or the first pass over a prompt
        elif T == 1:
            mask, is_causal = None, False  # one new token: every cached token is in its past
        else:
            # several new tokens on top of a cache: new token i sees cache + new tokens <= i
            mask = torch.ones(T, T_k, dtype=torch.bool, device=x.device).tril(diagonal=T_k - T)
            is_causal = False

        # scaled dot-product attention: softmax(q k^T / sqrt(hd)) v
        # PyTorch picks a fused kernel (FlashAttention on supported GPUs) automatically
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )

        # merge heads back: (B, nh, T, hd) -> (B, T, nh*hd)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        return self.resid_dropout(self.o_proj(y))


# =============================================================================
# Feed-forward (MLP)
# =============================================================================

class GeluMLP(nn.Module):
    """GPT-2:  down( gelu( up(x) ) ),  hidden size 4 * C"""

    def __init__(self, cfg):
        super().__init__()
        hidden = 4 * cfg.n_embd
        self.up = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.down = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.down(F.gelu(self.up(x))))


class SwiGLU(nn.Module):
    """LLaMA:  down( silu(gate(x)) * up(x) )

    A "gated" MLP: one branch (gate) decides, per hidden unit, how much of the other
    branch (up) passes through. silu(x) = x * sigmoid(x) is a smooth version of ReLU.

    It has 3 weight matrices instead of 2, so the hidden size shrinks from 4C to
    (2/3) * 4C to keep the parameter count equal to the GELU MLP:
        GELU:   2 matrices * C * 4C       = 8 C^2
        SwiGLU: 3 matrices * C * (8/3)C   = 8 C^2
    That keeps the ablation fair: same parameters, different design.
    """

    def __init__(self, cfg):
        super().__init__()
        hidden = int(2 * 4 * cfg.n_embd / 3)
        hidden = cfg.multiple_of * ((hidden + cfg.multiple_of - 1) // cfg.multiple_of)  # round up
        self.gate = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.up = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.down = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


# =============================================================================
# Transformer block
# =============================================================================

class Block(nn.Module):
    """Pre-norm residual block (GPT-2 and LLaMA both use this):
        x = x + attention(norm(x))
        x = x + mlp(norm(x))
    The residual "highway" (x + ...) lets gradients flow straight through deep stacks."""

    def __init__(self, cfg):
        super().__init__()
        self.norm1 = make_norm(cfg)
        self.attn = Attention(cfg)
        self.norm2 = make_norm(cfg)
        self.mlp = SwiGLU(cfg) if cfg.mlp == "swiglu" else GeluMLP(cfg)

    def forward(self, x, rope=None, kv_cache=None, start_pos=0):
        x = x + self.attn(self.norm1(x), rope, kv_cache, start_pos)
        x = x + self.mlp(self.norm2(x))
        return x
