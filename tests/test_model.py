"""
Tests for the model. Each one checks a property you should be able to explain in an interview.
    python -m pytest tests/test_model.py -v
"""

import itertools
import math

import pytest
import torch

from model.config import ModelConfig, get_config
from model.layers import RMSNorm, apply_rope, precompute_rope
from model.model import TinyLLaMA

torch.manual_seed(0)


def tiny(**overrides):
    return get_config("tiny", "llama", block_size=64, **overrides)


def batch(cfg, B=2, T=16):
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    return idx, torch.roll(idx, -1, dims=1)


# ---- every combination of switches builds and runs -----------------------------

ALL_VARIANTS = list(itertools.product(["rope", "learned"], ["rmsnorm", "layernorm"], ["swiglu", "gelu"], [4, 2, 1]))


@pytest.mark.parametrize("pos_emb,norm,mlp,n_kv_head", ALL_VARIANTS)
def test_every_variant_runs(pos_emb, norm, mlp, n_kv_head):
    cfg = tiny(pos_emb=pos_emb, norm=norm, mlp=mlp, n_kv_head=n_kv_head)
    model = TinyLLaMA(cfg)
    idx, targets = batch(cfg)
    logits, loss = model(idx, targets)
    assert logits.shape == (2, 16, cfg.vocab_size)
    loss.backward()


def test_initial_loss_is_uniform_guess():
    """A fresh model knows nothing, so it should give each of the 4096 tokens about equal
    probability. Cross-entropy of a uniform guess = ln(vocab_size) = ln(4096) = 8.32.
    A much higher initial loss would mean bad initialization."""
    cfg = tiny()
    _, loss = TinyLLaMA(cfg)(*batch(cfg))
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.3


# ---- causality ------------------------------------------------------------------

@pytest.mark.parametrize("arch", ["llama", "gpt2"])
def test_causal_no_peeking_at_the_future(arch):
    """Changing token 10 must not change predictions at positions 0..9."""
    cfg = get_config("tiny", arch, block_size=64)
    model = TinyLLaMA(cfg).eval()
    idx, targets = batch(cfg, B=1)
    changed = idx.clone()
    changed[0, 10] = (changed[0, 10] + 1) % cfg.vocab_size
    logits_a, _ = model(idx, targets)
    logits_b, _ = model(changed, targets)
    assert torch.allclose(logits_a[:, :10], logits_b[:, :10], atol=1e-5)
    assert not torch.allclose(logits_a[:, 10:], logits_b[:, 10:], atol=1e-5)


# ---- RoPE and RMSNorm ---------------------------------------------------------------

def test_rope_depends_only_on_relative_position():
    """q at position m, k at position n: q.k should depend only on m - n.
    So (m=5, n=2) and (m=25, n=22) must give the same score."""
    hd = 32
    cos, sin = precompute_rope(hd, 64)
    q = torch.randn(1, 1, 1, hd)
    k = torch.randn(1, 1, 1, hd)

    def score(m, n):
        q_rot = apply_rope(q, cos[m:m + 1], sin[m:m + 1])
        k_rot = apply_rope(k, cos[n:n + 1], sin[n:n + 1])
        return (q_rot * k_rot).sum().item()

    assert score(5, 2) == pytest.approx(score(25, 22), abs=1e-4)
    assert score(5, 2) != pytest.approx(score(5, 4), abs=1e-3)  # different distance, different score


def test_rope_keeps_vector_length():
    """A rotation never changes a vector's length, only its direction."""
    cos, sin = precompute_rope(32, 64)
    x = torch.randn(2, 4, 64, 32)
    assert torch.allclose(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1), atol=1e-4)


def test_rmsnorm_output_has_unit_rms():
    x = torch.randn(4, 10, 64) * 5 + 3
    y = RMSNorm(64)(x)
    assert torch.allclose(y.pow(2).mean(-1).sqrt(), torch.ones(4, 10), atol=1e-3)


# ---- parameter counts ---------------------------------------------------------------

def count_mlp_params(mlp):
    cfg = ModelConfig(n_embd=384, n_head=6, n_kv_head=6, mlp=mlp)
    return sum(p.numel() for p in TinyLLaMA(cfg).blocks[0].mlp.parameters())


def test_swiglu_has_same_params_as_gelu():
    """Fair ablation: SwiGLU's 2/3 hidden size keeps its parameters equal to the GELU MLP."""
    assert count_mlp_params("swiglu") == count_mlp_params("gelu")


def test_gqa_shrinks_attention():
    def attn_params(n_kv_head):
        cfg = ModelConfig(n_embd=384, n_head=6, n_kv_head=n_kv_head)
        return sum(p.numel() for p in TinyLLaMA(cfg).blocks[0].attn.parameters())
    # MHA: q,k,v,o all 384x384. GQA with 2 kv heads: k and v shrink to 384x128.
    assert attn_params(6) == 4 * 384 * 384
    assert attn_params(2) == 2 * 384 * 384 + 2 * 384 * 128


def test_weight_tying_shares_one_matrix():
    model = TinyLLaMA(tiny())
    assert model.lm_head.weight is model.tok_emb.weight
    untied = TinyLLaMA(tiny(tie_weights=False))
    assert untied.num_params() - model.num_params() == 4096 * 128


# ---- KV cache ---------------------------------------------------------------------

@pytest.mark.parametrize("arch,n_kv_head", [("llama", 2), ("llama", 1), ("gpt2", 4)])
def test_kv_cache_matches_full_recompute(arch, n_kv_head):
    """The most important inference test: processing tokens one at a time with a cache
    must give the SAME logits as running the full sequence at once."""
    cfg = get_config("tiny", arch, block_size=64, n_kv_head=n_kv_head)
    model = TinyLLaMA(cfg).eval()
    idx, targets = batch(cfg, B=2, T=20)

    full_logits, _ = model(idx, targets)  # all 20 positions in one pass

    caches = model.make_kv_caches(batch_size=2)
    step_logits = [model(idx[:, :8], kv_caches=caches, start_pos=0)[0]]  # prefill 8 tokens
    for t in range(8, 20):                                                # then 1 at a time
        step_logits.append(model(idx[:, t:t + 1], kv_caches=caches, start_pos=t)[0])

    assert torch.allclose(step_logits[0][:, 0], full_logits[:, 7], atol=1e-4)
    for i, t in enumerate(range(8, 20), start=1):
        assert torch.allclose(step_logits[i][:, 0], full_logits[:, t], atol=1e-4), f"mismatch at {t}"


@pytest.mark.parametrize("arch", ["llama", "gpt2"])
def test_kv_cache_logits_match_over_full_context(arch):
    """Decode one token at a time through the ENTIRE context window (up to position
    block_size - 1, where the RoPE table / learned position table ends) and compare every
    step's logits with a single uncached forward pass.

    The comparison is RELATIVE to the size of the logits: an untrained model's logits are
    tiny, so a fixed tolerance like 1e-4 could hide a large relative error."""
    torch.manual_seed(1)
    cfg = get_config("tiny", arch, block_size=64)
    model = TinyLLaMA(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 64))
    full_logits, _ = model(idx, idx)                       # (2, 64, vocab) in one pass

    caches = model.make_kv_caches(batch_size=2)
    worst = 0.0
    for t in range(64):                                    # position 0 .. 63
        step, _ = model(idx[:, t:t + 1], kv_caches=caches, start_pos=t)
        ref = full_logits[:, t]
        rel = ((step[:, 0] - ref).abs().max() / ref.abs().max()).item()
        worst = max(worst, rel)
        assert torch.equal(step[:, 0].argmax(-1), ref.argmax(-1)), f"argmax differs at position {t}"
    assert worst < 1e-5, f"worst relative logit error {worst:.2e}"


def test_kv_cache_chunked_prefill():
    """Several new tokens on top of an existing cache (the explicit-mask branch)."""
    cfg = tiny()
    model = TinyLLaMA(cfg).eval()
    idx, targets = batch(cfg, B=1, T=20)
    full_logits, _ = model(idx, targets)
    caches = model.make_kv_caches(batch_size=1)
    model(idx[:, :10], kv_caches=caches, start_pos=0)
    last, _ = model(idx[:, 10:20], kv_caches=caches, start_pos=10)
    assert torch.allclose(last[:, 0], full_logits[:, 19], atol=1e-4)


def test_generate_same_with_and_without_cache():
    cfg = tiny()
    model = TinyLLaMA(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 5))
    with_cache = model.generate(prompt, 30, temperature=0, use_cache=True)
    without = model.generate(prompt, 30, temperature=0, use_cache=False)
    assert torch.equal(with_cache, without)
    assert with_cache.shape == (1, 35)


# ---- it can learn ------------------------------------------------------------------

@pytest.mark.parametrize("arch", ["llama", "gpt2"])
def test_can_overfit_one_batch(arch):
    """A model that can't memorize one small batch has a bug somewhere."""
    torch.manual_seed(0)
    cfg = get_config("tiny", arch, block_size=64)
    model = TinyLLaMA(cfg)
    opt = model.configure_optimizer(0.0, 3e-3, (0.9, 0.95), "cpu")
    idx, targets = batch(cfg, B=4, T=32)
    first = None
    for _ in range(150):
        _, loss = model(idx, targets)
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.2 * first, f"loss only went {first:.2f} -> {loss.item():.2f}"
