"""
Model configuration.

Every architecture upgrade is a switch, so ONE model class can be either a GPT-2-style
baseline or a LLaMA-style model. That is what makes the ablation study possible:
change one switch, train, compare the loss.

    switch       GPT-2 baseline    mini-LLaMA
    pos_emb      "learned"         "rope"
    norm         "layernorm"       "rmsnorm"
    mlp          "gelu"            "swiglu"
    n_kv_head    = n_head (MHA)    < n_head (GQA)
"""

from dataclasses import dataclass, replace


@dataclass
class ModelConfig:
    # ---- sizes ----
    vocab_size: int = 4096    # must match the tokenizer
    block_size: int = 512     # max context length in tokens
    n_layer: int = 6
    n_head: int = 6           # query heads
    n_kv_head: int = 2        # key/value heads. = n_head: MHA, 1: MQA, in between: GQA
    n_embd: int = 384         # model width; head_dim = n_embd // n_head
    dropout: float = 0.0      # 0 for pretraining (lots of data), 0.1+ for small-data finetuning
    bias: bool = False        # bias in Linear/LayerNorm. LLaMA uses none: slightly faster, no worse

    # ---- architecture switches (for ablations) ----
    pos_emb: str = "rope"      # "rope" | "learned"
    norm: str = "rmsnorm"      # "rmsnorm" | "layernorm"
    mlp: str = "swiglu"        # "swiglu" | "gelu"
    tie_weights: bool = True   # share the input embedding and output layer weights

    # ---- details ----
    rope_theta: float = 10000.0  # RoPE base frequency (LLaMA 1/2 use 10k, LLaMA 3 uses 500k)
    norm_eps: float = 1e-5
    multiple_of: int = 64        # round SwiGLU hidden size up to this (GPU-friendly shapes)

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"
        assert self.pos_emb in ("rope", "learned")
        assert self.norm in ("rmsnorm", "layernorm")
        assert self.mlp in ("swiglu", "gelu")

    @property
    def head_dim(self):
        return self.n_embd // self.n_head


# ---- sizes ------------------------------------------------------------------------
SIZES = {
    # for debugging on the laptop CPU (~2M params)
    "tiny":  dict(n_layer=4, n_head=4, n_kv_head=2, n_embd=128),
    # for ablation runs on Kaggle (~12M params)
    "small": dict(n_layer=6, n_head=6, n_kv_head=2, n_embd=384),
    # the final model (~27M params)
    "base":  dict(n_layer=8, n_head=8, n_kv_head=4, n_embd=512),
    # draft model for speculative decoding (~2.5M params): as few layers as possible, since
    # on a CPU each layer costs a fixed overhead per step whatever its width
    "draft": dict(n_layer=2, n_head=4, n_kv_head=2, n_embd=256),
}

# ---- architectures ------------------------------------------------------------------
GPT2_BASELINE = dict(pos_emb="learned", norm="layernorm", mlp="gelu")
LLAMA = dict(pos_emb="rope", norm="rmsnorm", mlp="swiglu")


def get_config(size="small", arch="llama", **overrides):
    """get_config("small", "gpt2")                 -> baseline
       get_config("small", "llama")                -> mini-LLaMA
       get_config("small", "gpt2", pos_emb="rope") -> baseline + one upgrade (an ablation)"""
    arch_args = {"gpt2": GPT2_BASELINE, "llama": LLAMA}[arch]
    cfg = ModelConfig(**SIZES[size], **arch_args)
    if arch == "gpt2":
        cfg = replace(cfg, n_kv_head=cfg.n_head)  # plain multi-head attention
    return replace(cfg, **overrides)
