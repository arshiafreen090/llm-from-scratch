"""Draws docs/architecture.png (and .svg): how the pieces of the project fit together.
    python docs/make_architecture.py
Plain matplotlib, so the diagram is reproducible and editable like code.

Layout: three horizontal lanes (train, package, serve), each with its title in a column on the
left and its steps as boxes. Lane 2 runs right-to-left so that the arrows between lanes are
short vertical drops (SFT -> checkpoint, Hugging Face repo -> server and browser)."""

import os
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture")

# colours per kind of box: (fill, border)
C = {
    "data":  ("#E3EEFF", "#2F6FED"),
    "model": ("#ECE8FF", "#6D5DFC"),
    "pack":  ("#FFF2D9", "#D98B00"),
    "serve": ("#DFF5E8", "#1F9D55"),
    "user":  ("#F2F2F5", "#6B6B78"),
}
TEXT, MUTED, RESULT, LINE = "#1C1C22", "#5B5B68", "#15803D", "#4B4B5C"

fig, ax = plt.subplots(figsize=(17, 10.5))
ax.set_xlim(0, 170)
ax.set_ylim(-5, 105)
ax.axis("off")
fig.patch.set_facecolor("white")


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def left(self): return (self.x, self.y + self.h / 2)
    @property
    def right(self): return (self.x + self.w, self.y + self.h / 2)
    @property
    def top(self): return (self.x + self.w / 2, self.y + self.h)
    @property
    def bottom(self): return (self.x + self.w / 2, self.y)


def box(x, y, w, h, title, body, kind, result=None):
    """A rounded box: bold title, body lines, and the measured result in green italics."""
    fill, edge = C[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                                facecolor=fill, edgecolor=edge, linewidth=1.8))
    ax.text(x + w / 2, y + h - 2.4, title, ha="center", va="top", fontsize=11, fontweight="bold", color=TEXT)
    ax.text(x + w / 2, y + h - 6.6, body, ha="center", va="top", fontsize=8.8, color=TEXT, linespacing=1.35)
    if result:
        ax.text(x + w / 2, y + 1.6, result, ha="center", va="bottom", fontsize=8.4, color=RESULT,
                fontstyle="italic", linespacing=1.3)
    return Box(x, y, w, h)


def arrow(a, b, label=None, label_at=None, style="-|>", lw=1.6):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=15, color=LINE, linewidth=lw,
                                 shrinkA=1.5, shrinkB=1.5))
    if label:
        lx, ly = label_at or ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        ax.text(lx, ly, label, ha="center", va="center", fontsize=8.4, color=MUTED,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))


def lane(y, h, number, title, notes):
    ax.add_patch(FancyBboxPatch((2, y), 166, h, boxstyle="round,pad=0,rounding_size=2",
                                facecolor="#FAFAFC", edgecolor="#E4E4EA", linewidth=1.2))
    ax.text(5, y + h - 3, f"{number} · {title}", ha="left", va="top", fontsize=13, fontweight="bold", color=TEXT)
    ax.text(5, y + h - 8.4, notes, ha="left", va="top", fontsize=8.8, color=MUTED, linespacing=1.4)


# ---- title --------------------------------------------------------------------------------
ax.text(2, 104, "TinyLLaMA from scratch: how the pieces fit together", fontsize=19, fontweight="bold",
        color=TEXT, va="top")
ax.text(2, 99.4, "An 11M-parameter LLaMA-style language model: tokenizer, model, training, compression and "
        "serving, all written from scratch in PyTorch. Green italics are measured results.",
        fontsize=10.5, color=MUTED, va="top")

X0, GAP = 29, 3.7          # boxes start right of the title column
H1, H2, H3 = 19, 17, 20    # box heights per lane

# ---- lane 1: train (left to right) -------------------------------------------------------
lane(69, 27, 1, "Train", "Kaggle T4 GPUs (free)\n\nevery run shares the data,\ndata order, token budget,\noptimizer, evaluation\nwindows and precision")
w = (166 - X0 - 3 * GAP) / 4
y = 73
d1 = box(X0, y, w, H1, "Data & tokenizer", "TinyStories: 2.1M short children's\nstories (Hugging Face dataset)\nbyte-level BPE, 4,096 tokens,\ntrained from scratch → .bin files", "data")
m1 = box(X0 + (w + GAP), y, w, H1, "Mini-LLaMA, 11M params", "6 layers × 384 dims\nRoPE · RMSNorm · SwiGLU · GQA\n+ a GPT-2 twin for the ablations", "model")
t1 = box(X0 + 2 * (w + GAP), y, w, H1, "Pretraining", "100M tokens per run\n6 configurations × 2 seeds:\nGPT-2 → +1 change each → LLaMA", "model",
         result="LLaMA −0.107 val loss vs GPT-2,\nsame loss with 62% of the tokens")
t2 = box(X0 + 3 * (w + GAP), y, w, H1, "Instruction tuning (SFT)", "300k TinyStories-Instruct examples\nphrased the way people type;\nloss on response tokens only", "model",
         result="62% of requested words (v1: 39%),\n53% of plot words; unseen phrasings: same")
for a, b in ((d1, m1), (m1, t1), (t1, t2)):
    arrow(a.right, b.left)

# ---- lane 2: package (right to left, so the SFT weights drop straight down) --------------
lane(40, 25, 2, "Package", "laptop CPU\n\nthe same weights in\nthree forms; every step\nis checked against\nthe original")
y = 44
xs = [X0 + i * (w + GAP) for i in range(4)]
p1 = box(xs[3], y, w, H2, "Checkpoint (float32)", "weights + model config\n+ tokenizer name · 44 MB", "pack")
p2 = box(xs[2], y, w, H2, "INT8 quantization", "weight-only, one scale per output\nchannel; tied embedding and\noutput layer share one matrix", "pack",
         result="4× smaller (11 MB), val loss +0.0001")
p3 = box(xs[1], y, w, H2, "ONNX export", "same graph, KV cache as inputs\nand outputs, so a browser\nruntime can run it", "pack",
         result="greedy output identical to PyTorch")
p4 = box(xs[0], y, w, H2, "Hugging Face model repo", ".pt (int8) · .onnx · tokenizer.json\nafeening/tinyllama-from-scratch", "pack")
arrow(t2.bottom, p1.top, label="trained weights", label_at=(t2.bottom[0] + 13, 67))
for a, b in ((p1, p2), (p2, p3), (p3, p4)):
    arrow(a.left, b.right)

# ---- lane 3: serve and use (left to right) -----------------------------------------------
lane(4, 32, 3, "Serve & use", "a server with an API,\nor the model running\nentirely inside the\nvisitor's browser")
y = 8
ws = [36, 38, 19, 36.6]   # engine, server, clients, browser
gx = (166 - X0 - sum(ws)) / 3
x = X0
cols = []
for wi in ws:
    cols.append(x)
    x += wi + gx
s1 = box(cols[0], y, ws[0], H3, "Inference engine (Python)", "KV cache · temperature, top-k,\ntop-p, min-p · continuous batching:\none forward pass serves many\nrequests · speculative decoding", "serve",
         result="batching 5× throughput, speculative\n1.3–1.4×, output identical to one-at-a-time")
s2 = box(cols[1], y, ws[1], H3, "FastAPI server", "chat web UI at /\nOpenAI-compatible /v1/chat/completions\nSSE streaming · API keys · rate limits\nDocker image", "serve",
         result="the official OpenAI SDK works\nby changing base_url")
u1 = box(cols[2], y + 4, ws[2], H3 - 8, "Clients", "curl · openai SDK\nany OpenAI-style app", "user")
s3 = box(cols[3], y, ws[3], H3, "In the browser (live demo)", "static Hugging Face Space (free)\nONNX Runtime Web (WebAssembly)\nin a Web Worker; tokenizer in JS", "serve",
         result="~250 tokens/s on a laptop; nothing\ntyped leaves the device")
arrow(s1.right, s2.left)
arrow(s2.right, u1.left, label="HTTP", label_at=((s2.right[0] + u1.left[0]) / 2, u1.left[1] + 2.2))

# model files flow from the Hugging Face repo down to both ways of serving (an orthogonal "bus")
bus_y = 38
arrow(p4.bottom, (p4.bottom[0], bus_y), style="-")
arrow((p4.bottom[0], bus_y), (s3.top[0], bus_y), style="-")
arrow((s2.top[0], bus_y), s2.top, label=".pt (int8)", label_at=(s2.top[0] + 9, bus_y - 2.6))
arrow((s3.top[0], bus_y), s3.top, label=".onnx + tokenizer", label_at=(s3.top[0] - 14, bus_y - 2.6))

# ---- legend -------------------------------------------------------------------------------
for i, (label, kind) in enumerate((("data", "data"), ("model & training", "model"), ("packaging", "pack"),
                                   ("serving", "serve"), ("people & clients", "user"))):
    fill, edge = C[kind]
    ax.add_patch(FancyBboxPatch((4 + i * 24, -3.3), 3, 2.2, boxstyle="round,pad=0,rounding_size=0.5",
                                facecolor=fill, edgecolor=edge, linewidth=1.2))
    ax.text(8 + i * 24, -2.2, label, va="center", fontsize=9, color=MUTED)
ax.text(168, -2.2, "github.com/arshiafreen090/llm-from-scratch", ha="right", va="center", fontsize=9, color=MUTED)

fig.tight_layout(pad=0.4)
fig.savefig(OUT + ".png", dpi=170, facecolor="white")
fig.savefig(OUT + ".svg", facecolor="white")
print("wrote", OUT + ".png", "and .svg")
