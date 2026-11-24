"""
A web demo: type an instruction, watch the story appear, see which required words it used.

    python -m serve.demo                                    # the int8 SFT model at http://127.0.0.1:7860
    python -m serve.demo --ckpt checkpoints/sft-small-llama-s1.pt --quant int8
    python -m serve.demo --share                            # also a temporary public link (*.gradio.live)

The default checkpoint is the int8 file made by
    python -m inference.quantize --ckpt checkpoints/sft-small-llama-s1.pt
(11 MB instead of 44 MB, same quality: see the README's INT8 section).

HOW IT RUNS THE MODEL
    Through the same ModelWorker as the HTTP API (serve/api.py): ONE thread owns the model,
    and every request is a job in its queue. People using the demo at the same time are
    decoded together (continuous batching), each with their own KV-cache row and seed.

    Why not just call the model from each Gradio request thread? Speed. PyTorch on CPU
    splits each op across a team of OpenMP threads, and every thread that calls PyTorch gets
    its own team. With two teams there are more threads than CPU cores, so OpenMP stops
    busy-waiting between ops and puts its threads to sleep instead, and waking them for each
    of the thousands of small ops per token made generation ~10x slower (18 instead of ~190
    tokens/s). Keeping every model call on one thread keeps one team.

    The page is refreshed about UPDATES_PER_S times a second with whatever text has arrived,
    rather than once per token, so the browser never holds the model up.
"""

import argparse
import os
import queue
import re
import time

import gradio as gr

from experiments.eval_instruct import word_present
from inference.generate import GenerationResult, Generator
from inference.quantize import MODES as QUANT_MODES
from inference.sampling import SamplingParams
from serve.api import ModelWorker

DEFAULT_CKPT = "checkpoints/sft-small-llama-s1-int8.pt"
REPO_URL = "https://github.com/arshiafreen090/llm-from-scratch"

INSTRUCTION, COMPLETION = "Instruction", "Continue a story"
UPDATES_PER_S = 20  # how often the story on the page is refreshed while generating

# written in the same styles as the TinyStories-Instruct training instructions
EXAMPLES = [
    "Write a story. Use the words: rain, ball, happy.",
    "Write a story for children. Include these words: meet, waffle, new.",
    "Write a short story. The story must use the words call, leak, foolish. Also, include some dialogue.",
    "Please write a simple story with a happy ending. Include these words: dog, moon, brave.",
    "Write a story about a girl who finds a lost kitten. Use the words: box, warm, smile.",
]


def required_words(instruction):
    """The words an instruction asks for, in the phrasings TinyStories-Instruct uses:
    "Use the words: a, b, c."  "Include these words: a, b, c."  "...use the words a, b, c."
    Returns [] when the instruction doesn't ask for any."""
    m = re.search(r"words:?\s+([\w\s,'-]+?)(?:\.|$)", instruction, flags=re.IGNORECASE)
    if not m:
        return []
    parts = re.split(r",|\band\b", m.group(1))
    return [p.strip().lower() for p in parts if p.strip() and " " not in p.strip()]


def word_report(instruction, text):
    words = required_words(instruction)
    if not words:
        return ""
    marks = " · ".join(f"{'✅' if word_present(w, text) else '❌'} {w}" for w in words)
    used = sum(word_present(w, text) for w in words)
    return f"**Required words: {used}/{len(words)} used** · {marks}"


def stats_line(result: GenerationResult, queue_ms):
    waited = f"waited {queue_ms:.0f} ms for a free slot · " if queue_ms >= 50 else ""
    return (f"{len(result.token_ids)} tokens · {waited}first token after {result.ttft_ms:.0f} ms · "
            f"{result.decode_tokens_per_s:.0f} tokens/s · finished: {result.finish_reason} · seed {result.seed}")


def stream_story(worker: ModelWorker, prompt, mode, temperature, top_p, max_new_tokens, seed):
    """Yields (story so far, word report, stats) while the model worker generates."""
    gen = worker.generator
    prompt = (prompt or "").strip()
    chat = mode == INSTRUCTION
    if chat and not prompt:
        yield "", "", "Type an instruction first."
        return
    prompt_ids = gen.encode_prompt(prompt, chat=chat)  # the tokenizer is plain Python: fine on any thread
    if len(prompt_ids) >= gen.model.cfg.block_size:
        yield "", "", f"⚠ That's {len(prompt_ids)} tokens; the model's context is {gen.model.cfg.block_size}."
        return
    params = SamplingParams(max_new_tokens=int(max_new_tokens), temperature=float(temperature),
                            top_p=float(top_p) if top_p < 1 else None,
                            seed=int(seed) if seed is not None and seed >= 0 else None)
    try:
        job = worker.submit(prompt_ids, params)
    except Exception:  # the waiting queue is full
        yield "", "", "⚠ Too many people are using the demo right now; try again in a moment."
        return

    start = "" if chat else prompt  # completion mode: the story begins with the user's own words
    pieces = []
    last_update = 0.0
    try:
        while True:
            events = [job.out.get()]  # wait for the next event, then take everything already there
            while True:
                try:
                    events.append(job.out.get_nowait())
                except queue.Empty:
                    break
            for kind, value in events:
                if kind == "token":
                    pieces.append(value)
                elif kind == "done":
                    story = start + value.text
                    yield story, (word_report(prompt, story) if chat else ""), stats_line(value, job.queue_ms)
                    return
                else:  # "error"
                    yield start + "".join(pieces), "", f"⚠ {value}"
                    return
            if time.perf_counter() - last_update >= 1 / UPDATES_PER_S:
                story = start + "".join(pieces)
                yield story, (word_report(prompt, story) if chat else ""), "generating…"
                last_update = time.perf_counter()
    finally:
        # also runs when Gradio closes this stream early (the Stop button, or the page was
        # closed): the worker drops the job and gives its slot to someone else
        job.cancelled.set()


def build_demo(worker: ModelWorker, ckpt_name=""):
    gen = worker.generator
    cfg = gen.model.cfg
    quant = getattr(gen.model, "quant", None) or "float32"
    header = (
        "# TinyLLaMA from scratch\n"
        f"A LLaMA-style model with {gen.model.num_params() / 1e6:.0f}M parameters (RoPE, RMSNorm, SwiGLU, GQA), "
        f"pretrained on TinyStories and instruction-tuned, all from scratch. Running on CPU with {quant} "
        f"weights; context {cfg.block_size} tokens. It writes simple children's stories and uses about half "
        f"of the words an instruction asks for. [Code and results]({REPO_URL})"
    )

    def run(prompt, mode, temperature, top_p, max_new_tokens, seed):
        yield from stream_story(worker, prompt, mode, temperature, top_p, max_new_tokens, seed)

    with gr.Blocks(title="TinyLLaMA from scratch", analytics_enabled=False) as demo:
        gr.Markdown(header)
        with gr.Row():
            with gr.Column(scale=2):
                mode = gr.Radio([INSTRUCTION, COMPLETION], value=INSTRUCTION if gen.is_sft else COMPLETION,
                                label="Mode", info="Instruction: the model follows your request. "
                                                   "Continue a story: it carries on from your first words.")
                prompt = gr.Textbox(label="Your instruction or story start", lines=3, value=EXAMPLES[0])
                with gr.Row():
                    go = gr.Button("Generate", variant="primary")
                    stop = gr.Button("Stop")
                with gr.Accordion("Sampling settings", open=False):
                    temperature = gr.Slider(0, 1.5, value=0.7, step=0.05, label="Temperature",
                                            info="0 = always the most likely token (deterministic)")
                    top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p",
                                      info="sample only from the most likely tokens covering this much probability")
                    max_new = gr.Slider(16, 400, value=300, step=8, label="Max new tokens")
                    seed = gr.Number(value=-1, precision=0, label="Seed",
                                     info="-1 = random; the seed used is shown under the story")
                gr.Examples(EXAMPLES, inputs=prompt, label="Example instructions")
            with gr.Column(scale=3):
                story = gr.Textbox(label="Story", lines=16, max_lines=30, autoscroll=True, interactive=False)
                words = gr.Markdown()
                stats = gr.Markdown()

        inputs = [prompt, mode, temperature, top_p, max_new, seed]
        event = go.click(run, inputs, [story, words, stats])
        submit = prompt.submit(run, inputs, [story, words, stats])
        # cancelling closes stream_story, whose `finally` tells the worker to drop the job
        stop.click(lambda: "⏹ stopped", None, stats, cancels=[event, submit])
        if ckpt_name:
            gr.Markdown(f"<sub>checkpoint: `{ckpt_name}`</sub>")
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--quant", default=None, choices=QUANT_MODES, help="quantize a float checkpoint on load")
    parser.add_argument("--max_batch", type=int, default=8, help="people served at once (continuous batching)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="also create a temporary public gradio.live link")
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        hint = (f"\nmake it with: python -m inference.quantize --ckpt checkpoints/sft-small-llama-s1.pt"
                if args.ckpt == DEFAULT_CKPT else "")
        raise SystemExit(f"checkpoint not found: {args.ckpt}{hint}")
    # the model is loaded ON the worker thread (see "HOW IT RUNS THE MODEL" above)
    worker = ModelWorker(lambda: Generator.from_checkpoint(args.ckpt, "cpu", args.quant),
                         max_waiting=16, max_batch=args.max_batch)
    # warm-up: the first generation after loading runs ~3x slower (PyTorch sets up its
    # kernels and threads), so do it now instead of on the first visitor's request
    for _ in stream_story(worker, EXAMPLES[0], INSTRUCTION, 0.7, 0.9, 64, 0):
        pass
    demo = build_demo(worker, os.path.basename(args.ckpt))
    # Gradio's request threads only wait on job queues, so let as many run as there are batch rows
    demo.queue(default_concurrency_limit=args.max_batch, max_size=32)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
