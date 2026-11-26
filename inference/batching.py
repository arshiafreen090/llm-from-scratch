"""
Continuous batching: many requests share every forward pass.

Why batching helps
------------------
Generating one token means reading ALL the model's weights once, to do very little math
with them (one token's worth). That makes decoding limited by memory reads, not by compute.
Put B requests in the same forward pass and the SAME weight reads produce B tokens.
On a GPU a decode step costs almost the same for 1 or 32 rows; on a CPU the cost grows,
but much slower than B.

Static vs continuous batching
-----------------------------
    static       collect B requests, run them together until ALL of them finish.
                 A 5-token reply sits idle (wasting its row) while a 300-token reply finishes,
                 and a new request waits for the whole batch to end.
    continuous   the batch is re-formed at EVERY step (Orca, OSDI '22; used by vLLM and TGI).
                 A finished request leaves at once, and a waiting one takes its place at the
                 next step. Nobody waits for anyone else's long reply.

One engine step
---------------
    1. admit    waiting requests into free slots: one prefill pass per prompt fills that
                slot's KV cache
    2. sample   one token per running request, with ITS OWN sampling params, RNG and history
    3. retire   requests that finished: free their slot
    4. decode   ONE forward pass for all running requests together: input (n, 1), and every
                row sits at its own position in its own sequence

Slots: one KV-cache tensor with max_batch rows, one row per running request

    row 0  [ req A: 12 tokens cached ......... ]   next token at position 12
    row 1  [ req B: 40 tokens cached ........................................ ]   position 40
    row 2  [ req C: 7 tokens ...  | leftovers from an older request (masked out) ]   position 7

The decode pass gets positions = [12, 40, 7]. RoPE rotates each row by its own position, the
new k/v are written at each row's own position, and the attention mask lets row b see keys
0..positions[b] only (model/layers.py). Rows never interact: every operation in the model is
per-row, so a request gets the same tokens it would get alone (tests/test_batching.py checks this).

Running requests always occupy rows 0..n-1 ("compaction"): when the request in row i finishes,
the request in the last row moves into row i (a copy of its cached keys/values). So the decode
pass can use the first n rows as one contiguous batch.

Not done here (the next steps for a real server):
    * chunked prefill: a long prompt's prefill pauses decoding for everyone else for one pass.
      Splitting it into chunks interleaved with decode steps smooths that out.
    * paged KV cache (vLLM's PagedAttention): every slot reserves block_size positions even if
      the request is short. Allocating the cache in small pages lets far more requests fit.
"""

import codecs
import time
from collections import deque

import torch

from inference.generate import GenerationResult, Generator
from inference.sampling import SamplingParams, random_seed, sample_next


class Sequence:
    """Everything that belongs to ONE request. None of it is shared with other rows."""

    def __init__(self, prompt_ids, params: SamplingParams, tag=None):
        self.prompt_ids = list(prompt_ids)
        self.params = params
        self.tag = tag  # the caller's handle (the API stores its Job here)

        self.result = GenerationResult(text="", prompt_tokens=len(self.prompt_ids))
        # same seed rule as Generator: own RNG per request, seed reported for reproduction
        self.result.seed = params.seed if params.seed is not None else random_seed()

        # filled in by the engine
        self.rng = None       # torch.Generator, created on the model's device at admission
        self.history = None   # (1, T) every token so far (repetition penalty needs it)
        self.logits = None    # (1, vocab) scores for the next token
        self.next_id = None   # (1, 1) the token just sampled, fed to the next decode pass
        self.pos = 0          # tokens in this request's cache = position of its next token
        self.steps = 0        # tokens sampled (including special tokens that aren't shown)
        self.max_new = 0
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.pieces = []
        self.t_start = None   # admission time (prefill starts)
        self.t_first = None   # first token sampled


class BatchEngine:
    """Runs many Sequences at once. Single-threaded: the caller calls step() in a loop.

        engine = BatchEngine(generator, max_batch=8)
        engine.add(Sequence(ids, SamplingParams(...)))
        while engine.has_work():
            for seq, kind, value in engine.step():
                ...   # kind: "token" (text piece), "done" (GenerationResult) or "error" (message)

    max_batch=1 turns batching off: requests run one after another (the baseline).
    """

    def __init__(self, generator: Generator, max_batch: int = 8):
        self.gen = generator
        self.model = generator.model
        self.cfg = self.model.cfg
        self.device = generator.device
        self.max_batch = max_batch
        self.caches = self.model.make_kv_caches(max_batch)  # one row ("slot") per running request
        self.running = []       # running[i] lives in row i of every cache
        self.waiting = deque()  # added, not admitted yet

    # ---- bookkeeping ---------------------------------------------------------------

    def add(self, seq: Sequence):
        if len(seq.prompt_ids) >= self.cfg.block_size:
            raise ValueError(f"prompt is {len(seq.prompt_ids)} tokens, the model's context is {self.cfg.block_size}")
        self.waiting.append(seq)

    def has_work(self):
        return bool(self.running or self.waiting)

    def free_slots(self):
        return self.max_batch - len(self.running) - len(self.waiting)

    def sequences(self):
        return list(self.running) + list(self.waiting)

    def abort(self, seq: Sequence):
        """Drop a request (e.g. its client disconnected). No event is emitted for it."""
        if seq in self.waiting:
            self.waiting.remove(seq)
        elif seq in self.running:
            self._free_slot(self.running.index(seq))

    def _free_slot(self, i):
        """Remove running[i], moving the last running request into its row (compaction)."""
        last = len(self.running) - 1
        if i != last:
            moved = self.running[last]
            for cache in self.caches:
                cache.copy_row(last, i, moved.pos)
            self.running[i] = moved
        self.running.pop()

    # ---- the step ------------------------------------------------------------------

    @torch.inference_mode()
    def step(self):
        """One scheduler iteration. Returns a list of (seq, kind, value) events."""
        events = []

        # 1. admit waiting requests into free rows
        while self.waiting and len(self.running) < self.max_batch:
            seq = self.waiting.popleft()
            try:
                self._prefill(seq, row=len(self.running))
            except Exception as e:  # a bad request fails alone
                events.append((seq, "error", str(e)))
                continue
            self.running.append(seq)

        # 2. sample one token per running request
        finished = []
        for i, seq in enumerate(self.running):
            try:
                done = self._sample(seq, events)
            except Exception as e:
                events.append((seq, "error", str(e)))
                done = True
            if done:
                finished.append(i)

        # 3. retire finished requests (highest row first, so compaction never moves one
        #    that is about to be removed)
        for i in reversed(finished):
            self._free_slot(i)

        # 4. one decode pass for everyone still running
        if self.running:
            try:
                self._decode()
            except Exception as e:
                # this pass was SHARED, so its failure hits every request in it. Report it to
                # each of them and start clean; the engine keeps serving new requests.
                for seq in self.running:
                    events.append((seq, "error", f"batched decode step failed: {e}"))
                self.running = []
        return events

    def _prefill(self, seq: Sequence, row: int):
        seq.t_start = time.perf_counter()
        seq.rng = torch.Generator(device=self.device).manual_seed(seq.result.seed)
        seq.history = torch.tensor([seq.prompt_ids], dtype=torch.long, device=self.device)
        # the prompt goes through the model alone, writing into this request's row only
        caches = [cache.rows(row, row + 1) for cache in self.caches]
        logits, _ = self.model(seq.history, kv_caches=caches, start_pos=0)
        seq.logits = logits[:, -1, :]
        seq.pos = len(seq.prompt_ids)
        seq.max_new = min(seq.params.max_new_tokens, self.cfg.block_size - seq.pos)
        seq.result.finish_reason = "context" if seq.max_new < seq.params.max_new_tokens else "length"

    def _sample(self, seq: Sequence, events):
        """Pick this request's next token. Same rules as Generator.stream_ids.
        Returns True when the request is finished."""
        if seq.steps >= seq.max_new:
            return self._finish(seq, events)
        next_id = sample_next(seq.logits, seq.history, seq.params, seq.rng)  # (1, 1)
        seq.steps += 1
        if seq.t_first is None:
            seq.t_first = time.perf_counter()
            seq.result.ttft_ms = (seq.t_first - seq.t_start) * 1000

        token = int(next_id)
        if token in self.gen.special_ids:
            if seq.params.stop_at_special:
                seq.result.finish_reason = "stop"
                return self._finish(seq, events)
            # not stopping: the model still sees it, but a control token is never shown as text
        else:
            seq.result.token_ids.append(token)
            piece = seq.decoder.decode(self.gen.tok.vocab[token])
            if piece:
                seq.pieces.append(piece)
                events.append((seq, "token", piece))

        seq.history = torch.cat([seq.history, next_id], dim=1)
        seq.next_id = next_id
        if seq.steps >= seq.max_new:
            return self._finish(seq, events)
        return False

    def _finish(self, seq: Sequence, events):
        tail = seq.decoder.decode(b"", final=True)  # flush an incomplete UTF-8 character
        if tail:
            seq.pieces.append(tail)
            events.append((seq, "token", tail))
        result = seq.result
        result.text = "".join(seq.pieces)
        t_end = time.perf_counter()
        result.total_ms = (t_end - seq.t_start) * 1000
        if seq.t_first is not None and len(result.token_ids) > 1:
            result.decode_tokens_per_s = (len(result.token_ids) - 1) / (t_end - seq.t_first)
        events.append((seq, "done", result))
        return True

    def _decode(self):
        """All running requests through the model together: rows 0..n-1, one token each."""
        n = len(self.running)
        idx = torch.cat([seq.next_id for seq in self.running], dim=0)                      # (n, 1)
        if n == 1:
            # a lone request: the plain integer-position path (no per-row mask, gather or
            # scatter), so running unbatched costs exactly what Generator costs
            positions = self.running[0].pos
        else:
            positions = torch.tensor([seq.pos for seq in self.running], device=self.device)  # (n,)
        caches = [cache.rows(0, n) for cache in self.caches]
        logits, _ = self.model(idx, kv_caches=caches, start_pos=positions)                 # (n, 1, vocab)
        for i, seq in enumerate(self.running):
            seq.logits = logits[i:i + 1, -1, :]
            seq.pos += 1
