// In-browser model engine, run as a Web Worker (a background thread) so the page never freezes.
// It downloads the ONNX model (made by inference/export_onnx.py) and runs it with ONNX Runtime
// Web (WebAssembly) on the visitor's own CPU: nothing is sent to a server.
//
// Messages in:   {type: "load", modelUrl, metaUrl, tokenizerUrl}
//                {type: "generate", id, instruction, chat, params: {temperature, topP, maxTokens, seed}}
//                {type: "stop", id}
// Messages out:  {type: "progress", loaded, total}   {type: "ready", meta}
//                {type: "token", id, text}   {type: "done", id, stats}   {type: "error", id, message}
"use strict";

const ORT_VERSION = "1.20.1";
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
importScripts(ORT_BASE + "ort.wasm.min.js", "tokenizer.js");
ort.env.wasm.wasmPaths = ORT_BASE;
// several threads need SharedArrayBuffer, which needs special server headers that free static
// hosting doesn't send; one thread is plenty for an 11M-parameter model
ort.env.wasm.numThreads = 1;

let session, tok, meta;
const stopped = new Set();   // ids of requests the page asked to stop

// ---------- loading, with download progress ----------
async function fetchWithProgress(url, expectedBytes) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`couldn't download the model (HTTP ${res.status})`);
  const total = Number(res.headers.get("content-length")) || expectedBytes || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    postMessage({ type: "progress", loaded, total });
  }
  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const c of chunks) { bytes.set(c, offset); offset += c.length; }
  return bytes;
}

async function load({ modelUrl, metaUrl, tokenizerUrl }) {
  meta = await (await fetch(metaUrl)).json();
  if (meta.format !== "tinyllama-onnx-v1") throw new Error("unknown model format");
  [tok] = await Promise.all([Tokenizer.load(tokenizerUrl)]);
  const bytes = await fetchWithProgress(modelUrl, meta.bytes);
  // graphOptimizationLevel "all" folds the int8 -> float32 weight conversion once, at load
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  postMessage({ type: "ready", meta });
}

// ---------- sampling (same rules as inference/sampling.py: temperature, then top-p) ----------
function mulberry32(seed) {   // small seeded random generator: same seed, same story
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function sample(logits, temperature, topP, random) {
  if (temperature === 0) {                       // greedy: the single most likely token
    let best = 0;
    for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
    return best;
  }
  const order = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]);
  const max = logits[order[0]];
  const probs = order.map((i) => Math.exp((logits[i] - max) / temperature));
  const sum = probs.reduce((s, p) => s + p, 0);
  // top-p: keep the smallest set of top tokens whose probability adds up to topP
  let kept = 0, cum = 0;
  while (kept < order.length && cum < topP) { cum += probs[kept] / sum; kept++; }
  let r = random() * probs.slice(0, kept).reduce((s, p) => s + p, 0);
  for (let i = 0; i < kept; i++) { r -= probs[i]; if (r <= 0) return order[i]; }
  return order[kept - 1];
}

// ---------- generation ----------
const ids64 = (ids) => new ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, ids.length]);

// Single-threaded WebAssembly computes each step synchronously, so `await session.run()` never
// lets the worker read its messages, and a "stop" would only arrive after the whole story.
// Posting to a MessageChannel and waiting for it queues behind any waiting messages: a pause
// of ~0 ms (setTimeout(0) would be throttled to ~4 ms when called repeatedly).
const channel = new MessageChannel();
const yieldToMessages = () => new Promise((resolve) => { channel.port1.onmessage = resolve; channel.port2.postMessage(0); });

async function generate({ id, instruction, chat, params }) {
  const prompt = chat ? tok.promptIds(instruction) : tok.completionIds(instruction);
  if (prompt.length >= meta.block_size) {
    throw new Error(`That's ${prompt.length} tokens; the model's context is ${meta.block_size}.`);
  }
  const maxNew = Math.min(params.maxTokens, meta.block_size - prompt.length);
  const seed = Number.isInteger(params.seed) ? params.seed : crypto.getRandomValues(new Uint32Array(1))[0];
  const random = mulberry32(seed);
  const empty = new ort.Tensor("float32", new Float32Array(0), [meta.n_layer, 1, meta.n_kv_head, 0, meta.head_dim]);
  const decoder = new TextDecoder("utf-8");      // holds back half-finished multi-byte characters

  let feeds = { input_ids: ids64(prompt), past_key: empty, past_value: empty };
  let finish = "length", tokens = 0, tFirst = null;
  const tStart = performance.now();
  for (let step = 0; step < maxNew; step++) {
    if (step % 4 === 0) await yieldToMessages();   // every 4 tokens: let a "stop" message in
    if (stopped.has(id)) { finish = "stopped"; break; }
    const out = await session.run(feeds);
    const token = sample(out.logits.data, params.temperature, params.topP, random);
    if (tFirst === null) tFirst = performance.now();
    if (tok.specialIds.has(token)) { finish = "stop"; break; }   // <|endoftext|>: the story is done
    tokens++;
    const text = decoder.decode(tok.vocab[token], { stream: true });
    if (text) postMessage({ type: "token", id, text });
    feeds = { input_ids: ids64([token]), past_key: out.present_key, past_value: out.present_value };
  }
  const tail = decoder.decode();
  if (tail) postMessage({ type: "token", id, text: tail });
  const tEnd = performance.now();
  const tps = tFirst !== null && tokens > 1 ? (tokens - 1) / ((tEnd - tFirst) / 1000) : 0;
  postMessage({ type: "done", id, stats: { tokens, tps, seed, finish, ttft_ms: tFirst ? tFirst - tStart : 0 } });
}

// ---------- messages ----------
let queue = Promise.resolve();   // one generation at a time
onmessage = (e) => {
  const msg = e.data;
  if (msg.type === "stop") { stopped.add(msg.id); return; }
  if (msg.type === "load") {
    load(msg).catch((err) => postMessage({ type: "error", id: null, message: err.message }));
    return;
  }
  if (msg.type === "generate") {
    queue = queue.then(() => generate(msg))
      .catch((err) => postMessage({ type: "error", id: msg.id, message: err.message }))
      .finally(() => stopped.delete(msg.id));
  }
};
