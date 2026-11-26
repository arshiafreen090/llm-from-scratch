// Byte-level BPE tokenizer: a JavaScript port of tokenizer/tokenizer.py (encode and decode)
// plus the chat template of tokenizer/chat.py. Loaded by the in-browser engine (engine.js).
// tests/test_web_tokenizer.py checks that it gives exactly the same ids as the Python one.
"use strict";

// tokenizer.py's GPT-4 split pattern, written for JavaScript's regex engine:
//  - JavaScript has no possessive quantifiers (?+ ++). Here they never change a match: after
//    "[^\r\n\p{L}\p{N}]?" comes "\p{L}+", which can't start at a non-letter anyway, and after
//    "[^\s\p{L}\p{N}]+" comes "[\r\n]*", which can't fail. So plain ? and + give the same chunks.
//  - "(?i:...)" becomes the flag i on the whole pattern: the only other letters in the pattern
//    are inside \p{...}, which case-insensitivity doesn't change.
const PYTHON_PATTERN = String.raw`'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+`;
const JS_PATTERN = /'(?:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+/giu;

class Tokenizer {
  constructor(data) {
    if (data.pattern !== PYTHON_PATTERN) throw new Error("tokenizer uses a split pattern this port doesn't know");
    this.ranks = new Map();                 // pair (a, b) -> the id it merges into
    this.vocab = [];                        // id -> bytes
    for (let i = 0; i < 256; i++) this.vocab.push(Uint8Array.of(i));
    data.merges.forEach(([a, b], i) => {
      this.ranks.set(a * 65536 + b, 256 + i);
      const bytes = new Uint8Array(this.vocab[a].length + this.vocab[b].length);
      bytes.set(this.vocab[a]);
      bytes.set(this.vocab[b], this.vocab[a].length);
      this.vocab.push(bytes);
    });
    this.special = {};
    data.special_tokens.forEach((s, i) => {
      this.special[s] = 256 + data.merges.length + i;
      this.vocab.push(new TextEncoder().encode(s));
    });
    this.specialIds = new Set(Object.values(this.special));
    this.eot = this.special["<|endoftext|>"];
    this.cache = new Map();
  }

  static async load(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`couldn't load the tokenizer (HTTP ${res.status})`);
    return new Tokenizer(await res.json());
  }

  // One regex chunk -> ids, replaying merges in the order they were learned (lowest id first),
  // exactly like Tokenizer._encode_chunk.
  encodeChunk(chunk) {
    const cached = this.cache.get(chunk);
    if (cached) return cached;
    let ids = Array.from(new TextEncoder().encode(chunk));
    while (ids.length >= 2) {
      let best = Infinity, bestKey = -1;
      for (let i = 0; i < ids.length - 1; i++) {
        const rank = this.ranks.get(ids[i] * 65536 + ids[i + 1]);
        if (rank !== undefined && rank < best) { best = rank; bestKey = ids[i] * 65536 + ids[i + 1]; }
      }
      if (bestKey < 0) break;                 // no learned merge applies any more
      const a = Math.floor(bestKey / 65536), b = bestKey % 65536, out = [];
      for (let i = 0; i < ids.length; i++) {
        if (i < ids.length - 1 && ids[i] === a && ids[i + 1] === b) { out.push(best); i++; }
        else out.push(ids[i]);
      }
      ids = out;
    }
    if (this.cache.size < 50000) this.cache.set(chunk, ids);
    return ids;
  }

  // Special-token strings typed by a user stay plain text (as in encode_ordinary)
  encodeOrdinary(text) {
    const ids = [];
    for (const m of text.matchAll(JS_PATTERN)) ids.push(...this.encodeChunk(m[0]));
    return ids;
  }

  // tokenizer/chat.py prompt_ids: <|endoftext|> <|user|> instruction <|assistant|>
  promptIds(instruction) {
    return [this.eot, this.special["<|user|>"], ...this.encodeOrdinary(instruction.trim()), this.special["<|assistant|>"]];
  }

  // Generator.encode_prompt(chat=False): a story continues after <|endoftext|>
  completionIds(text) {
    return [this.eot, ...this.encodeOrdinary(text)];
  }
}

if (typeof self !== "undefined") self.Tokenizer = Tokenizer;
