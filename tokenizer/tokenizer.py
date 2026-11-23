"""
Byte-level BPE tokenizer, written from scratch.

This is the same family of tokenizer used by GPT-2, GPT-4 and LLaMA 3:
  1. Split text into chunks with a regex (words, numbers, punctuation, spaces).
  2. Turn each chunk into its UTF-8 bytes (ids 0-255), so nothing is ever "unknown".
  3. Repeatedly merge the most frequent adjacent pair of ids into a new id.

Usage:
    tok = Tokenizer()
    tok.train(text, vocab_size=4096)
    ids = tok.encode("Once upon a time<|endoftext|>")
    text = tok.decode(ids)
    tok.save("tokenizer/tinystories_4k.json")
    tok = Tokenizer.load("tokenizer/tinystories_4k.json")
"""

import json
import time
from collections import Counter, defaultdict

import regex as re  # the `regex` package, not built-in `re`: we need \p{L} (any letter) support

# -----------------------------------------------------------------------------
# GPT-4's split pattern (tiktoken's cl100k_base). Every regex match becomes one chunk,
# and BPE merges never cross chunk boundaries. The alternatives, left to right:
#   '(?i:[sdmt]|ll|ve|re)          contractions: 's 't 'm 'd 'll 've 're
#   [^\r\n\p{L}\p{N}]?+\p{L}+      a word, plus one optional leading char (usually a space)
#   \p{N}{1,3}                     numbers, at most 3 digits per chunk
#    ?[^\s\p{L}\p{N}]++[\r\n]*     punctuation / symbols
#   \s*[\r\n]                      newlines
#   \s+(?!\S)|\s+                  other whitespace (leaves the last space to the next word)
# Result: "Hello world!" -> ["Hello", " world", "!"], so " world" can become ONE token.
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# Reserved tokens. They are never produced by merges and never seen during training.
#   <|endoftext|>              separates stories during pretraining (Part 1)
#   <|user|>, <|assistant|>    chat template for instruction tuning (Part 2)
# Reserving them now means the vocabulary doesn't change between pretraining and SFT.
SPECIAL_TOKENS = ["<|endoftext|>", "<|user|>", "<|assistant|>"]


# -----------------------------------------------------------------------------
# The three BPE building blocks

def get_pair_counts(ids, counts=None, weight=1):
    """Count adjacent pairs. [1, 2, 3, 1, 2] -> {(1,2): 2, (2,3): 1, (3,1): 1}
    `weight` lets us count a chunk that appears many times in one go."""
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + weight
    return counts


def merge(ids, pair, new_id):
    """Replace every occurrence of `pair` with `new_id`, scanning left to right.
    merge([1, 2, 3, 1, 2], (1, 2), 99) -> [99, 3, 99]"""
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2  # skip both halves of the pair
        else:
            out.append(ids[i])
            i += 1
    return out


def best_pair(counts):
    """The most frequent pair. On a tie the smallest pair wins, so training is deterministic."""
    return max(counts, key=lambda p: (counts[p], -p[0], -p[1]))


# -----------------------------------------------------------------------------

class Tokenizer:

    def __init__(self, pattern=GPT4_SPLIT_PATTERN, special_tokens=SPECIAL_TOKENS):
        self.pattern = pattern
        self.compiled_pattern = re.compile(pattern)
        self.special_list = list(special_tokens)
        self.merges = {}  # (id_a, id_b) -> new_id, kept in the order they were learned
        self._build_vocab()

    # ---- vocabulary --------------------------------------------------------

    def _build_vocab(self):
        """Rebuild id -> bytes from scratch: 256 raw bytes, then the merges, then specials."""
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in self.merges.items():
            # merges are stored in learning order, so a and b already exist in vocab
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]

        # special tokens get the ids right after the last merge
        first_special = 256 + len(self.merges)
        self.special_tokens = {s: first_special + i for i, s in enumerate(self.special_list)}
        for s, idx in self.special_tokens.items():
            self.vocab[idx] = s.encode("utf-8")

        # regex that finds special tokens; the (...) group makes re.split() keep them
        if self.special_list:
            self.special_pattern = re.compile("(" + "|".join(re.escape(s) for s in self.special_list) + ")")
        else:
            self.special_pattern = None

        self._cache = {}  # chunk string -> ids, makes encoding repeated words fast

    @property
    def vocab_size(self):
        return len(self.vocab)

    def __len__(self):
        return self.vocab_size

    @property
    def eot_id(self):
        return self.special_tokens["<|endoftext|>"]

    # ---- training ----------------------------------------------------------

    def train(self, text, vocab_size, verbose=True):
        """Learn merges until the vocabulary has `vocab_size` tokens (specials included).

        Doing BPE naively means recounting every pair in the whole text after each merge,
        which is far too slow. Two tricks make it fast:
          1. Deduplicate chunks. " the" may appear 500k times, but we store it ONCE
             along with its count, and every pair count is weighted by that count.
          2. Update counts incrementally. After merging (a, b), only the chunks that
             contained (a, b) change, so only their pair counts get updated.
        tests/test_tokenizer.py checks this gives the same merges as the naive version.
        """
        num_merges = vocab_size - 256 - len(self.special_list)
        assert num_merges >= 0, f"vocab_size must be at least {256 + len(self.special_list)}"

        # special tokens are not part of the training text
        if self.special_pattern is not None:
            text = self.special_pattern.sub("\n", text)

        # trick 1: unique chunks with their frequencies
        chunk_counts = Counter(self.compiled_pattern.findall(text))
        words = [list(chunk.encode("utf-8")) for chunk in chunk_counts]
        freqs = list(chunk_counts.values())

        # initial pair counts, plus an index: pair -> which words contain it
        pair_counts = {}
        where = defaultdict(set)
        for i, (word, freq) in enumerate(zip(words, freqs)):
            for pair in zip(word, word[1:]):
                pair_counts[pair] = pair_counts.get(pair, 0) + freq
                where[pair].add(i)

        if verbose:
            print(f"training on {len(text.encode('utf-8')) / 1e6:.1f} MB of text: "
                  f"{sum(freqs):,} chunks, {len(words):,} unique. Learning {num_merges:,} merges...")

        self.merges = {}
        t0 = time.time()
        for step in range(num_merges):
            if not pair_counts:  # every chunk is already a single token
                break
            pair = best_pair(pair_counts)
            count = pair_counts[pair]
            new_id = 256 + step
            self.merges[pair] = new_id

            # trick 2: only touch the words that contain this pair
            for i in where.pop(pair):
                old = words[i]
                new = merge(old, pair, new_id)
                if len(new) == len(old):
                    continue  # stale index entry: this word no longer contains the pair
                freq = freqs[i]
                # remove the old word's pairs, then add the new word's pairs
                for p in zip(old, old[1:]):
                    pair_counts[p] -= freq
                    if pair_counts[p] == 0:
                        del pair_counts[p]
                for p in zip(new, new[1:]):
                    pair_counts[p] = pair_counts.get(p, 0) + freq
                    where[p].add(i)
                words[i] = new

            if verbose and (step % 256 == 0 or step == num_merges - 1):
                a, b = (self._bytes_so_far(x) for x in pair)
                print(f"  merge {step + 1:5d}/{num_merges}: {a + b!r:<20} "
                      f"(seen {count:,} times)  {time.time() - t0:.1f}s")

        self._build_vocab()

    def _bytes_so_far(self, idx):
        """Bytes for an id while training is still running (for printing only)."""
        if idx < 256:
            return bytes([idx])
        a, b = next(p for p, i in self.merges.items() if i == idx)
        return self._bytes_so_far(a) + self._bytes_so_far(b)

    # ---- encoding / decoding ----------------------------------------------

    def _encode_chunk(self, chunk):
        """Encode one regex chunk by replaying merges in the order they were learned."""
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached

        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # of all pairs present, apply the one that was learned EARLIEST (lowest new id).
            # Later merges are built on top of earlier ones, so order matters.
            pairs = set(zip(ids, ids[1:]))
            pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # no learned merge applies any more
            ids = merge(ids, pair, self.merges[pair])

        if len(self._cache) < 1_000_000:  # cap memory use
            self._cache[chunk] = ids
        return ids

    def encode_ordinary(self, text):
        """Encode text, treating special-token strings as plain text."""
        ids = []
        for chunk in self.compiled_pattern.findall(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def encode(self, text, allow_special=True):
        """Encode text. If allow_special, "<|endoftext|>" etc. become their single special id."""
        if not allow_special or self.special_pattern is None:
            return self.encode_ordinary(text)
        ids = []
        # "hi<|endoftext|>bye" -> ["hi", "<|endoftext|>", "bye"]
        for part in self.special_pattern.split(text):
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            elif part:
                ids.extend(self.encode_ordinary(part))
        return ids

    def decode(self, ids):
        """ids -> text. errors="replace" matters: a model can generate a byte sequence that
        isn't valid UTF-8 (e.g. half of an emoji), and that must not crash decoding."""
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def token_strings(self, ids):
        """Each id as its own readable string. Useful for seeing how text was split."""
        return [self.vocab[i].decode("utf-8", errors="replace") for i in ids]

    # ---- save / load ---------------------------------------------------------

    def save(self, path):
        """Save to JSON. Also writes <name>.vocab.txt, a readable list of every token."""
        data = {
            "type": "byte-level-bpe",
            "pattern": self.pattern,
            "special_tokens": self.special_list,
            "merges": [list(pair) for pair in self.merges],  # in learning order
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        vocab_path = path[:-5] + ".vocab.txt" if path.endswith(".json") else path + ".vocab.txt"
        with open(vocab_path, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                f.write(f"{idx}\t{token.decode('utf-8', errors='replace')!r}\n")

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(pattern=data["pattern"], special_tokens=data["special_tokens"])
        # the i-th merge created id 256 + i
        tok.merges = {(a, b): 256 + i for i, (a, b) in enumerate(data["merges"])}
        tok._build_vocab()
        return tok
