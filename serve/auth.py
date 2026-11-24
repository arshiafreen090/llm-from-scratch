"""
API keys and rate limits for the HTTP API.

    python -m serve.auth new-key                 # print a fresh key (give it to a user)
    TINYLLAMA_API_KEYS=sk-tl-abc,sk-tl-def python -m serve.api --ckpt ...   # only these keys work

HOW KEYS WORK
    The server owner creates keys and passes them in the TINYLLAMA_API_KEYS environment
    variable (comma-separated) or a file (--api_keys_file, one per line). On a host like
    Hugging Face Spaces that's a "secret". Clients send one as a header:
        Authorization: Bearer sk-tl-...
    exactly like the OpenAI API, so the OpenAI client libraries work unchanged.
    With no keys configured, the API is open (and rate-limited per IP address).

    The server keeps only SHA-256 hashes of the keys, and compares hashes with
    hmac.compare_digest, whose running time doesn't depend on how many characters match
    (a plain == can leak, through timing, how much of a guessed key was right).

    Deliberately NOT here: self-service sign-up. Letting strangers create their own keys
    needs accounts, a database, email checks and abuse handling. For one CPU serving a
    small model, owner-issued keys are the reliable option.

RATE LIMITS (token bucket, per key, or per IP address when the API is open)
    Each client has a bucket of `burst` tokens that refills at `per_minute` tokens per
    minute. A request takes one token; with none left the answer is HTTP 429 plus a
    Retry-After header saying how many seconds until the next token. Short bursts are fine;
    a sustained flood is capped at per_minute requests per minute.
"""

import hashlib
import hmac
import math
import os
import secrets
import sys
import threading
import time

from fastapi import Request

ENV_VAR = "TINYLLAMA_API_KEYS"


class APIError(Exception):
    """An error returned in the OpenAI format: {"error": {"message", "type", "code"}}."""

    def __init__(self, status, message, type_="invalid_request_error", code=None, headers=None):
        super().__init__(message)
        self.status, self.message, self.type, self.code = status, message, type_, code
        self.headers = headers or {}

    def body(self):
        return {"error": {"message": self.message, "type": self.type, "code": self.code}}


def hash_key(key):
    return hashlib.sha256(key.strip().encode()).hexdigest()


def new_key():
    return "sk-tl-" + secrets.token_urlsafe(24)  # 192 random bits


def load_key_hashes(keys_file=None, env=None):
    """Hashes of every key in the environment variable and the optional file."""
    env = os.environ if env is None else env
    keys = [k for k in env.get(ENV_VAR, "").split(",")]
    if keys_file:
        with open(keys_file, encoding="utf-8") as f:
            keys += [line for line in f if not line.strip().startswith("#")]
    return {hash_key(k) for k in keys if k.strip()}


class RateLimiter:
    """Token bucket per client id. Thread-safe."""

    def __init__(self, per_minute, burst=None):
        self.rate = per_minute / 60.0            # tokens added per second
        self.burst = burst or max(1, per_minute // 4)
        self.buckets = {}                         # client -> (tokens, time of last update)
        self.lock = threading.Lock()

    def check(self, client, now=None):
        """Take one token. Returns 0 if allowed, else the seconds to wait."""
        now = time.monotonic() if now is None else now
        with self.lock:
            tokens, last = self.buckets.get(client, (self.burst, now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            if tokens >= 1:
                self.buckets[client] = (tokens - 1, now)
                return 0
            self.buckets[client] = (tokens, now)
            return math.ceil((1 - tokens) / self.rate)


class Guard:
    """FastAPI dependency: checks the API key (if keys are configured) and the rate limit.
    Returns an id for the client: "key:<hash prefix>" or "ip:<address>"."""

    def __init__(self, key_hashes=(), limiter: RateLimiter = None, trust_proxy=False):
        self.key_hashes = set(key_hashes)
        self.limiter = limiter
        self.trust_proxy = trust_proxy  # behind a proxy (e.g. HF Spaces) the real IP is in X-Forwarded-For

    @property
    def keys_required(self):
        return bool(self.key_hashes)

    def client_ip(self, request):
        forwarded = request.headers.get("x-forwarded-for")
        if self.trust_proxy and forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def __call__(self, request: Request):
        if self.key_hashes:
            auth = request.headers.get("authorization", "")
            key = auth[7:] if auth.lower().startswith("bearer ") else ""
            if not key:
                raise APIError(401, "Missing API key. Send it as 'Authorization: Bearer <key>'.",
                               code="missing_api_key")
            h = hash_key(key)
            if not any(hmac.compare_digest(h, k) for k in self.key_hashes):
                raise APIError(401, "Invalid API key.", code="invalid_api_key")
            client = "key:" + h[:12]
        else:
            client = "ip:" + self.client_ip(request)
        if self.limiter is not None:
            wait = self.limiter.check(client)
            if wait:
                raise APIError(429, f"Rate limit reached. Try again in {wait} s.", type_="rate_limit_error",
                               code="rate_limit_exceeded", headers={"Retry-After": str(wait)})
        return client


if __name__ == "__main__":
    if sys.argv[1:] == ["new-key"]:
        key = new_key()
        print(key)
        print(f"# add it to {ENV_VAR} (comma-separated) or to an --api_keys_file; share it only with its user",
              file=sys.stderr)
    else:
        print("usage: python -m serve.auth new-key", file=sys.stderr)
