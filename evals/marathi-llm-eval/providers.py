"""
One interface over every model provider, plus a response cache and cost accounting.

Model specs:   provider:model[@base_url]
  anthropic:claude-opus-5-5          Claude (Anthropic SDK)                         ANTHROPIC_API_KEY
  openai:<model>                     OpenAI                                         OPENAI_API_KEY
  gemini:<model>                     Google Gemini (OpenAI-compatible endpoint)     GEMINI_API_KEY
  xai:<model>                        xAI Grok (OpenAI-compatible endpoint)          XAI_API_KEY
  local:<model>[@http://host:port/v1]  any vLLM / SGLang / Ollama server
  mock:<name>                        canned functions for tests and dry runs

Every response is cached in SQLite, keyed by (spec, settings, system, prompt). Re-running an eval
costs nothing, and the exact responses behind every number can be inspected afterwards.
"""

import hashlib
import json
import os
import sqlite3
import threading
from dataclasses import asdict, dataclass

# USD per million tokens (input, output). Claude prices from Anthropic's published list; add the
# other providers' current prices with --prices prices.json (cost is reported as unknown otherwise).
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}


@dataclass
class LLMResponse:
    """What every client returns."""

    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    stop_reason: str = None
    refused: bool = False
    error: str = None
    cached: bool = False


# mock providers: name -> fn(system, user) -> text   (tests can register more)
MOCKS = {
    "always-a": lambda system, user: "A",
    "echo": lambda system, user: user,
    "judge-3": lambda system, user: json.dumps(
        {"rationale": "ok", "helpfulness": 3, "fluency": 3, "language": "marathi"}
    ),
}


class MockClient:
    """Deterministic fake model."""

    def __init__(self, name):
        self.fn = MOCKS[name]

    def complete(self, system, user, max_tokens=None, json_schema=None):
        """Call the registered function."""
        text = self.fn(system, user)
        return LLMResponse(text=text, input_tokens=len(user) // 4, output_tokens=len(text) // 4)


def parse_spec(spec):
    """'gemini:gemini-x@https://..' -> ('gemini', 'gemini-x', 'https://..')."""
    provider, _, rest = spec.partition(":")
    model, _, base_url = rest.partition("@")
    return provider, model, base_url or None


def make_client(spec, effort="medium", temperature=None):
    """Instantiate the right client for a model spec (provider SDKs are imported lazily)."""
    provider, model, base_url = parse_spec(spec)
    if provider == "anthropic":
        from anthropic_client import AnthropicClient

        return AnthropicClient(model, effort=effort)
    if provider in ("openai", "gemini", "xai", "local"):
        from openai_compat_client import OpenAICompatClient

        return OpenAICompatClient(provider, model, base_url=base_url, temperature=temperature)
    if provider == "mock":
        return MockClient(model)
    raise ValueError(f"unknown provider in {spec!r}")


class CachedClient:
    """Wraps a client with a thread-safe SQLite response cache. Errors are not cached, so they are retried
    on the next run."""

    def __init__(self, spec, client, cache_path, settings=""):
        self.spec, self.client, self.settings = spec, client, settings
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        self.db = sqlite3.connect(cache_path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT)")
        self.lock = threading.Lock()

    def _key(self, system, user, max_tokens, json_schema):
        raw = json.dumps(
            [self.spec, self.settings, system, user, max_tokens, json_schema], sort_keys=True, ensure_ascii=False
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    def complete(self, system, user, max_tokens=4096, json_schema=None):
        """Cached completion."""
        key = self._key(system, user, max_tokens, json_schema)
        with self.lock:
            row = self.db.execute("SELECT value FROM responses WHERE key = ?", (key,)).fetchone()
        if row:
            return LLMResponse(**{**json.loads(row[0]), "cached": True})
        resp = self.client.complete(system, user, max_tokens=max_tokens, json_schema=json_schema)
        if resp.error is None:
            with self.lock:
                self.db.execute("INSERT OR REPLACE INTO responses VALUES (?, ?)", (key, json.dumps(asdict(resp))))
                self.db.commit()
        return resp


def cost_usd(spec, input_tokens, output_tokens, prices=None):
    """Dollar cost of a token count, or None if the model's price is unknown."""
    table = {**PRICES, **(prices or {})}
    _, model, _ = parse_spec(spec)
    if model not in table:
        return None
    p_in, p_out = table[model]
    return (input_tokens * p_in + output_tokens * p_out) / 1e6


def slug(spec):
    """Filesystem-safe name for a model spec."""
    return parse_spec(spec)[0] + "_" + parse_spec(spec)[1].replace("/", "_").replace(":", "_")
