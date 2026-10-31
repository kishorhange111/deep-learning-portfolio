"""
OpenAI, Google Gemini, xAI Grok and self-hosted open models, through the OpenAI SDK.

Gemini and Grok both expose OpenAI-compatible Chat Completions endpoints, and so do vLLM, SGLang
and Ollama servers for open-weight models. One client therefore covers them all; only the base URL and
API-key variable differ.
"""

import os
import sys
import time

import openai

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from providers import LLMResponse  # noqa: E402

ENDPOINTS = {  # provider -> (base_url, API-key environment variable)
    "openai": (None, "OPENAI_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
    "local": ("http://localhost:8000/v1", None),  # e.g. `vllm serve <model>`; pass another URL with model@url
}


class OpenAICompatClient:
    """complete(system, user, max_tokens, json_schema) -> LLMResponse."""

    def __init__(self, provider, model, base_url=None, temperature=None, max_retries=4):
        default_url, key_var = ENDPOINTS[provider]
        self.provider, self.model, self.temperature = provider, model, temperature
        api_key = os.environ.get(key_var, "") if key_var else os.environ.get("LOCAL_API_KEY", "EMPTY")
        self.client = openai.OpenAI(api_key=api_key or None, base_url=base_url or default_url, max_retries=max_retries)

    def build_request(self, system, user, max_tokens=4096, json_schema=None):
        """Chat Completions request body."""
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        params = {"model": self.model, "messages": messages}
        # OpenAI's reasoning models only accept max_completion_tokens; the compatible endpoints use max_tokens
        params["max_completion_tokens" if self.provider == "openai" else "max_tokens"] = max_tokens
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if json_schema is not None:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "judgement", "schema": json_schema, "strict": True},
            }
        return params

    @staticmethod
    def parse_response(resp, latency):
        """Message text, usage and refusal flag from a ChatCompletion."""
        choice = resp.choices[0]
        refusal = getattr(choice.message, "refusal", None)
        usage = resp.usage
        return LLMResponse(
            text=choice.message.content or "",
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_s=latency,
            stop_reason=choice.finish_reason,
            refused=bool(refusal) or choice.finish_reason == "content_filter",
        )

    def complete(self, system, user, max_tokens=4096, json_schema=None):
        """One request; API errors are returned in the response (the SDK already retried 429/5xx)."""
        params = self.build_request(system, user, max_tokens, json_schema)
        t0 = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(**params)
        except openai.BadRequestError as e:
            return LLMResponse(error=f"bad_request: {e.message}", latency_s=time.perf_counter() - t0)
        except openai.RateLimitError as e:
            return LLMResponse(error=f"rate_limited: {e.message}", latency_s=time.perf_counter() - t0)
        except openai.APIStatusError as e:
            return LLMResponse(error=f"api_status_{e.status_code}: {e.message}", latency_s=time.perf_counter() - t0)
        except openai.APIConnectionError as e:
            return LLMResponse(error=f"connection: {e}", latency_s=time.perf_counter() - t0)
        return self.parse_response(resp, time.perf_counter() - t0)
