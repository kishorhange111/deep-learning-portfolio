"""
Claude through the official Anthropic SDK.

Choices that matter for an eval:
  * Current Claude models think adaptively. Depth is set with output_config.effort, which is passed
    explicitly and recorded, because it changes both accuracy and cost. Sampling parameters
    (temperature, top_p) are not sent: current models and API versions reject them.
  * No server-side refusal fallback. A fallback would silently answer with a *different* model,
    which would corrupt the measurement. A refusal is recorded as a refusal (stop_reason "refusal").
  * The judge's scores come back through structured outputs (output_config.format with a JSON
    schema), so they always parse.
"""

import os
import sys
import time

import anthropic

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from providers import LLMResponse  # noqa: E402


class AnthropicClient:
    """complete(system, user, max_tokens, json_schema) -> LLMResponse."""

    def __init__(self, model, effort="medium", max_retries=4):
        self.model, self.effort = model, effort
        self.client = anthropic.Anthropic(max_retries=max_retries)

    def build_request(self, system, user, max_tokens=16000, json_schema=None):
        """The Messages API request body (kept separate so it can be tested without network access)."""
        output_config = {"effort": self.effort}
        if json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        params = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": user}],
            "output_config": output_config,
        }
        if system:
            params["system"] = system
        return params

    @staticmethod
    def parse_response(resp, latency):
        """Text blocks joined (thinking blocks are skipped), usage, and the refusal flag."""
        text = "".join(block.text for block in resp.content if block.type == "text")
        return LLMResponse(
            text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_s=latency,
            stop_reason=resp.stop_reason,
            refused=resp.stop_reason == "refusal",
        )

    def complete(self, system, user, max_tokens=16000, json_schema=None):
        """One request; API errors are returned in the response (the SDK already retried 429/5xx)."""
        params = self.build_request(system, user, max_tokens, json_schema)
        t0 = time.perf_counter()
        try:
            resp = self.client.messages.create(**params)
        except anthropic.BadRequestError as e:  # not retryable: a problem with the request itself
            return LLMResponse(error=f"bad_request: {e.message}", latency_s=time.perf_counter() - t0)
        except anthropic.RateLimitError as e:
            return LLMResponse(error=f"rate_limited: {e.message}", latency_s=time.perf_counter() - t0)
        except anthropic.APIStatusError as e:
            return LLMResponse(error=f"api_status_{e.status_code}: {e.message}", latency_s=time.perf_counter() - t0)
        except anthropic.APIConnectionError as e:
            return LLMResponse(error=f"connection: {e}", latency_s=time.perf_counter() - t0)
        return self.parse_response(resp, time.perf_counter() - t0)
