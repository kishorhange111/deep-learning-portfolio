"""
OpenAI, Gemini, Grok and self-hosted open models as the agent's model, through the OpenAI SDK's Chat
Completions function calling (Gemini and Grok expose OpenAI-compatible endpoints, as do vLLM and Ollama).

OpenAI caches long prompt prefixes automatically. The cached share is read from
usage.prompt_tokens_details.cached_tokens when the provider reports it.
"""

import json
import os
import sys

import openai

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent import ToolCall, Turn  # noqa: E402

ENDPOINTS = {
    "openai": (None, "OPENAI_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
    "local": ("http://localhost:8000/v1", None),
}


def to_openai_tools(specs):
    """Provider-neutral specs -> Chat Completions tool definitions."""
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["parameters"],
                "strict": True,
            },
        }
        for s in specs
    ]


class OpenAIBackend:
    """Holds one conversation in Chat Completions format."""

    def __init__(self, provider, model, base_url=None, max_tokens=8000, client=None):
        default_url, key_var = ENDPOINTS[provider]
        self.provider, self.model, self.max_tokens = provider, model, max_tokens
        key = os.environ.get(key_var, "") if key_var else os.environ.get("LOCAL_API_KEY", "EMPTY")
        self.client = client or openai.OpenAI(api_key=key or None, base_url=base_url or default_url, max_retries=4)

    def start(self, system, tool_specs, question):
        """Begin a conversation."""
        self.tools = to_openai_tools(tool_specs)
        self.messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]

    def request(self):
        """The next request body."""
        params = {"model": self.model, "messages": self.messages, "tools": self.tools}
        params["max_completion_tokens" if self.provider == "openai" else "max_tokens"] = self.max_tokens
        return params

    def next_turn(self):
        """Call the model; append its reply; return the parsed turn."""
        try:
            resp = self.client.chat.completions.create(**self.request())
        except openai.BadRequestError as e:
            return Turn(error=f"bad_request: {e.message}")
        except openai.RateLimitError as e:
            return Turn(error=f"rate_limited: {e.message}")
        except openai.APIStatusError as e:
            return Turn(error=f"api_status_{e.status_code}: {e.message}")
        except openai.APIConnectionError as e:
            return Turn(error=f"connection: {e}")
        choice = resp.choices[0]
        msg = choice.message
        calls = msg.tool_calls or []
        assistant = {"role": "assistant", "content": msg.content or ""}
        if calls:
            assistant["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.function.name, "arguments": c.function.arguments},
                }
                for c in calls
            ]
        self.messages.append(assistant)
        parsed = []
        for c in calls:
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_invalid_json": c.function.arguments}
            parsed.append(ToolCall(c.id, c.function.name, args))
        usage = resp.usage
        details = getattr(usage, "prompt_tokens_details", None)
        cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
        return Turn(
            text=msg.content or "",
            tool_calls=parsed,
            stop_reason=choice.finish_reason,
            refused=bool(getattr(msg, "refusal", None)),
            input_tokens=(usage.prompt_tokens or 0) - cached,
            output_tokens=usage.completion_tokens or 0,
            cache_read_tokens=cached,
            served_by=resp.model or self.model,
        )

    def add_tool_results(self, results):
        """One 'tool' message per call."""
        for call_id, text, _ in results:
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})

    def add_user_text(self, text):
        """A plain user message."""
        self.messages.append({"role": "user", "content": text})
