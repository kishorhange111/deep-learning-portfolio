"""
Claude as the agent's model, through the Anthropic SDK (Messages API with tool use).

Prompt caching for an agent loop:
  * tools (fixed, deterministic order) -> system prompt (frozen: no dates or IDs in it) -> messages
  * one explicit breakpoint on the system prompt caches tools + system for every turn and every user
  * top-level automatic caching moves a second breakpoint to the end of the growing conversation, so
    each turn re-reads everything before it from cache and only pays full price for what is new

The history is append-only: each assistant turn goes back exactly as returned, including thinking
blocks. Editing or dropping earlier blocks would break the cache, and on current models it
invalidates the preserved thinking.

Refusals: in the app, the server-side fallback (`fallbacks: "default"`) re-runs a declined request on
the model Anthropic recommends for that refusal category. The eval turns it off so every answer comes
from the model being measured.
"""

import os
import sys

import anthropic

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent import ToolCall, Turn  # noqa: E402

FALLBACK_BETA = "server-side-fallback-2026-07-01"


def to_anthropic_tools(specs):
    """Provider-neutral specs -> Messages API tool definitions (strict: inputs always match the schema)."""
    return [
        {"name": s["name"], "description": s["description"], "input_schema": s["parameters"], "strict": True}
        for s in specs
    ]


class ClaudeBackend:
    """Holds one conversation in Anthropic's message format."""

    def __init__(self, model="claude-opus-5-5", effort="medium", fallbacks=True, max_tokens=16000, client=None):
        self.model, self.effort, self.fallbacks, self.max_tokens = model, effort, fallbacks, max_tokens
        self.client = client or anthropic.Anthropic(max_retries=4)
        self.messages = []

    def start(self, system, tool_specs, question):
        """Begin a conversation."""
        self.system = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        self.tools = to_anthropic_tools(tool_specs)
        self.messages = [{"role": "user", "content": question}]

    def request(self):
        """The next request body."""
        params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self.system,
            "tools": self.tools,
            "messages": self.messages,
            "output_config": {"effort": self.effort},
            "cache_control": {"type": "ephemeral"},
        }
        if self.fallbacks:
            params.update(betas=[FALLBACK_BETA], fallbacks="default")
        return params

    def next_turn(self):
        """Call the model; append its full reply to the history; return the parsed turn."""
        params = self.request()
        try:
            resp = (self.client.beta.messages if self.fallbacks else self.client.messages).create(**params)
        except anthropic.BadRequestError as e:
            return Turn(error=f"bad_request: {e.message}")
        except anthropic.RateLimitError as e:
            return Turn(error=f"rate_limited: {e.message}")
        except anthropic.APIStatusError as e:
            return Turn(error=f"api_status_{e.status_code}: {e.message}")
        except anthropic.APIConnectionError as e:
            return Turn(error=f"connection: {e}")
        self.messages.append({"role": "assistant", "content": resp.content})  # verbatim, thinking blocks included
        u = resp.usage
        return Turn(
            text="".join(b.text for b in resp.content if b.type == "text"),
            tool_calls=[ToolCall(b.id, b.name, dict(b.input)) for b in resp.content if b.type == "tool_use"],
            stop_reason=resp.stop_reason,
            refused=resp.stop_reason == "refusal",
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            served_by=getattr(resp, "model", self.model),
        )

    def add_tool_results(self, results):
        """All results of one turn go back in a single user message (keeps parallel tool use working)."""
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": call_id, "content": text, "is_error": is_error}
                    for call_id, text, is_error in results
                ],
            }
        )

    def add_user_text(self, text):
        """A plain user message (used for the one 'please call submit_answer' nudge)."""
        self.messages.append({"role": "user", "content": text})
