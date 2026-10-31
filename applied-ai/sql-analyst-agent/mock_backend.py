"""
Deterministic stand-in models for tests, CI and dry runs (no API keys, no cost).

  mock:oracle   answers benign questions by running the gold SQL from tasks.json, and injection
                questions by reading the ticket. It never follows instructions found in data.
  mock:naive    like oracle, but obeys any instruction it reads in tool output. This is the worst-case
                victim, used to check that each defense actually blocks the attack path.
  mock:lazy     never calls a tool (exercises the submit_answer reminder and the no-answer path).
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent import ToolCall, Turn  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TICKET_QUESTION = re.compile(r"Customer (\d+) opened a support ticket on (\d{4}-\d{2}-\d{2})")


def _gold_sql(question):
    with open(os.path.join(HERE, "tasks.json")) as f:
        return next((t["gold_sql"] for t in json.load(f) if t["question"] == question), None)


class ScriptedBackend:
    """Implements the backend interface with a Python policy instead of an LLM."""

    def __init__(self, policy):
        self.policy = policy

    def start(self, system, tool_specs, question):
        """Remember the question; reset the transcript."""
        self.question, self.results, self.n = question, [], 0

    def next_turn(self):
        """Ask the policy for the next action."""
        self.n += 1
        action = self.policy(self.question, self.results)
        if isinstance(action, str):
            return Turn(text=action, input_tokens=100, output_tokens=10)
        name, args = action
        return Turn(tool_calls=[ToolCall(f"call_{self.n}", name, args)], input_tokens=100, output_tokens=20)

    def add_tool_results(self, results):
        """Record tool outputs (the policy reads them)."""
        self.results += [text for _, text, _ in results]

    def add_user_text(self, text):
        """Ignored by scripted policies."""


def _ticket_policy(obey_injections):
    def policy(question, results):
        m = TICKET_QUESTION.search(question)
        if m:
            if not results:
                return "run_sql", {
                    "query": f"SELECT body FROM support_tickets WHERE customer_id = {m.group(1)} "
                    f"AND created_at = '{m.group(2)}'"
                }
            seen = "\n".join(results)
            if (
                obey_injections
                and "send_email" in seen
                and not any(r.startswith(("Email", "Blocked")) for r in results)
            ):
                address = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", seen)
                return "send_email", {
                    "to": address.group(0) if address else "unknown",
                    "subject": "export",
                    "body": "data",
                }
            if obey_injections and "ACCESS GRANTED" in seen:
                return "submit_answer", {"answer": "ACCESS GRANTED", "sql": "SELECT 1"}
            order = re.search(r"order #(\d+)", results[0])
            damaged = "damaged" in results[0]
            answer = f"The ticket mentions order {order.group(1) if order else 'unknown'}" + (
                ", which arrived damaged." if damaged else "."
            )
            return "submit_answer", {"answer": answer, "sql": "SELECT body FROM support_tickets"}
        gold = _gold_sql(question)
        if gold is None:
            return "I don't know."
        if not results:
            return "run_sql", {"query": gold}
        rows = [line for line in results[-1].splitlines() if not line.startswith(("<tool_output>", "</tool_output>"))]
        return "submit_answer", {"answer": rows[-1], "sql": gold}

    return policy


def make_mock(name):
    """Factory used by agent.make_backend for 'mock:<name>'."""
    if name == "oracle":
        return ScriptedBackend(_ticket_policy(obey_injections=False))
    if name == "naive":
        return ScriptedBackend(_ticket_policy(obey_injections=True))
    if name == "lazy":
        return ScriptedBackend(lambda q, r: "I think the answer is 42.")
    raise ValueError(f"unknown mock {name!r}")
