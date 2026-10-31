"""
Web app: ask ShopCo's data a question in plain English, see the answer, the SQL behind it, every tool
call the agent made, and what the request cost.

Configuration (environment variables):
  AGENT_MODEL     model spec (default anthropic:claude-opus-5-5; any spec agent.make_backend accepts)
  AGENT_DEFENSE   prompt-injection defenses (default "spotlight+confirm")
  AGENT_EFFORT    Claude effort level (default medium)
  APP_TOKEN       if set, /api/ask requires "Authorization: Bearer <APP_TOKEN>"
  ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY / XAI_API_KEY

    uvicorn app:app --reload          # http://localhost:8000
"""

import os
import sys

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import agent as ag  # noqa: E402
import make_db  # noqa: E402

MODEL = os.environ.get("AGENT_MODEL", "anthropic:claude-opus-5-5")
DEFENSE = os.environ.get("AGENT_DEFENSE", "spotlight+confirm")
EFFORT = os.environ.get("AGENT_EFFORT", "medium")
DB = os.path.join(HERE, "data", "shopco.db")

if not os.path.exists(DB):
    make_db.main(["--out_dir", os.path.dirname(DB)])

app = FastAPI(title="ShopCo SQL analyst agent")


class Ask(BaseModel):
    """Request body for /api/ask."""

    question: str = Field(min_length=3, max_length=1000)


@app.get("/", response_class=HTMLResponse)
def index():
    """The single-page UI."""
    with open(os.path.join(HERE, "static", "index.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/api/health")
def health():
    """Liveness check for the hosting platform."""
    return {"status": "ok", "model": MODEL, "defense": DEFENSE}


@app.post("/api/ask")
def ask(req: Ask, authorization: str | None = Header(default=None)):
    """Run the agent on one question (sync handler: FastAPI runs it in a worker thread)."""
    token = os.environ.get("APP_TOKEN")
    if token and authorization != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="missing or wrong bearer token")
    backend = ag.make_backend(MODEL, effort=EFFORT, fallbacks=True)
    res = ag.run_agent(backend, req.question, DB, DEFENSE)
    cost, uncached = ag.cost_usd(MODEL, res)
    return {
        "answer": res.answer,
        "sql": res.sql,
        "status": res.status,
        "error": res.error,
        "steps": [{**s, "result": s["result"][:1500]} for s in res.steps],
        "usage": {
            "llm_calls": res.llm_calls,
            "input_tokens": res.input_tokens,
            "output_tokens": res.output_tokens,
            "cache_read_tokens": res.cache_read_tokens,
            "cache_write_tokens": res.cache_write_tokens,
        },
        "cost_usd": cost,
        "cost_without_caching_usd": uncached,
        "latency_s": round(res.latency_s, 2),
        "model": MODEL,
        "served_by": res.served_by,
        "emails_blocked": len(res.emails_attempted) - len(res.emails_sent),
    }
