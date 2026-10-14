"""Single entry point used by the API and the evaluation script."""
from __future__ import annotations

import asyncio
import time
import uuid

try:
    from langsmith import traceable
    from langsmith.run_helpers import get_current_run_tree
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

    def get_current_run_tree():
        return None

from app.ai import tools
from app.ai.context import EvidenceContext
from app.ai.validation import insufficient
from app.core.config import settings

MODE_STANDARD = "standard"
MODE_RESEARCH = "research"


@traceable(name="rbi_ask", run_type="chain")
def run_pipeline(question: str, mode: str = MODE_STANDARD, years: list[str] | None = None) -> dict:
    started = time.perf_counter()
    run_tree = get_current_run_tree()
    run_id = str(run_tree.id) if run_tree is not None else uuid.uuid4().hex

    available = tools.available_report_years()
    ctx = EvidenceContext(allowed_years=years, available_years=available)

    extra: dict = {}
    if not available:
        outcome = {"final": insufficient("no_reports"), "attempts": 0}
    elif mode == MODE_RESEARCH:
        from app.ai.research_agent import run_research  # lazy: only needed in research mode

        outcome = asyncio.run(run_research(question, ctx))
        extra["steps"] = outcome.get("steps", [])
    else:
        from app.ai.graph import run_standard

        outcome = run_standard(question, ctx)
        extra["plan"] = outcome.get("plan", [])

    final = outcome["final"]
    validation = outcome.get("validation") or {}
    meta = {
        "mode": mode,
        "outcome": final["outcome"],
        "attempts": outcome.get("attempts", 0),
        "retrieved": [e.public(with_text=False) for e in ctx.all()],
        "retrieved_texts": [e.text for e in ctx.all()],
        "unsupported_numbers": validation.get("unsupported_numbers", []),
        "grounding": validation.get("grounding"),
        "usage": ctx.usage,
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "run_id": run_id,
        "models": {
            "answer": settings.OPENAI_AGENT_MODEL if mode == MODE_RESEARCH else settings.OPENAI_CHAT_MODEL,
            "embedding": settings.EMBEDDING_MODEL_NAME,
        },
        **extra,
    }
    if final.get("note"):
        meta["note"] = final["note"]

    return {
        "answer": final["answer"],
        "sufficient": final["sufficient"],
        "citations": final["citations"],
        "meta": meta,
    }
