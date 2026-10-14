"""Answer validation shared by the LangGraph workflow and the research agent.

Layer 1 (deterministic): every [S#] marker must map to retrieved evidence and
carry a quote that is actually present in that chunk.
Layer 2 (optional LLM judge): every claim must be supported by the cited chunks.
"""
from __future__ import annotations

import logging

from app.ai import tools
from app.ai.context import EvidenceContext, format_evidence, verify_citations
from app.ai.llm import LLMError, call_tools
from app.ai.prompts import render
from app.core.config import settings

log = logging.getLogger(__name__)


def validate_answer(ctx: EvidenceContext, answer: str, citations: list[dict]) -> dict:
    result = verify_citations(ctx, answer, citations)
    problems = list(result.problems)
    grounding: dict | None = None

    if result.ok and settings.VALIDATION_LLM_CHECK:
        cited = [ctx.get(c["source_id"]) for c in result.citations]
        try:
            calls = call_tools(
                ctx=ctx,
                model=settings.OPENAI_VALIDATOR_MODEL,
                system=render("validator"),
                user=render("validator_user", answer=answer, evidence=format_evidence([e for e in cited if e])),
                tools=[tools.REPORT_GROUNDING_SCHEMA],
                force=tools.REPORT_GROUNDING,
            )
            args = calls[0].arguments
            unsupported = [str(c) for c in (args.get("unsupported_claims") or [])]
            grounding = {"grounded": bool(args.get("grounded")), "unsupported_claims": unsupported}
            if not grounding["grounded"]:
                detail = "; ".join(unsupported) if unsupported else "some claims are not supported by the cited evidence"
                problems.append(f"Unsupported claims: {detail}. Remove them or support them with cited evidence.")
        except LLMError as exc:
            log.warning("Grounding check failed: %s", exc)
            grounding = {"grounded": False, "error": "validator_error"}
            problems.append("The grounding check could not be completed.")

    return {
        "ok": not problems,
        "problems": problems,
        "citations": result.citations,
        "unsupported_numbers": result.unsupported_numbers,
        "grounding": grounding,
    }


def answered(answer: str, citations: list[dict]) -> dict:
    return {"answer": answer, "sufficient": True, "citations": citations, "outcome": "answered"}


def insufficient(reason: str, note: str | None = None) -> dict:
    return {
        "answer": settings.INSUFFICIENT_EVIDENCE_MESSAGE,
        "sufficient": False,
        "citations": [],
        "outcome": reason,
        "note": note,
    }
