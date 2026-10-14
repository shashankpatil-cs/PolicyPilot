"""Function-calling tools shared by the LangGraph workflow and the OpenAI Agents SDK agent.

Tools:
  search_reports       semantic + keyword search over indexed reports
  compare_years        balanced retrieval for the same topic across report years
  list_available_reports
  verify_citations     deterministic citation check against retrieved evidence
Plus two "structured output" functions used by the LangGraph nodes:
  submit_answer, report_grounding
"""
from __future__ import annotations

import json
from typing import Any

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

from app.ai.context import EvidenceContext, verify_citations as _verify
from app.core.config import settings
from app.db.session import SessionLocal
from app.rag.retriever import PgVectorRetriever, list_indexed_reports

SEARCH = "search_reports"
COMPARE = "compare_years"
LIST_REPORTS = "list_available_reports"
VERIFY = "verify_citations"
SUBMIT_ANSWER = "submit_answer"
REPORT_GROUNDING = "report_grounding"


class ToolError(ValueError):
    """Bad arguments from the model; the message is safe to show back to it."""


# ----------------------------------------------------------------------
# Implementations
# ----------------------------------------------------------------------
def available_report_years() -> list[str]:
    with SessionLocal() as db:
        seen: list[str] = []
        for r in list_indexed_reports(db):
            if r["report_year"] not in seen:
                seen.append(r["report_year"])
        return seen


def list_available_reports() -> list[dict]:
    with SessionLocal() as db:
        return list_indexed_reports(db)


@traceable(name="tool_search_reports", run_type="tool")
def search_reports(ctx: EvidenceContext, query: str, years: list[str] | None = None, top_k: int | None = None) -> list[dict]:
    query = (query or "").strip()
    if not query:
        raise ToolError("query must not be empty")
    effective = ctx.effective_years(years)
    if effective is not None and not effective:
        raise ToolError(f"None of the requested years are available. Available years: {ctx.available_years}")
    k = max(1, min(int(top_k or settings.RETRIEVAL_TOP_K), settings.MAX_EVIDENCE_CHUNKS))
    docs = PgVectorRetriever(years=effective, top_k=k).invoke(query)
    return [e.public() for e in ctx.add_documents(docs)]


@traceable(name="tool_compare_years", run_type="tool")
def compare_years(ctx: EvidenceContext, query: str, years: list[str], top_k_per_year: int | None = None) -> dict[str, list[dict]]:
    query = (query or "").strip()
    if not query:
        raise ToolError("query must not be empty")
    effective = ctx.effective_years(years) or []
    if len(effective) < 2:
        raise ToolError(f"compare_years needs at least two available years. Available years: {ctx.available_years}")
    k = max(1, min(int(top_k_per_year or settings.COMPARE_TOP_K_PER_YEAR), settings.MAX_EVIDENCE_CHUNKS))
    out: dict[str, list[dict]] = {}
    for year in effective:
        docs = PgVectorRetriever(years=[year], top_k=k).invoke(query)
        out[year] = [e.public() for e in ctx.add_documents(docs)]
    return out


@traceable(name="tool_verify_citations", run_type="tool")
def verify_citations(ctx: EvidenceContext, answer: str, citations: list[dict]) -> dict:
    return _verify(ctx, answer, citations).as_dict()


# ----------------------------------------------------------------------
# JSON schemas for OpenAI function calling
# ----------------------------------------------------------------------
def retrieval_tool_schemas(available_years: list[str]) -> list[dict]:
    year_item: dict[str, Any] = {"type": "string"}
    if available_years:
        year_item["enum"] = available_years
    return [
        {
            "name": SEARCH,
            "description": "Search the indexed RBI Annual Reports for passages relevant to a query. "
            "Optionally restrict to specific report years.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Focused retrieval query."},
                    "years": {
                        "type": "array",
                        "items": year_item,
                        "description": "Optional report years to restrict the search to.",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": COMPARE,
            "description": "Retrieve balanced evidence for the same topic from each of several report years, "
            "for comparisons, changes and trends.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Focused retrieval query."},
                    "years": {"type": "array", "items": year_item, "minItems": 2},
                },
                "required": ["query", "years"],
            },
        },
    ]


SUBMIT_ANSWER_SCHEMA = {
    "name": SUBMIT_ANSWER,
    "description": "Submit the final grounded answer with verbatim supporting quotes.",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "sufficient": {
                "type": "boolean",
                "description": "False if the evidence cannot answer the main question.",
            },
            "answer": {"type": "string", "description": "Answer text with inline [S#] markers."},
            "citations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_id": {"type": "string", "description": "Evidence id such as S1."},
                        "quote": {"type": "string", "description": "Verbatim contiguous excerpt from that evidence."},
                    },
                    "required": ["source_id", "quote"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["sufficient", "answer", "citations"],
        "additionalProperties": False,
    },
}

REPORT_GROUNDING_SCHEMA = {
    "name": REPORT_GROUNDING,
    "description": "Report whether every claim in the answer is supported by the cited evidence.",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "grounded": {"type": "boolean"},
            "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["grounded", "unsupported_claims"],
        "additionalProperties": False,
    },
}


# ----------------------------------------------------------------------
# Dispatcher (used by the LangGraph retrieve node)
# ----------------------------------------------------------------------
def execute_retrieval_tool(ctx: EvidenceContext, name: str, args: dict) -> Any:
    if name == SEARCH:
        return search_reports(ctx, args.get("query", ""), args.get("years"))
    if name == COMPARE:
        return compare_years(ctx, args.get("query", ""), args.get("years") or [])
    raise ToolError(f"Unknown retrieval tool: {name}")


def to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)
