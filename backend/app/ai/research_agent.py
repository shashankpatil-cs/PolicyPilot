"""Research agent built on the OpenAI Agents SDK.

The agent decides which tools to call (search_reports, compare_years,
list_available_reports, verify_citations), possibly many times, then returns a
structured answer. Its output is NEVER trusted as-is: citations are re-verified
server-side against the evidence the tools actually returned.

NOTE: no `from __future__ import annotations` here - the Agents SDK builds tool
schemas from the real type hints.
"""
import logging
from typing import Optional

from agents import (
    Agent,
    ModelSettings,
    RunContextWrapper,
    Runner,
    function_tool,
    set_default_openai_client,
    set_default_openai_key,
)
from agents.exceptions import MaxTurnsExceeded
from agents.models.openai_provider import OpenAIProvider
from agents.run_config import RunConfig
from openai import AsyncOpenAI
from pydantic import BaseModel


from app.ai import tools
from app.ai.context import EvidenceContext
from app.ai.prompts import render
from app.ai.validation import answered, insufficient, validate_answer
from app.core.config import settings

log = logging.getLogger(__name__)

_configured = False


class CitationIn(BaseModel):
    source_id: str
    quote: str


class ResearchOutput(BaseModel):
    sufficient: bool
    answer: str
    citations: list[CitationIn]


@function_tool(name_override=tools.LIST_REPORTS)
def list_available_reports_tool(ctx: RunContextWrapper[EvidenceContext]) -> str:
    """List the RBI Annual Reports that are indexed (report year, title, page count)."""
    allowed = ctx.context.allowed_years
    reports = [r for r in tools.list_available_reports() if not allowed or r["report_year"] in allowed]
    return tools.to_json(reports)


@function_tool(name_override=tools.SEARCH)
def search_reports_tool(
    ctx: RunContextWrapper[EvidenceContext], query: str, years: Optional[list[str]] = None
) -> str:
    """Search the indexed RBI Annual Reports for passages relevant to a query.
    Returns passages with ids (S1, S2, ...), report year and page number.

    Args:
        query: A focused retrieval query with key terms, names and indicators.
        years: Optional report years (e.g. "2022-23") to restrict the search to.
    """
    try:
        return tools.to_json(tools.search_reports(ctx.context, query, years))
    except tools.ToolError as exc:
        return tools.to_json({"error": str(exc)})


@function_tool(name_override=tools.COMPARE)
def compare_years_tool(ctx: RunContextWrapper[EvidenceContext], query: str, years: list[str]) -> str:
    """Retrieve balanced evidence on the same topic from each of several report years.
    Use for comparisons, year-on-year changes and trends.

    Args:
        query: A focused retrieval query about the topic to compare.
        years: Two or more report years to compare (e.g. "2021-22", "2022-23").
    """
    try:
        return tools.to_json(tools.compare_years(ctx.context, query, years))
    except tools.ToolError as exc:
        return tools.to_json({"error": str(exc)})


@function_tool(name_override=tools.VERIFY)
def verify_citations_tool(
    ctx: RunContextWrapper[EvidenceContext], answer: str, citations: list[CitationIn]
) -> str:
    """Check a draft answer and its citations against the evidence retrieved so far.
    Reports problems such as unknown ids or quotes that are not verbatim in the evidence.

    Args:
        answer: Draft answer text with inline [S#] markers.
        citations: One entry per marker: source_id and a verbatim supporting quote.
    """
    result = tools.verify_citations(ctx.context, answer, [c.model_dump() for c in citations])
    # Do not echo full citation objects back; the model only needs the problems.
    return tools.to_json({"ok": result["ok"], "problems": result["problems"]})


def _configure_sdk() -> None:
    global _configured
    if _configured:
        return
    if settings.OPENAI_BASE_URL:
        from openai import AsyncOpenAI

        set_default_openai_client(
            AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL,
                timeout=settings.OPENAI_TIMEOUT_SECONDS,
                max_retries=settings.OPENAI_MAX_RETRIES,
            )
        )
    else:
        set_default_openai_key(settings.OPENAI_API_KEY)
    _configured = True


def build_agent() -> Agent[EvidenceContext]:
    model_settings = (
        ModelSettings(temperature=settings.openai_temperature)
        if settings.openai_temperature is not None
        else ModelSettings()
    )
    return Agent[EvidenceContext](
        name="RBI Research Agent",
        instructions=render("research_agent", min_quote_chars=settings.MIN_QUOTE_CHARS),
        model=settings.OPENAI_AGENT_MODEL,
        model_settings=model_settings,
        tools=[list_available_reports_tool, search_reports_tool, compare_years_tool, verify_citations_tool],
        output_type=ResearchOutput,
    )


def _tool_trace(result) -> list[dict]:
    trace: list[dict] = []
    try:
        for item in result.new_items:
            raw = getattr(item, "raw_item", None)
            if item.__class__.__name__ == "ToolCallItem" and raw is not None:
                trace.append({"tool": getattr(raw, "name", "?"), "arguments": getattr(raw, "arguments", "")})
    except Exception:
        log.debug("Could not extract tool trace", exc_info=True)
    return trace


async def run_research(question: str, ctx: EvidenceContext) -> dict:
    _configure_sdk()
    client = AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL or None,
        timeout=settings.OPENAI_TIMEOUT_SECONDS,
        max_retries=settings.OPENAI_MAX_RETRIES,
    )
    provider = OpenAIProvider(openai_client=client)
    run_config = RunConfig(model_provider=provider)

    agent = build_agent()
    feedback: list[str] = []
    steps: list[dict] = []
    last_validation: dict | None = None

    try:
        for attempt in range(1, settings.AGENT_MAX_REPAIR_ATTEMPTS + 1):
            prompt = question
            if ctx.allowed_years:
                prompt += f"\n\n(Only use these report years: {', '.join(ctx.allowed_years)}.)"
            if feedback:
                prompt += (
                    "\n\nYour previous final answer failed automatic verification:\n"
                    + "\n".join(f"- {p}" for p in feedback)
                    + "\nEvidence ids already retrieved keep the same ids if you search again. "
                    "Fix the problems and answer again using only retrieved evidence."
                )
            try:
                result = await Runner.run(
                    agent,
                    prompt,
                    context=ctx,
                    max_turns=settings.AGENT_MAX_TURNS,
                    run_config=run_config,
                )
            except MaxTurnsExceeded:
                return {
                    "final": insufficient("agent_max_turns"),
                    "attempts": attempt,
                    "steps": steps,
                    "validation": last_validation,
                }

            usage = getattr(result.context_wrapper, "usage", None)
            if usage is not None:
                ctx.usage["llm_calls"] += getattr(usage, "requests", 0) or 0
                ctx.usage["prompt_tokens"] += getattr(usage, "input_tokens", 0) or 0
                ctx.usage["completion_tokens"] += getattr(usage, "output_tokens", 0) or 0
            steps.extend(_tool_trace(result))

            output: ResearchOutput = result.final_output_as(ResearchOutput)
            if not output.sufficient:
                return {
                    "final": insufficient("model_abstained", output.answer),
                    "attempts": attempt,
                    "steps": steps,
                    "validation": last_validation,
                }

            last_validation = validate_answer(ctx, output.answer.strip(), [c.model_dump() for c in output.citations])
            if last_validation["ok"]:
                return {
                    "final": answered(output.answer.strip(), last_validation["citations"]),
                    "attempts": attempt,
                    "steps": steps,
                    "validation": last_validation,
                }
            feedback = last_validation["problems"]

        return {
            "final": insufficient("validation_failed", "; ".join(feedback)),
            "attempts": settings.AGENT_MAX_REPAIR_ATTEMPTS,
            "steps": steps,
            "validation": last_validation,
        }
    finally:
        await client.close()

