"""LangGraph workflow: plan (function calling) -> retrieve (tools) -> generate -> validate -> finalize.

    START -> analyze -> retrieve -+-> generate -> validate -+-> finalize -> END
                                  |              ^         |
                                  |              +---------+ (retry with feedback)
                                  +-> insufficient -> END  <- abstain / validation failed

Only the question and the retrieved chunks are ever sent to OpenAI.
"""
from __future__ import annotations

import logging
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.ai import tools
from app.ai.context import EvidenceContext, format_evidence
from app.ai.llm import LLMError, call_tools
from app.ai.prompts import render
from app.ai.validation import answered, insufficient, validate_answer
from app.core.config import settings

log = logging.getLogger(__name__)


class State(TypedDict, total=False):
    question: str
    plan: list[dict]
    draft: dict
    feedback: list[str]
    attempts: int
    validation: dict
    final: dict


def build_graph(ctx: EvidenceContext):
    # ------------------------------------------------------------ nodes
    def analyze(state: State) -> dict:
        constraint = (
            f"The user restricted the search to these report years only: {', '.join(ctx.allowed_years)}."
            if ctx.allowed_years
            else ""
        )
        system = render(
            "planner",
            available_years=", ".join(ctx.available_years),
            year_constraint=constraint,
            max_calls=settings.MAX_PLANNED_TOOL_CALLS,
        )
        try:
            calls = call_tools(
                ctx=ctx,
                model=settings.OPENAI_PLANNER_MODEL,
                system=system,
                user=state["question"],
                tools=tools.retrieval_tool_schemas(ctx.available_years),
            )
            plan = [{"name": c.name, "arguments": c.arguments} for c in calls[: settings.MAX_PLANNED_TOOL_CALLS]]
        except LLMError as exc:
            log.warning("Planner failed (%s); falling back to a plain search", exc)
            plan = [{"name": tools.SEARCH, "arguments": {"query": state["question"]}}]
        return {"plan": plan}

    def retrieve(state: State) -> dict:
        errors: list[str] = []
        for call in state["plan"]:
            try:
                tools.execute_retrieval_tool(ctx, call["name"], call["arguments"])
            except tools.ToolError as exc:
                errors.append(str(exc))
        if not ctx.all():
            # Planner produced unusable arguments: fall back to a plain search of the raw question.
            try:
                tools.search_reports(ctx, state["question"])
            except tools.ToolError as exc:
                errors.append(str(exc))
        ctx.trim_to(settings.MAX_EVIDENCE_CHUNKS)
        plan = [dict(c) for c in state["plan"]]
        if errors:
            plan.append({"name": "_errors", "arguments": {"errors": errors}})
        return {"plan": plan}

    def generate(state: State) -> dict:
        feedback = ""
        if state.get("feedback"):
            feedback = render("feedback_block", problems="\n".join(f"- {p}" for p in state["feedback"]))
        calls = call_tools(
            ctx=ctx,
            model=settings.OPENAI_CHAT_MODEL,
            system=render("answer_system", min_quote_chars=settings.MIN_QUOTE_CHARS),
            user=render(
                "answer_user",
                question=state["question"],
                evidence=format_evidence(ctx.all()),
                feedback=feedback,
            ),
            tools=[tools.SUBMIT_ANSWER_SCHEMA],
            force=tools.SUBMIT_ANSWER,
        )
        args = calls[0].arguments
        draft = {
            "sufficient": bool(args.get("sufficient")),
            "answer": str(args.get("answer", "")).strip(),
            "citations": [c for c in (args.get("citations") or []) if isinstance(c, dict)],
        }
        return {"draft": draft, "attempts": state.get("attempts", 0) + 1}

    def validate(state: State) -> dict:
        draft = state["draft"]
        outcome = validate_answer(ctx, draft["answer"], draft["citations"])
        return {"validation": outcome, "feedback": outcome["problems"]}

    def finalize(state: State) -> dict:
        return {"final": answered(state["draft"]["answer"], state["validation"]["citations"])}

    def abstain(state: State) -> dict:
        draft = state.get("draft")
        if not ctx.all():
            reason, note = "no_evidence", None
        elif draft and not draft["sufficient"]:
            reason, note = "model_abstained", draft["answer"]
        else:
            reason, note = "validation_failed", "; ".join(state.get("feedback") or [])
        return {"final": insufficient(reason, note)}

    # ---------------------------------------------------------- routing
    def after_retrieve(state: State) -> str:
        return "generate" if ctx.all() else "abstain"

    def after_generate(state: State) -> str:
        return "validate" if state["draft"]["sufficient"] else "abstain"

    def after_validate(state: State) -> str:
        if not state["validation"]["problems"]:
            return "finalize"
        if state.get("attempts", 0) < settings.MAX_GENERATION_ATTEMPTS:
            return "generate"
        return "abstain"

    graph = StateGraph(State)
    graph.add_node("analyze", analyze)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("validate", validate)
    graph.add_node("finalize", finalize)
    graph.add_node("abstain", abstain)

    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "retrieve")
    graph.add_conditional_edges("retrieve", after_retrieve, {"generate": "generate", "abstain": "abstain"})
    graph.add_conditional_edges("generate", after_generate, {"validate": "validate", "abstain": "abstain"})
    graph.add_conditional_edges(
        "validate", after_validate, {"finalize": "finalize", "generate": "generate", "abstain": "abstain"}
    )
    graph.add_edge("finalize", END)
    graph.add_edge("abstain", END)
    return graph.compile()


def run_standard(question: str, ctx: EvidenceContext) -> dict:
    """Runs the workflow; returns {"final": ..., "state": ...}."""
    graph = build_graph(ctx)
    state = graph.invoke({"question": question, "attempts": 0}, config={"run_name": "rbi_standard_workflow"})
    return {
        "final": state["final"],
        "plan": state.get("plan", []),
        "attempts": state.get("attempts", 0),
        "validation": state.get("validation"),
    }
