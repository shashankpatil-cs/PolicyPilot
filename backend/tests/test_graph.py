"""LangGraph workflow + research agent with a scripted OpenAI client and in-memory retriever."""
import asyncio
from types import SimpleNamespace

import pytest

from app.ai import tools
from app.ai.context import EvidenceContext
from app.ai.pipeline import run_pipeline
from app.core.config import settings

Q1 = "kept the policy repo rate unchanged at 6.50 per cent during the year"
Q3 = "kept the policy repo rate unchanged at 4.00 per cent for most of the year"


def answer(text, citations, sufficient=True):
    return {"sufficient": sufficient, "answer": text, "citations": citations}


GOOD = answer("The repo rate was unchanged at 6.50 per cent [S1].", [{"source_id": "S1", "quote": Q1}])
BAD = answer("The repo rate was cut sharply [S1].",
             [{"source_id": "S1", "quote": "The Reserve Bank cut the repo rate sharply during the year"}])


def test_happy_path_returns_verified_year_and_page(fake_llm, fake_retriever):
    fake_llm.answers = [GOOD]
    r = run_pipeline("What happened to the repo rate?", "standard", None)
    assert r["sufficient"] is True
    c = r["citations"][0]
    assert (c["report_year"], c["page_number"], c["verified"]) == ("2022-23", 10, True)
    assert r["meta"]["outcome"] == "answered" and r["meta"]["attempts"] == 1
    assert fake_llm.count("report_grounding") == 1


def test_only_question_and_chunks_reach_openai(fake_llm, fake_retriever):
    fake_llm.answers = [GOOD]
    run_pipeline("What happened to the repo rate?", "standard", None)
    answer_call = next(kw for names, kw in fake_llm.calls if "submit_answer" in names)
    user_msg = answer_call["messages"][1]["content"]
    assert "What happened to the repo rate?" in user_msg and "[S1] Report 2022-23, PDF page 10" in user_msg
    assert len(answer_call["messages"]) == 2  # system + user only: no chat history


def test_fabricated_quote_is_retried_with_feedback_then_accepted(fake_llm, fake_retriever):
    fake_llm.answers = [BAD, GOOD]
    r = run_pipeline("repo rate?", "standard", None)
    assert r["sufficient"] and r["meta"]["attempts"] == 2
    retry_prompt = [kw for names, kw in fake_llm.calls if "submit_answer" in names][1]["messages"][1]["content"]
    assert "failed automatic verification" in retry_prompt


def test_persistently_bad_answer_becomes_insufficient_not_shown(fake_llm, fake_retriever):
    fake_llm.answers = [BAD]
    r = run_pipeline("repo rate?", "standard", None)
    assert r["sufficient"] is False and r["citations"] == []
    assert r["answer"] == settings.INSUFFICIENT_EVIDENCE_MESSAGE
    assert r["meta"]["outcome"] == "validation_failed" and "cut" not in r["answer"]


def test_model_abstention_hides_model_text_from_user(fake_llm, fake_retriever):
    fake_llm.answers = [answer("Evidence silent; but I believe it was 9 per cent.", [], sufficient=False)]
    r = run_pipeline("something else", "standard", None)
    assert r["sufficient"] is False and r["answer"] == settings.INSUFFICIENT_EVIDENCE_MESSAGE
    assert r["meta"]["outcome"] == "model_abstained" and "9 per cent" in r["meta"]["note"]
    assert fake_llm.count("report_grounding") == 0


def test_no_evidence_never_calls_the_answer_model(fake_llm, fake_retriever, monkeypatch):
    monkeypatch.setattr(fake_retriever, "corpus", [])
    fake_llm.answers = [GOOD]
    r = run_pipeline("anything", "standard", None)
    assert r["sufficient"] is False and r["meta"]["outcome"] == "no_evidence"
    assert fake_llm.count("submit_answer") == 0


def test_no_indexed_reports(fake_llm, monkeypatch):
    monkeypatch.setattr("app.ai.tools.available_report_years", lambda: [])
    r = run_pipeline("anything", "standard", None)
    assert r["meta"]["outcome"] == "no_reports" and not fake_llm.calls


def test_llm_grounding_failure_blocks_answer(fake_llm, fake_retriever):
    fake_llm.answers = [GOOD]
    fake_llm.grounding = {"grounded": False, "unsupported_claims": ["stance claim"]}
    r = run_pipeline("repo rate?", "standard", None)
    assert r["sufficient"] is False and r["meta"]["outcome"] == "validation_failed"
    assert fake_llm.count("submit_answer") == settings.MAX_GENERATION_ATTEMPTS


def test_user_year_restriction_is_enforced_over_planner(fake_llm, fake_retriever):
    fake_llm.planner_calls = [("search_reports", {"query": "repo", "years": ["2022-23"]})]
    fake_llm.answers = [answer("Unchanged at 4.00 per cent [S1].", [{"source_id": "S1", "quote": Q3}])]
    run_pipeline("repo rate?", "standard", ["2021-22"])
    assert all(c["years"] == [] or c["years"] == ["2021-22"] for c in fake_retriever.calls)


def test_compare_years_pulls_balanced_evidence(fake_llm, fake_retriever):
    fake_llm.planner_calls = [("compare_years", {"query": "repo rate", "years": ["2021-22", "2022-23"]})]
    # ids follow retrieval order: 2021-22 is searched first -> S1; 2022-23 -> S2 (repo) and S3 (inflation)
    fake_llm.answers = [answer(
        "Repo rate: 4.00 per cent [S1] versus 6.50 per cent [S2].",
        [{"source_id": "S1", "quote": Q3}, {"source_id": "S2", "quote": Q1}],
    )]
    r = run_pipeline("Compare the repo rate", "standard", None)
    assert r["sufficient"]
    assert {c["report_year"] for c in r["citations"]} == {"2021-22", "2022-23"}
    assert {c["years"][0] for c in fake_retriever.calls if c["years"]} == {"2021-22", "2022-23"}


def test_planner_with_invalid_years_falls_back_to_plain_search(fake_llm, fake_retriever):
    fake_llm.planner_calls = [("search_reports", {"query": "repo", "years": ["1850-51"]})]
    fake_llm.answers = [GOOD]
    assert run_pipeline("repo rate?", "standard", None)["sufficient"]


def test_research_agent_verifies_agent_output(fake_retriever, monkeypatch, fake_llm):
    """Agent runs (stubbed Runner) -> its citations are re-verified server-side."""
    from app.ai import research_agent as ra

    def make_runner(output):
        async def run(agent, prompt, context, max_turns):
            tools.search_reports(context, "repo rate")  # what the agent's tool would do
            return SimpleNamespace(
                context_wrapper=SimpleNamespace(usage=SimpleNamespace(requests=2, input_tokens=100, output_tokens=20)),
                new_items=[],
                final_output_as=lambda cls: output,
            )
        return run

    good = ra.ResearchOutput(sufficient=True, answer=GOOD["answer"], citations=[ra.CitationIn(**GOOD["citations"][0])])
    monkeypatch.setattr(ra.Runner, "run", staticmethod(make_runner(good)))
    monkeypatch.setattr(ra, "_configure_sdk", lambda: None)
    fake_llm.answers = [GOOD]
    r = run_pipeline("repo rate?", "research", None)
    assert r["sufficient"] and r["citations"][0]["verified"] and r["meta"]["mode"] == "research"

    fake = ra.ResearchOutput(sufficient=True, answer=BAD["answer"], citations=[ra.CitationIn(**BAD["citations"][0])])
    monkeypatch.setattr(ra.Runner, "run", staticmethod(make_runner(fake)))
    r = run_pipeline("repo rate?", "research", None)
    assert r["sufficient"] is False and r["meta"]["outcome"] == "validation_failed"


def test_quote_attributed_to_the_wrong_source_is_rejected(fake_llm, fake_retriever):
    """A real quote, but cited against a different chunk than it came from."""
    fake_llm.planner_calls = [("compare_years", {"query": "repo rate", "years": ["2021-22", "2022-23"]})]
    swapped = answer(
        "Repo rate: 4.00 per cent [S3] versus 6.50 per cent [S1].",
        [{"source_id": "S3", "quote": Q3}, {"source_id": "S1", "quote": Q1}],
    )
    fake_llm.answers = [swapped]
    r = run_pipeline("Compare the repo rate", "standard", None)
    assert r["sufficient"] is False and r["meta"]["outcome"] == "validation_failed"
