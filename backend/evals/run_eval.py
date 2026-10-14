"""LangSmith evaluation of the full pipeline.

    cd backend
    python -m evals.run_eval

Dataset rows (JSONL, path from EVAL_DATASET_PATH):
  question            required
  years               optional list of report years to restrict to
  expect_answerable   true -> must answer with citations, false -> must abstain
  expected_keywords   optional list; fraction found in the answer is scored
  expected_years      optional list; fraction of these years cited is scored

Edit dataset.example.jsonl (or point EVAL_DATASET_PATH elsewhere) so the
questions match the reports you have actually indexed.
"""
from __future__ import annotations

import json
import os
import sys

from app.core.config import settings

# Evaluation always needs LangSmith, regardless of LANGSMITH_TRACING.
if not settings.LANGSMITH_API_KEY:
    sys.exit("LANGSMITH_API_KEY must be set in .env to run evaluations.")
os.environ.update(
    {
        "LANGSMITH_TRACING": "true",
        "LANGSMITH_API_KEY": settings.LANGSMITH_API_KEY,
        "LANGSMITH_PROJECT": settings.LANGSMITH_PROJECT,
        "LANGSMITH_ENDPOINT": settings.LANGSMITH_ENDPOINT,
    }
)

from langsmith import Client  # noqa: E402
from langsmith.evaluation import evaluate  # noqa: E402

from app.ai.pipeline import run_pipeline  # noqa: E402
from app.ai.tracing import setup_tracing  # noqa: E402


def load_rows() -> list[dict]:
    path = settings.eval_dataset_path
    if not path.exists():
        sys.exit(f"Eval dataset not found: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        sys.exit(f"Eval dataset is empty: {path}")
    return rows


def ensure_dataset(client: Client) -> None:
    name = settings.EVAL_DATASET_NAME
    if client.has_dataset(dataset_name=name):
        print(f"Using existing LangSmith dataset '{name}' (delete it in LangSmith to re-upload the file).")
        return
    rows = load_rows()
    dataset = client.create_dataset(name, description="RBI Intelligence AI evaluation questions")
    client.create_examples(
        inputs=[{"question": r["question"], "years": r.get("years")} for r in rows],
        outputs=[
            {
                "expect_answerable": bool(r.get("expect_answerable", True)),
                "expected_keywords": r.get("expected_keywords", []),
                "expected_years": r.get("expected_years", []),
            }
            for r in rows
        ],
        dataset_id=dataset.id,
    )
    print(f"Created dataset '{name}' with {len(rows)} examples.")


def target(inputs: dict) -> dict:
    return run_pipeline(inputs["question"], settings.EVAL_MODE, inputs.get("years") or None)


# ---------------------------------------------------------------- evaluators
def abstention_correct(outputs: dict, reference_outputs: dict) -> dict:
    expected = bool(reference_outputs.get("expect_answerable", True))
    return {"key": "abstention_correct", "score": int(bool(outputs["sufficient"]) == expected)}


def verified_citations(outputs: dict) -> dict:
    if not outputs["sufficient"]:
        return {"key": "verified_citations", "score": None, "comment": "abstained"}
    citations = outputs.get("citations") or []
    ok = bool(citations) and all(c.get("verified") for c in citations)
    return {"key": "verified_citations", "score": int(ok)}


def keyword_coverage(outputs: dict, reference_outputs: dict) -> dict:
    keywords = reference_outputs.get("expected_keywords") or []
    if not keywords or not outputs["sufficient"]:
        return {"key": "keyword_coverage", "score": None, "comment": "not applicable"}
    text = outputs["answer"].lower()
    return {"key": "keyword_coverage", "score": sum(k.lower() in text for k in keywords) / len(keywords)}


def year_coverage(outputs: dict, reference_outputs: dict) -> dict:
    years = reference_outputs.get("expected_years") or []
    if not years or not outputs["sufficient"]:
        return {"key": "year_coverage", "score": None, "comment": "not applicable"}
    cited = {c.get("report_year") for c in outputs.get("citations") or []}
    return {"key": "year_coverage", "score": sum(y in cited for y in years) / len(years)}


def numbers_supported(outputs: dict) -> dict:
    if not outputs["sufficient"]:
        return {"key": "numbers_supported", "score": None, "comment": "abstained"}
    unsupported = outputs["meta"].get("unsupported_numbers") or []
    return {
        "key": "numbers_supported",
        "score": int(not unsupported),
        "comment": ", ".join(unsupported) if unsupported else "",
    }


def latency_ok(outputs: dict) -> dict:
    return {"key": "latency_seconds", "score": outputs["meta"]["latency_ms"] / 1000}


def main() -> None:
    setup_tracing()
    client = Client()
    ensure_dataset(client)
    results = evaluate(
        target,
        data=settings.EVAL_DATASET_NAME,
        evaluators=[
            abstention_correct,
            verified_citations,
            keyword_coverage,
            year_coverage,
            numbers_supported,
            latency_ok,
        ],
        experiment_prefix=settings.EVAL_EXPERIMENT_PREFIX,
        max_concurrency=settings.EVAL_MAX_CONCURRENCY,
        metadata={
            "mode": settings.EVAL_MODE,
            "answer_model": settings.OPENAI_CHAT_MODEL,
            "agent_model": settings.OPENAI_AGENT_MODEL,
            "embedding_model": settings.EMBEDDING_MODEL_NAME,
            "top_k": settings.RETRIEVAL_TOP_K,
            "hybrid": settings.HYBRID_SEARCH,
        },
    )
    print("Evaluation finished. Open the experiment in LangSmith to review results:")
    print(getattr(results, "experiment_name", "(see LangSmith)"))


if __name__ == "__main__":
    main()
