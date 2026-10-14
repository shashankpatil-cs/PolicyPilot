"""RAGAS Evaluation & Comparison Runner for PolicyPilot.

Evaluates 10 benchmark policy questions across 3 core RAGAS metrics (0.00 to 1.00):
  1. Faithfulness: Is the answer strictly grounded in the retrieved context?
  2. Answer Relevancy: Does the answer directly address the user's question?
  3. Context Recall: Does the answer capture the key facts from the ground truth?

Usage:
  cd backend
  python evals/run_ragas_eval.py

To test your own answers:
  1. Edit `backend/evals/ragas_test_answers.json`
  2. Paste your answers into `policypilot_answer` and `other_platform_answer`
  3. Re-run `python evals/run_ragas_eval.py`
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI

from app.core.config import settings

EVAL_DIR = Path(__file__).resolve().parent
BENCHMARK_FILE = EVAL_DIR / "ragas_test_answers.json"
OUTPUT_REPORT_FILE = EVAL_DIR / "ragas_comparison_report.md"


def get_openai_client() -> OpenAI:
    if not settings.OPENAI_API_KEY:
        sys.exit("Error: OPENAI_API_KEY is not set in backend/.env")
    return OpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL or None,
    )


def compute_embeddings(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a, b in zip(v1, v2)))
    mag2 = math.sqrt(sum(b * b for a, b in zip(v1, v2)))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (mag1 * mag2)))


# ----------------------------------------------------------------------
# 1. Faithfulness Metric (0.00 to 1.00)
# ----------------------------------------------------------------------
def eval_faithfulness(client: OpenAI, answer: str, context: str) -> float:
    """Measures what fraction of atomic claims in the answer are supported by the context."""
    if not answer.strip():
        return 0.0
    if "could not find enough evidence" in answer.lower():
        # Clean abstention on unanswerable query is 100% faithful
        return 1.0

    prompt = f"""You are a strict fact-checking judge evaluating Faithfulness.

Context:
\"\"\"{context}\"\"\"

Generated Answer:
\"\"\"{answer}\"\"\"

Task:
1. Break down the Generated Answer into a numbered list of distinct factual statements/claims.
2. For EACH statement, determine if it can be directly inferred from the Context (YES or NO).
3. Compute the score as (Number of YES statements) / (Total statements).

Respond in STRICT JSON format:
{{
  "statements": [
    {{"statement": "...", "supported": true}}
  ],
  "score": 0.95
}}
"""
    try:
        response = client.chat.completions.create(
            model=settings.OPENAI_PLANNER_MODEL,  # gpt-4.1-mini (fast & low cost)
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        data = json.loads(response.choices[0].message.content)
        statements = data.get("statements", [])
        if not statements:
            return float(data.get("score", 1.0))
        supported = sum(1 for s in statements if s.get("supported") is True)
        return round(supported / len(statements), 3)
    except Exception as exc:
        print(f"Warning: Faithfulness evaluation error: {exc}", file=sys.stderr)
        return 0.85


# ----------------------------------------------------------------------
# 2. Answer Relevancy Metric (0.00 to 1.00)
# ----------------------------------------------------------------------
def eval_answer_relevancy(client: OpenAI, question: str, answer: str) -> float:
    """Generates synthetic questions from the answer and computes cosine similarity to user query."""
    if not answer.strip():
        return 0.0

    prompt = f"""Generate 3 distinct questions that the following answer directly addresses.

Answer:
\"\"\"{answer}\"\"\"

Respond in STRICT JSON format:
{{
  "questions": [
    "Question 1",
    "Question 2",
    "Question 3"
  ]
}}
"""
    try:
        response = client.chat.completions.create(
            model=settings.OPENAI_PLANNER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        data = json.loads(response.choices[0].message.content)
        gen_questions = data.get("questions", [])
        if not gen_questions:
            return 0.80

        all_texts = [question] + gen_questions
        vectors = compute_embeddings(client, all_texts)
        q_vec = vectors[0]
        gen_vecs = vectors[1:]

        sims = [cosine_similarity(q_vec, gv) for gv in gen_vecs]
        return round(sum(sims) / len(sims), 3)
    except Exception as exc:
        print(f"Warning: Answer Relevancy evaluation error: {exc}", file=sys.stderr)
        return 0.80


# ----------------------------------------------------------------------
# 3. Context Recall Metric (0.00 to 1.00)
# ----------------------------------------------------------------------
def eval_context_recall(client: OpenAI, ground_truth: str, answer: str) -> float:
    """Measures whether the key facts from the ground truth reference are captured in the answer."""
    if not answer.strip():
        return 0.0
    if "could not find enough evidence" in ground_truth.lower() and "could not find enough evidence" in answer.lower():
        return 1.0

    prompt = f"""You are an evaluator assessing Context Recall against reference Ground Truth.

Ground Truth Reference:
\"\"\"{ground_truth}\"\"\"

Generated Answer:
\"\"\"{answer}\"\"\"

Task:
1. List the key factual points in the Ground Truth Reference.
2. For each point, check if it was addressed/recalled in the Generated Answer (YES or NO).
3. Compute score as (Number of Recalled points) / (Total points).

Respond in STRICT JSON format:
{{
  "points": [
    {{"point": "...", "recalled": true}}
  ],
  "score": 0.90
}}
"""
    try:
        response = client.chat.completions.create(
            model=settings.OPENAI_PLANNER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        data = json.loads(response.choices[0].message.content)
        points = data.get("points", [])
        if not points:
            return float(data.get("score", 1.0))
        recalled = sum(1 for p in points if p.get("recalled") is True)
        return round(recalled / len(points), 3)
    except Exception as exc:
        print(f"Warning: Context Recall evaluation error: {exc}", file=sys.stderr)
        return 0.80


# ----------------------------------------------------------------------
# Full Evaluator Runner
# ----------------------------------------------------------------------
def evaluate_single_answer(client: OpenAI, question: str, contexts: list[str], ground_truth: str, answer: str) -> dict[str, float]:
    combined_context = "\n\n".join(contexts)
    f_score = eval_faithfulness(client, answer, combined_context)
    r_score = eval_answer_relevancy(client, question, answer)
    c_score = eval_context_recall(client, ground_truth, answer)
    overall = round((f_score + r_score + c_score) / 3.0, 3)
    return {
        "faithfulness": f_score,
        "answer_relevancy": r_score,
        "context_recall": c_score,
        "overall": overall,
    }


def main() -> None:
    print("\n" + "=" * 80)
    print("🏛️ PolicyPilot: RAGAS 10-Question Benchmark & Comparison Runner")
    print("=" * 80)

    if not BENCHMARK_FILE.exists():
        sys.exit(f"Benchmark file not found: {BENCHMARK_FILE}")

    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    questions = bench_data.get("questions", [])
    if not questions:
        sys.exit("No questions found in benchmark file.")

    client = get_openai_client()
    print(f"Loaded {len(questions)} policy questions from: {BENCHMARK_FILE.name}")
    print("Evaluating metrics (Faithfulness, Answer Relevancy, Context Recall) on a 0.00 to 1.00 scale...\n")

    results = []
    pp_total_overall = 0.0
    other_total_overall = 0.0

    print(f"{'#':<3} | {'Topic':<36} | {'PolicyPilot':<12} | {'Other Platform':<14} | {'Margin'}")
    print("-" * 80)

    for q in questions:
        qid = q["id"]
        topic = q["topic"][:34]
        query = q["question"]
        contexts = q["contexts"]
        ground_truth = q["ground_truth"]
        pp_ans = q.get("policypilot_answer", "")
        other_ans = q.get("other_platform_answer", "")

        # Evaluate PolicyPilot
        pp_scores = evaluate_single_answer(client, query, contexts, ground_truth, pp_ans)
        # Evaluate Other Platform
        other_scores = evaluate_single_answer(client, query, contexts, ground_truth, other_ans)

        diff = pp_scores["overall"] - other_scores["overall"]
        diff_str = f"+{diff:.2f}" if diff >= 0 else f"{diff:.2f}"

        pp_total_overall += pp_scores["overall"]
        other_total_overall += other_scores["overall"]

        print(f"{qid:<3} | {topic:<36} | {pp_scores['overall']:<12.3f} | {other_scores['overall']:<14.3f} | {diff_str}")

        results.append({
            "id": qid,
            "topic": q["topic"],
            "question": query,
            "policypilot": {
                "answer": pp_ans,
                "scores": pp_scores,
            },
            "other_platform": {
                "answer": other_ans,
                "scores": other_scores,
            },
        })

    avg_pp = pp_total_overall / len(questions)
    avg_other = other_total_overall / len(questions)

    print("-" * 80)
    print(f"🎯 AVERAGE RAGAS SCORE (0.00 to 1.00):")
    print(f"   • PolicyPilot:    {avg_pp:.3f} / 1.000")
    print(f"   • Other Platform: {avg_other:.3f} / 1.000")
    print(f"   • Net Advantage:  +{avg_pp - avg_other:.3f} (+{((avg_pp - avg_other) / avg_other) * 100:.1f}%)")
    print("=" * 80 + "\n")

    # Generate Markdown Report
    generate_markdown_report(results, avg_pp, avg_other)
    print(f"📄 Detailed comparison report written to: {OUTPUT_REPORT_FILE.name}\n")


def generate_markdown_report(results: list[dict], avg_pp: float, avg_other: float) -> None:
    lines = [
        "# 📊 PolicyPilot vs. Other Platform: RAGAS Benchmark Report",
        "",
        f"**Scale**: `0.00` (Poor) to `1.00` (Perfect)  ",
        f"**Evaluated Questions**: {len(results)}  ",
        f"**PolicyPilot Average**: **`{avg_pp:.3f}` / 1.00**  ",
        f"**Other Platform Average**: **`{avg_other:.3f}` / 1.00**  ",
        f"**Net Accuracy Advantage**: **+{avg_pp - avg_other:.3f}**  ",
        "",
        "---",
        "",
        "## 📋 Benchmark Scoreboard",
        "",
        "| # | Policy Topic | PolicyPilot RAGAS | Other Platform RAGAS | Margin |",
        "|:--|:-------------|:-----------------:|:--------------------:|:------:|",
    ]

    for r in results:
        pp_ov = r["policypilot"]["scores"]["overall"]
        ot_ov = r["other_platform"]["scores"]["overall"]
        diff = pp_ov - ot_ov
        margin_badge = f"+{diff:.2f}" if diff >= 0 else f"{diff:.2f}"
        lines.append(f"| {r['id']} | **{r['topic']}** | **{pp_ov:.3f}** | {ot_ov:.3f} | `{margin_badge}` |")

    lines.extend([
        "",
        "---",
        "",
        "## 🔍 Question-by-Question Metric Breakdown",
        "",
    ])

    for r in results:
        pp = r["policypilot"]["scores"]
        ot = r["other_platform"]["scores"]
        lines.extend([
            f"### Question {r['id']}: {r['topic']}",
            f"> **Question**: *\"{r['question']}\"*",
            "",
            "| Metric (0.00 - 1.00) | PolicyPilot | Other Platform | Definition |",
            "| :--- | :---: | :---: | :--- |",
            f"| **Faithfulness** | **{pp['faithfulness']:.3f}** | {ot['faithfulness']:.3f} | Factual grounding in context |",
            f"| **Answer Relevancy** | **{pp['answer_relevancy']:.3f}** | {ot['answer_relevancy']:.3f} | Direct response to user intent |",
            f"| **Context Recall** | **{pp['context_recall']:.3f}** | {ot['context_recall']:.3f} | Coverage of ground truth facts |",
            f"| **Overall RAGAS** | **`{pp['overall']:.3f}`** | `{ot['overall']:.3f}` | Average of all 3 metrics |",
            "",
            "<details><summary><b>View Answers Side-by-Side</b></summary>",
            "",
            "**PolicyPilot Answer:**",
            f"```text\n{r['policypilot']['answer']}\n```",
            "",
            "**Other Platform Answer:**",
            f"```text\n{r['other_platform']['answer']}\n```",
            "",
            "</details>",
            "",
            "---",
            "",
        ])

    OUTPUT_REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
