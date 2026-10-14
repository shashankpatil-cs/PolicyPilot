"""Test doubles: scripted OpenAI client, in-memory retriever, hashed bag-of-words embeddings."""
import hashlib
import json
import math
from types import SimpleNamespace

from app.core.config import settings


def hash_embed(text: str) -> list[float]:
    vec = [0.0] * settings.EMBEDDING_DIM
    for token in text.lower().split():
        token = "".join(ch for ch in token if ch.isalnum())
        if len(token) < 3:
            continue
        idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % settings.EMBEDDING_DIM
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class FakeLLM:
    """Looks like OpenAI(); behaviour is chosen by which tool the code asked for."""

    def __init__(self):
        self.answers: list[dict] = []        # consumed in order; the last one repeats
        self.grounding = {"grounded": True, "unsupported_claims": []}
        self.planner_calls: list[tuple[str, dict]] | None = None  # None -> plain search of the question
        self.calls: list[tuple[list[str], dict]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    @staticmethod
    def _response(calls):
        tool_calls = [
            SimpleNamespace(function=SimpleNamespace(name=n, arguments=json.dumps(a))) for n, a in calls
        ]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=tool_calls))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )

    def count(self, tool: str) -> int:
        return sum(1 for names, _ in self.calls if tool in names)

    def _create(self, **kw):
        names = [t["function"]["name"] for t in kw["tools"]]
        self.calls.append((names, kw))
        if "submit_answer" in names:
            args = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
            return self._response([("submit_answer", args)])
        if "report_grounding" in names:
            return self._response([("report_grounding", self.grounding)])
        if self.planner_calls is not None:
            return self._response(self.planner_calls)
        return self._response([("search_reports", {"query": kw["messages"][1]["content"]})])


CORPUS = [
    # (chunk_id, year, page, text)
    (1, "2022-23", 10, "The Reserve Bank kept the policy repo rate unchanged at 6.50 per cent during the year, "
                       "while retaining its stance of withdrawal of accommodation."),
    (2, "2022-23", 22, "Consumer price index inflation averaged 6.7 per cent in 2022-23, above the upper "
                       "tolerance band for several months."),
    (3, "2021-22", 14, "The Reserve Bank kept the policy repo rate unchanged at 4.00 per cent for most of the "
                       "year before the off-cycle increase in May."),
]


class FakeRetriever:
    YEARS = ["2021-22", "2022-23"]
    calls: list[dict] = []
    corpus = CORPUS

    def __init__(self, years=None, top_k=0):
        self.years, self.top_k = years, top_k

    def invoke(self, query):
        FakeRetriever.calls.append({"query": query, "years": self.years})
        docs = []
        for cid, year, page, text in self.corpus:
            if self.years and year not in self.years:
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "chunk_id": cid, "report_id": 1, "report_title": f"RBI Annual Report {year}",
                        "report_year": year, "page_number": page, "page_label": None, "score": 0.8,
                    },
                )
            )
        return docs
