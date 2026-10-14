"""Per-request evidence store and deterministic citation verification.

The LLM never gets to decide what counts as a valid citation: a citation is
"verified" only if its source id exists in the retrieved evidence AND its quote
is actually found in that chunk's text. Report year and page always come from
the database row, never from the model.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from langchain_core.documents import Document
from rapidfuzz import fuzz

from app.ai.prompts import render
from app.core.config import settings

_MARKER_GROUP = re.compile(r"\[\s*(S\d+(?:\s*,\s*S\d+)*)\s*\]")
_SID = re.compile(r"S\d+")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


@dataclass
class Evidence:
    source_id: str
    chunk_id: int
    report_id: int
    report_title: str
    report_year: str
    page_number: int
    page_label: str | None
    text: str
    score: float

    def public(self, with_text: bool = True) -> dict:
        data = {
            "source_id": self.source_id,
            "report_year": self.report_year,
            "report_title": self.report_title,
            "page_number": self.page_number,
            "page_label": self.page_label,
            "relevance": self.score,
        }
        if with_text:
            data["text"] = self.text
        return data


class EvidenceContext:
    """Holds retrieved chunks (with stable ids S1, S2, ...) for one request."""

    def __init__(self, allowed_years: list[str] | None = None, available_years: list[str] | None = None):
        self.allowed_years = allowed_years or None  # user-selected restriction
        self.available_years = available_years or []
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0}
        self._by_chunk: dict[int, Evidence] = {}
        self._by_sid: dict[str, Evidence] = {}

    # ---- years --------------------------------------------------------
    def effective_years(self, requested: list[str] | None) -> list[str] | None:
        """None = no restriction. [] = nothing allowed."""
        if requested:
            pool = [y for y in requested if y in self.available_years]
        else:
            pool = None
        if self.allowed_years:
            if pool is None:
                return list(self.allowed_years)
            return [y for y in pool if y in self.allowed_years]
        return pool

    # ---- evidence -----------------------------------------------------
    def add_documents(self, docs: list[Document]) -> list[Evidence]:
        added: list[Evidence] = []
        for doc in docs:
            m = doc.metadata
            existing = self._by_chunk.get(m["chunk_id"])
            if existing is None:
                existing = Evidence(
                    source_id=f"S{len(self._by_chunk) + 1}",
                    chunk_id=m["chunk_id"],
                    report_id=m["report_id"],
                    report_title=m["report_title"],
                    report_year=m["report_year"],
                    page_number=m["page_number"],
                    page_label=m.get("page_label"),
                    text=doc.page_content,
                    score=m.get("score", 0.0),
                )
                self._by_chunk[existing.chunk_id] = existing
                self._by_sid[existing.source_id] = existing
            added.append(existing)
        return added

    def get(self, source_id: str) -> Evidence | None:
        return self._by_sid.get(source_id)

    def all(self) -> list[Evidence]:
        return list(self._by_sid.values())

    def trim_to(self, limit: int) -> None:
        """Keep only the best `limit` chunks (by relevance) but keep their ids stable."""
        if len(self._by_sid) <= limit:
            return
        keep = sorted(self._by_sid.values(), key=lambda e: e.score, reverse=True)[:limit]
        keep_ids = {e.source_id for e in keep}
        self._by_sid = {sid: e for sid, e in self._by_sid.items() if sid in keep_ids}
        self._by_chunk = {e.chunk_id: e for e in self._by_sid.values()}

    def add_usage(self, usage) -> None:
        if usage is None:
            return
        self.usage["llm_calls"] += 1
        self.usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        self.usage["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0


def format_evidence(items: list[Evidence]) -> str:
    return "\n\n".join(
        render(
            "evidence_block",
            source_id=e.source_id,
            report_year=e.report_year,
            page_number=e.page_number,
            text=e.text,
        )
        for e in items
    )


# ----------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------
_TRANSLATE = str.maketrans(
    {
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", "−": "-", "­": "",
        "•": " ", " ": " ",
    }
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_TRANSLATE).lower()
    text = re.sub(r"-\s*\n\s*", "", text)
    return re.sub(r"\s+", " ", text).strip()


def quote_in_text(quote: str, text: str) -> bool:
    q, t = normalize(quote), normalize(text)
    if not q:
        return False
    if q in t:
        return True
    q_no_punct = re.sub(r"[^\w\s]", "", q)
    t_no_punct = re.sub(r"[^\w\s]", "", t)
    if q_no_punct and q_no_punct in t_no_punct:
        return True
    parts = [p.strip() for p in re.split(r"\.{2,}|\…|\[\.+\]", q) if len(p.strip()) >= 12]
    if len(parts) > 1 and all(
        p in t or re.sub(r"[^\w\s]", "", p) in t_no_punct or fuzz.partial_ratio(p, t) >= settings.QUOTE_FUZZY_THRESHOLD
        for p in parts
    ):
        return True
    return (
        fuzz.partial_ratio(q, t) >= settings.QUOTE_FUZZY_THRESHOLD
        or fuzz.partial_ratio(q_no_punct, t_no_punct) >= settings.QUOTE_FUZZY_THRESHOLD
    )


def extract_markers(answer: str) -> list[str]:
    seen: list[str] = []
    for group in _MARKER_GROUP.findall(answer):
        for sid in _SID.findall(group):
            if sid not in seen:
                seen.append(sid)
    return seen


def _numbers(text: str) -> set[str]:
    out = set()
    for raw in _NUMBER.findall(text):
        cleaned = raw.replace(",", "").rstrip(".")
        if len(cleaned.replace(".", "")) >= 2:  # ignore list numbers / single digits
            out.add(cleaned)
    return out


@dataclass
class VerificationResult:
    ok: bool
    citations: list[dict] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "verified_citations": self.citations,
            "problems": self.problems,
            "unsupported_numbers": self.unsupported_numbers,
        }


def verify_citations(ctx: EvidenceContext, answer: str, citations: list[dict]) -> VerificationResult:
    problems: list[str] = []
    markers = extract_markers(answer)
    if not markers:
        problems.append("The answer has no inline citation markers like [S1].")

    by_sid: dict[str, list[str]] = {}
    for c in citations:
        by_sid.setdefault(str(c.get("source_id", "")).strip(), []).append(str(c.get("quote", "")))

    verified: list[dict] = []
    for sid in markers:
        evidence = ctx.get(sid)
        if evidence is None:
            problems.append(f"[{sid}] does not exist in the evidence.")
            continue
        quotes = by_sid.get(sid)
        if not quotes:
            problems.append(f"[{sid}] is used in the answer but has no supporting quote in citations.")
            continue
        good_quote = None
        target_ev = evidence
        for quote in quotes:
            if len(quote.strip()) < settings.MIN_QUOTE_CHARS:
                continue
            if quote_in_text(quote, evidence.text):
                good_quote = quote.strip()
                break

        # Check if the quote is in another evidence block retrieved in this context
        if good_quote is None:
            for quote in quotes:
                if len(quote.strip()) < settings.MIN_QUOTE_CHARS:
                    continue
                for other_ev in ctx.all():
                    if other_ev.source_id != sid and quote_in_text(quote, other_ev.text):
                        good_quote = quote.strip()
                        target_ev = other_ev
                        break
                if good_quote is not None:
                    break

        if good_quote is None:
            problems.append(
                f"The quote given for [{sid}] is missing, too short (<{settings.MIN_QUOTE_CHARS} chars) "
                f"or not found verbatim in that evidence block."
            )
            continue
        verified.append(
            {
                "source_id": sid,
                "report_id": target_ev.report_id,
                "chunk_id": target_ev.chunk_id,
                "report_title": target_ev.report_title,
                "report_year": target_ev.report_year,
                "page_number": target_ev.page_number,
                "page_label": target_ev.page_label,
                "quote": good_quote,
                "verified": True,
            }
        )

    unsupported: list[str] = []
    if settings.NUMERIC_CHECK != "off" and verified:
        cited_text = " ".join(
            normalize(ctx.get(c["source_id"]).text).replace(",", "") for c in verified  # type: ignore[union-attr]
        )
        stripped = _MARKER_GROUP.sub(" ", answer)
        for number in sorted(_numbers(stripped)):
            if number not in cited_text:
                unsupported.append(number)
        if unsupported and settings.NUMERIC_CHECK == "strict":
            problems.append(
                "These numbers do not appear in the cited evidence: "
                + ", ".join(unsupported)
                + ". Use only figures present in the evidence."
            )

    return VerificationResult(ok=not problems, citations=verified, problems=problems, unsupported_numbers=unsupported)
