import pytest
from langchain_core.documents import Document

from app.ai.context import EvidenceContext, extract_markers, verify_citations
from app.core.config import settings

TEXT = (
    "The Reserve Bank kept the policy repo rate unchanged at 6.50 per cent during the year, "
    "while retaining its stance of with-\ndrawal of accommodation. Inflation averaged 5,400 basis points."
)


@pytest.fixture
def ctx():
    c = EvidenceContext(available_years=["2022-23"])
    c.add_documents([Document(page_content=TEXT, metadata={
        "chunk_id": 1, "report_id": 9, "report_title": "T", "report_year": "2022-23",
        "page_number": 10, "page_label": "xii", "score": 0.9})])
    return c


GOOD = {"source_id": "S1", "quote": "kept the policy repo rate unchanged at 6.50 per cent during the year"}


def test_valid_citation_gets_year_and_page_from_database(ctx):
    r = verify_citations(ctx, "The rate was unchanged at 6.50 per cent [S1].", [GOOD])
    assert r.ok
    c = r.citations[0]
    assert (c["report_year"], c["page_number"], c["page_label"], c["verified"]) == ("2022-23", 10, "xii", True)


def test_model_cannot_choose_year_or_page(ctx):
    forged = {**GOOD, "report_year": "1999-00", "page_number": 999}
    c = verify_citations(ctx, "Unchanged [S1].", [forged]).citations[0]
    assert (c["report_year"], c["page_number"]) == ("2022-23", 10)


def test_fabricated_quote_rejected(ctx):
    bad = {"source_id": "S1", "quote": "The Reserve Bank cut the repo rate by one hundred basis points"}
    r = verify_citations(ctx, "Cut [S1].", [bad])
    assert not r.ok and not r.citations


def test_unknown_source_id_rejected(ctx):
    r = verify_citations(ctx, "Something [S7].", [{"source_id": "S7", "quote": GOOD["quote"]}])
    assert not r.ok and "does not exist" in r.problems[0]


def test_marker_without_quote_rejected(ctx):
    assert not verify_citations(ctx, "Claim [S1].", []).ok


def test_no_markers_rejected(ctx):
    assert not verify_citations(ctx, "Uncited claim.", [GOOD]).ok


def test_short_quote_rejected(ctx):
    assert not verify_citations(ctx, "x [S1].", [{"source_id": "S1", "quote": "repo rate"}]).ok


def test_pdf_hyphenation_noise_tolerated(ctx):
    q = {"source_id": "S1", "quote": "retaining its stance of withdrawal of accommodation"}
    assert verify_citations(ctx, "Stance [S1].", [q]).ok


def test_markers_grouped_and_deduped():
    assert extract_markers("a [S1][S3] b [S1, S2] c [S10]") == ["S1", "S3", "S2", "S10"]


def test_numeric_strict_flags_unsupported_number(ctx, monkeypatch):
    monkeypatch.setattr(settings, "NUMERIC_CHECK", "strict")
    r = verify_citations(ctx, "The repo rate was 7.25 per cent [S1].", [GOOD])
    assert not r.ok and "7.25" in r.unsupported_numbers


def test_numeric_accepts_supported_numbers_with_commas(ctx, monkeypatch):
    monkeypatch.setattr(settings, "NUMERIC_CHECK", "strict")
    assert verify_citations(ctx, "Unchanged at 6.50 per cent [S1].", [GOOD]).ok
    q = {"source_id": "S1", "quote": "Inflation averaged 5,400 basis points."}
    assert verify_citations(ctx, "About 5400 basis points [S1].", [q]).ok


def test_numeric_warn_reports_but_passes(ctx, monkeypatch):
    monkeypatch.setattr(settings, "NUMERIC_CHECK", "warn")
    r = verify_citations(ctx, "The rate was 7.25 per cent [S1].", [GOOD])
    assert r.ok and r.unsupported_numbers == ["7.25"]


def test_effective_years_restrictions():
    c = EvidenceContext(allowed_years=["2022-23"], available_years=["2021-22", "2022-23"])
    assert c.effective_years(None) == ["2022-23"]
    assert c.effective_years(["2021-22"]) == []          # outside the user's restriction
    assert c.effective_years(["2022-23", "1900"]) == ["2022-23"]
    assert EvidenceContext(available_years=["2022-23"]).effective_years(None) is None
