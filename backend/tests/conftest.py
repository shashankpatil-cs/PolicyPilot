"""Test configuration.

Unit tests need only backend/.env. Integration tests (marked `integration`) need a
THROWAWAY Postgres database with pgvector and RBI_TEST_DB_NAME set to its name
(create it with:  DB_NAME=<name> python scripts/setup_db.py).
They refuse to run against the database named in .env.
"""
import os
import tempfile

_test_db = os.environ.get("RBI_TEST_DB_NAME")
if _test_db:
    os.environ["DB_NAME"] = _test_db
os.environ["UPLOAD_DIR"] = tempfile.mkdtemp(prefix="rbi_test_uploads_")

import pytest  # noqa: E402

from app.core.config import settings  # noqa: E402
from tests.fakes import FakeLLM, FakeRetriever, hash_embed  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: needs a throwaway Postgres (RBI_TEST_DB_NAME)")


def pytest_collection_modifyitems(config, items):
    if _test_db:
        return
    skip = pytest.mark.skip(reason="set RBI_TEST_DB_NAME to run integration tests")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def fake_llm(monkeypatch):
    llm = FakeLLM()
    monkeypatch.setattr("app.ai.llm.get_client", lambda: llm)
    return llm


@pytest.fixture
def fake_retriever(monkeypatch):
    FakeRetriever.calls = []
    monkeypatch.setattr("app.ai.tools.PgVectorRetriever", FakeRetriever)
    monkeypatch.setattr("app.ai.tools.available_report_years", lambda: FakeRetriever.YEARS)
    return FakeRetriever


@pytest.fixture
def fake_embeddings(monkeypatch):
    monkeypatch.setattr("app.rag.retriever.embed_query", hash_embed)
    monkeypatch.setattr("app.rag.indexer.embed_documents", lambda texts: [hash_embed(t) for t in texts])
    monkeypatch.setattr(settings, "MIN_SIMILARITY", 0.05)  # hashed bag-of-words gives low cosine values
