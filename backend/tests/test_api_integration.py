"""End-to-end API tests against a real throwaway Postgres + pgvector.

BGE-M3 and OpenAI are replaced by a hashed bag-of-words embedder and a scripted LLM;
everything else (SQL, pgvector, FTS, indexing thread, auth, routes) is real.
"""
import time
import uuid

import pymupdf
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text

from app.core.config import settings
from app.core.ratelimit import _failures
from app.db.models import Chunk, Report, User
from app.db.session import SessionLocal
from app.main import app
from tests.test_pdf import PAGES

pytestmark = pytest.mark.integration

Q = "repo rate was held steady while liquidity conditions were managed carefully"


def make_pdf(path, pages=PAGES):
    doc = pymupdf.open()
    for t in pages:
        doc.new_page().insert_textbox(pymupdf.Rect(72, 72, 520, 700), t, fontsize=11)
    doc.save(path)
    doc.close()


def uname(prefix):
    """Unique username per call so tests can be re-run against the same database."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        with SessionLocal() as db:  # clean slate: leftovers from an aborted earlier run
            for report in db.scalars(select(Report)):
                db.delete(report)
            db.commit()
        yield c


@pytest.fixture(autouse=True)
def _clean_throttle():
    _failures.clear()


def auth(token):
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="module")
def admin_token(client):
    r = client.post("/api/auth/admin/login", json={"username": settings.ADMIN_USERNAME, "password": settings.ADMIN_PASSWORD})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def register(client, name):
    r = client.post("/api/auth/register", json={"username": name, "password": "correct-horse-1"})
    assert r.status_code == 201, r.text
    return r.json()["access_token"]


def wait_indexed(client, token, report_id, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        rep = next(r for r in client.get("/api/admin/reports", headers=auth(token)).json() if r["id"] == report_id)
        if rep["status"] in ("indexed", "failed"):
            return rep
        time.sleep(0.3)
    raise AssertionError("indexing timed out")


# ----------------------------------------------------------------- auth
def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_admin_password_stored_only_as_bcrypt_hash(client):
    with SessionLocal() as db:
        admin = db.scalar(select(User).where(User.username == settings.ADMIN_USERNAME.lower()))
    assert admin.role == "admin"
    assert admin.password_hash.startswith("$2") and settings.ADMIN_PASSWORD not in admin.password_hash


def test_roles_and_access_control(client, admin_token):
    alice = uname("alice")
    user_token = register(client, alice)
    assert client.get("/api/admin/reports").status_code == 401
    assert client.get("/api/admin/reports", headers=auth(user_token)).status_code == 403
    assert client.get("/api/admin/reports", headers=auth(admin_token)).status_code == 200
    # a normal user cannot use the admin login endpoint
    r = client.post("/api/auth/admin/login", json={"username": alice, "password": "correct-horse-1"})
    assert r.status_code == 401
    assert client.post("/api/auth/login", json={"username": alice, "password": "nope"}).status_code == 401
    assert client.get("/api/chat/conversations").status_code == 401
    assert client.get("/api/auth/me", headers=auth("garbage")).status_code == 401


def test_registration_validation(client):
    assert client.post("/api/auth/register", json={"username": "ab", "password": "correct-horse-1"}).status_code == 422
    assert client.post("/api/auth/register", json={"username": "bobby_x", "password": "short"}).status_code == 422
    assert client.post("/api/auth/register", json={"username": "bobby_x", "password": "x" * 73}).status_code == 422
    dupe = uname("dupe")
    register(client, dupe)
    assert client.post("/api/auth/register", json={"username": dupe.upper(), "password": "correct-horse-1"}).status_code == 409
    # the admin name cannot be taken or used to self-escalate
    assert client.post("/api/auth/register", json={"username": settings.ADMIN_USERNAME, "password": "correct-horse-1"}).status_code == 409


def test_login_throttle(client):
    for _ in range(settings.LOGIN_MAX_ATTEMPTS):
        assert client.post("/api/auth/login", json={"username": "ghost_user", "password": "x"}).status_code == 401
    assert client.post("/api/auth/login", json={"username": "ghost_user", "password": "x"}).status_code == 429


# ------------------------------------------------- upload, index, retrieve
def test_full_flow(client, admin_token, fake_embeddings, fake_llm, tmp_path):
    h = auth(admin_token)
    pdf = tmp_path / "ar.pdf"
    make_pdf(pdf)

    # non-PDF rejected, bad year rejected
    bad = client.post("/api/admin/reports", headers=h, data={"title": "x", "report_year": "2023-24"},
                      files={"file": ("x.pdf", b"not a pdf at all", "application/pdf")})
    assert bad.status_code == 415
    badyear = client.post("/api/admin/reports", headers=h, data={"title": "x", "report_year": "twenty"},
                          files={"file": ("x.pdf", pdf.read_bytes(), "application/pdf")})
    assert badyear.status_code == 422

    # upload -> background indexing
    up = client.post("/api/admin/reports", headers=h, data={"title": "RBI Annual Report 2023-24", "report_year": "2023-24"},
                     files={"file": ("ar.pdf", pdf.read_bytes(), "application/pdf")})
    assert up.status_code == 201, up.text
    rid = up.json()["id"]
    rep = wait_indexed(client, admin_token, rid)
    assert rep["status"] == "indexed", rep
    assert rep["page_count"] == 3 and rep["chunks_total"] == rep["chunks_done"] > 0

    # duplicate content rejected
    dup = client.post("/api/admin/reports", headers=h, data={"title": "again", "report_year": "2023-24"},
                      files={"file": ("ar.pdf", pdf.read_bytes(), "application/pdf")})
    assert dup.status_code == 409

    # chunks in the DB carry exact page numbers + vectors
    with SessionLocal() as db:
        pages = {p for (p,) in db.execute(select(Chunk.page_number).where(Chunk.report_id == rid))}
        assert pages == {1, 2, 3}
        dim = db.scalar(text("SELECT vector_dims(embedding) FROM chunks WHERE report_id = :r LIMIT 1"), {"r": rid})
        assert dim == settings.EMBEDDING_DIM
        idx = {r[0] for r in db.execute(text("SELECT indexname FROM pg_indexes WHERE tablename='chunks'"))}
        assert {"ix_chunks_embedding_hnsw", "ix_chunks_fts"} <= idx

    # users see only indexed years; unknown year filter rejected
    token = register(client, uname("carol"))
    years = client.get("/api/chat/reports", headers=auth(token)).json()
    assert [y["report_year"] for y in years] == ["2023-24"]
    r = client.post("/api/chat/ask", headers=auth(token), json={"question": "repo rate?", "years": ["1999-00"]})
    assert r.status_code == 422

    # standard mode: retrieval is real, LLM scripted; citation is verified against the real chunk
    fake_llm.answers = [{"sufficient": True,
                         "answer": "The repo rate was held steady [S1].",
                         "citations": [{"source_id": "S1", "quote": Q}]}]
    ask = client.post("/api/chat/ask", headers=auth(token),
                      json={"question": "What happened to the repo rate and liquidity?", "mode": "standard"})
    assert ask.status_code == 200, ask.text
    body = ask.json()
    msg = body["message"]
    assert msg["sufficient"] is True
    cite = msg["citations"][0]
    assert (cite["report_year"], cite["page_number"], cite["verified"]) == ("2023-24", 1, True)
    assert "run_id" not in msg["meta"] and "note" not in msg["meta"]
    assert msg["meta"]["retrieved"] and msg["meta"]["retrieved"][0]["report_year"] == "2023-24"
    conv_id = body["conversation"]["id"]

    # follow-up in same conversation; history persisted in order
    fake_llm.answers = [{"sufficient": False, "answer": "n/a", "citations": []}]
    ask2 = client.post("/api/chat/ask", headers=auth(token),
                       json={"question": "What is a good butter chicken recipe?", "conversation_id": conv_id})
    assert ask2.json()["message"]["sufficient"] is False
    detail = client.get(f"/api/chat/conversations/{conv_id}", headers=auth(token)).json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant", "user", "assistant"]
    assert detail["messages"][3]["content"] == settings.INSUFFICIENT_EVIDENCE_MESSAGE

    # unrelated question never reached the answer model's evidence (below similarity floor / no FTS hit)
    assert ask2.json()["message"]["meta"]["outcome"] in ("no_evidence", "model_abstained")

    # history is private to the owner
    other = register(client, uname("dave"))
    assert client.get(f"/api/chat/conversations/{conv_id}", headers=auth(other)).status_code == 404
    assert client.delete(f"/api/chat/conversations/{conv_id}", headers=auth(other)).status_code == 404
    assert client.post("/api/chat/ask", headers=auth(other), json={"question": "hi", "conversation_id": conv_id}).status_code == 404

    # feedback
    mid = msg["id"]
    assert client.post(f"/api/chat/messages/{mid}/feedback", headers=auth(token), json={"rating": 1}).json()["feedback"] == 1
    assert client.post(f"/api/chat/messages/{mid}/feedback", headers=auth(other), json={"rating": 1}).status_code == 404
    convs = client.get("/api/chat/conversations", headers=auth(token)).json()
    assert [c["id"] for c in convs] == [conv_id]

    # empty / oversize questions
    assert client.post("/api/chat/ask", headers=auth(token), json={"question": "   "}).status_code == 422
    assert client.post("/api/chat/ask", headers=auth(token),
                       json={"question": "x" * (settings.QUESTION_MAX_CHARS + 1)}).status_code == 422

    # reindex works and keeps data consistent
    assert client.post(f"/api/admin/reports/{rid}/reindex", headers=h).status_code == 200
    assert wait_indexed(client, admin_token, rid)["status"] == "indexed"
    with SessionLocal() as db:
        n = db.scalar(select(func.count()).select_from(Chunk).where(Chunk.report_id == rid))
        assert n == rep["chunks_total"]  # no duplicates after reindex

    stats = client.get("/api/admin/stats", headers=h).json()
    assert stats["reports_indexed"] >= 1 and stats["chunks"] >= n

    # delete cascades to chunks
    assert client.delete(f"/api/admin/reports/{rid}", headers=h).status_code == 204
    with SessionLocal() as db:
        assert db.scalar(select(func.count()).select_from(Chunk).where(Chunk.report_id == rid)) == 0
    assert client.delete(f"/api/admin/reports/{rid}", headers=h).status_code == 404


def test_year_filtered_hybrid_retrieval(client, admin_token, fake_embeddings, tmp_path):
    """Two reports; search must honour year filters, status filter and the similarity floor."""
    from app.rag.retriever import search

    h = auth(admin_token)
    ids = {}
    for year, pages in (("2020-21", ["Alpha year. Gross domestic product contracted sharply during the pandemic "
                                     "year as lockdowns disrupted economic activity across all major sectors."]),
                        ("2021-22", ["Beta year. Gross domestic product rebounded strongly as activity normalised "
                                     "and vaccination coverage widened across the country this financial year."])):
        pdf = tmp_path / f"{year}.pdf"
        make_pdf(pdf, pages)
        r = client.post("/api/admin/reports", headers=h, data={"title": f"AR {year}", "report_year": year},
                        files={"file": (pdf.name, pdf.read_bytes(), "application/pdf")})
        assert r.status_code == 201, r.text
        ids[year] = r.json()["id"]
    for rid in ids.values():
        assert wait_indexed(client, admin_token, rid)["status"] == "indexed"

    with SessionLocal() as db:
        both = search(db, "gross domestic product activity", None, 10)
        assert {x.report_year for x in both} == {"2020-21", "2021-22"}
        only = search(db, "gross domestic product activity", ["2021-22"], 10)
        assert only and {x.report_year for x in only} == {"2021-22"}
        assert search(db, "gross domestic product", [], 10) == []             # empty restriction -> nothing
        assert search(db, "butter chicken recipe", None, 10) == []            # below floor, no keyword hit
        assert all(x.page_number == 1 for x in both)

    # a report that is not 'indexed' is invisible to retrieval
    with SessionLocal() as db:
        db.execute(text("UPDATE reports SET status='failed' WHERE id=:i"), {"i": ids["2021-22"]})
        db.commit()
        assert {x.report_year for x in search(db, "gross domestic product activity", None, 10)} == {"2020-21"}

    for rid in ids.values():
        client.post(f"/api/admin/reports/{rid}/reindex", headers=h)  # failed one can be retried
    for rid in ids.values():
        assert wait_indexed(client, admin_token, rid)["status"] == "indexed"
    for rid in ids.values():
        assert client.delete(f"/api/admin/reports/{rid}", headers=h).status_code == 204


def test_unreadable_pdf_marks_report_failed_and_leaves_no_chunks(client, admin_token, fake_embeddings, tmp_path):
    h = auth(admin_token)
    pdf = tmp_path / "blank.pdf"
    doc = pymupdf.open()
    doc.new_page()  # no text -> "scanned" PDF
    doc.save(pdf)
    doc.close()
    r = client.post("/api/admin/reports", headers=h, data={"title": "Scanned", "report_year": "2019-20"},
                    files={"file": ("blank.pdf", pdf.read_bytes(), "application/pdf")})
    rid = r.json()["id"]
    rep = wait_indexed(client, admin_token, rid)
    assert rep["status"] == "failed" and "OCR" in rep["error"]
    with SessionLocal() as db:
        assert db.scalar(select(func.count()).select_from(Chunk).where(Chunk.report_id == rid)) == 0
    assert client.delete(f"/api/admin/reports/{rid}", headers=h).status_code == 204
