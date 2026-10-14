"""pgvector retrieval: dense (BGE-M3) + optional keyword (Postgres FTS), fused with RRF.

Exposed as a LangChain `BaseRetriever` so retrieval shows up as a retriever
run in LangSmith traces.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import STATUS_INDEXED, Chunk, Report
from app.db.session import SessionLocal
from app.rag.embeddings import embed_query

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]{3,}")


@dataclass
class Hit:
    chunk_id: int
    report_id: int
    report_title: str
    report_year: str
    page_number: int
    page_label: str | None
    content: str
    similarity: float
    fused: float = 0.0


def _tune_session(db: Session) -> None:
    """Best-effort pgvector HNSW settings; older pgvector versions lack some GUCs."""
    settings_sql = [f"SET LOCAL hnsw.ef_search = {int(settings.HNSW_EF_SEARCH)}"]
    settings_sql.append("SET LOCAL hnsw.iterative_scan = relaxed_order")
    for stmt in settings_sql:
        try:
            with db.begin_nested():
                db.execute(text(stmt))
        except Exception:
            log.debug("Skipping unsupported setting: %s", stmt)


def _base_columns(sim_expr):
    return [
        Chunk.id,
        Chunk.report_id,
        Report.title,
        Chunk.report_year,
        Chunk.page_number,
        Chunk.page_label,
        Chunk.content,
        sim_expr.label("sim"),
    ]


def _row_to_hit(row) -> Hit:
    return Hit(
        chunk_id=row.id,
        report_id=row.report_id,
        report_title=row.title,
        report_year=row.report_year,
        page_number=row.page_number,
        page_label=row.page_label,
        content=row.content,
        similarity=float(row.sim),
    )


def _fts_expression(query: str) -> str:
    seen: list[str] = []
    for token in _TOKEN_RE.findall(query.lower()):
        if token not in seen:
            seen.append(token)
    return " | ".join(seen[: settings.FTS_MAX_TERMS])


def search(db: Session, query: str, years: list[str] | None, top_k: int) -> list[Hit]:
    if years is not None and not years:
        return []
    qvec = embed_query(query)
    sim = 1 - Chunk.embedding.cosine_distance(qvec)
    candidates = settings.RETRIEVAL_CANDIDATES
    _tune_session(db)

    dense_stmt = (
        select(*_base_columns(sim))
        .join(Report, Report.id == Chunk.report_id)
        .where(Report.status == STATUS_INDEXED)
        .order_by(Chunk.embedding.cosine_distance(qvec))
        .limit(candidates)
    )
    if years:
        dense_stmt = dense_stmt.where(Chunk.report_year.in_(years))
    dense = [_row_to_hit(r) for r in db.execute(dense_stmt)]

    keyword: list[Hit] = []
    expr = _fts_expression(query) if settings.HYBRID_SEARCH else ""
    if expr:
        cfg = settings.FTS_CONFIG
        tsq = func.to_tsquery(cfg, expr)
        tsv = func.to_tsvector(cfg, Chunk.content)
        fts_stmt = (
            select(*_base_columns(sim))
            .join(Report, Report.id == Chunk.report_id)
            .where(Report.status == STATUS_INDEXED, tsv.op("@@")(tsq))
            .order_by(func.ts_rank_cd(tsv, tsq).desc())
            .limit(candidates)
        )
        if years:
            fts_stmt = fts_stmt.where(Chunk.report_year.in_(years))
        keyword = [_row_to_hit(r) for r in db.execute(fts_stmt)]

    # Reciprocal Rank Fusion over the two ranked lists.
    fused: dict[int, Hit] = {}
    for ranked in (dense, keyword):
        for rank, hit in enumerate(ranked, start=1):
            entry = fused.setdefault(hit.chunk_id, hit)
            entry.fused += 1.0 / (settings.RRF_K + rank)

    kept = [h for h in fused.values() if h.similarity >= settings.MIN_SIMILARITY]
    kept.sort(key=lambda h: h.fused, reverse=True)
    return kept[:top_k]


def list_indexed_reports(db: Session) -> list[dict]:
    rows = db.execute(
        select(Report.id, Report.title, Report.report_year, Report.page_count)
        .where(Report.status == STATUS_INDEXED)
        .order_by(Report.sort_key, Report.id)
    )
    return [
        {"report_id": r.id, "title": r.title, "report_year": r.report_year, "pages": r.page_count}
        for r in rows
    ]


class PgVectorRetriever(BaseRetriever):
    """LangChain retriever over the pgvector `chunks` table."""

    years: list[str] | None = None
    top_k: int = 0

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        with SessionLocal() as db:
            hits = search(db, query, self.years, self.top_k or settings.RETRIEVAL_TOP_K)
        return [
            Document(
                page_content=h.content,
                metadata={
                    "chunk_id": h.chunk_id,
                    "report_id": h.report_id,
                    "report_title": h.report_title,
                    "report_year": h.report_year,
                    "page_number": h.page_number,
                    "page_label": h.page_label,
                    "score": round(h.similarity, 4),
                },
            )
            for h in hits
        ]
