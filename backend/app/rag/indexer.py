"""Background indexing: PDF -> pages -> chunks -> BGE-M3 vectors -> pgvector."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import delete, insert

from app.core.config import settings
from app.db.models import (
    STATUS_FAILED,
    STATUS_INDEXED,
    STATUS_INDEXING,
    Chunk,
    Report,
)
from app.db.session import SessionLocal
from app.rag.embeddings import embed_documents
from app.rag.pdf_processor import chunk_pages, extract_pages

log = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=settings.INDEXING_MAX_WORKERS, thread_name_prefix="indexer")


def submit_index(report_id: int) -> None:
    _executor.submit(index_report, report_id)


def index_report(report_id: int) -> None:
    db = SessionLocal()
    try:
        report = db.get(Report, report_id)
        if report is None:
            return
        report.status = STATUS_INDEXING
        report.error = None
        report.chunks_done = 0
        report.chunks_total = 0
        report.indexed_at = None
        db.execute(delete(Chunk).where(Chunk.report_id == report_id))
        db.commit()

        pages = extract_pages(Path(report.file_path))
        chunks = chunk_pages(pages)
        if not chunks:
            raise ValueError("No extractable text found. Scanned PDFs need OCR before indexing.")

        report.page_count = len(pages)
        report.chunks_total = len(chunks)
        db.commit()

        batch = settings.EMBEDDING_BATCH_SIZE * 4
        for start in range(0, len(chunks), batch):
            part = chunks[start : start + batch]
            vectors = embed_documents([c.content for c in part])
            db.execute(
                insert(Chunk),
                [
                    {
                        "report_id": report_id,
                        "report_year": report.report_year,
                        "page_number": c.page_number,
                        "page_label": c.page_label,
                        "chunk_index": c.chunk_index,
                        "content": c.content,
                        "embedding": vec,
                    }
                    for c, vec in zip(part, vectors)
                ],
            )
            report.chunks_done = start + len(part)
            db.commit()

        report.status = STATUS_INDEXED
        report.indexed_at = datetime.now(timezone.utc)
        db.commit()
        log.info("Indexed report %s: %s pages, %s chunks", report_id, len(pages), len(chunks))
    except Exception as exc:
        log.exception("Indexing failed for report %s", report_id)
        db.rollback()
        # Remove partial vectors so retrieval never sees half a report.
        db.execute(delete(Chunk).where(Chunk.report_id == report_id))
        report = db.get(Report, report_id)
        if report is not None:
            report.status = STATUS_FAILED
            report.error = str(exc)[:2000]
            report.chunks_done = 0
        db.commit()
    finally:
        db.close()
