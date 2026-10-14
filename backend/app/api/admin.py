from __future__ import annotations

import hashlib
import logging
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import require_admin
from app.core.config import settings
from app.db.models import (
    STATUS_INDEXED,
    STATUS_INDEXING,
    STATUS_UPLOADED,
    Chunk,
    Conversation,
    Message,
    Report,
    User,
)
from app.db.session import get_db
from app.rag.indexer import submit_index
from app.schemas import ReportOut, StatsOut

log = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])

_READ_BLOCK = 1024 * 1024
_PDF_MAGIC = b"%PDF-"


def _get_report(db: Session, report_id: int) -> Report:
    report = db.get(Report, report_id)
    if report is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Report not found")
    return report


def _sort_key(report_year: str) -> int:
    m = re.match(r"\d{4}", report_year)
    return int(m.group()) if m else 0


def _remove_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        log.warning("Could not delete %s", path)


@router.get("/stats", response_model=StatsOut)
def stats(db: Session = Depends(get_db)) -> StatsOut:
    count = lambda col: db.scalar(select(func.count()).select_from(col)) or 0  # noqa: E731
    return StatsOut(
        users=count(User),
        reports=count(Report),
        reports_indexed=db.scalar(select(func.count()).select_from(Report).where(Report.status == STATUS_INDEXED)) or 0,
        chunks=count(Chunk),
        conversations=count(Conversation),
        messages=count(Message),
    )


@router.get("/reports", response_model=list[ReportOut])
def list_reports(db: Session = Depends(get_db)) -> list[Report]:
    return list(db.scalars(select(Report).order_by(Report.sort_key.desc(), Report.id.desc())))


@router.post("/reports", response_model=ReportOut, status_code=status.HTTP_201_CREATED)
def upload_report(
    title: str = Form(...),
    report_year: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
) -> Report:
    title = title.strip()
    report_year = report_year.strip()
    if not title or len(title) > 255:
        raise HTTPException(422, "Title is required (max 255 characters)")
    if not re.fullmatch(settings.REPORT_YEAR_REGEX, report_year):
        raise HTTPException(422, "Report year format is not valid")

    settings.upload_path.mkdir(parents=True, exist_ok=True)
    dest = settings.upload_path / f"{uuid.uuid4().hex}.pdf"
    digest = hashlib.sha256()
    size = 0
    try:
        with dest.open("wb") as out:
            first = True
            while block := file.file.read(_READ_BLOCK):
                if first:
                    if not block.startswith(_PDF_MAGIC):
                        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "File is not a PDF")
                    first = False
                size += len(block)
                if size > settings.max_upload_bytes:
                    raise HTTPException(
                        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        f"File exceeds the {settings.MAX_UPLOAD_MB} MB limit",
                    )
                digest.update(block)
                out.write(block)
        if size == 0:
            raise HTTPException(422, "Uploaded file is empty")

        file_hash = digest.hexdigest()
        existing = db.scalar(select(Report).where(Report.file_hash == file_hash))
        if existing is not None:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"This PDF is already uploaded as '{existing.title}' ({existing.report_year})",
            )

        report = Report(
            title=title,
            report_year=report_year,
            sort_key=_sort_key(report_year),
            filename=Path(file.filename or "report.pdf").name[:255],
            file_path=str(dest),
            file_hash=file_hash,
            file_size=size,
            status=STATUS_UPLOADED,
            uploaded_by=admin.id,
        )
        db.add(report)
        db.commit()
    except Exception:
        _remove_file(dest)
        raise

    report.status = STATUS_INDEXING
    db.commit()
    submit_index(report.id)
    return report


@router.post("/reports/{report_id}/reindex", response_model=ReportOut)
def reindex_report(report_id: int, db: Session = Depends(get_db)) -> Report:
    report = _get_report(db, report_id)
    if report.status == STATUS_INDEXING:
        raise HTTPException(status.HTTP_409_CONFLICT, "Report is already being indexed")
    if not Path(report.file_path).exists():
        raise HTTPException(status.HTTP_410_GONE, "The stored PDF file is missing; delete and re-upload")
    report.status = STATUS_INDEXING
    report.error = None
    report.chunks_done = 0
    db.commit()
    submit_index(report.id)
    return report


@router.delete("/reports/{report_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_report(report_id: int, db: Session = Depends(get_db)) -> None:
    report = _get_report(db, report_id)
    if report.status == STATUS_INDEXING:
        raise HTTPException(status.HTTP_409_CONFLICT, "Wait for indexing to finish before deleting")
    path = Path(report.file_path)
    db.delete(report)  # chunks are removed by ON DELETE CASCADE
    db.commit()
    _remove_file(path)
