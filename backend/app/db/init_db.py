"""Schema creation, vector/FTS indexes and admin bootstrap."""
from __future__ import annotations

import logging

from sqlalchemy import select, text, update
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import hash_password
from app.db.models import (
    ROLE_ADMIN,
    STATUS_FAILED,
    STATUS_INDEXING,
    Base,
    Report,
    User,
)
from app.db.session import SessionLocal, engine

log = logging.getLogger(__name__)


def create_schema() -> None:
    with engine.begin() as conn:
        try:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        except Exception as exc:  # needs superuser unless already installed
            log.warning("Could not CREATE EXTENSION vector (%s). Run scripts/setup_db.py as a superuser.", exc)
            raise
    Base.metadata.create_all(engine)
    cfg = settings.FTS_CONFIG  # validated as [a-z_]+ in Settings
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw "
                "ON chunks USING hnsw (embedding vector_cosine_ops)"
            )
        )
        conn.execute(
            text(
                f"CREATE INDEX IF NOT EXISTS ix_chunks_fts "
                f"ON chunks USING gin (to_tsvector('{cfg}', content))"
            )
        )


def bootstrap_admin(db: Session) -> None:
    """Create (or optionally reset) the admin account from .env. Only a bcrypt hash is stored."""
    username = settings.ADMIN_USERNAME.strip().lower()
    password_hash = settings.ADMIN_PASSWORD_HASH.strip() or hash_password(settings.ADMIN_PASSWORD)

    admin = db.scalar(select(User).where(User.username == username))
    if admin is None:
        db.add(User(username=username, password_hash=password_hash, role=ROLE_ADMIN, is_active=True))
        db.commit()
        log.info("Admin account '%s' created", username)
        return

    if admin.role != ROLE_ADMIN:
        admin.role = ROLE_ADMIN
        db.commit()
    if settings.ADMIN_RESET_PASSWORD_ON_START:
        admin.password_hash = password_hash
        admin.is_active = True
        db.commit()
        log.info("Admin password reset from environment")


def mark_interrupted_indexing(db: Session) -> None:
    """Reports left in 'indexing' by a crashed/stopped process cannot finish; flag them for re-index."""
    result = db.execute(
        update(Report)
        .where(Report.status == STATUS_INDEXING)
        .values(status=STATUS_FAILED, error="Indexing was interrupted by a server restart. Re-index to retry.")
    )
    db.commit()
    if result.rowcount:
        log.warning("Marked %s interrupted report(s) as failed", result.rowcount)


def init_app_data() -> None:
    if settings.AUTO_CREATE_TABLES:
        create_schema()
    with SessionLocal() as db:
        bootstrap_admin(db)
        mark_interrupted_indexing(db)
