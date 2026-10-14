from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.ai.tracing import setup_tracing
from app.api import admin, auth, chat
from app.core.config import settings
from app.db.init_db import init_app_data
from app.db.session import engine

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

setup_tracing()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    init_app_data()
    if settings.PRELOAD_EMBEDDING_MODEL:
        from app.rag.embeddings import get_embeddings

        get_embeddings()
    log.info("%s started (%s)", settings.APP_NAME, settings.APP_ENV)
    yield


app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,  # JWT is sent in the Authorization header, not cookies
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix=settings.API_PREFIX)
app.include_router(admin.router, prefix=settings.API_PREFIX)
app.include_router(chat.router, prefix=settings.API_PREFIX)


@app.get(f"{settings.API_PREFIX}/health", tags=["meta"])
def health() -> dict:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return {"status": "ok", "app": settings.APP_NAME}
