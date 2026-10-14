"""BGE-M3 dense embeddings via LangChain's HuggingFaceEmbeddings (loaded once, lazily)."""
from __future__ import annotations

import logging
import threading
from functools import lru_cache

from app.core.config import settings

log = logging.getLogger(__name__)

_lock = threading.Lock()
_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        with _lock:
            if _embeddings is None:
                from langchain_huggingface import HuggingFaceEmbeddings

                log.info("Loading embedding model %s on %s", settings.EMBEDDING_MODEL_NAME, settings.EMBEDDING_DEVICE)
                model = HuggingFaceEmbeddings(
                    model_name=settings.EMBEDDING_MODEL_NAME,
                    model_kwargs={"device": settings.EMBEDDING_DEVICE},
                    encode_kwargs={
                        "normalize_embeddings": True,
                        "batch_size": settings.EMBEDDING_BATCH_SIZE,
                    },
                )
                st_model = getattr(model, "_client", None) or getattr(model, "client", None)
                if st_model is not None:
                    st_model.max_seq_length = settings.EMBEDDING_MAX_SEQ_LENGTH
                probe = model.embed_query("dimension probe")
                if len(probe) != settings.EMBEDDING_DIM:
                    raise RuntimeError(
                        f"EMBEDDING_DIM={settings.EMBEDDING_DIM} but {settings.EMBEDDING_MODEL_NAME} "
                        f"produces {len(probe)}-dim vectors"
                    )
                _embeddings = model
    return _embeddings


def embed_documents(texts: list[str]) -> list[list[float]]:
    return get_embeddings().embed_documents(texts)


@lru_cache(maxsize=256)
def _embed_query_cached(text: str) -> tuple[float, ...]:
    return tuple(get_embeddings().embed_query(text))


def embed_query(text: str) -> list[float]:
    # Cached: comparisons search the same query once per report year.
    return list(_embed_query_cached(text))
