"""Central configuration.

Provides sensible defaults for local development (PostgreSQL + JWT auth) while
allowing any variable to be overridden via backend/.env or environment variables.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL

BACKEND_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BACKEND_DIR / ".env", override=True)


def _resolve(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (BACKEND_DIR / p).resolve()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # App
    APP_NAME: str = "RBI Intelligence AI"
    APP_ENV: str = "development"
    API_PREFIX: str = "/api"
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: str = "http://localhost:5173"
    PRELOAD_EMBEDDING_MODEL: bool = False

    # Database (strictly local PostgreSQL by default)
    DATABASE_URL: str | None = None
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "rbi_intelligence"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "postgres"
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    AUTO_CREATE_TABLES: bool = True
    FTS_CONFIG: str = "english"

    # Auth (strictly JWT auth)
    JWT_SECRET: str = "rbi_intelligence_dev_jwt_secret_key_at_least_32_characters_long_for_auth"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480
    USERNAME_REGEX: str = r"^[a-zA-Z0-9_.-]{3,32}$"
    PASSWORD_MIN_LENGTH: int = 8
    LOGIN_MAX_ATTEMPTS: int = 5
    LOGIN_WINDOW_SECONDS: int = 300

    # Admin
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = "admin282005"
    ADMIN_PASSWORD_HASH: str = ""
    ADMIN_RESET_PASSWORD_ON_START: bool = False

    # Uploads
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_MB: int = 200
    REPORT_YEAR_REGEX: str = r"^\d{4}(-\d{2})?$"

    # PDF / chunking
    PDF_EXTRACTOR: str = "pymupdf"
    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 200
    MIN_CHUNK_CHARS: int = 80

    # Embeddings
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    EMBEDDING_DIM: int = 1024
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_BATCH_SIZE: int = 16
    EMBEDDING_MAX_SEQ_LENGTH: int = 512
    INDEXING_MAX_WORKERS: int = 1

    # Retrieval
    RETRIEVAL_TOP_K: int = 8
    RETRIEVAL_CANDIDATES: int = 40
    MIN_SIMILARITY: float = 0.35
    HYBRID_SEARCH: bool = True
    RRF_K: int = 60
    FTS_MAX_TERMS: int = 12
    COMPARE_TOP_K_PER_YEAR: int = 4
    MAX_EVIDENCE_CHUNKS: int = 16
    HNSW_EF_SEARCH: int = 100

    # OpenAI
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = ""
    OPENAI_CHAT_MODEL: str = "gpt-4.1"
    OPENAI_PLANNER_MODEL: str = "gpt-4.1-mini"
    OPENAI_VALIDATOR_MODEL: str = "gpt-4.1-mini"
    OPENAI_AGENT_MODEL: str = "gpt-4.1"
    OPENAI_TEMPERATURE: str = "0"
    OPENAI_TIMEOUT_SECONDS: float = 60.0
    OPENAI_MAX_RETRIES: int = 2

    # Answering / validation
    QUESTION_MAX_CHARS: int = 1500
    MAX_PLANNED_TOOL_CALLS: int = 3
    MAX_GENERATION_ATTEMPTS: int = 2
    VALIDATION_LLM_CHECK: bool = True
    NUMERIC_CHECK: str = "warn"
    MIN_QUOTE_CHARS: int = 25
    QUOTE_FUZZY_THRESHOLD: float = 92.0
    INSUFFICIENT_EVIDENCE_MESSAGE: str = (
        "I could not find enough evidence in the indexed RBI Annual Reports to answer this reliably. "
        "Try rephrasing the question, naming a report year, or ask an administrator to index the relevant report."
    )

    # Research agent
    AGENT_MAX_TURNS: int = 10
    AGENT_MAX_REPAIR_ATTEMPTS: int = 2
    AGENTS_OPENAI_TRACING: bool = False

    # Prompts / chat
    PROMPTS_DIR: str = "prompts"
    CONVERSATIONS_PAGE_LIMIT: int = 100

    # LangSmith / evals
    LANGSMITH_TRACING: bool = False
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "rbi-intelligence-ai"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    EVAL_DATASET_PATH: str = "evals/dataset.example.jsonl"
    EVAL_DATASET_NAME: str = "rbi-intelligence-eval"
    EVAL_EXPERIMENT_PREFIX: str = "rbi-rag"
    EVAL_MAX_CONCURRENCY: int = 2
    EVAL_MODE: str = "standard"

    @model_validator(mode="after")
    def _validate(self) -> "Settings":
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.PDF_EXTRACTOR not in {"pymupdf", "pdfplumber"}:
            raise ValueError("PDF_EXTRACTOR must be 'pymupdf' or 'pdfplumber'")
        if self.NUMERIC_CHECK not in {"off", "warn", "strict"}:
            raise ValueError("NUMERIC_CHECK must be off, warn or strict")
        if self.EVAL_MODE not in {"standard", "research"}:
            raise ValueError("EVAL_MODE must be standard or research")
        if len(self.JWT_SECRET) < 32 or self.JWT_SECRET.startswith("change-me"):
            raise ValueError(
                "JWT_SECRET must be a random string of at least 32 characters "
                "(python -c \"import secrets; print(secrets.token_urlsafe(48))\")"
            )
        if not (self.ADMIN_PASSWORD or self.ADMIN_PASSWORD_HASH):
            raise ValueError("Set ADMIN_PASSWORD or ADMIN_PASSWORD_HASH")
        if not re.fullmatch(r"[a-z_]+", self.FTS_CONFIG):
            raise ValueError("FTS_CONFIG must be a plain lowercase identifier")
        if self.LANGSMITH_TRACING and not self.LANGSMITH_API_KEY:
            raise ValueError("LANGSMITH_TRACING=true requires LANGSMITH_API_KEY")
        for pattern_name in ("USERNAME_REGEX", "REPORT_YEAR_REGEX"):
            re.compile(getattr(self, pattern_name))
        return self

    # ---- derived values -------------------------------------------------
    @property
    def database_url(self) -> URL | str:
        if self.DATABASE_URL:
            url = self.DATABASE_URL
            if url.startswith("postgres://"):
                url = url.replace("postgres://", "postgresql+psycopg2://", 1)
            elif url.startswith("postgresql://") and not url.startswith("postgresql+"):
                url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
            return url
        return URL.create(
            "postgresql+psycopg2",
            username=self.DB_USER,
            password=self.DB_PASSWORD,
            host=self.DB_HOST,
            port=self.DB_PORT,
            database=self.DB_NAME,
        )

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def upload_path(self) -> Path:
        return _resolve(self.UPLOAD_DIR)

    @property
    def prompts_path(self) -> Path:
        return _resolve(self.PROMPTS_DIR)

    @property
    def eval_dataset_path(self) -> Path:
        return _resolve(self.EVAL_DATASET_PATH)

    @property
    def openai_temperature(self) -> float | None:
        raw = self.OPENAI_TEMPERATURE.strip()
        return float(raw) if raw else None

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


settings = get_settings()
