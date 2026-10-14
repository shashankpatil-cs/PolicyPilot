from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ---- auth -------------------------------------------------------------
class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class UserOut(ORMModel):
    id: int
    username: str
    role: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserOut


# ---- reports ----------------------------------------------------------
class ReportOut(ORMModel):
    id: int
    title: str
    report_year: str
    filename: str
    file_size: int
    page_count: int
    status: str
    error: str | None
    chunks_total: int
    chunks_done: int
    created_at: datetime
    indexed_at: datetime | None


class ReportYearOut(BaseModel):
    report_year: str
    title: str


class StatsOut(BaseModel):
    users: int
    reports: int
    reports_indexed: int
    chunks: int
    conversations: int
    messages: int


# ---- chat -------------------------------------------------------------
class AskRequest(BaseModel):
    question: str
    conversation_id: int | None = None
    mode: Literal["standard", "research"] = "standard"
    years: list[str] | None = None


class MessageOut(ORMModel):
    id: int
    role: str
    content: str
    sufficient: bool | None
    citations: list[dict[str, Any]] | None
    meta: dict[str, Any] | None
    feedback: int | None
    created_at: datetime


class ConversationOut(ORMModel):
    id: int
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationDetail(ConversationOut):
    messages: list[MessageOut]


class AskResponse(BaseModel):
    conversation: ConversationOut
    message: MessageOut


class FeedbackRequest(BaseModel):
    rating: Literal[1, -1]
