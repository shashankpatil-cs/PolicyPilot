from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from openai import OpenAIError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ai import tools
from app.ai.llm import LLMError
from app.ai.pipeline import run_pipeline
from app.api.deps import get_current_user
from app.core.config import settings
from app.db.models import Conversation, Message, User
from app.db.session import get_db
from app.rag.retriever import list_indexed_reports
from app.schemas import (
    AskRequest,
    AskResponse,
    ConversationDetail,
    ConversationOut,
    FeedbackRequest,
    MessageOut,
    ReportYearOut,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

_HIDDEN_META_KEYS = {"note", "run_id"}


def _public_meta(meta: dict | None) -> dict | None:
    if meta is None:
        return None
    return {k: v for k, v in meta.items() if k not in _HIDDEN_META_KEYS}


def _message_out(msg: Message) -> MessageOut:
    out = MessageOut.model_validate(msg)
    out.meta = _public_meta(out.meta)
    return out


def _owned_conversation(db: Session, conversation_id: int, user: User) -> Conversation:
    conv = db.get(Conversation, conversation_id)
    if conv is None or conv.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found")
    return conv


@router.get("/reports", response_model=list[ReportYearOut])
def available_reports(db: Session = Depends(get_db), _: User = Depends(get_current_user)) -> list[ReportYearOut]:
    return [ReportYearOut(report_year=r["report_year"], title=r["title"]) for r in list_indexed_reports(db)]


@router.get("/conversations", response_model=list[ConversationOut])
def list_conversations(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[Conversation]:
    stmt = (
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
        .limit(settings.CONVERSATIONS_PAGE_LIMIT)
    )
    return list(db.scalars(stmt))


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(
    conversation_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> ConversationDetail:
    conv = _owned_conversation(db, conversation_id, user)
    return ConversationDetail(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        messages=[_message_out(m) for m in conv.messages],
    )


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> None:
    conv = _owned_conversation(db, conversation_id, user)
    db.delete(conv)
    db.commit()


@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> AskResponse:
    question = body.question.strip()
    if not question:
        raise HTTPException(422, "Question must not be empty")
    if len(question) > settings.QUESTION_MAX_CHARS:
        raise HTTPException(
            422,
            f"Question is too long (max {settings.QUESTION_MAX_CHARS} characters)",
        )

    years: list[str] | None = None
    if body.years:
        available = tools.available_report_years()
        unknown = [y for y in body.years if y not in available]
        if unknown:
            raise HTTPException(
                422,
                f"Report year(s) not indexed: {', '.join(unknown)}",
            )
        years = list(dict.fromkeys(body.years))

    if body.conversation_id is not None:
        conv = _owned_conversation(db, body.conversation_id, user)
    else:
        conv = Conversation(user_id=user.id, title=question[:60])
        db.add(conv)
        db.flush()

    db.add(Message(conversation_id=conv.id, role="user", content=question))
    db.commit()  # keep the question even if answering fails

    try:
        result = run_pipeline(question, body.mode, years)
    except (OpenAIError, LLMError):
        log.exception("AI provider failure while answering")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "The AI service failed to answer. Please try again.")

    assistant = Message(
        conversation_id=conv.id,
        role="assistant",
        content=result["answer"],
        sufficient=result["sufficient"],
        citations=result["citations"],
        meta=result["meta"],
    )
    db.add(assistant)
    conv.title = conv.title  # touch: onupdate bumps updated_at
    db.commit()
    db.refresh(conv)
    db.refresh(assistant)
    return AskResponse(conversation=ConversationOut.model_validate(conv), message=_message_out(assistant))


@router.post("/messages/{message_id}/feedback", response_model=MessageOut)
def message_feedback(
    message_id: int,
    body: FeedbackRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> MessageOut:
    msg = db.get(Message, message_id)
    if msg is None or msg.role != "assistant":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Message not found")
    _owned_conversation(db, msg.conversation_id, user)
    msg.feedback = body.rating
    db.commit()

    run_id = (msg.meta or {}).get("run_id")
    if run_id and settings.LANGSMITH_TRACING:
        try:
            from langsmith import Client

            Client().create_feedback(run_id, key="user_rating", score=1 if body.rating == 1 else 0)
        except Exception:
            log.warning("Could not send feedback to LangSmith", exc_info=True)
    return _message_out(msg)
