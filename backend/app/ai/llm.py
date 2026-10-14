"""Thin OpenAI function-calling helper (Chat Completions + forced / required tool calls)."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache

from openai import OpenAI

from app.ai.context import EvidenceContext
from app.core.config import settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


@dataclass
class ToolCall:
    name: str
    arguments: dict


@lru_cache
def get_client() -> OpenAI:
    client = OpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL or None,
        timeout=settings.OPENAI_TIMEOUT_SECONDS,
        max_retries=settings.OPENAI_MAX_RETRIES,
    )
    if settings.LANGSMITH_TRACING:
        try:
            from langsmith.wrappers import wrap_openai

            client = wrap_openai(client)
        except ImportError:
            pass
    return client


def call_tools(
    *,
    ctx: EvidenceContext,
    model: str,
    system: str,
    user: str,
    tools: list[dict],
    force: str | None = None,
) -> list[ToolCall]:
    """Call the model with tools. `force` names a function that must be called;
    otherwise the model must call at least one tool."""
    tool_choice = {"type": "function", "function": {"name": force}} if force else "required"
    kwargs: dict = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "tools": [{"type": "function", "function": t} for t in tools],
        "tool_choice": tool_choice,
    }
    if settings.openai_temperature is not None:
        kwargs["temperature"] = settings.openai_temperature

    response = get_client().chat.completions.create(**kwargs)
    ctx.add_usage(response.usage)

    message = response.choices[0].message
    calls: list[ToolCall] = []
    for tc in message.tool_calls or []:
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError as exc:
            raise LLMError(f"Model returned invalid JSON arguments for {tc.function.name}") from exc
        if not isinstance(args, dict):
            raise LLMError(f"Model returned non-object arguments for {tc.function.name}")
        calls.append(ToolCall(tc.function.name, args))
    if not calls:
        raise LLMError("Model did not call any function")
    return calls
