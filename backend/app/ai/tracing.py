"""LangSmith wiring (LangChain/LangGraph auto-trace via env; OpenAI + Agents SDK via wrappers)."""
from __future__ import annotations

import logging
import os

from app.core.config import settings

log = logging.getLogger(__name__)


def setup_tracing() -> None:
    enabled = settings.LANGSMITH_TRACING
    flag = "true" if enabled else "false"
    os.environ["LANGSMITH_TRACING"] = flag
    os.environ["LANGCHAIN_TRACING_V2"] = flag  # older library versions
    if enabled:
        os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
        os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
        os.environ["LANGSMITH_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
        os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT
        os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
        log.info("LangSmith tracing enabled (project=%s)", settings.LANGSMITH_PROJECT)

    try:
        from agents import set_trace_processors, set_tracing_disabled

        if enabled:
            from langsmith.wrappers import OpenAIAgentsTracingProcessor

            set_trace_processors([OpenAIAgentsTracingProcessor()])
        else:
            set_tracing_disabled(not settings.AGENTS_OPENAI_TRACING)
    except ImportError:
        pass
    except Exception:
        log.exception("Could not configure Agents SDK tracing")
