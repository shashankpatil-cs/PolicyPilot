"""In-memory failed-login throttle (single-process; use Redis if you scale out)."""
from __future__ import annotations

import threading
import time

from fastapi import HTTPException, status

from app.core.config import settings

_lock = threading.Lock()
_failures: dict[str, list[float]] = {}


def _prune(key: str, now: float) -> list[float]:
    window = settings.LOGIN_WINDOW_SECONDS
    kept = [t for t in _failures.get(key, []) if now - t < window]
    if kept:
        _failures[key] = kept
    else:
        _failures.pop(key, None)
    return kept


def check_login_allowed(key: str) -> None:
    now = time.monotonic()
    with _lock:
        if len(_prune(key, now)) >= settings.LOGIN_MAX_ATTEMPTS:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                "Too many failed login attempts. Please wait and try again.",
            )


def record_login_failure(key: str) -> None:
    now = time.monotonic()
    with _lock:
        _prune(key, now)
        _failures.setdefault(key, []).append(now)


def clear_login_failures(key: str) -> None:
    with _lock:
        _failures.pop(key, None)
