"""Prompt templates are plain text files in PROMPTS_DIR (configured in .env)."""
from __future__ import annotations

import re
from functools import lru_cache

from app.core.config import settings

_VAR = re.compile(r"\{\{(\w+)\}\}")


@lru_cache(maxsize=None)
def _read(name: str) -> str:
    return (settings.prompts_path / f"{name}.txt").read_text(encoding="utf-8").strip()


def render(name: str, **values: object) -> str:
    template = _read(name)

    def sub(match: re.Match) -> str:
        key = match.group(1)
        if key not in values:
            raise KeyError(f"Prompt '{name}' needs variable '{key}'")
        return str(values[key])

    return _VAR.sub(sub, template)
