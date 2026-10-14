"""PDF text extraction (page-aware) and chunking.

Chunks never span pages, so every chunk carries an exact page number that
citations can be verified against.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings


@dataclass
class PageText:
    page_number: int  # 1-based physical page in the PDF
    page_label: str | None  # printed label (e.g. "xii", "45") when the PDF defines one
    text: str


@dataclass
class ChunkRecord:
    page_number: int
    page_label: str | None
    chunk_index: int
    content: str


_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_MULTI_BLANK = re.compile(r"\n{3,}")
_SPACES = re.compile(r"[ \t ]+")
_LIGATURE_REPLACEMENTS = {
    "\ufb01 ": "fi",
    "\ufb02 ": "fl",
    "\ufb00 ": "ff",
    "\ufb03 ": "ffi",
    "\ufb04 ": "ffl",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb00": "ff",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}


def clean_text(raw: str) -> str:
    for lig, rep in _LIGATURE_REPLACEMENTS.items():
        raw = raw.replace(lig, rep)
    raw = unicodedata.normalize("NFKC", raw)
    text = raw.replace("\x00", "")  # Postgres rejects NUL characters
    text = _HYPHEN_BREAK.sub(r"\1\2", text)  # re-join words split across lines
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


def _extract_page_text_pymupdf(page) -> str:
    rect = page.rect
    w, h = rect.width, rect.height
    mid_x = w / 2.0
    raw_blocks = page.get_text("blocks")
    # Filter text blocks (block[6] == 0 for text, 1 for image)
    text_blocks = [b for b in raw_blocks if b[6] == 0 and b[4].strip()]
    if not text_blocks:
        return ""

    # Detect 2-column layout: sufficient text blocks on both left and right
    left_blocks = [b for b in text_blocks if b[2] <= mid_x + 30 and b[1] > 70 and b[3] < h - 40]
    right_blocks = [b for b in text_blocks if b[0] >= mid_x - 30 and b[1] > 70 and b[3] < h - 40]
    is_two_column = len(left_blocks) >= 2 and len(right_blocks) >= 2

    if is_two_column:
        def block_sort_key(b):
            x0, y0, x1, y1 = b[0], b[1], b[2], b[3]
            # Headers at very top
            if y0 <= 70:
                return (0, 0, y0, x0)
            # Footers at very bottom
            if y1 >= h - 40:
                return (3, 0, y0, x0)
            # Full-width spanning blocks (e.g., wide tables, section banners)
            if x0 < mid_x - 30 and x1 > mid_x + 30:
                return (1, y0, 0, x0)
            # Left column
            if x1 <= mid_x + 30:
                return (1, 0, y0, x0)
            # Right column
            return (1, 1, y0, x0)

        sorted_blocks = sorted(text_blocks, key=block_sort_key)
    else:
        # Single-column: natural reading order top-to-bottom, left-to-right
        sorted_blocks = sorted(text_blocks, key=lambda b: (b[1], b[0]))

    return "\n\n".join(b[4].strip() for b in sorted_blocks)


def _extract_pymupdf(path: Path) -> list[PageText]:
    import pymupdf

    pages: list[PageText] = []
    with pymupdf.open(path) as doc:
        for index, page in enumerate(doc, start=1):
            label = None
            try:
                label = page.get_label() or None
            except Exception:
                label = None
            raw_text = _extract_page_text_pymupdf(page)
            pages.append(PageText(index, label, clean_text(raw_text)))
    return pages


def _extract_pdfplumber(path: Path) -> list[PageText]:
    import pdfplumber

    pages: list[PageText] = []
    with pdfplumber.open(path) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            pages.append(PageText(index, None, clean_text(page.extract_text() or "")))
    return pages


def extract_pages(path: Path) -> list[PageText]:
    if settings.PDF_EXTRACTOR == "pdfplumber":
        return _extract_pdfplumber(path)
    return _extract_pymupdf(path)


def chunk_pages(pages: list[PageText]) -> list[ChunkRecord]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    records: list[ChunkRecord] = []
    for page in pages:
        if not page.text:
            continue
        for piece in splitter.split_text(page.text):
            piece = piece.strip()
            if len(piece) < settings.MIN_CHUNK_CHARS:
                continue
            records.append(ChunkRecord(page.page_number, page.page_label, len(records), piece))
    return records
