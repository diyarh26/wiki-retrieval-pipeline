"""Section-aware preprocessing and chunking."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, Iterable, List, Tuple

CHUNK_WORDS = 300
CHUNK_OVERLAP = 60
SHORT_PAGE_WORDS = 320
DEFAULT_SECTION = "Introduction"

_SPACE_RE = re.compile(r"\s+")


@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    title: str
    section: str
    text: str


def _normalize_text(text: str) -> str:
    return _SPACE_RE.sub(" ", str(text or "")).strip()


def _words(text: str) -> List[str]:
    text = _normalize_text(text)
    return text.split() if text else []


def _looks_like_heading(text: str) -> bool:
    candidate = _normalize_text(text).strip(".:")
    if not candidate or len(candidate) > 80:
        return False
    words = candidate.split()
    if not 1 <= len(words) <= 8:
        return False
    if any(mark in candidate for mark in "?!;"):
        return False
    return candidate[0].isupper()


def _split_leading_heading(paragraph: str) -> Tuple[str | None, str]:
    paragraph = paragraph.strip()
    if "." not in paragraph:
        return None, paragraph

    first, rest = paragraph.split(".", 1)
    if _looks_like_heading(first):
        return first.strip(), rest.strip()
    return None, paragraph


def split_sections(content: str) -> List[Tuple[str, str]]:
    """Best-effort section split for Wikipedia-style article text."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", str(content or "")) if p.strip()]
    if not paragraphs:
        return [(DEFAULT_SECTION, "")]

    sections: List[Tuple[str, List[str]]] = []
    current_section = DEFAULT_SECTION
    current_parts: List[str] = []

    def flush() -> None:
        nonlocal current_parts
        if current_parts:
            sections.append((current_section, current_parts))
            current_parts = []

    for paragraph in paragraphs:
        if _looks_like_heading(paragraph):
            flush()
            current_section = paragraph.strip().strip(".:")
            continue

        heading, body = _split_leading_heading(paragraph)
        if heading and body:
            flush()
            current_section = heading
            current_parts.append(body)
            continue

        current_parts.append(paragraph)

    flush()
    if not sections:
        return [(DEFAULT_SECTION, _normalize_text(content))]
    return [(section, _normalize_text(" ".join(parts))) for section, parts in sections]


def _window_words(words: List[str], *, size: int, overlap: int) -> Iterable[List[str]]:
    if not words:
        return
    stride = max(1, size - overlap)
    for start in range(0, len(words), stride):
        window = words[start : start + size]
        if window:
            yield window
        if start + size >= len(words):
            break


def _format_chunk(title: str, section: str, body: str) -> str:
    title = _normalize_text(title)
    section = _normalize_text(section)
    body = _normalize_text(body)
    if section:
        return f"Title: {title}\nSection: {section}\nText: {body}".strip()
    return f"Title: {title}\nText: {body}".strip()


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    """Split one corpus entry into title-prefixed section-aware chunks."""
    page_id = int(record["page_id"])
    title = _normalize_text(record.get("title", ""))
    content = str(record.get("content", "") or "")
    content_words = _words(content)

    if len(content_words) <= SHORT_PAGE_WORDS:
        text = _format_chunk(title, DEFAULT_SECTION, content)
        return [Chunk(page_id=page_id, chunk_id=0, title=title, section=DEFAULT_SECTION, text=text)]

    chunks: List[Chunk] = []
    for section, section_text in split_sections(content):
        section_words = _words(section_text)
        if not section_words:
            continue
        for window in _window_words(section_words, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
            text = _format_chunk(title, section, " ".join(window))
            chunks.append(
                Chunk(
                    page_id=page_id,
                    chunk_id=len(chunks),
                    title=title,
                    section=section,
                    text=text,
                )
            )

    if chunks:
        return chunks

    text = _format_chunk(title, DEFAULT_SECTION, content)
    return [Chunk(page_id=page_id, chunk_id=0, title=title, section=DEFAULT_SECTION, text=text)]


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record))
    return chunks
