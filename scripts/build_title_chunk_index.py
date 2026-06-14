"""Build an alternate title-prepended chunk FAISS index.

This experiment keeps the main `faiss.index` untouched.  It writes separate
artifacts so retrieval can opt into the variant and fall back if absent.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List

import faiss
import numpy as np

STUDENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDENT_ROOT))

from chunk import DEFAULT_SECTION, _format_chunk, _normalize_text, _window_words, _words, split_sections
from embed import embed_texts
from utils import ARTIFACTS_DIR, EMBEDDING_MODEL_NAME, iter_entries

TITLE_CHUNK_INDEX_NAME = "faiss_title190.index"
TITLE_CHUNK_META_NAME = "chunk_meta_title190.json"
BUILD_TEXT_BATCH_SIZE = 2048
MODEL_ENCODE_BATCH_SIZE = 128


@dataclass
class ChunkRow:
    page_id: int
    chunk_id: int
    section: str
    text: str


def chunk_entry_variant(
    record: Dict[str, Any],
    *,
    chunk_words: int,
    chunk_overlap: int,
    short_page_words: int,
) -> List[ChunkRow]:
    page_id = int(record["page_id"])
    title = _normalize_text(record.get("title", ""))
    content = str(record.get("content", "") or "")
    content_words = _words(content)

    if len(content_words) <= short_page_words:
        return [
            ChunkRow(
                page_id=page_id,
                chunk_id=0,
                section=DEFAULT_SECTION,
                text=_format_chunk(title, DEFAULT_SECTION, content),
            )
        ]

    chunks: List[ChunkRow] = []
    for section, section_text in split_sections(content):
        section_words = _words(section_text)
        if not section_words:
            continue
        for window in _window_words(section_words, size=chunk_words, overlap=chunk_overlap):
            chunks.append(
                ChunkRow(
                    page_id=page_id,
                    chunk_id=len(chunks),
                    section=section,
                    text=_format_chunk(title, section, " ".join(window)),
                )
            )

    if chunks:
        return chunks
    return [
        ChunkRow(
            page_id=page_id,
            chunk_id=0,
            section=DEFAULT_SECTION,
            text=_format_chunk(title, DEFAULT_SECTION, content),
        )
    ]


def chunk_corpus_variant(
    records: Iterable[Dict[str, Any]],
    *,
    chunk_words: int,
    chunk_overlap: int,
    short_page_words: int,
) -> List[ChunkRow]:
    rows: List[ChunkRow] = []
    for record in records:
        rows.extend(
            chunk_entry_variant(
                record,
                chunk_words=chunk_words,
                chunk_overlap=chunk_overlap,
                short_page_words=short_page_words,
            )
        )
    return rows


def build_index(chunks: List[ChunkRow]) -> faiss.Index:
    index: faiss.Index | None = None
    total = len(chunks)
    print(f"embedding_chunks={total}", flush=True)
    for start in range(0, total, BUILD_TEXT_BATCH_SIZE):
        end = min(start + BUILD_TEXT_BATCH_SIZE, total)
        vectors = embed_texts(
            [chunk.text for chunk in chunks[start:end]],
            batch_size=MODEL_ENCODE_BATCH_SIZE,
        )
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if index is None:
            index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        print(f"embedded={end}/{total}", flush=True)
    if index is None:
        index = faiss.IndexFlatIP(384)
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-words", type=int, default=190)
    parser.add_argument("--chunk-overlap", type=int, default=45)
    parser.add_argument("--short-page-words", type=int, default=220)
    parser.add_argument("--index-name", default=TITLE_CHUNK_INDEX_NAME)
    parser.add_argument("--meta-name", default=TITLE_CHUNK_META_NAME)
    args = parser.parse_args()

    records = list(iter_entries())
    chunks = chunk_corpus_variant(
        records,
        chunk_words=args.chunk_words,
        chunk_overlap=args.chunk_overlap,
        short_page_words=args.short_page_words,
    )
    print(f"pages={len(records)}")
    print(
        f"chunk_words={args.chunk_words} "
        f"chunk_overlap={args.chunk_overlap} "
        f"short_page_words={args.short_page_words}"
    )

    index = build_index(chunks)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(ARTIFACTS_DIR / args.index_name))
    meta = {
        "page_ids": [row.page_id for row in chunks],
        "chunk_ids": [row.chunk_id for row in chunks],
        "sections": [row.section for row in chunks],
        "model": EMBEDDING_MODEL_NAME,
        "index_type": "faiss.IndexFlatIP",
        "num_vectors": len(chunks),
        "config": {
            "chunk_words": args.chunk_words,
            "chunk_overlap": args.chunk_overlap,
            "short_page_words": args.short_page_words,
            "score_formula": "max_chunk_score + 0.10 * average_top_3_chunk_scores",
            "context": "Title and section prepended to every chunk",
        },
    }
    (ARTIFACTS_DIR / args.meta_name).write_text(
        json.dumps(meta, ensure_ascii=True),
        encoding="utf-8",
    )
    print(f"index={ARTIFACTS_DIR / args.index_name}")
    print(f"meta={ARTIFACTS_DIR / args.meta_name}")


if __name__ == "__main__":
    main()
