"""Offline FAISS index build and load (not timed at grading)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from chunk import CHUNK_OVERLAP, CHUNK_WORDS, SHORT_PAGE_WORDS, Chunk, chunk_corpus
from embed import embed_texts
from utils import ARTIFACTS_DIR, EMBEDDING_MODEL_NAME, ensure_artifacts_dir, iter_entries

FAISS_INDEX_NAME = "faiss.index"
CHUNK_META_NAME = "chunk_meta.json"
PAGE_META_NAME = "page_meta.json"
BUILD_TEXT_BATCH_SIZE = 2048
MODEL_ENCODE_BATCH_SIZE = 128


def _build_faiss_index(chunks: List[Chunk]) -> faiss.Index:
    """Embed chunks in batches and add them incrementally to FAISS."""
    index: faiss.Index | None = None
    total = len(chunks)
    print(f"Embedding {total} chunks...", flush=True)

    for start in range(0, total, BUILD_TEXT_BATCH_SIZE):
        end = min(start + BUILD_TEXT_BATCH_SIZE, total)
        texts = [chunk.text for chunk in chunks[start:end]]
        vectors = embed_texts(texts, batch_size=MODEL_ENCODE_BATCH_SIZE)
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        if index is None:
            index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        print(f"Embedded chunks {end}/{total}", flush=True)

    if index is None:
        index = faiss.IndexFlatIP(384)
    return index


def build_index(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
) -> Tuple[faiss.Index, Dict[str, Any]]:
    """
    Embed the chunked corpus and persist FAISS + metadata artifacts.

    Row i in the FAISS index corresponds to chunk_meta["page_ids"][i].
    """
    out_dir = artifacts_dir or ensure_artifacts_dir()
    records = list(iter_entries(entries_dir))
    chunks: List[Chunk] = chunk_corpus(records)
    print(f"Loaded {len(records)} pages and created {len(chunks)} chunks.", flush=True)
    index = _build_faiss_index(chunks)
    faiss.write_index(index, str(out_dir / FAISS_INDEX_NAME))

    page_meta = [
        {
            "page_id": int(record["page_id"]),
            "title": str(record.get("title", "")),
            "content_words": len(str(record.get("content", "") or "").split()),
        }
        for record in records
    ]

    chunk_meta: Dict[str, Any] = {
        "page_ids": [c.page_id for c in chunks],
        "chunk_ids": [c.chunk_id for c in chunks],
        "sections": [c.section for c in chunks],
        "model": EMBEDDING_MODEL_NAME,
        "index_type": "faiss.IndexFlatIP",
        "num_vectors": len(chunks),
        "config": {
            "chunk_words": CHUNK_WORDS,
            "chunk_overlap": CHUNK_OVERLAP,
            "short_page_words": SHORT_PAGE_WORDS,
            "score_formula": "max_chunk_score + 0.10 * average_top_3_chunk_scores",
        },
    }
    (out_dir / CHUNK_META_NAME).write_text(
        json.dumps(chunk_meta, ensure_ascii=True), encoding="utf-8"
    )
    (out_dir / PAGE_META_NAME).write_text(
        json.dumps(page_meta, ensure_ascii=True), encoding="utf-8"
    )
    return index, chunk_meta


def load_index(
    artifacts_dir: Optional[Path] = None,
) -> Tuple[faiss.Index, Dict[str, Any], Dict[int, Dict[str, Any]]]:
    """Load precomputed FAISS index and metadata from artifacts/."""
    root = artifacts_dir or ARTIFACTS_DIR
    index = faiss.read_index(str(root / FAISS_INDEX_NAME))
    chunk_meta = json.loads((root / CHUNK_META_NAME).read_text(encoding="utf-8"))
    page_rows = json.loads((root / PAGE_META_NAME).read_text(encoding="utf-8"))
    page_meta = {int(row["page_id"]): row for row in page_rows}
    chunk_meta["page_ids"] = [int(x) for x in chunk_meta["page_ids"]]
    chunk_meta["chunk_ids"] = [int(x) for x in chunk_meta["chunk_ids"]]
    return index, chunk_meta, page_meta
