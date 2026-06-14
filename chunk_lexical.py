"""Chunk-level BM25 artifacts and retrieval."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from chunk import chunk_corpus
from lexical import BM25_B, BM25_K1, tokenize
from utils import ARTIFACTS_DIR, iter_entries

CHUNK_BM25_META_NAME = "chunk_bm25_meta.json"
CHUNK_BM25_TERMS_NAME = "chunk_bm25_terms.json"
CHUNK_BM25_DOCS_NAME = "chunk_bm25_doc_indices.npy"
CHUNK_BM25_TFS_NAME = "chunk_bm25_term_freqs.npy"
CHUNK_BM25_DOC_LENGTHS_NAME = "chunk_bm25_doc_lengths.npy"
CHUNK_BM25_PAGE_IDS_NAME = "chunk_bm25_page_ids.npy"

TOP_CHUNKS_PER_PAGE = 3
MULTI_CHUNK_WEIGHT = 0.10


@dataclass
class ChunkBM25Index:
    page_ids: np.ndarray
    doc_lengths: np.ndarray
    doc_indices: np.ndarray
    term_freqs: np.ndarray
    terms: Dict[str, Tuple[int, int, float]]
    avg_doc_length: float
    k1: float = BM25_K1
    b: float = BM25_B


def _chunk_page_score(scores: List[float]) -> float:
    ordered = sorted(scores, reverse=True)
    return ordered[0] + MULTI_CHUNK_WEIGHT * float(np.mean(ordered[:TOP_CHUNKS_PER_PAGE]))


def build_chunk_bm25_index() -> ChunkBM25Index:
    """Build a BM25 index where each document is a title-prefixed chunk."""
    records = list(iter_entries())
    chunks = chunk_corpus(records)
    postings: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    page_ids: List[int] = []
    doc_lengths: List[float] = []

    print(f"chunk_bm25_pages={len(records)}", flush=True)
    print(f"chunk_bm25_chunks={len(chunks)}", flush=True)

    for doc_idx, chunk in enumerate(chunks):
        page_ids.append(int(chunk.page_id))
        counts = Counter(tokenize(chunk.text))
        doc_lengths.append(float(sum(counts.values())) or 1.0)
        for term, term_freq in counts.items():
            postings[term].append((doc_idx, float(term_freq)))
        if doc_idx and doc_idx % 50000 == 0:
            print(f"indexed_chunks={doc_idx}/{len(chunks)}", flush=True)

    num_docs = len(page_ids)
    avg_doc_length = float(np.mean(doc_lengths)) if doc_lengths else 1.0
    terms: Dict[str, Tuple[int, int, float]] = {}
    flat_doc_indices: List[int] = []
    flat_term_freqs: List[float] = []
    offset = 0

    for term in sorted(postings):
        rows = postings[term]
        doc_freq = len(rows)
        idf = math.log(1.0 + (num_docs - doc_freq + 0.5) / (doc_freq + 0.5))
        for doc_idx, term_freq in rows:
            flat_doc_indices.append(doc_idx)
            flat_term_freqs.append(term_freq)
        terms[term] = (offset, doc_freq, float(idf))
        offset += doc_freq

    return ChunkBM25Index(
        page_ids=np.asarray(page_ids, dtype=np.int32),
        doc_lengths=np.asarray(doc_lengths, dtype=np.float32),
        doc_indices=np.asarray(flat_doc_indices, dtype=np.int32),
        term_freqs=np.asarray(flat_term_freqs, dtype=np.float32),
        terms=terms,
        avg_doc_length=avg_doc_length,
    )


def save_chunk_bm25_index(
    index: ChunkBM25Index,
    artifacts_dir: Optional[Path] = None,
) -> None:
    root = artifacts_dir or ARTIFACTS_DIR
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / CHUNK_BM25_PAGE_IDS_NAME, index.page_ids)
    np.save(root / CHUNK_BM25_DOC_LENGTHS_NAME, index.doc_lengths)
    np.save(root / CHUNK_BM25_DOCS_NAME, index.doc_indices)
    np.save(root / CHUNK_BM25_TFS_NAME, index.term_freqs)
    (root / CHUNK_BM25_TERMS_NAME).write_text(
        json.dumps(
            {term: [offset, length, idf] for term, (offset, length, idf) in index.terms.items()},
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    (root / CHUNK_BM25_META_NAME).write_text(
        json.dumps(
            {
                "version": 1,
                "num_chunks": int(index.page_ids.shape[0]),
                "num_terms": len(index.terms),
                "num_postings": int(index.doc_indices.shape[0]),
                "avg_doc_length": index.avg_doc_length,
                "k1": index.k1,
                "b": index.b,
                "source_chunks": "chunk.chunk_corpus current title-prefixed section chunks",
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )


def chunk_bm25_artifacts_exist(artifacts_dir: Optional[Path] = None) -> bool:
    root = artifacts_dir or ARTIFACTS_DIR
    required = [
        CHUNK_BM25_META_NAME,
        CHUNK_BM25_TERMS_NAME,
        CHUNK_BM25_DOCS_NAME,
        CHUNK_BM25_TFS_NAME,
        CHUNK_BM25_DOC_LENGTHS_NAME,
        CHUNK_BM25_PAGE_IDS_NAME,
    ]
    return all((root / name).exists() for name in required)


def load_chunk_bm25_index(artifacts_dir: Optional[Path] = None) -> Optional[ChunkBM25Index]:
    root = artifacts_dir or ARTIFACTS_DIR
    if not chunk_bm25_artifacts_exist(root):
        return None
    try:
        meta = json.loads((root / CHUNK_BM25_META_NAME).read_text(encoding="utf-8"))
        raw_terms = json.loads((root / CHUNK_BM25_TERMS_NAME).read_text(encoding="utf-8"))
        terms = {
            term: (int(row[0]), int(row[1]), float(row[2]))
            for term, row in raw_terms.items()
        }
        return ChunkBM25Index(
            page_ids=np.load(root / CHUNK_BM25_PAGE_IDS_NAME),
            doc_lengths=np.load(root / CHUNK_BM25_DOC_LENGTHS_NAME),
            doc_indices=np.load(root / CHUNK_BM25_DOCS_NAME),
            term_freqs=np.load(root / CHUNK_BM25_TFS_NAME),
            terms=terms,
            avg_doc_length=float(meta["avg_doc_length"]),
            k1=float(meta.get("k1", BM25_K1)),
            b=float(meta.get("b", BM25_B)),
        )
    except Exception:
        return None


def rank_chunk_bm25(
    index: ChunkBM25Index,
    query: str,
    *,
    top_k: int = 100,
    chunk_top_k: int = 2000,
) -> List[Tuple[int, float]]:
    query_terms = set(tokenize(query))
    if not query_terms or index.page_ids.size == 0:
        return []

    scores = np.zeros(index.page_ids.shape[0], dtype=np.float32)
    length_norm = index.k1 * (
        1.0 - index.b + index.b * index.doc_lengths / index.avg_doc_length
    )

    for term in query_terms:
        term_row = index.terms.get(term)
        if term_row is None:
            continue
        offset, length, idf = term_row
        end = offset + length
        doc_rows = index.doc_indices[offset:end]
        term_freqs = index.term_freqs[offset:end]
        scores[doc_rows] += idf * (
            term_freqs * (index.k1 + 1.0)
        ) / (term_freqs + length_norm[doc_rows])

    candidate_count = min(chunk_top_k, int(index.page_ids.shape[0]))
    if candidate_count <= 0 or float(scores.max(initial=0.0)) <= 0.0:
        return []

    if candidate_count < index.page_ids.shape[0]:
        top_indices = np.argpartition(-scores, candidate_count - 1)[:candidate_count]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
    else:
        top_indices = np.argsort(-scores)

    by_page: Dict[int, List[float]] = {}
    for idx in top_indices:
        idx_int = int(idx)
        score = float(scores[idx_int])
        if score <= 0.0:
            continue
        by_page.setdefault(int(index.page_ids[idx_int]), []).append(score)

    page_scores = {
        page_id: _chunk_page_score(page_scores)
        for page_id, page_scores in by_page.items()
    }
    ordered = sorted(
        page_scores,
        key=lambda page_id: (page_scores[page_id], -page_id),
        reverse=True,
    )
    return [(page_id, page_scores[page_id]) for page_id in ordered[:top_k]]


def rank_chunk_bm25_batch(
    index: ChunkBM25Index,
    queries: Sequence[str],
    *,
    top_k: int = 100,
    chunk_top_k: int = 2000,
) -> List[List[Tuple[int, float]]]:
    return [
        rank_chunk_bm25(index, query, top_k=top_k, chunk_top_k=chunk_top_k)
        for query in queries
    ]
