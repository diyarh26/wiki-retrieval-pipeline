"""Query-time retrieval (timed portion includes query embedding)."""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from embed import embed_queries
from index import load_index
from utils import K_EVAL

DEFAULT_CHUNK_CANDIDATES = 300
DEFAULT_PAGE_CANDIDATES = 200
TOP_CHUNKS_PER_PAGE = 3
MULTI_CHUNK_WEIGHT = 0.10
RRF_K = 60.0
CHUNK_RRF_WEIGHT = 1.00
PAGE_RRF_WEIGHT = 1.15
TITLE_TOKEN_WEIGHT = 0.006
TITLE_COVERAGE_WEIGHT = 0.015
TITLE_PHRASE_WEIGHT = 0.020
PAGE_VECTORS_NAME = "index_vectors.npy"
PAGE_META_NAME = "index_meta.json"

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "before",
    "between",
    "did",
    "does",
    "during",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "into",
    "its",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whose",
    "with",
}

_INDEX_CACHE: Dict[
    Path,
    Tuple[
        Any,
        Dict[str, Any],
        Dict[int, Dict[str, Any]],
        Optional[np.ndarray],
        Optional[List[int]],
    ],
] = {}


def _load_cached_index(
    artifacts_dir: Optional[Path],
) -> Tuple[
    Any,
    Dict[str, Any],
    Dict[int, Dict[str, Any]],
    Optional[np.ndarray],
    Optional[List[int]],
]:
    root = (artifacts_dir or Path(__file__).resolve().parent / "artifacts").resolve()
    if root not in _INDEX_CACHE:
        index, chunk_meta, page_meta = load_index(root)
        page_vectors, page_ids = _try_load_page_vectors(root)
        _INDEX_CACHE[root] = (index, chunk_meta, page_meta, page_vectors, page_ids)
    return _INDEX_CACHE[root]


def _page_score(scores: List[float]) -> float:
    ordered = sorted(scores, reverse=True)
    return ordered[0] + MULTI_CHUNK_WEIGHT * float(np.mean(ordered[:TOP_CHUNKS_PER_PAGE]))


def _try_load_page_vectors(root: Path) -> Tuple[Optional[np.ndarray], Optional[List[int]]]:
    vectors_path = root / PAGE_VECTORS_NAME
    meta_path = root / PAGE_META_NAME
    if not vectors_path.exists() or not meta_path.exists():
        return None, None
    if vectors_path.stat().st_size < 1024:
        return None, None

    try:
        vectors = np.load(vectors_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None, None

    page_ids = [int(x) for x in meta.get("page_ids", [])]
    if vectors.ndim != 2 or len(page_ids) != vectors.shape[0]:
        return None, None
    return np.ascontiguousarray(vectors, dtype=np.float32), page_ids


def _tokens(text: str) -> List[str]:
    return [
        token
        for token in _TOKEN_RE.findall(str(text).lower())
        if len(token) > 2 and token not in _STOPWORDS
    ]


def _title_boost(query_tokens: set[str], query_text: str, title: str) -> float:
    title_tokens = _tokens(title)
    if not query_tokens or not title_tokens:
        return 0.0

    title_set = set(title_tokens)
    overlap = query_tokens & title_set
    if not overlap:
        return 0.0

    coverage = len(overlap) / len(title_set)
    boost = TITLE_TOKEN_WEIGHT * len(overlap) + TITLE_COVERAGE_WEIGHT * coverage
    if " ".join(title_tokens) in query_text.lower():
        boost += TITLE_PHRASE_WEIGHT
    return boost


def _rank_page_vectors(
    query_vector: np.ndarray,
    page_vectors: Optional[np.ndarray],
    page_ids: Optional[List[int]],
) -> List[Tuple[int, float]]:
    if page_vectors is None or page_ids is None or page_vectors.size == 0:
        return []

    scores = page_vectors @ query_vector
    candidate_count = min(DEFAULT_PAGE_CANDIDATES, len(page_ids))
    if candidate_count <= 0:
        return []

    if candidate_count < len(page_ids):
        top_indices = np.argpartition(-scores, candidate_count - 1)[:candidate_count]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
    else:
        top_indices = np.argsort(-scores)

    return [(page_ids[int(idx)], float(scores[int(idx)])) for idx in top_indices]


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:
    """
    Return ranked page_id lists (best first) for each query.

    Search chunk embeddings with FAISS, then aggregate chunk scores into
    page-level rankings because grading is by page_id.
    """
    if not queries:
        return []

    index, chunk_meta, page_meta, page_vectors, dense_page_ids = _load_cached_index(artifacts_dir)
    query_vectors = np.ascontiguousarray(embed_queries(queries), dtype=np.float32)
    if query_vectors.size == 0:
        return [[] for _ in queries]

    candidate_count = min(DEFAULT_CHUNK_CANDIDATES, int(index.ntotal))
    scores, indices = index.search(query_vectors, candidate_count)
    page_ids = chunk_meta["page_ids"]

    ranked: List[List[int]] = []
    for query_text, query_vector, row_scores, row_indices in zip(queries, query_vectors, scores, indices):
        by_page: Dict[int, List[float]] = {}
        for score, idx in zip(row_scores, row_indices):
            if idx < 0:
                continue
            page_id = page_ids[int(idx)]
            by_page.setdefault(page_id, []).append(float(score))

        chunk_ranked = sorted(
            by_page.items(),
            key=lambda item: (_page_score(item[1]), max(item[1]), -item[0]),
            reverse=True,
        )
        dense_ranked = _rank_page_vectors(query_vector, page_vectors, dense_page_ids)

        fused: Dict[int, Dict[str, float]] = {}
        for rank, (page_id, page_scores) in enumerate(chunk_ranked, start=1):
            row = fused.setdefault(page_id, {"score": 0.0, "raw": 0.0})
            raw_score = _page_score(page_scores)
            row["score"] += CHUNK_RRF_WEIGHT / (RRF_K + rank)
            row["raw"] = max(row["raw"], raw_score)

        for rank, (page_id, dense_score) in enumerate(dense_ranked, start=1):
            row = fused.setdefault(page_id, {"score": 0.0, "raw": 0.0})
            row["score"] += PAGE_RRF_WEIGHT / (RRF_K + rank)
            row["raw"] = max(row["raw"], dense_score)

        query_tokens = set(_tokens(query_text))
        for page_id, row in fused.items():
            title = page_meta.get(page_id, {}).get("title", "")
            row["score"] += _title_boost(query_tokens, query_text, title)

        ordered_pages = sorted(
            fused.items(),
            key=lambda item: (item[1]["score"], item[1]["raw"], -item[0]),
            reverse=True,
        )
        ranked.append([page_id for page_id, _row in ordered_pages[:top_k]])
    return ranked
