"""Query-time retrieval (timed portion includes query embedding)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from embed import embed_queries
from index import load_index
from utils import K_EVAL

DEFAULT_CHUNK_CANDIDATES = 300
TOP_CHUNKS_PER_PAGE = 3
MULTI_CHUNK_WEIGHT = 0.10

_INDEX_CACHE: Dict[Path, Tuple[Any, Dict[str, Any], Dict[int, Dict[str, Any]]]] = {}


def _load_cached_index(
    artifacts_dir: Optional[Path],
) -> Tuple[Any, Dict[str, Any], Dict[int, Dict[str, Any]]]:
    root = (artifacts_dir or Path(__file__).resolve().parent / "artifacts").resolve()
    if root not in _INDEX_CACHE:
        _INDEX_CACHE[root] = load_index(root)
    return _INDEX_CACHE[root]


def _page_score(scores: List[float]) -> float:
    ordered = sorted(scores, reverse=True)
    return ordered[0] + MULTI_CHUNK_WEIGHT * float(np.mean(ordered[:TOP_CHUNKS_PER_PAGE]))


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

    index, chunk_meta, _page_meta = _load_cached_index(artifacts_dir)
    query_vectors = np.ascontiguousarray(embed_queries(queries), dtype=np.float32)
    if query_vectors.size == 0:
        return [[] for _ in queries]

    candidate_count = min(DEFAULT_CHUNK_CANDIDATES, int(index.ntotal))
    scores, indices = index.search(query_vectors, candidate_count)
    page_ids = chunk_meta["page_ids"]

    ranked: List[List[int]] = []
    for row_scores, row_indices in zip(scores, indices):
        by_page: Dict[int, List[float]] = {}
        for score, idx in zip(row_scores, row_indices):
            if idx < 0:
                continue
            page_id = page_ids[int(idx)]
            by_page.setdefault(page_id, []).append(float(score))

        ordered_pages = sorted(
            by_page.items(),
            key=lambda item: (_page_score(item[1]), max(item[1]), -item[0]),
            reverse=True,
        )
        ranked.append([page_id for page_id, _scores in ordered_pages[:top_k]])
    return ranked
