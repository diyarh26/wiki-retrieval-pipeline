"""Query-time retrieval (timed portion includes query embedding)."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from chunk_lexical import (
    ChunkBM25Index,
    load_chunk_bm25_index,
    rank_chunk_bm25_batch,
)
from embed import embed_queries
from index import load_index
from lexical import BM25Index, expand_query, load_bm25_index, rank_bm25_batch
from page_features import PageFeature, classify_query_type, load_page_features
from utils import ENTRIES_DIR, K_EVAL

DEFAULT_CHUNK_CANDIDATES = 2000
DEFAULT_PAGE_CANDIDATES = 100
DEFAULT_BM25_CANDIDATES = 100
DEFAULT_CHUNK_BM25_CANDIDATES = 100
DEFAULT_RERANK_CANDIDATES = 100
TOP_CHUNKS_PER_PAGE = 3
MULTI_CHUNK_WEIGHT = 0.10

# Clean weighted fusion.  All positive weights represent evidence that should
# help a page: semantic similarity, lexical evidence, query/title overlap, and
# agreement between independent retrievers.
DENSE_SCORE_WEIGHT = 0.45
PAGE_BM25_SCORE_WEIGHT = 0.30
EXPANDED_BM25_SCORE_WEIGHT = 0.05
CHUNK_BM25_SCORE_WEIGHT = 0.16
CHUNK_DENSE_SCORE_WEIGHT = 0.08
TITLE_OVERLAP_WEIGHT = 0.03
RARE_TOKEN_WEIGHT = 0.03
PHRASE_MATCH_WEIGHT = 0.01
SOURCE_COUNT_WEIGHT = 0.03
DENSE_RANK_WEIGHT = 0.03
PAGE_BM25_RANK_WEIGHT = 0.03
EXPANDED_BM25_RANK_WEIGHT = 0.01
CHUNK_BM25_RANK_WEIGHT = 0.02
CHUNK_DENSE_RANK_WEIGHT = 0.02
PAGE_TYPE_MATCH_WEIGHT = 0.02
GENERIC_PAGE_PENALTY_WEIGHT = 0.00

TITLE_TOKEN_WEIGHT = 0.006
TITLE_COVERAGE_WEIGHT = 0.015
TITLE_PHRASE_WEIGHT = 0.020
PAGE_VECTORS_NAME = "index_vectors.npy"
PAGE_META_NAME = "index_meta.json"
TITLE_CHUNK_INDEX_NAME = "faiss_title190.index"
TITLE_CHUNK_META_NAME = "chunk_meta_title190.json"
USE_PAGE_DENSE_ENV = "WIKI_USE_PAGE_DENSE"
USE_CHUNK_DENSE_ENV = "WIKI_USE_CHUNK_DENSE"
USE_PAGE_BM25_ENV = "WIKI_USE_PAGE_BM25"
USE_EXPANDED_BM25_ENV = "WIKI_USE_EXPANDED_BM25"
USE_TITLE_CHUNKS_ENV = "WIKI_USE_TITLE_CHUNKS"
USE_CHUNK_BM25_ENV = "WIKI_USE_CHUNK_BM25"

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
_BM25_CACHE: Dict[Path, Optional[BM25Index]] = {}
_CHUNK_BM25_CACHE: Dict[Path, Optional[ChunkBM25Index]] = {}
_PAGE_TEXT_CACHE: Dict[int, Tuple[Counter[str], Counter[str], str]] = {}
_PAGE_FEATURE_CACHE: Dict[Path, Optional[Tuple[Dict[int, PageFeature], Dict[str, List[int]]]]] = {}
_OPTIONAL_CHUNK_INDEX_CACHE: Dict[Tuple[Path, str, str], Optional[Tuple[Any, List[int]]]] = {}


def _env_enabled(name: str, *, default: bool = True) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() not in {"0", "false", "no", "off"}


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


def _load_cached_bm25(artifacts_dir: Optional[Path]) -> Optional[BM25Index]:
    root = (artifacts_dir or Path(__file__).resolve().parent / "artifacts").resolve()
    if root not in _BM25_CACHE:
        _BM25_CACHE[root] = load_bm25_index(root)
    return _BM25_CACHE[root]


def _load_cached_chunk_bm25(artifacts_dir: Optional[Path]) -> Optional[ChunkBM25Index]:
    root = (artifacts_dir or Path(__file__).resolve().parent / "artifacts").resolve()
    if root not in _CHUNK_BM25_CACHE:
        _CHUNK_BM25_CACHE[root] = load_chunk_bm25_index(root)
    return _CHUNK_BM25_CACHE[root]


def _load_cached_page_features(
    artifacts_dir: Optional[Path],
) -> Optional[Tuple[Dict[int, PageFeature], Dict[str, List[int]]]]:
    root = (artifacts_dir or Path(__file__).resolve().parent / "artifacts").resolve()
    if root not in _PAGE_FEATURE_CACHE:
        _PAGE_FEATURE_CACHE[root] = load_page_features(root)
    return _PAGE_FEATURE_CACHE[root]


def _load_optional_chunk_index(
    artifacts_dir: Optional[Path],
    *,
    index_name: str,
    meta_name: str,
) -> Optional[Tuple[Any, List[int]]]:
    root = (artifacts_dir or Path(__file__).resolve().parent / "artifacts").resolve()
    cache_key = (root, index_name, meta_name)
    if cache_key in _OPTIONAL_CHUNK_INDEX_CACHE:
        return _OPTIONAL_CHUNK_INDEX_CACHE[cache_key]

    index_path = root / index_name
    meta_path = root / meta_name
    if not index_path.exists() or not meta_path.exists():
        _OPTIONAL_CHUNK_INDEX_CACHE[cache_key] = None
        return None

    try:
        index = faiss.read_index(str(index_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        page_ids = [int(page_id) for page_id in meta["page_ids"]]
    except Exception:
        _OPTIONAL_CHUNK_INDEX_CACHE[cache_key] = None
        return None

    _OPTIONAL_CHUNK_INDEX_CACHE[cache_key] = (index, page_ids)
    return _OPTIONAL_CHUNK_INDEX_CACHE[cache_key]


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


def _load_page_text_features(page_id: int) -> Tuple[Counter[str], Counter[str], str]:
    if page_id not in _PAGE_TEXT_CACHE:
        try:
            record = json.loads((ENTRIES_DIR / f"{page_id}.json").read_text(encoding="utf-8"))
        except Exception:
            _PAGE_TEXT_CACHE[page_id] = (Counter(), Counter(), "")
        else:
            title = str(record.get("title", ""))
            content = str(record.get("content", "") or "")
            _PAGE_TEXT_CACHE[page_id] = (
                Counter(_tokens(title)),
                Counter(_tokens(content)),
                f"{title} {content}".lower(),
            )
    return _PAGE_TEXT_CACHE[page_id]


def _text_feature_scores(
    query_text: str,
    expanded_query_text: str,
    page_id: int,
) -> Tuple[float, float, float, float]:
    title_tokens, content_tokens, full_text = _load_page_text_features(page_id)
    query_tokens = set(_tokens(query_text))
    expanded_tokens = set(_tokens(expanded_query_text))
    rare_tokens = [token for token in query_tokens if len(token) > 5]
    expanded_rare_tokens = [token for token in expanded_tokens if len(token) > 5]

    rare_score = 0.0
    if rare_tokens:
        coverage = sum(
            1
            for token in rare_tokens
            if title_tokens.get(token, 0) > 0 or content_tokens.get(token, 0) > 0
        ) / len(rare_tokens)
        frequency = sum(math.log1p(content_tokens.get(token, 0)) for token in rare_tokens)
        rare_score = coverage + 0.05 * frequency

    expanded_rare_score = 0.0
    if expanded_rare_tokens:
        coverage = sum(
            1
            for token in expanded_rare_tokens
            if title_tokens.get(token, 0) > 0 or content_tokens.get(token, 0) > 0
        ) / len(expanded_rare_tokens)
        frequency = sum(
            math.log1p(content_tokens.get(token, 0)) for token in expanded_rare_tokens
        )
        expanded_rare_score = coverage + 0.03 * frequency

    title_overlap = 0.0
    if title_tokens:
        title_overlap = sum(
            1 for token in query_tokens if title_tokens.get(token, 0) > 0
        ) / len(title_tokens)

    phrase_hits = 0
    expanded_list = _tokens(expanded_query_text)
    for ngram_size in (2, 3, 4):
        for start in range(max(0, len(expanded_list) - ngram_size + 1)):
            if " ".join(expanded_list[start : start + ngram_size]) in full_text:
                phrase_hits += 1

    return rare_score, expanded_rare_score, title_overlap, min(phrase_hits, 8) / 8.0


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


def _rank_chunk_vectors(
    index: Any,
    query_vectors: np.ndarray,
    chunk_page_ids: List[int],
) -> List[Dict[int, float]]:
    if int(index.ntotal) <= 0:
        return [{} for _query in query_vectors]

    candidate_count = min(DEFAULT_CHUNK_CANDIDATES, int(index.ntotal))
    scores, indices = index.search(query_vectors, candidate_count)

    rows: List[Dict[int, float]] = []
    for row_scores, row_indices in zip(scores, indices):
        by_page: Dict[int, List[float]] = {}
        for score, idx in zip(row_scores, row_indices):
            if idx < 0:
                continue
            page_id = chunk_page_ids[int(idx)]
            by_page.setdefault(page_id, []).append(float(score))

        page_scores = {
            page_id: _page_score(page_scores)
            for page_id, page_scores in by_page.items()
        }
        ordered = sorted(
            page_scores,
            key=lambda page_id: (page_scores[page_id], -page_id),
            reverse=True,
        )
        rows.append(
            {
                page_id: page_scores[page_id]
                for page_id in ordered[:DEFAULT_RERANK_CANDIDATES]
            }
        )
    return rows


def _normalize_scores(
    score_map: Dict[int, float],
    candidates: List[int],
) -> Dict[int, float]:
    if not candidates:
        return {}
    values = [score_map.get(page_id, 0.0) for page_id in candidates]
    low = min(values)
    high = max(values)
    span = high - low
    if span <= 0.0:
        return {page_id: 0.0 for page_id in candidates}
    return {
        page_id: (score_map.get(page_id, 0.0) - low) / span
        for page_id in candidates
    }


def _merge_chunk_score_maps(
    primary: Dict[int, float],
    secondary: Optional[Dict[int, float]],
) -> Dict[int, float]:
    if not secondary:
        return primary

    merged = dict(primary)
    for page_id, score in secondary.items():
        merged[page_id] = max(merged.get(page_id, -1.0), score)

    ordered = sorted(
        merged,
        key=lambda page_id: (merged[page_id], -page_id),
        reverse=True,
    )
    return {
        page_id: merged[page_id]
        for page_id in ordered[: DEFAULT_RERANK_CANDIDATES * 2]
    }


def _candidate_seed_ids(
    dense_ranked: List[Tuple[int, float]],
    bm25_ranked: List[Tuple[int, float]],
    expanded_bm25_ranked: List[Tuple[int, float]],
    chunk_bm25_ranked: List[Tuple[int, float]],
    chunk_scores: Dict[int, float],
    candidate_count: int,
) -> List[int]:
    return list(
        dict.fromkeys(
            [page_id for page_id, _score in dense_ranked[:candidate_count]]
            + [page_id for page_id, _score in bm25_ranked[:candidate_count]]
            + [
                page_id
                for page_id, _score in expanded_bm25_ranked[:candidate_count]
            ]
            + [page_id for page_id, _score in chunk_bm25_ranked[:candidate_count]]
            + list(chunk_scores.keys())[:candidate_count]
        )
    )


def _inverse_rank_scores(ranked: List[Tuple[int, float]]) -> Dict[int, float]:
    return {
        page_id: 1.0 / (rank + 1.0)
        for rank, (page_id, _score) in enumerate(ranked)
    }


def _inverse_key_rank_scores(page_ids: List[int]) -> Dict[int, float]:
    return {page_id: 1.0 / (rank + 1.0) for rank, page_id in enumerate(page_ids)}


def _source_agreement(
    candidates: List[int],
    source_maps: List[Dict[int, float]],
) -> Dict[int, float]:
    available_sources = [source for source in source_maps if source]
    denominator = float(len(available_sources) or 1)
    return {
        page_id: sum(1 for source in available_sources if page_id in source) / denominator
        for page_id in candidates
    }


def _rerank_feature_union(
    query_text: str,
    dense_ranked: List[Tuple[int, float]],
    bm25_ranked: List[Tuple[int, float]],
    expanded_bm25_ranked: List[Tuple[int, float]],
    chunk_bm25_ranked: List[Tuple[int, float]],
    chunk_scores: Dict[int, float],
    page_features: Optional[Dict[int, PageFeature]],
    top_k: int,
) -> List[int]:
    candidates = _candidate_seed_ids(
        dense_ranked,
        bm25_ranked,
        expanded_bm25_ranked,
        chunk_bm25_ranked,
        chunk_scores,
        DEFAULT_RERANK_CANDIDATES,
    )

    if not candidates:
        return []

    dense_scores = dict(dense_ranked)
    bm25_scores = dict(bm25_ranked)
    expanded_bm25_scores = dict(expanded_bm25_ranked)
    chunk_bm25_scores = dict(chunk_bm25_ranked)

    dense_norm = _normalize_scores(dense_scores, candidates)
    bm25_norm = _normalize_scores(bm25_scores, candidates)
    expanded_bm25_norm = _normalize_scores(expanded_bm25_scores, candidates)
    chunk_bm25_norm = _normalize_scores(chunk_bm25_scores, candidates)
    chunk_norm = _normalize_scores(chunk_scores, candidates)

    dense_rank = _inverse_rank_scores(dense_ranked)
    bm25_rank = _inverse_rank_scores(bm25_ranked)
    expanded_bm25_rank = _inverse_rank_scores(expanded_bm25_ranked)
    chunk_bm25_rank = _inverse_rank_scores(chunk_bm25_ranked)
    chunk_rank = _inverse_key_rank_scores(list(chunk_scores.keys()))

    expanded_query_text = expand_query(query_text)
    raw_text_scores = {
        page_id: _text_feature_scores(query_text, expanded_query_text, page_id)
        for page_id in candidates
    }
    rare_norm = _normalize_scores(
        {page_id: row[0] for page_id, row in raw_text_scores.items()},
        candidates,
    )
    expanded_rare_norm = _normalize_scores(
        {page_id: row[1] for page_id, row in raw_text_scores.items()},
        candidates,
    )
    title_norm = _normalize_scores(
        {page_id: row[2] for page_id, row in raw_text_scores.items()},
        candidates,
    )
    phrase_norm = _normalize_scores(
        {page_id: row[3] for page_id, row in raw_text_scores.items()},
        candidates,
    )

    source_count = _source_agreement(
        candidates,
        [
            dense_scores,
            bm25_scores,
            expanded_bm25_scores,
            chunk_bm25_scores,
            chunk_scores,
        ],
    )
    query_type = classify_query_type(query_text) if page_features else "generic"

    scored: List[Tuple[float, int, int]] = []
    for rank, page_id in enumerate(candidates):
        feature = page_features.get(page_id) if page_features else None
        type_match = (
            1.0
            if feature is not None
            and query_type != "generic"
            and feature.page_type == query_type
            else 0.0
        )
        generic_penalty = feature.generic_penalty if feature is not None else 0.0
        score = (
            DENSE_SCORE_WEIGHT * dense_norm.get(page_id, 0.0)
            + PAGE_BM25_SCORE_WEIGHT * bm25_norm.get(page_id, 0.0)
            + EXPANDED_BM25_SCORE_WEIGHT * expanded_bm25_norm.get(page_id, 0.0)
            + CHUNK_BM25_SCORE_WEIGHT * chunk_bm25_norm.get(page_id, 0.0)
            + CHUNK_DENSE_SCORE_WEIGHT * chunk_norm.get(page_id, 0.0)
            + DENSE_RANK_WEIGHT * dense_rank.get(page_id, 0.0)
            + PAGE_BM25_RANK_WEIGHT * bm25_rank.get(page_id, 0.0)
            + EXPANDED_BM25_RANK_WEIGHT * expanded_bm25_rank.get(page_id, 0.0)
            + CHUNK_BM25_RANK_WEIGHT * chunk_bm25_rank.get(page_id, 0.0)
            + CHUNK_DENSE_RANK_WEIGHT * chunk_rank.get(page_id, 0.0)
            + RARE_TOKEN_WEIGHT * rare_norm.get(page_id, 0.0)
            + 0.02 * expanded_rare_norm.get(page_id, 0.0)
            + TITLE_OVERLAP_WEIGHT * title_norm.get(page_id, 0.0)
            + PHRASE_MATCH_WEIGHT * phrase_norm.get(page_id, 0.0)
            + SOURCE_COUNT_WEIGHT * source_count.get(page_id, 0.0)
            + PAGE_TYPE_MATCH_WEIGHT * type_match
            - GENERIC_PAGE_PENALTY_WEIGHT * generic_penalty
        )
        scored.append((score, -rank, page_id))

    scored.sort(reverse=True)
    return [page_id for _score, _rank, page_id in scored[:top_k]]


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

    page_ids = chunk_meta["page_ids"]
    use_page_dense = _env_enabled(USE_PAGE_DENSE_ENV)
    use_chunk_dense = _env_enabled(USE_CHUNK_DENSE_ENV)
    use_page_bm25 = _env_enabled(USE_PAGE_BM25_ENV)
    use_expanded_bm25 = _env_enabled(USE_EXPANDED_BM25_ENV)
    use_chunk_bm25 = _env_enabled(USE_CHUNK_BM25_ENV)

    bm25_index = (
        _load_cached_bm25(artifacts_dir)
        if use_page_bm25 or use_expanded_bm25
        else None
    )
    chunk_bm25_index = (
        _load_cached_chunk_bm25(artifacts_dir)
        if use_chunk_bm25
        else None
    )
    bm25_ranked_batch = (
        rank_bm25_batch(bm25_index, queries, top_k=DEFAULT_BM25_CANDIDATES)
        if bm25_index is not None and use_page_bm25
        else [[] for _query in queries]
    )
    expanded_queries = [expand_query(query) for query in queries]
    has_expanded_query = any(
        expanded_query != str(query)
        for expanded_query, query in zip(expanded_queries, queries)
    )
    if bm25_index is not None and use_expanded_bm25 and has_expanded_query:
        expanded_bm25_ranked_batch = rank_bm25_batch(
            bm25_index,
            expanded_queries,
            top_k=DEFAULT_BM25_CANDIDATES,
        )
    else:
        expanded_bm25_ranked_batch = [[] for _query in queries]

    chunk_bm25_ranked_batch = (
        rank_chunk_bm25_batch(
            chunk_bm25_index,
            expanded_queries,
            top_k=DEFAULT_CHUNK_BM25_CANDIDATES,
        )
        if chunk_bm25_index is not None
        else [[] for _query in queries]
    )
    chunk_score_batch = (
        _rank_chunk_vectors(index, query_vectors, page_ids)
        if use_chunk_dense
        else [{} for _query in queries]
    )
    optional_title_index = (
        _load_optional_chunk_index(
            artifacts_dir,
            index_name=TITLE_CHUNK_INDEX_NAME,
            meta_name=TITLE_CHUNK_META_NAME,
        )
        if _env_enabled(USE_TITLE_CHUNKS_ENV, default=False)
        else None
    )
    if use_chunk_dense and optional_title_index is not None:
        title_index, title_page_ids = optional_title_index
        title_chunk_score_batch = _rank_chunk_vectors(
            title_index,
            query_vectors,
            title_page_ids,
        )
        chunk_score_batch = [
            _merge_chunk_score_maps(primary, secondary)
            for primary, secondary in zip(chunk_score_batch, title_chunk_score_batch)
        ]
    feature_cache = _load_cached_page_features(artifacts_dir)
    page_features = feature_cache[0] if feature_cache is not None else None

    ranked: List[List[int]] = []
    for query_idx, (query_text, query_vector) in enumerate(zip(queries, query_vectors)):
        dense_ranked = (
            _rank_page_vectors(query_vector, page_vectors, dense_page_ids)
            if use_page_dense
            else []
        )
        ranked.append(
            _rerank_feature_union(
                query_text,
                dense_ranked,
                bm25_ranked_batch[query_idx],
                expanded_bm25_ranked_batch[query_idx],
                chunk_bm25_ranked_batch[query_idx],
                chunk_score_batch[query_idx],
                page_features,
                top_k,
            )
        )
    return ranked
