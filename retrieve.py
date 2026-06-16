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
from field_lexical import FieldBM25Index, load_field_bm25_indexes, rank_field_bm25_batch
from index import load_index
from lexical import BM25Index, expand_query, load_bm25_index, rank_bm25_batch
from page_features import PageFeature, classify_query_type, load_page_features
from utils import ENTRIES_DIR, K_EVAL

DEFAULT_CHUNK_CANDIDATES = 2000
DEFAULT_PAGE_CANDIDATES = 100
DEFAULT_BM25_CANDIDATES = 100
DEFAULT_CHUNK_BM25_CANDIDATES = 100
DEFAULT_FIELD_BM25_CANDIDATES = 100
DEFAULT_RERANK_CANDIDATES = 100
DEFAULT_FIELD_BM25_SOURCES = {"title_lead"}
DEFAULT_FIELD_BM25_MAX_RANK = 50
TOP_CHUNKS_PER_PAGE = 3
MULTI_CHUNK_WEIGHT = 0.10

# Clean weighted fusion.  All positive weights represent evidence that should
# help a page: semantic similarity, lexical evidence, query/title overlap, and
# agreement between independent retrievers.
DENSE_SCORE_WEIGHT = 0.748
PAGE_BM25_SCORE_WEIGHT = 0.085
EXPANDED_BM25_SCORE_WEIGHT = 0.020
CHUNK_BM25_SCORE_WEIGHT = 0.410
CHUNK_DENSE_SCORE_WEIGHT = 0.019
FIELD_BM25_SCORE_WEIGHT = 0.163
TITLE_OVERLAP_WEIGHT = 0.007
RARE_TOKEN_WEIGHT = 0.032
PHRASE_MATCH_WEIGHT = 0.068
SOURCE_COUNT_WEIGHT = 0.058
DENSE_RANK_WEIGHT = 0.033
PAGE_BM25_RANK_WEIGHT = 0.010
EXPANDED_BM25_RANK_WEIGHT = 0.030
CHUNK_BM25_RANK_WEIGHT = 0.021
CHUNK_DENSE_RANK_WEIGHT = 0.029
FIELD_BM25_RANK_WEIGHT = 0.089
PAGE_TYPE_MATCH_WEIGHT = 0.046
GENERIC_PAGE_PENALTY_WEIGHT = 0.00
DIPLOMACY_FIELD_BM25_SCORE_WEIGHT = 0.06
DIPLOMACY_FIELD_BM25_RANK_WEIGHT = 0.02
EXPANDED_RARE_TOKEN_WEIGHT = 0.059
LEXICAL_RRF_CONSENSUS_WEIGHT = 0.003
SOURCE_RRF_CONSENSUS_WEIGHT = 0.006
TOP20_SOURCE_CONSENSUS_WEIGHT = 0.019
TOP50_SOURCE_CONSENSUS_WEIGHT = 0.003
CONSENSUS_RRF_K = 60

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
USE_FIELD_BM25_ENV = "WIKI_USE_FIELD_BM25"
USE_EXPANDED_BM25_ENV = "WIKI_USE_EXPANDED_BM25"
USE_TITLE_CHUNKS_ENV = "WIKI_USE_TITLE_CHUNKS"
USE_CHUNK_BM25_ENV = "WIKI_USE_CHUNK_BM25"
USE_RRF_ENV = "WIKI_USE_RRF"
SCORE_MODE_ENV = "WIKI_SCORE_MODE"
FIELD_BM25_SOURCES_ENV = "WIKI_FIELD_BM25_SOURCES"
FIELD_BM25_AS_CANDIDATE_ENV = "WIKI_FIELD_BM25_AS_CANDIDATE"
FIELD_BM25_AS_FEATURE_ENV = "WIKI_FIELD_BM25_AS_FEATURE"
FIELD_BM25_AS_SOURCE_ENV = "WIKI_FIELD_BM25_AS_SOURCE"
FIELD_BM25_AS_RRF_ENV = "WIKI_FIELD_BM25_AS_RRF"
FIELD_BM25_SCORE_WEIGHT_ENV = "WIKI_FIELD_BM25_SCORE_WEIGHT"
FIELD_BM25_RANK_WEIGHT_ENV = "WIKI_FIELD_BM25_RANK_WEIGHT"
FIELD_BM25_MAX_RANK_ENV = "WIKI_FIELD_BM25_MAX_RANK"
FIELD_BM25_SCORE_CAP_ENV = "WIKI_FIELD_BM25_SCORE_CAP"
FIELD_BM25_FEATURE_REQUIRE_OTHER_SOURCE_ENV = "WIKI_FIELD_BM25_FEATURE_REQUIRE_OTHER_SOURCE"
FIELD_BM25_FEATURE_MIN_SOURCE_COUNT_ENV = "WIKI_FIELD_BM25_FEATURE_MIN_SOURCE_COUNT"
FIELD_BM25_CANDIDATE_REQUIRE_SOURCE_ENV = "WIKI_FIELD_BM25_CANDIDATE_REQUIRE_SOURCE"
FIELD_BM25_CANDIDATE_REQUIRE_TOP_N_ENV = "WIKI_FIELD_BM25_CANDIDATE_REQUIRE_TOP_N"
RRF_K_ENV = "WIKI_RRF_K"
RRF_SCORE_WEIGHT_ENV = "WIKI_RRF_SCORE_WEIGHT"
RRF_PAGE_DENSE_WEIGHT_ENV = "WIKI_RRF_PAGE_DENSE_WEIGHT"
RRF_CHUNK_DENSE_WEIGHT_ENV = "WIKI_RRF_CHUNK_DENSE_WEIGHT"
RRF_PAGE_BM25_WEIGHT_ENV = "WIKI_RRF_PAGE_BM25_WEIGHT"
RRF_CHUNK_BM25_WEIGHT_ENV = "WIKI_RRF_CHUNK_BM25_WEIGHT"
RRF_EXPANDED_BM25_WEIGHT_ENV = "WIKI_RRF_EXPANDED_BM25_WEIGHT"
RRF_FIELD_BM25_WEIGHT_ENV = "WIKI_RRF_FIELD_BM25_WEIGHT"

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
_FIELD_BM25_CACHE: Dict[Path, Dict[str, FieldBM25Index]] = {}
_PAGE_TEXT_CACHE: Dict[int, Tuple[Counter[str], Counter[str], str]] = {}
_PAGE_FEATURE_CACHE: Dict[Path, Optional[Tuple[Dict[int, PageFeature], Dict[str, List[int]]]]] = {}
_OPTIONAL_CHUNK_INDEX_CACHE: Dict[Tuple[Path, str, str], Optional[Tuple[Any, List[int]]]] = {}


def _env_enabled(name: str, *, default: bool = True) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() not in {"0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def _env_csv_set(name: str) -> Optional[set[str]]:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return None
    values = {
        value.strip()
        for value in raw_value.split(",")
        if value.strip()
    }
    return values or set()


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


def _load_cached_field_bm25(
    artifacts_dir: Optional[Path],
) -> Dict[str, FieldBM25Index]:
    root = (artifacts_dir or Path(__file__).resolve().parent / "artifacts").resolve()
    if root not in _FIELD_BM25_CACHE:
        _FIELD_BM25_CACHE[root] = load_field_bm25_indexes(root)
    selected = _env_csv_set(FIELD_BM25_SOURCES_ENV)
    if selected is None:
        selected = DEFAULT_FIELD_BM25_SOURCES
    return {
        name: index
        for name, index in _FIELD_BM25_CACHE[root].items()
        if name in selected
    }


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


def _merge_ranked_sources(
    source_rows: List[List[Tuple[int, float]]],
    *,
    top_k: int,
    rrf_k: int = 60,
) -> List[Tuple[int, float]]:
    scores: Dict[int, float] = {}
    for row in source_rows:
        for rank, (page_id, _score) in enumerate(row, start=1):
            scores[page_id] = scores.get(page_id, 0.0) + 1.0 / (rrf_k + rank)
    ordered = sorted(scores.items(), key=lambda item: (item[1], -item[0]), reverse=True)
    return ordered[:top_k]


def _candidate_seed_ids(
    dense_ranked: List[Tuple[int, float]],
    bm25_ranked: List[Tuple[int, float]],
    expanded_bm25_ranked: List[Tuple[int, float]],
    chunk_bm25_ranked: List[Tuple[int, float]],
    field_bm25_ranked: List[Tuple[int, float]],
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
            + [page_id for page_id, _score in field_bm25_ranked[:candidate_count]]
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


def _rank_positions(ranked: List[Tuple[int, float]]) -> Dict[int, int]:
    return {page_id: rank for rank, (page_id, _score) in enumerate(ranked, start=1)}


def _key_rank_positions(page_ids: List[int]) -> Dict[int, int]:
    return {page_id: rank for rank, page_id in enumerate(page_ids, start=1)}


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


def _source_hit_counts(
    candidates: List[int],
    source_maps: List[Dict[int, float]],
) -> Dict[int, int]:
    available_sources = [source for source in source_maps if source]
    return {
        page_id: sum(1 for source in available_sources if page_id in source)
        for page_id in candidates
    }


def _top_ranked_set(ranked: List[Tuple[int, float]], top_n: int) -> set[int]:
    if top_n <= 0:
        return set()
    return {page_id for page_id, _score in ranked[:top_n]}


def _field_candidate_allowed_ids(
    dense_ranked: List[Tuple[int, float]],
    bm25_ranked: List[Tuple[int, float]],
    chunk_bm25_ranked: List[Tuple[int, float]],
    chunk_scores: Dict[int, float],
) -> Optional[set[int]]:
    required_sources = _env_csv_set(FIELD_BM25_CANDIDATE_REQUIRE_SOURCE_ENV)
    if not required_sources:
        return None

    top_n = max(0, _env_int(FIELD_BM25_CANDIDATE_REQUIRE_TOP_N_ENV, DEFAULT_RERANK_CANDIDATES))
    source_sets = {
        "dense": _top_ranked_set(dense_ranked, top_n),
        "page_dense": _top_ranked_set(dense_ranked, top_n),
        "bm25": _top_ranked_set(bm25_ranked, top_n),
        "page_bm25": _top_ranked_set(bm25_ranked, top_n),
        "chunk_bm25": _top_ranked_set(chunk_bm25_ranked, top_n),
        "chunk_dense": set(list(chunk_scores.keys())[:top_n]),
    }
    source_sets["any"] = (
        source_sets["dense"]
        | source_sets["page_bm25"]
        | source_sets["chunk_bm25"]
    )

    allowed: set[int] = set()
    for source_name in required_sources:
        allowed.update(source_sets.get(source_name, set()))
    return allowed


def _rrf_add_ranked(
    scores: Dict[int, float],
    ranked_ids: List[int],
    *,
    weight: float,
    rrf_k: int,
) -> None:
    if weight <= 0.0:
        return
    for rank, page_id in enumerate(ranked_ids, start=1):
        scores[page_id] = scores.get(page_id, 0.0) + weight / (rrf_k + rank)


def _rrf_scores(
    candidates: List[int],
    dense_ranked: List[Tuple[int, float]],
    bm25_ranked: List[Tuple[int, float]],
    expanded_bm25_ranked: List[Tuple[int, float]],
    chunk_bm25_ranked: List[Tuple[int, float]],
    field_bm25_ranked: List[Tuple[int, float]],
    chunk_scores: Dict[int, float],
) -> Dict[int, float]:
    rrf_k = max(1, _env_int(RRF_K_ENV, 60))
    raw_scores: Dict[int, float] = {}
    _rrf_add_ranked(
        raw_scores,
        [page_id for page_id, _score in dense_ranked],
        weight=_env_float(RRF_PAGE_DENSE_WEIGHT_ENV, 1.0),
        rrf_k=rrf_k,
    )
    _rrf_add_ranked(
        raw_scores,
        list(chunk_scores.keys()),
        weight=_env_float(RRF_CHUNK_DENSE_WEIGHT_ENV, 0.8),
        rrf_k=rrf_k,
    )
    _rrf_add_ranked(
        raw_scores,
        [page_id for page_id, _score in bm25_ranked],
        weight=_env_float(RRF_PAGE_BM25_WEIGHT_ENV, 1.2),
        rrf_k=rrf_k,
    )
    _rrf_add_ranked(
        raw_scores,
        [page_id for page_id, _score in chunk_bm25_ranked],
        weight=_env_float(RRF_CHUNK_BM25_WEIGHT_ENV, 1.0),
        rrf_k=rrf_k,
    )
    _rrf_add_ranked(
        raw_scores,
        [page_id for page_id, _score in expanded_bm25_ranked],
        weight=_env_float(RRF_EXPANDED_BM25_WEIGHT_ENV, 0.5),
        rrf_k=rrf_k,
    )
    _rrf_add_ranked(
        raw_scores,
        [page_id for page_id, _score in field_bm25_ranked],
        weight=_env_float(RRF_FIELD_BM25_WEIGHT_ENV, 0.8),
        rrf_k=rrf_k,
    )
    return _normalize_scores(raw_scores, candidates)


def _consensus_rrf_scores(
    candidates: List[int],
    source_positions: List[Dict[int, int]],
    source_weights: List[float],
) -> Dict[int, float]:
    raw_scores: Dict[int, float] = {}
    for positions, weight in zip(source_positions, source_weights):
        if not positions or weight <= 0.0:
            continue
        for page_id in candidates:
            rank = positions.get(page_id)
            if rank is not None:
                raw_scores[page_id] = raw_scores.get(page_id, 0.0) + (
                    weight / (CONSENSUS_RRF_K + rank)
                )
    return _normalize_scores(raw_scores, candidates)


def _top_rank_consensus(
    candidates: List[int],
    source_positions: List[Dict[int, int]],
    *,
    top_n: int,
) -> Dict[int, float]:
    available_sources = [positions for positions in source_positions if positions]
    denominator = float(len(available_sources) or 1)
    return {
        page_id: sum(
            1
            for positions in available_sources
            if (rank := positions.get(page_id)) is not None and rank <= top_n
        ) / denominator
        for page_id in candidates
    }


def _field_bm25_weights(query_type: str) -> Tuple[float, float]:
    score_default = FIELD_BM25_SCORE_WEIGHT
    rank_default = FIELD_BM25_RANK_WEIGHT
    if query_type == "diplomacy":
        score_default = DIPLOMACY_FIELD_BM25_SCORE_WEIGHT
        rank_default = DIPLOMACY_FIELD_BM25_RANK_WEIGHT
    return (
        _env_float(FIELD_BM25_SCORE_WEIGHT_ENV, score_default),
        _env_float(FIELD_BM25_RANK_WEIGHT_ENV, rank_default),
    )


def _rerank_feature_rows(
    query_text: str,
    dense_ranked: List[Tuple[int, float]],
    bm25_ranked: List[Tuple[int, float]],
    expanded_bm25_ranked: List[Tuple[int, float]],
    chunk_bm25_ranked: List[Tuple[int, float]],
    field_bm25_ranked: List[Tuple[int, float]],
    chunk_scores: Dict[int, float],
    page_features: Optional[Dict[int, PageFeature]],
    top_k: int,
) -> List[Dict[str, Any]]:
    field_as_candidate = _env_enabled(FIELD_BM25_AS_CANDIDATE_ENV, default=True)
    field_as_feature = _env_enabled(FIELD_BM25_AS_FEATURE_ENV, default=True)
    field_as_source = _env_enabled(FIELD_BM25_AS_SOURCE_ENV, default=True)
    field_as_rrf = _env_enabled(FIELD_BM25_AS_RRF_ENV, default=True)
    field_max_rank = _env_int(FIELD_BM25_MAX_RANK_ENV, DEFAULT_FIELD_BM25_MAX_RANK)
    if field_max_rank > 0:
        field_evidence_ranked = field_bm25_ranked[:field_max_rank]
    else:
        field_evidence_ranked = field_bm25_ranked
    field_candidate_ranked = field_evidence_ranked
    field_candidate_allowed = _field_candidate_allowed_ids(
        dense_ranked,
        bm25_ranked,
        chunk_bm25_ranked,
        chunk_scores,
    )
    if field_candidate_allowed is not None:
        field_candidate_ranked = [
            (page_id, score)
            for page_id, score in field_candidate_ranked
            if page_id in field_candidate_allowed
        ]
    candidates = _candidate_seed_ids(
        dense_ranked,
        bm25_ranked,
        expanded_bm25_ranked,
        chunk_bm25_ranked,
        field_candidate_ranked if field_as_candidate else [],
        chunk_scores,
        DEFAULT_RERANK_CANDIDATES,
    )

    if not candidates:
        return []

    dense_scores = dict(dense_ranked)
    bm25_scores = dict(bm25_ranked)
    expanded_bm25_scores = dict(expanded_bm25_ranked)
    chunk_bm25_scores = dict(chunk_bm25_ranked)
    field_bm25_scores = dict(field_evidence_ranked)

    dense_norm = _normalize_scores(dense_scores, candidates)
    bm25_norm = _normalize_scores(bm25_scores, candidates)
    expanded_bm25_norm = _normalize_scores(expanded_bm25_scores, candidates)
    chunk_bm25_norm = _normalize_scores(chunk_bm25_scores, candidates)
    field_bm25_norm = _normalize_scores(field_bm25_scores, candidates)
    chunk_norm = _normalize_scores(chunk_scores, candidates)

    dense_rank = _inverse_rank_scores(dense_ranked)
    bm25_rank = _inverse_rank_scores(bm25_ranked)
    expanded_bm25_rank = _inverse_rank_scores(expanded_bm25_ranked)
    chunk_bm25_rank = _inverse_rank_scores(chunk_bm25_ranked)
    field_bm25_rank = _inverse_rank_scores(field_evidence_ranked)
    chunk_rank = _inverse_key_rank_scores(list(chunk_scores.keys()))
    dense_position = _rank_positions(dense_ranked)
    bm25_position = _rank_positions(bm25_ranked)
    expanded_bm25_position = _rank_positions(expanded_bm25_ranked)
    chunk_bm25_position = _rank_positions(chunk_bm25_ranked)
    field_bm25_position = _rank_positions(field_evidence_ranked)
    chunk_position = _key_rank_positions(list(chunk_scores.keys()))
    lexical_consensus_positions = [
        bm25_position,
        chunk_bm25_position,
        field_bm25_position,
    ]
    source_consensus_positions = [
        dense_position,
        bm25_position,
        chunk_bm25_position,
        field_bm25_position,
        chunk_position,
    ]
    lexical_rrf_norm = _consensus_rrf_scores(
        candidates,
        lexical_consensus_positions,
        [1.0, 1.0, 0.7],
    )
    source_rrf_norm = _consensus_rrf_scores(
        candidates,
        source_consensus_positions,
        [0.8, 1.1, 1.0, 0.7, 0.6],
    )
    top20_source_consensus = _top_rank_consensus(
        candidates,
        source_consensus_positions,
        top_n=20,
    )
    top50_source_consensus = _top_rank_consensus(
        candidates,
        source_consensus_positions,
        top_n=50,
    )

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
            field_bm25_scores if field_as_source else {},
            chunk_scores,
        ],
    )
    other_source_counts = _source_hit_counts(
        candidates,
        [
            dense_scores,
            bm25_scores,
            expanded_bm25_scores,
            chunk_bm25_scores,
            chunk_scores,
        ],
    )
    all_source_counts = _source_hit_counts(
        candidates,
        [
            dense_scores,
            bm25_scores,
            expanded_bm25_scores,
            chunk_bm25_scores,
            field_bm25_scores,
            chunk_scores,
        ],
    )
    query_type = classify_query_type(query_text)
    score_mode = os.environ.get(SCORE_MODE_ENV, "weighted").strip().lower()
    rrf_norm = (
        _rrf_scores(
            candidates,
            dense_ranked,
            bm25_ranked,
            expanded_bm25_ranked,
            chunk_bm25_ranked,
            field_evidence_ranked if field_as_rrf else [],
            chunk_scores,
        )
        if _env_enabled(USE_RRF_ENV, default=False) or score_mode == "rrf"
        else {}
    )
    rrf_weight = _env_float(RRF_SCORE_WEIGHT_ENV, 1.0)
    field_score_weight, field_rank_weight = _field_bm25_weights(query_type)
    field_score_cap = min(1.0, max(0.0, _env_float(FIELD_BM25_SCORE_CAP_ENV, 1.0)))
    field_require_other = _env_enabled(
        FIELD_BM25_FEATURE_REQUIRE_OTHER_SOURCE_ENV,
        default=False,
    )
    field_min_source_count = max(0, _env_int(FIELD_BM25_FEATURE_MIN_SOURCE_COUNT_ENV, 0))

    scored: List[Tuple[float, int, int, Dict[str, Any]]] = []
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
        field_feature_allowed = False
        field_bm25_feature = 0.0
        field_rank_feature = 0.0
        if score_mode == "rrf":
            score = rrf_norm.get(page_id, 0.0)
        else:
            field_feature_allowed = field_as_feature
            if field_require_other and other_source_counts.get(page_id, 0) < 1:
                field_feature_allowed = False
            if field_min_source_count > 0 and all_source_counts.get(page_id, 0) < field_min_source_count:
                field_feature_allowed = False
            field_bm25_feature = (
                min(field_bm25_norm.get(page_id, 0.0), field_score_cap)
                if field_feature_allowed
                else 0.0
            )
            field_rank_feature = (
                field_bm25_rank.get(page_id, 0.0)
                if field_feature_allowed
                else 0.0
            )
            score = (
                DENSE_SCORE_WEIGHT * dense_norm.get(page_id, 0.0)
                + PAGE_BM25_SCORE_WEIGHT * bm25_norm.get(page_id, 0.0)
                + EXPANDED_BM25_SCORE_WEIGHT * expanded_bm25_norm.get(page_id, 0.0)
                + CHUNK_BM25_SCORE_WEIGHT * chunk_bm25_norm.get(page_id, 0.0)
                + field_score_weight * field_bm25_feature
                + CHUNK_DENSE_SCORE_WEIGHT * chunk_norm.get(page_id, 0.0)
                + DENSE_RANK_WEIGHT * dense_rank.get(page_id, 0.0)
                + PAGE_BM25_RANK_WEIGHT * bm25_rank.get(page_id, 0.0)
                + EXPANDED_BM25_RANK_WEIGHT * expanded_bm25_rank.get(page_id, 0.0)
                + CHUNK_BM25_RANK_WEIGHT * chunk_bm25_rank.get(page_id, 0.0)
                + field_rank_weight * field_rank_feature
                + CHUNK_DENSE_RANK_WEIGHT * chunk_rank.get(page_id, 0.0)
                + RARE_TOKEN_WEIGHT * rare_norm.get(page_id, 0.0)
                + EXPANDED_RARE_TOKEN_WEIGHT * expanded_rare_norm.get(page_id, 0.0)
                + TITLE_OVERLAP_WEIGHT * title_norm.get(page_id, 0.0)
                + PHRASE_MATCH_WEIGHT * phrase_norm.get(page_id, 0.0)
                + SOURCE_COUNT_WEIGHT * source_count.get(page_id, 0.0)
                + LEXICAL_RRF_CONSENSUS_WEIGHT * lexical_rrf_norm.get(page_id, 0.0)
                + SOURCE_RRF_CONSENSUS_WEIGHT * source_rrf_norm.get(page_id, 0.0)
                + TOP20_SOURCE_CONSENSUS_WEIGHT * top20_source_consensus.get(page_id, 0.0)
                + TOP50_SOURCE_CONSENSUS_WEIGHT * top50_source_consensus.get(page_id, 0.0)
                + PAGE_TYPE_MATCH_WEIGHT * type_match
                - GENERIC_PAGE_PENALTY_WEIGHT * generic_penalty
                + rrf_weight * rrf_norm.get(page_id, 0.0)
            )
        row: Dict[str, Any] = {
            "page_id": page_id,
            "candidate_rank": rank + 1,
            "score": score,
            "query_type": query_type,
            "dense_score": dense_scores.get(page_id, 0.0),
            "dense_norm": dense_norm.get(page_id, 0.0),
            "dense_rank": dense_position.get(page_id),
            "page_bm25_score": bm25_scores.get(page_id, 0.0),
            "page_bm25_norm": bm25_norm.get(page_id, 0.0),
            "page_bm25_rank": bm25_position.get(page_id),
            "expanded_bm25_score": expanded_bm25_scores.get(page_id, 0.0),
            "expanded_bm25_norm": expanded_bm25_norm.get(page_id, 0.0),
            "expanded_bm25_rank": expanded_bm25_position.get(page_id),
            "chunk_bm25_score": chunk_bm25_scores.get(page_id, 0.0),
            "chunk_bm25_norm": chunk_bm25_norm.get(page_id, 0.0),
            "chunk_bm25_rank": chunk_bm25_position.get(page_id),
            "field_bm25_score": field_bm25_scores.get(page_id, 0.0),
            "field_bm25_norm": field_bm25_norm.get(page_id, 0.0),
            "field_bm25_rank": field_bm25_position.get(page_id),
            "field_bm25_feature": field_bm25_feature,
            "field_rank_feature": field_rank_feature,
            "field_feature_allowed": field_feature_allowed,
            "chunk_dense_score": chunk_scores.get(page_id, 0.0),
            "chunk_dense_norm": chunk_norm.get(page_id, 0.0),
            "chunk_dense_rank": chunk_position.get(page_id),
            "rare_norm": rare_norm.get(page_id, 0.0),
            "expanded_rare_norm": expanded_rare_norm.get(page_id, 0.0),
            "title_norm": title_norm.get(page_id, 0.0),
            "phrase_norm": phrase_norm.get(page_id, 0.0),
            "source_count": source_count.get(page_id, 0.0),
            "lexical_rrf_norm": lexical_rrf_norm.get(page_id, 0.0),
            "source_rrf_norm": source_rrf_norm.get(page_id, 0.0),
            "top20_source_consensus": top20_source_consensus.get(page_id, 0.0),
            "top50_source_consensus": top50_source_consensus.get(page_id, 0.0),
            "other_source_count": other_source_counts.get(page_id, 0),
            "all_source_count": all_source_counts.get(page_id, 0),
            "type_match": type_match,
            "generic_penalty": generic_penalty,
            "rrf_norm": rrf_norm.get(page_id, 0.0),
        }
        scored.append((score, -rank, page_id, row))

    scored.sort(key=lambda item: item[:3], reverse=True)
    rows: List[Dict[str, Any]] = []
    for final_rank, (_score, _rank, _page_id, row) in enumerate(scored, start=1):
        row["final_rank"] = final_rank
        rows.append(row)
    return rows


def _rerank_feature_union(
    query_text: str,
    dense_ranked: List[Tuple[int, float]],
    bm25_ranked: List[Tuple[int, float]],
    expanded_bm25_ranked: List[Tuple[int, float]],
    chunk_bm25_ranked: List[Tuple[int, float]],
    field_bm25_ranked: List[Tuple[int, float]],
    chunk_scores: Dict[int, float],
    page_features: Optional[Dict[int, PageFeature]],
    top_k: int,
) -> List[int]:
    rows = _rerank_feature_rows(
        query_text,
        dense_ranked,
        bm25_ranked,
        expanded_bm25_ranked,
        chunk_bm25_ranked,
        field_bm25_ranked,
        chunk_scores,
        page_features,
        top_k,
    )
    return [int(row["page_id"]) for row in rows[:top_k]]


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
    use_field_bm25 = _env_enabled(USE_FIELD_BM25_ENV)
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
    field_bm25_indexes = (
        _load_cached_field_bm25(artifacts_dir)
        if use_field_bm25
        else {}
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
    field_bm25_batches = [
        rank_field_bm25_batch(
            index,
            queries,
            top_k=DEFAULT_FIELD_BM25_CANDIDATES,
        )
        for index in field_bm25_indexes.values()
    ]
    field_bm25_ranked_batch = [
        _merge_ranked_sources(
            [batch[query_idx] for batch in field_bm25_batches],
            top_k=DEFAULT_FIELD_BM25_CANDIDATES,
        )
        for query_idx in range(len(queries))
    ] if field_bm25_batches else [[] for _query in queries]
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
                field_bm25_ranked_batch[query_idx],
                chunk_score_batch[query_idx],
                page_features,
                top_k,
            )
        )
    return ranked


def debug_search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
    candidate_limit: Optional[int] = None,
) -> List[List[Dict[str, Any]]]:
    """Return reranker feature rows for diagnostics; not used by grading."""
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
    use_field_bm25 = _env_enabled(USE_FIELD_BM25_ENV)
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
    field_bm25_indexes = (
        _load_cached_field_bm25(artifacts_dir)
        if use_field_bm25
        else {}
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
    field_bm25_batches = [
        rank_field_bm25_batch(
            field_index,
            queries,
            top_k=DEFAULT_FIELD_BM25_CANDIDATES,
        )
        for field_index in field_bm25_indexes.values()
    ]
    field_bm25_ranked_batch = [
        _merge_ranked_sources(
            [batch[query_idx] for batch in field_bm25_batches],
            top_k=DEFAULT_FIELD_BM25_CANDIDATES,
        )
        for query_idx in range(len(queries))
    ] if field_bm25_batches else [[] for _query in queries]
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

    debug_rows: List[List[Dict[str, Any]]] = []
    for query_idx, (query_text, query_vector) in enumerate(zip(queries, query_vectors)):
        dense_ranked = (
            _rank_page_vectors(query_vector, page_vectors, dense_page_ids)
            if use_page_dense
            else []
        )
        rows = _rerank_feature_rows(
            query_text,
            dense_ranked,
            bm25_ranked_batch[query_idx],
            expanded_bm25_ranked_batch[query_idx],
            chunk_bm25_ranked_batch[query_idx],
            field_bm25_ranked_batch[query_idx],
            chunk_score_batch[query_idx],
            page_features,
            top_k,
        )
        debug_rows.append(rows[:candidate_limit] if candidate_limit is not None else rows)
    return debug_rows
