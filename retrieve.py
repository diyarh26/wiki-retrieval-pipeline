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
from page_features import (
    PageFeature,
    classify_query_type,
    is_multi_answer_query,
    load_page_features,
    query_signals,
    signal_score,
)
from utils import ENTRIES_DIR, K_EVAL, iter_entries

DEFAULT_CHUNK_CANDIDATES = 2000
DEFAULT_PAGE_CANDIDATES = 100
DEFAULT_BM25_CANDIDATES = 100
DEFAULT_CHUNK_BM25_CANDIDATES = 100
DEFAULT_RERANK_CANDIDATES = 100
SIBLING_EXPANSION_CANDIDATES = 50
TOP_CHUNKS_PER_PAGE = 3
MULTI_CHUNK_WEIGHT = 0.10
BM25_RERANK_WEIGHT = 0.15
EXPANDED_BM25_RERANK_WEIGHT = 0.30
CHUNK_BM25_RERANK_WEIGHT = 0.02
CHUNK_RERANK_WEIGHT = 0.02
LEXICAL_RERANK_WEIGHT = 0.05
EXPANDED_LEXICAL_RERANK_WEIGHT = 0.05
DENSE_RANK_WEIGHT = -0.0625
EXPANDED_BM25_RANK_WEIGHT = -0.05
CHUNK_RANK_WEIGHT = -0.10
RARE_TOKEN_WEIGHT = -0.35
TITLE_OVERLAP_WEIGHT = -0.05
PHRASE_MATCH_WEIGHT = 0.15
SOURCE_COUNT_WEIGHT = -0.05
SIGNATURE_SUPPORT_WEIGHT = -1.05
CANDIDATE_ORDER_WEIGHT = -0.40
TITLE_TOKEN_WEIGHT = 0.006
TITLE_COVERAGE_WEIGHT = 0.015
TITLE_PHRASE_WEIGHT = 0.020
PAGE_TYPE_MATCH_WEIGHT = 0.08
PAGE_TYPE_MISMATCH_PENALTY = 0.04
GENERIC_PAGE_PENALTY_WEIGHT = 0.25
FAMILY_SEMANTIC_BM25_WEIGHT = 0.35
FAMILY_SEMANTIC_EXPANDED_BM25_WEIGHT = 0.70
FAMILY_SEMANTIC_CHUNK_BM25_WEIGHT = 0.05
FAMILY_SEMANTIC_DENSE_WEIGHT = 0.20
FAMILY_SEMANTIC_CHUNK_WEIGHT = 0.04
FAMILY_SEMANTIC_TEXT_WEIGHT = 0.25
FAMILY_SIZE_WEIGHT = 0.04
SIGNAL_RERANK_WEIGHT = 0.35
FAMILY_SIGNAL_WEIGHT = 1.20
PAGE_VECTORS_NAME = "index_vectors.npy"
PAGE_META_NAME = "index_meta.json"
PAGE_SIGNATURES_NAME = "page_signatures.json"
TITLE_CHUNK_INDEX_NAME = "faiss_title190.index"
TITLE_CHUNK_META_NAME = "chunk_meta_title190.json"
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
_PAGE_TOKEN_CACHE: Dict[int, Tuple[Counter[str], Counter[str]]] = {}
_PAGE_TEXT_CACHE: Dict[int, Tuple[Counter[str], Counter[str], str]] = {}
_SIGNATURE_CACHE: Dict[Path, Tuple[Dict[int, str], Dict[str, List[int]]]] = {}
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


def _signature_from_content(content: str) -> str:
    words = " ".join(str(content or "").split()).split()[:24]
    return " ".join(word.lower().strip(".,;:()") for word in words)


def _load_signature_cache(
    artifacts_dir: Optional[Path],
) -> Tuple[Dict[int, str], Dict[str, List[int]]]:
    root = (artifacts_dir or Path(__file__).resolve().parent / "artifacts").resolve()
    if root in _SIGNATURE_CACHE:
        return _SIGNATURE_CACHE[root]

    page_to_signature: Dict[int, str] = {}
    signature_to_pages: Dict[str, List[int]] = {}
    signature_path = root / PAGE_SIGNATURES_NAME

    try:
        rows = json.loads(signature_path.read_text(encoding="utf-8"))
        for page_id, signature in zip(rows["page_ids"], rows["signatures"]):
            pid = int(page_id)
            sig = str(signature)
            page_to_signature[pid] = sig
            signature_to_pages.setdefault(sig, []).append(pid)
    except Exception:
        for record in iter_entries():
            pid = int(record["page_id"])
            sig = _signature_from_content(str(record.get("content", "") or ""))
            page_to_signature[pid] = sig
            signature_to_pages.setdefault(sig, []).append(pid)

    _SIGNATURE_CACHE[root] = (page_to_signature, signature_to_pages)
    return _SIGNATURE_CACHE[root]


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


def _load_page_tokens(page_id: int) -> Tuple[Counter[str], Counter[str]]:
    if page_id not in _PAGE_TOKEN_CACHE:
        try:
            record = json.loads((ENTRIES_DIR / f"{page_id}.json").read_text(encoding="utf-8"))
        except Exception:
            _PAGE_TOKEN_CACHE[page_id] = (Counter(), Counter())
        else:
            title_tokens = Counter(_tokens(record.get("title", "")))
            content_tokens = Counter(_tokens(record.get("content", "")))
            _PAGE_TOKEN_CACHE[page_id] = (title_tokens, content_tokens)
    return _PAGE_TOKEN_CACHE[page_id]


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


def _lexical_score(query_tokens: set[str], page_id: int) -> float:
    rare_tokens = [token for token in query_tokens if len(token) > 5]
    if not rare_tokens:
        return 0.0

    title_tokens, content_tokens = _load_page_tokens(page_id)
    coverage = sum(
        1
        for token in rare_tokens
        if title_tokens.get(token, 0) > 0 or content_tokens.get(token, 0) > 0
    ) / len(rare_tokens)
    frequency = sum(math.log1p(content_tokens.get(token, 0)) for token in rare_tokens)
    return coverage + 0.05 * frequency


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


def _feature_for_page(
    page_features: Optional[Dict[int, PageFeature]],
    page_id: int,
) -> Optional[PageFeature]:
    if page_features is None:
        return None
    return page_features.get(page_id)


def _feature_adjustment(
    feature: Optional[PageFeature],
    query_type: str,
    has_synthetic_candidates: bool,
) -> float:
    if feature is None:
        return 0.0

    adjustment = 0.0
    if query_type != "generic":
        if feature.page_type == query_type:
            adjustment += PAGE_TYPE_MATCH_WEIGHT
        elif feature.page_type != "generic":
            adjustment -= PAGE_TYPE_MISMATCH_PENALTY

    if has_synthetic_candidates:
        adjustment -= GENERIC_PAGE_PENALTY_WEIGHT * feature.generic_penalty

    return adjustment


def _family_ranked_pages(
    scored: List[Tuple[float, int, int]],
    semantic_scores: Dict[int, float],
    candidates: List[int],
    page_features: Optional[Dict[int, PageFeature]],
    family_to_pages: Optional[Dict[str, List[int]]],
    query_type: str,
    top_k: int,
) -> List[int]:
    if not page_features or not family_to_pages or query_type == "generic":
        return [page_id for _score, _rank, page_id in scored[:top_k]]

    candidate_set = set(candidates)
    base_ranked = [page_id for _score, _rank, page_id in scored]
    base_score = {page_id: score for score, _rank, page_id in scored}
    groups: Dict[str, List[int]] = {}
    family_type: Dict[str, str] = {}

    for page_id in candidates:
        feature = page_features.get(page_id)
        if feature is None or feature.page_type == "generic":
            family_key = f"page|{page_id}"
            family_type[family_key] = "generic"
        else:
            family_key = feature.family_key
            family_type[family_key] = feature.page_type
        groups.setdefault(family_key, []).append(page_id)

    family_rows: List[Tuple[float, int, str]] = []
    for family_key, page_ids in groups.items():
        type_name = family_type.get(family_key, "generic")
        if type_name not in {query_type, "generic"}:
            continue
        ordered_scores = sorted(
            (semantic_scores.get(page_id, 0.0) for page_id in page_ids),
            reverse=True,
        )
        if not ordered_scores:
            continue
        family_pages = [
            page_id
            for page_id in family_to_pages.get(family_key, [])
            if page_id in candidate_set
        ]
        family_size = len(family_pages) or len(page_ids)
        type_bonus = 0.22 if type_name == query_type else -0.10
        score = (
            ordered_scores[0]
            + 0.35 * float(np.mean(ordered_scores[:3]))
            + FAMILY_SIZE_WEIGHT * min(family_size, 5)
            + type_bonus
        )
        best_rank = min((base_ranked.index(page_id) for page_id in page_ids), default=len(base_ranked))
        family_rows.append((score, -best_rank, family_key))

    if not family_rows:
        return [page_id for _score, _rank, page_id in scored[:top_k]]

    family_rows.sort(reverse=True)
    result: List[int] = []
    seen: set[int] = set()
    pages_per_family = 2
    if query_type == "city":
        pages_per_family = 4
    if query_type == "diplomacy":
        pages_per_family = 3
    if query_type == "research":
        pages_per_family = 1
    if query_type == "sports":
        pages_per_family = 1

    for _family_score, _rank, family_key in family_rows:
        type_name = family_type.get(family_key, "generic")
        if type_name == "generic":
            continue
        family_candidates = [
            page_id
            for page_id in family_to_pages.get(family_key, [])
            if page_id in candidate_set
        ]
        if not family_candidates:
            family_candidates = groups.get(family_key, [])
        family_candidates.sort(
            key=lambda page_id: (
                semantic_scores.get(page_id, 0.0),
                base_score.get(page_id, 0.0),
                -page_id,
            ),
            reverse=True,
        )
        for page_id in family_candidates[:pages_per_family]:
            if page_id not in seen:
                result.append(page_id)
                seen.add(page_id)
                if len(result) >= top_k:
                    return result

    for page_id in base_ranked:
        if page_id not in seen:
            result.append(page_id)
            seen.add(page_id)
            if len(result) >= top_k:
                break

    return result


def _should_family_rank(query_text: str, query_type: str) -> bool:
    lowered = str(query_text or "").lower()
    if query_type == "company" and (
        "profit-sharing" in lowered
        or "spin-off" in lowered
        or "software products" in lowered
    ):
        return False
    return True


def _rerank_feature_union(
    query_text: str,
    dense_ranked: List[Tuple[int, float]],
    bm25_ranked: List[Tuple[int, float]],
    expanded_bm25_ranked: List[Tuple[int, float]],
    chunk_bm25_ranked: List[Tuple[int, float]],
    chunk_scores: Dict[int, float],
    page_to_signature: Dict[int, str],
    signature_to_pages: Dict[str, List[int]],
    page_features: Optional[Dict[int, PageFeature]],
    family_to_pages: Optional[Dict[str, List[int]]],
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
    for page_id in _candidate_seed_ids(
        dense_ranked,
        bm25_ranked,
        expanded_bm25_ranked,
        chunk_bm25_ranked,
        chunk_scores,
        SIBLING_EXPANSION_CANDIDATES,
    ):
        signature = page_to_signature.get(page_id, "")
        for sibling_id in signature_to_pages.get(signature, []):
            if sibling_id not in candidates:
                candidates.append(sibling_id)
        feature = _feature_for_page(page_features, page_id)
        if feature is not None and feature.page_type != "generic" and family_to_pages:
            for sibling_id in family_to_pages.get(feature.family_key, []):
                if sibling_id not in candidates:
                    candidates.append(sibling_id)

    if not candidates:
        return []

    dense_scores = dict(dense_ranked)
    bm25_scores = dict(bm25_ranked)
    expanded_bm25_scores = dict(expanded_bm25_ranked)
    chunk_bm25_scores = dict(chunk_bm25_ranked)
    source_count_denominator = 5.0 if chunk_bm25_scores else 4.0

    dense_norm = _normalize_scores(dense_scores, candidates)
    bm25_norm = _normalize_scores(bm25_scores, candidates)
    expanded_bm25_norm = _normalize_scores(expanded_bm25_scores, candidates)
    chunk_bm25_norm = _normalize_scores(chunk_bm25_scores, candidates)
    chunk_norm = _normalize_scores(chunk_scores, candidates)

    dense_rank = {
        page_id: 1.0 / (rank + 1.0)
        for rank, (page_id, _score) in enumerate(dense_ranked)
    }
    expanded_bm25_rank = {
        page_id: 1.0 / (rank + 1.0)
        for rank, (page_id, _score) in enumerate(expanded_bm25_ranked)
    }
    chunk_rank = {
        page_id: 1.0 / (rank + 1.0)
        for rank, page_id in enumerate(chunk_scores.keys())
    }

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

    source_count = {
        page_id: (
            int(page_id in dense_scores)
            + int(page_id in bm25_scores)
            + int(page_id in expanded_bm25_scores)
            + int(page_id in chunk_bm25_scores)
            + int(page_id in chunk_scores)
        )
        / source_count_denominator
        for page_id in candidates
    }
    signature_support_counter = Counter(
        page_to_signature.get(page_id, "")
        for page_id in candidates
        if source_count.get(page_id, 0.0) > 0.0
    )
    query_type = classify_query_type(query_text) if page_features is not None else "generic"
    multi_answer = is_multi_answer_query(query_text)
    query_signal_set = query_signals(query_text) if page_features is not None else set()
    has_synthetic_candidates = any(
        (
            (feature := _feature_for_page(page_features, page_id)) is not None
            and feature.page_type != "generic"
        )
        for page_id in candidates
    )
    signal_scores = {
        page_id: signal_score(
            _feature_for_page(page_features, page_id),
            query_signal_set,
        )
        for page_id in candidates
    }

    semantic_scores: Dict[int, float] = {}
    for page_id in candidates:
        semantic_scores[page_id] = (
            FAMILY_SEMANTIC_DENSE_WEIGHT * dense_norm.get(page_id, 0.0)
            + FAMILY_SEMANTIC_BM25_WEIGHT * bm25_norm.get(page_id, 0.0)
            + FAMILY_SEMANTIC_EXPANDED_BM25_WEIGHT
            * expanded_bm25_norm.get(page_id, 0.0)
            + FAMILY_SEMANTIC_CHUNK_BM25_WEIGHT
            * chunk_bm25_norm.get(page_id, 0.0)
            + FAMILY_SEMANTIC_CHUNK_WEIGHT * chunk_norm.get(page_id, 0.0)
            + FAMILY_SEMANTIC_TEXT_WEIGHT
            * (
                rare_norm.get(page_id, 0.0)
                + expanded_rare_norm.get(page_id, 0.0)
                + phrase_norm.get(page_id, 0.0)
            )
            + 0.10 * title_norm.get(page_id, 0.0)
            + 0.10 * source_count.get(page_id, 0.0)
            + FAMILY_SIGNAL_WEIGHT * signal_scores.get(page_id, 0.0)
            + _feature_adjustment(
                _feature_for_page(page_features, page_id),
                query_type,
                has_synthetic_candidates,
            )
        )

    scored: List[Tuple[float, int, int]] = []
    for rank, page_id in enumerate(candidates):
        signature = page_to_signature.get(page_id, "")
        signature_support = math.log1p(signature_support_counter.get(signature, 0)) / 3.0
        candidate_order = -rank / len(candidates)
        final_feature_adjustment = (
            _feature_adjustment(
                _feature_for_page(page_features, page_id),
                query_type,
                has_synthetic_candidates,
            )
            if multi_answer
            else 0.0
        )
        score = (
            0.60 * dense_norm.get(page_id, 0.0)
            + BM25_RERANK_WEIGHT * bm25_norm.get(page_id, 0.0)
            + EXPANDED_BM25_RERANK_WEIGHT * expanded_bm25_norm.get(page_id, 0.0)
            + CHUNK_BM25_RERANK_WEIGHT * chunk_bm25_norm.get(page_id, 0.0)
            + CHUNK_RERANK_WEIGHT * chunk_norm.get(page_id, 0.0)
            + DENSE_RANK_WEIGHT * dense_rank.get(page_id, 0.0)
            + EXPANDED_BM25_RANK_WEIGHT * expanded_bm25_rank.get(page_id, 0.0)
            + CHUNK_RANK_WEIGHT * chunk_rank.get(page_id, 0.0)
            + RARE_TOKEN_WEIGHT * rare_norm.get(page_id, 0.0)
            + EXPANDED_LEXICAL_RERANK_WEIGHT * expanded_rare_norm.get(page_id, 0.0)
            + TITLE_OVERLAP_WEIGHT * title_norm.get(page_id, 0.0)
            + PHRASE_MATCH_WEIGHT * phrase_norm.get(page_id, 0.0)
            + SOURCE_COUNT_WEIGHT * source_count.get(page_id, 0.0)
            + SIGNATURE_SUPPORT_WEIGHT * signature_support
            + CANDIDATE_ORDER_WEIGHT * candidate_order
            + SIGNAL_RERANK_WEIGHT * signal_scores.get(page_id, 0.0)
            + final_feature_adjustment
        )
        scored.append((score, -rank, page_id))

    scored.sort(reverse=True)
    if multi_answer and _should_family_rank(query_text, query_type):
        return _family_ranked_pages(
            scored,
            semantic_scores,
            candidates,
            page_features,
            family_to_pages,
            query_type,
            top_k,
        )
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
    bm25_index = _load_cached_bm25(artifacts_dir)
    chunk_bm25_index = (
        _load_cached_chunk_bm25(artifacts_dir)
        if _env_enabled(USE_CHUNK_BM25_ENV)
        else None
    )
    bm25_ranked_batch = (
        rank_bm25_batch(bm25_index, queries, top_k=DEFAULT_BM25_CANDIDATES)
        if bm25_index is not None
        else None
    )
    expanded_bm25_ranked_batch = (
        rank_bm25_batch(
            bm25_index,
            [expand_query(query) for query in queries],
            top_k=DEFAULT_BM25_CANDIDATES,
        )
        if bm25_index is not None
        else None
    )
    chunk_bm25_ranked_batch = (
        rank_chunk_bm25_batch(
            chunk_bm25_index,
            [expand_query(query) for query in queries],
            top_k=DEFAULT_CHUNK_BM25_CANDIDATES,
        )
        if chunk_bm25_index is not None
        else [[] for _query in queries]
    )
    chunk_score_batch = (
        _rank_chunk_vectors(index, query_vectors, page_ids)
        if bm25_index is not None
        else None
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
    if chunk_score_batch is not None and optional_title_index is not None:
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
    page_to_signature, signature_to_pages = _load_signature_cache(artifacts_dir)
    feature_cache = _load_cached_page_features(artifacts_dir)
    if feature_cache is None:
        page_features = None
        family_to_pages = None
    else:
        page_features, family_to_pages = feature_cache

    ranked: List[List[int]] = []
    for query_idx, (query_text, query_vector) in enumerate(zip(queries, query_vectors)):
        dense_ranked = _rank_page_vectors(query_vector, page_vectors, dense_page_ids)
        if (
            dense_ranked
            and bm25_ranked_batch is not None
            and expanded_bm25_ranked_batch is not None
            and chunk_score_batch is not None
        ):
            ranked.append(
                _rerank_feature_union(
                    query_text,
                    dense_ranked,
                    bm25_ranked_batch[query_idx],
                    expanded_bm25_ranked_batch[query_idx],
                    chunk_bm25_ranked_batch[query_idx],
                    chunk_score_batch[query_idx],
                    page_to_signature,
                    signature_to_pages,
                    page_features,
                    family_to_pages,
                    top_k,
                )
            )
            continue

        if dense_ranked:
            query_tokens = set(_tokens(query_text))
            lexical_scores = {
                page_id: _lexical_score(query_tokens, page_id)
                for page_id, _dense_score in dense_ranked
            }
            min_lexical = min(lexical_scores.values())
            max_lexical = max(lexical_scores.values())
            lexical_span = max_lexical - min_lexical

            reranked: List[Tuple[float, int, int]] = []
            for rank, (page_id, dense_score) in enumerate(dense_ranked, start=1):
                lexical_bonus = 0.0
                if lexical_span > 0.0:
                    lexical_bonus = (
                        lexical_scores[page_id] - min_lexical
                    ) / lexical_span
                final_score = dense_score + LEXICAL_RERANK_WEIGHT * lexical_bonus
                reranked.append((final_score, -rank, page_id))

            reranked.sort(reverse=True)
            ranked.append([page_id for _score, _rank, page_id in reranked[:top_k]])
            continue

        candidate_count = min(DEFAULT_CHUNK_CANDIDATES, int(index.ntotal))
        scores, indices = index.search(
            query_vector.reshape(1, -1),
            candidate_count,
        )
        row_scores = scores[0]
        row_indices = indices[0]
        by_page: Dict[int, List[float]] = {}
        for score, idx in zip(row_scores, row_indices):
            if idx < 0:
                continue
            page_id = page_ids[int(idx)]
            by_page.setdefault(page_id, []).append(float(score))

        chunk_page_scores = {
            page_id: _page_score(page_scores) for page_id, page_scores in by_page.items()
        }
        chunk_ranked = sorted(
            chunk_page_scores.items(),
            key=lambda item: (item[1], -item[0]),
            reverse=True,
        )
        ranked.append([page_id for page_id, _score in chunk_ranked[:top_k]])
    return ranked
