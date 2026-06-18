"""Diagnostics for Section B retrieval experiments.

This script is not used by the autograder.  It measures candidate recall,
ideal rerank upper bounds, and optional reranking grids on the public queries.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import faiss
import numpy as np

STUDENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDENT_ROOT))

from embed import embed_queries
from eval import mean_ndcg_at_k, ndcg_at_k
from lexical import (
    BM25Index,
    bm25_artifacts_exist,
    build_bm25_index,
    expand_query,
    load_bm25_index,
    rank_bm25_batch,
    tokenize,
)
from chunk_lexical import (
    chunk_bm25_artifacts_exist,
    load_chunk_bm25_index,
    rank_chunk_bm25_batch,
)
from field_lexical import load_field_bm25_indexes, rank_field_bm25_batch
from main import run
from retrieve import debug_search_batch
from retrieve import _page_score
from utils import ARTIFACTS_DIR, ENTRIES_DIR, K_EVAL, PUBLIC_QUERIES_PATH

KS = (10, 50, 100, 200)
PAGE_VECTORS_NAME = "index_vectors.npy"
PAGE_META_NAME = "index_meta.json"
FAISS_INDEX_NAME = "faiss.index"
CHUNK_META_NAME = "chunk_meta.json"


def _load_public_queries() -> Tuple[List[str], List[Set[int]], List[str]]:
    rows = json.loads(PUBLIC_QUERIES_PATH.read_text(encoding="utf-8"))
    queries = [str(row["query"]) for row in rows]
    relevant = [{int(pid) for pid in row["relevant_page_ids"]} for row in rows]
    query_ids = [str(row.get("query_id", i)) for i, row in enumerate(rows)]
    return queries, relevant, query_ids


def _load_page_vectors() -> Tuple[np.ndarray, List[int]]:
    vectors = np.load(ARTIFACTS_DIR / PAGE_VECTORS_NAME)
    meta = json.loads((ARTIFACTS_DIR / PAGE_META_NAME).read_text(encoding="utf-8"))
    return np.ascontiguousarray(vectors, dtype=np.float32), [int(x) for x in meta["page_ids"]]


def _rank_dense(
    query_vectors: np.ndarray,
    page_vectors: np.ndarray,
    page_ids: Sequence[int],
    *,
    top_k: int,
) -> Tuple[List[List[int]], List[Dict[int, float]]]:
    ranked: List[List[int]] = []
    score_maps: List[Dict[int, float]] = []
    for query_vector in query_vectors:
        scores = page_vectors @ query_vector
        count = min(top_k, len(page_ids))
        indices = np.argpartition(-scores, count - 1)[:count]
        indices = indices[np.argsort(-scores[indices])]
        ranked_ids = [page_ids[int(idx)] for idx in indices]
        ranked.append(ranked_ids)
        score_maps.append({page_ids[int(idx)]: float(scores[int(idx)]) for idx in indices})
    return ranked, score_maps


def _rank_chunks(query_vectors: np.ndarray, *, top_k: int) -> Tuple[List[List[int]], List[Dict[int, float]]]:
    index = faiss.read_index(str(ARTIFACTS_DIR / FAISS_INDEX_NAME))
    chunk_meta = json.loads((ARTIFACTS_DIR / CHUNK_META_NAME).read_text(encoding="utf-8"))
    chunk_page_ids = [int(x) for x in chunk_meta["page_ids"]]

    chunk_search_k = min(int(index.ntotal), max(2000, top_k * 20))
    scores, indices = index.search(query_vectors, chunk_search_k)

    ranked: List[List[int]] = []
    score_maps: List[Dict[int, float]] = []
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
        ordered = sorted(page_scores, key=lambda pid: (page_scores[pid], -pid), reverse=True)
        ranked.append(ordered[:top_k])
        score_maps.append({pid: page_scores[pid] for pid in ordered[:top_k]})
    return ranked, score_maps


def _load_or_build_bm25(use_saved: bool) -> BM25Index:
    if use_saved and bm25_artifacts_exist(ARTIFACTS_DIR):
        loaded = load_bm25_index(ARTIFACTS_DIR)
        if loaded is not None:
            print("bm25_source=artifacts")
            return loaded

    print("bm25_source=temporary_in_memory")
    t0 = time.perf_counter()
    index = build_bm25_index()
    print(f"bm25_temp_build_time={time.perf_counter() - t0:.2f}s")
    return index


def _rank_bm25(
    index: BM25Index,
    queries: Sequence[str],
    *,
    top_k: int,
) -> Tuple[List[List[int]], List[Dict[int, float]]]:
    rows = rank_bm25_batch(index, queries, top_k=top_k)
    ranked = [[page_id for page_id, _score in row] for row in rows]
    score_maps = [{page_id: score for page_id, score in row} for row in rows]
    return ranked, score_maps


def _rank_chunk_bm25(
    queries: Sequence[str],
    *,
    top_k: int,
) -> Tuple[List[List[int]], List[Dict[int, float]]]:
    if not chunk_bm25_artifacts_exist(ARTIFACTS_DIR):
        empty_rows = [[] for _query in queries]
        empty_scores = [{} for _query in queries]
        return empty_rows, empty_scores

    index = load_chunk_bm25_index(ARTIFACTS_DIR)
    if index is None:
        empty_rows = [[] for _query in queries]
        empty_scores = [{} for _query in queries]
        return empty_rows, empty_scores

    rows = rank_chunk_bm25_batch(index, queries, top_k=top_k)
    ranked = [[page_id for page_id, _score in row] for row in rows]
    score_maps = [{page_id: score for page_id, score in row} for row in rows]
    return ranked, score_maps


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


def _rank_field_bm25(
    queries: Sequence[str],
    *,
    top_k: int,
) -> Tuple[List[List[int]], List[Dict[int, float]]]:
    indexes = load_field_bm25_indexes(ARTIFACTS_DIR)
    if not indexes:
        empty_rows = [[] for _query in queries]
        empty_scores = [{} for _query in queries]
        return empty_rows, empty_scores

    ranked_batches = [
        rank_field_bm25_batch(index, queries, top_k=top_k)
        for index in indexes.values()
    ]
    merged_rows = [
        _merge_ranked_sources(
            [batch[query_idx] for batch in ranked_batches],
            top_k=top_k,
        )
        for query_idx in range(len(queries))
    ]
    ranked = [[page_id for page_id, _score in row] for row in merged_rows]
    score_maps = [{page_id: score for page_id, score in row} for row in merged_rows]
    return ranked, score_maps


def _candidate_union(rows: Sequence[Sequence[int]], k: int) -> Set[int]:
    pool: Set[int] = set()
    for row in rows:
        pool.update(row[:k])
    return pool


def _ideal_ndcg(candidate_ids: Iterable[int], relevant: Set[int]) -> float:
    hits = [pid for pid in candidate_ids if pid in relevant]
    hit_count = min(len(set(hits)), K_EVAL)
    if hit_count <= 0:
        return 0.0
    ranked = list(sorted(relevant))[:hit_count]
    filler = [-i - 1 for i in range(K_EVAL - hit_count)]
    return ndcg_at_k(ranked + filler, relevant)


def _recall_at(ranked: Sequence[int], relevant: Set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def _print_recall_table(
    label: str,
    ranked_rows: Sequence[Sequence[int]],
    relevant_rows: Sequence[Set[int]],
) -> None:
    print(f"\n[{label}]")
    for k in KS:
        recalls = [
            _recall_at(ranked, relevant, k)
            for ranked, relevant in zip(ranked_rows, relevant_rows)
        ]
        hit_rate = sum(1 for value in recalls if value > 0.0) / len(recalls)
        ideal = [
            _ideal_ndcg(ranked[:k], relevant)
            for ranked, relevant in zip(ranked_rows, relevant_rows)
        ]
        print(
            f"k={k:<3} mean_recall={np.mean(recalls):.4f} "
            f"hit_rate={hit_rate:.4f} ideal_ndcg={np.mean(ideal):.4f}"
        )


def _print_union_table(
    label: str,
    all_ranked_rows: Sequence[Sequence[Sequence[int]]],
    relevant_rows: Sequence[Set[int]],
) -> None:
    print(f"\n[{label}]")
    for k in KS:
        recalls = []
        ideals = []
        for rows, relevant in zip(all_ranked_rows, relevant_rows):
            pool = _candidate_union(rows, k)
            recalls.append(len(pool & relevant) / len(relevant) if relevant else 0.0)
            ideals.append(_ideal_ndcg(pool, relevant))
        hit_rate = sum(1 for value in recalls if value > 0.0) / len(recalls)
        print(
            f"k={k:<3} mean_recall={np.mean(recalls):.4f} "
            f"hit_rate={hit_rate:.4f} ideal_ndcg={np.mean(ideals):.4f}"
        )


def _load_page_text(page_id: int) -> Tuple[str, str]:
    record = json.loads((ENTRIES_DIR / f"{page_id}.json").read_text(encoding="utf-8"))
    return str(record.get("title", "")), str(record.get("content", "") or "")


class TextFeatureCache:
    def __init__(self) -> None:
        self._cache: Dict[int, Tuple[Counter[str], Counter[str], str]] = {}

    def get(self, page_id: int) -> Tuple[Counter[str], Counter[str], str]:
        if page_id not in self._cache:
            title, content = _load_page_text(page_id)
            self._cache[page_id] = (
                Counter(tokenize(title)),
                Counter(tokenize(content)),
                f"{title} {content}".lower(),
            )
        return self._cache[page_id]


def _text_features(query: str, page_id: int, cache: TextFeatureCache) -> Tuple[float, float, float]:
    query_tokens = set(tokenize(query))
    rare_tokens = [token for token in query_tokens if len(token) > 5]
    title_counts, content_counts, full_text = cache.get(page_id)

    rare_score = 0.0
    if rare_tokens:
        coverage = sum(
            1
            for token in rare_tokens
            if title_counts.get(token, 0) > 0 or content_counts.get(token, 0) > 0
        ) / len(rare_tokens)
        frequency = sum(math.log1p(content_counts.get(token, 0)) for token in rare_tokens)
        rare_score = coverage + 0.05 * frequency

    title_score = 0.0
    if query_tokens and title_counts:
        overlap = sum(1 for token in query_tokens if title_counts.get(token, 0) > 0)
        title_score = overlap / max(1, len(title_counts))

    exact_score = 0.0
    query_phrase = " ".join(query.lower().split())
    if query_phrase and query_phrase in full_text:
        exact_score = 1.0

    return rare_score, title_score, exact_score


def _normalize_scores(score_map: Dict[int, float], candidates: Sequence[int]) -> Dict[int, float]:
    values = [score_map.get(pid, 0.0) for pid in candidates]
    if not values:
        return {}
    low = min(values)
    high = max(values)
    span = high - low
    if span <= 0.0:
        return {pid: 0.0 for pid in candidates}
    return {pid: (score_map.get(pid, 0.0) - low) / span for pid in candidates}


def _grid_search(
    queries: Sequence[str],
    relevant_rows: Sequence[Set[int]],
    dense_rows: Sequence[Sequence[int]],
    bm25_rows: Sequence[Sequence[int]],
    chunk_rows: Sequence[Sequence[int]],
    dense_scores: Sequence[Dict[int, float]],
    bm25_scores: Sequence[Dict[int, float]],
    chunk_scores: Sequence[Dict[int, float]],
) -> None:
    cache = TextFeatureCache()
    pool_sizes = (20, 50, 100, 200)
    bm25_weights = (0.0, 0.03, 0.06, 0.10, 0.15, 0.25, 0.40, 0.60)
    chunk_weights = (0.0, 0.02, 0.05, 0.10)
    rare_weights = (0.0, 0.05, 0.10, 0.15, 0.25)
    title_weights = (0.0, 0.02, 0.05, 0.10)
    exact_weights = (0.0, 0.05, 0.10)

    best: List[Tuple[float, Tuple[int, float, float, float, float, float]]] = []
    for pool_size in pool_sizes:
        prepared = []
        for query, dense, bm25, chunk, ds, bs, cs in zip(
            queries,
            dense_rows,
            bm25_rows,
            chunk_rows,
            dense_scores,
            bm25_scores,
            chunk_scores,
        ):
            ordered_candidates = list(dict.fromkeys(
                list(dense[:pool_size])
                + list(bm25[:pool_size])
                + list(chunk[:pool_size])
            ))
            dense_norm = _normalize_scores(ds, ordered_candidates)
            bm25_norm = _normalize_scores(bs, ordered_candidates)
            chunk_norm = _normalize_scores(cs, ordered_candidates)
            text_maps = {
                pid: _text_features(query, pid, cache)
                for pid in ordered_candidates
            }
            rare_norm = _normalize_scores(
                {pid: row[0] for pid, row in text_maps.items()}, ordered_candidates
            )
            title_norm = _normalize_scores(
                {pid: row[1] for pid, row in text_maps.items()}, ordered_candidates
            )
            exact_norm = _normalize_scores(
                {pid: row[2] for pid, row in text_maps.items()}, ordered_candidates
            )
            prepared.append(
                (
                    ordered_candidates,
                    dense_norm,
                    bm25_norm,
                    chunk_norm,
                    rare_norm,
                    title_norm,
                    exact_norm,
                )
            )

        for bm25_weight in bm25_weights:
            for chunk_weight in chunk_weights:
                for rare_weight in rare_weights:
                    for title_weight in title_weights:
                        for exact_weight in exact_weights:
                            ranked_rows: List[List[int]] = []
                            for row in prepared:
                                (
                                    candidates,
                                    dense_norm,
                                    bm25_norm,
                                    chunk_norm,
                                    rare_norm,
                                    title_norm,
                                    exact_norm,
                                ) = row
                                scored = []
                                for rank, page_id in enumerate(candidates):
                                    score = (
                                        dense_norm.get(page_id, 0.0)
                                        + bm25_weight * bm25_norm.get(page_id, 0.0)
                                        + chunk_weight * chunk_norm.get(page_id, 0.0)
                                        + rare_weight * rare_norm.get(page_id, 0.0)
                                        + title_weight * title_norm.get(page_id, 0.0)
                                        + exact_weight * exact_norm.get(page_id, 0.0)
                                    )
                                    scored.append((score, -rank, page_id))
                                scored.sort(reverse=True)
                                ranked_rows.append([page_id for _score, _rank, page_id in scored[:K_EVAL]])
                            score = mean_ndcg_at_k(ranked_rows, relevant_rows)
                            best.append(
                                (
                                    score,
                                    (
                                        pool_size,
                                        bm25_weight,
                                        chunk_weight,
                                        rare_weight,
                                        title_weight,
                                        exact_weight,
                                    ),
                                )
                            )

    print("\n[rerank_grid_top20]")
    print("columns: ndcg pool bm25_w chunk_w rare_w title_w exact_w")
    for score, config in sorted(best, reverse=True)[:20]:
        print(f"{score:.4f} {config}")


def _source_ranks(
    ranked_sources: Dict[str, Sequence[int]],
    page_id: int,
) -> Dict[str, int]:
    ranks: Dict[str, int] = {}
    for source_name, ranked in ranked_sources.items():
        try:
            ranks[source_name] = list(ranked).index(page_id) + 1
        except ValueError:
            continue
    return ranks


def _print_per_query(
    query_ids: Sequence[str],
    queries: Sequence[str],
    relevant_rows: Sequence[Set[int]],
    predicted_rows: Sequence[Sequence[int]],
    source_rows: Dict[str, Sequence[Sequence[int]]],
) -> None:
    print("\n[per_query_error_report]")
    for i, (query_id, query, relevant) in enumerate(zip(query_ids, queries, relevant_rows)):
        predicted = list(predicted_rows[i][:K_EVAL])
        predicted_hits = set(predicted) & relevant
        union_200 = set()
        for rows in source_rows.values():
            union_200.update(rows[i][:200])
        candidate_hits = union_200 & relevant
        missing_from_candidates = sorted(relevant - union_200)
        candidates_ranked_too_low = sorted(candidate_hits - predicted_hits)
        source_hit_rows = {}
        for page_id in sorted(relevant):
            source_hit_rows[page_id] = _source_ranks(
                {name: rows[i][:200] for name, rows in source_rows.items()},
                page_id,
            )

        print(f"{i:02d} {query_id} ndcg={ndcg_at_k(predicted, relevant):.4f}")
        print(f"   query={query}")
        print(f"   relevant={sorted(relevant)}")
        print(f"   predicted_top10={predicted}")
        print(f"   missing_from_candidates={missing_from_candidates}")
        print(f"   candidates_ranked_too_low={candidates_ranked_too_low}")
        print(f"   relevant_source_ranks={source_hit_rows}")


def _fmt_rank(value: object) -> str:
    return "-" if value is None else str(value)


def _fmt_float(value: object) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "-"


def _print_feature_dump(
    query_ids: Sequence[str],
    queries: Sequence[str],
    relevant_rows: Sequence[Set[int]],
    predicted_rows: Sequence[Sequence[int]],
    debug_rows: Sequence[Sequence[Dict[str, object]]],
    *,
    worst_count: int,
    rows_per_query: int,
) -> None:
    scored_queries = sorted(
        (
            ndcg_at_k(predicted_rows[i], relevant_rows[i]),
            i,
        )
        for i in range(len(queries))
    )
    print("\n[worst_query_candidate_feature_dump]")
    for ndcg, query_idx in scored_queries[:worst_count]:
        relevant = relevant_rows[query_idx]
        rows = list(debug_rows[query_idx])
        selected: List[Dict[str, object]] = []
        seen: Set[int] = set()
        for row in rows[:rows_per_query]:
            page_id = int(row["page_id"])
            selected.append(row)
            seen.add(page_id)
        for row in rows:
            page_id = int(row["page_id"])
            if page_id in relevant and page_id not in seen:
                selected.append(row)
                seen.add(page_id)

        print(f"{query_idx:02d} {query_ids[query_idx]} ndcg={ndcg:.4f}")
        print(f"   query={queries[query_idx]}")
        print(f"   relevant={sorted(relevant)}")
        print(
            "   columns: final_rank rel page_id score "
            "dense_r page_bm25_r chunk_bm25_r field_bm25_r chunk_dense_r "
            "dense_n page_bm25_n chunk_bm25_n field_bm25_n chunk_dense_n "
            "source lex_rrf src_rrf top20 top50 other_sources rare title phrase type"
        )
        for row in selected:
            page_id = int(row["page_id"])
            print(
                "   "
                f"{int(row['final_rank']):>3} "
                f"{'*' if page_id in relevant else '-'} "
                f"{page_id:<6} "
                f"{_fmt_float(row.get('score'))} "
                f"{_fmt_rank(row.get('dense_rank')):>4} "
                f"{_fmt_rank(row.get('page_bm25_rank')):>4} "
                f"{_fmt_rank(row.get('chunk_bm25_rank')):>4} "
                f"{_fmt_rank(row.get('field_bm25_rank')):>4} "
                f"{_fmt_rank(row.get('chunk_dense_rank')):>4} "
                f"{_fmt_float(row.get('dense_norm')):>5} "
                f"{_fmt_float(row.get('page_bm25_norm')):>5} "
                f"{_fmt_float(row.get('chunk_bm25_norm')):>5} "
                f"{_fmt_float(row.get('field_bm25_norm')):>5} "
                f"{_fmt_float(row.get('chunk_dense_norm')):>5} "
                f"{_fmt_float(row.get('source_count')):>5} "
                f"{_fmt_float(row.get('lexical_rrf_norm')):>5} "
                f"{_fmt_float(row.get('source_rrf_norm')):>5} "
                f"{_fmt_float(row.get('top20_source_consensus')):>5} "
                f"{_fmt_float(row.get('top50_source_consensus')):>5} "
                f"{int(row.get('other_source_count', 0)):>2} "
                f"{_fmt_float(row.get('rare_norm')):>5} "
                f"{_fmt_float(row.get('title_norm')):>5} "
                f"{_fmt_float(row.get('phrase_norm')):>5} "
                f"{_fmt_float(row.get('type_match')):>4}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--use-bm25-artifact", action="store_true")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--dump-features", action="store_true")
    parser.add_argument("--dump-worst", type=int, default=5)
    parser.add_argument("--dump-candidates", type=int, default=15)
    args = parser.parse_args()

    t0 = time.perf_counter()
    queries, relevant_rows, query_ids = _load_public_queries()
    query_vectors = np.ascontiguousarray(embed_queries(queries), dtype=np.float32)
    print(f"queries={len(queries)} embed_time={time.perf_counter() - t0:.2f}s")

    t1 = time.perf_counter()
    page_vectors, page_ids = _load_page_vectors()
    dense_rows, dense_scores = _rank_dense(
        query_vectors,
        page_vectors,
        page_ids,
        top_k=args.top_k,
    )
    print(f"dense_rank_time={time.perf_counter() - t1:.2f}s")

    t2 = time.perf_counter()
    bm25_index = _load_or_build_bm25(args.use_bm25_artifact)
    bm25_rows, bm25_scores = _rank_bm25(bm25_index, queries, top_k=args.top_k)
    print(f"bm25_rank_time={time.perf_counter() - t2:.2f}s")

    t2b = time.perf_counter()
    expanded_queries = [expand_query(query) for query in queries]
    expanded_bm25_rows, expanded_bm25_scores = _rank_bm25(
        bm25_index,
        expanded_queries,
        top_k=args.top_k,
    )
    print(f"expanded_bm25_rank_time={time.perf_counter() - t2b:.2f}s")

    t3 = time.perf_counter()
    chunk_rows, chunk_scores = _rank_chunks(query_vectors, top_k=args.top_k)
    print(f"chunk_rank_time={time.perf_counter() - t3:.2f}s")

    t4 = time.perf_counter()
    chunk_bm25_rows, chunk_bm25_scores = _rank_chunk_bm25(
        expanded_queries,
        top_k=args.top_k,
    )
    print(f"chunk_bm25_rank_time={time.perf_counter() - t4:.2f}s")

    t4b = time.perf_counter()
    field_bm25_rows, field_bm25_scores = _rank_field_bm25(
        queries,
        top_k=args.top_k,
    )
    print(f"field_bm25_rank_time={time.perf_counter() - t4b:.2f}s")

    t5 = time.perf_counter()
    predicted_rows = run(queries)
    print(f"current_run_time={time.perf_counter() - t5:.2f}s")

    _print_recall_table("dense_page", dense_rows, relevant_rows)
    _print_recall_table("bm25_page", bm25_rows, relevant_rows)
    _print_recall_table("expanded_bm25_page", expanded_bm25_rows, relevant_rows)
    _print_recall_table("dense_chunk", chunk_rows, relevant_rows)
    _print_recall_table("chunk_bm25", chunk_bm25_rows, relevant_rows)
    _print_recall_table("field_bm25", field_bm25_rows, relevant_rows)
    _print_union_table(
        "union_all_sources",
        list(zip(dense_rows, bm25_rows, expanded_bm25_rows, chunk_rows, chunk_bm25_rows, field_bm25_rows)),
        relevant_rows,
    )

    print("\n[current_ranker_reference]")
    print(f"dense_top10_ndcg={mean_ndcg_at_k([row[:10] for row in dense_rows], relevant_rows):.4f}")
    print(f"bm25_top10_ndcg={mean_ndcg_at_k([row[:10] for row in bm25_rows], relevant_rows):.4f}")
    print(f"expanded_bm25_top10_ndcg={mean_ndcg_at_k([row[:10] for row in expanded_bm25_rows], relevant_rows):.4f}")
    print(f"dense_chunk_top10_ndcg={mean_ndcg_at_k([row[:10] for row in chunk_rows], relevant_rows):.4f}")
    print(f"chunk_bm25_top10_ndcg={mean_ndcg_at_k([row[:10] for row in chunk_bm25_rows], relevant_rows):.4f}")
    print(f"field_bm25_top10_ndcg={mean_ndcg_at_k([row[:10] for row in field_bm25_rows], relevant_rows):.4f}")
    print(f"current_run_top10_ndcg={mean_ndcg_at_k(predicted_rows, relevant_rows):.4f}")

    _print_per_query(
        query_ids,
        queries,
        relevant_rows,
        predicted_rows,
        {
            "page_dense": dense_rows,
            "page_bm25": bm25_rows,
            "expanded_bm25": expanded_bm25_rows,
            "chunk_dense": chunk_rows,
            "chunk_bm25": chunk_bm25_rows,
            "field_bm25": field_bm25_rows,
        },
    )

    if args.dump_features:
        t6 = time.perf_counter()
        debug_rows = debug_search_batch(queries)
        print(f"feature_dump_build_time={time.perf_counter() - t6:.2f}s")
        _print_feature_dump(
            query_ids,
            queries,
            relevant_rows,
            predicted_rows,
            debug_rows,
            worst_count=args.dump_worst,
            rows_per_query=args.dump_candidates,
        )

    if args.tune:
        _grid_search(
            queries,
            relevant_rows,
            dense_rows,
            bm25_rows,
            chunk_rows,
            dense_scores,
            bm25_scores,
            chunk_scores,
        )


if __name__ == "__main__":
    main()
