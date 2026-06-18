"""BM25 indexes over generic page fields such as title and lead text."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from lexical import BM25_B, BM25_K1, TITLE_WEIGHT, tokenize
from utils import ARTIFACTS_DIR, ENTRIES_DIR, iter_entries

FIELD_BM25_NAMES = ("title", "lead", "title_lead")


@dataclass
class FieldBM25Index:
    name: str
    page_ids: np.ndarray
    doc_lengths: np.ndarray
    doc_indices: np.ndarray
    term_freqs: np.ndarray
    terms: Dict[str, Tuple[int, int, float]]
    avg_doc_length: float
    k1: float = BM25_K1
    b: float = BM25_B


def _artifact_name(field_name: str, suffix: str) -> str:
    return f"{field_name}_bm25_{suffix}"


def _lead_text(content: Any, *, max_words: int = 160) -> str:
    text = str(content or "")
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    lead = paragraphs[0] if paragraphs else text
    words = lead.split()
    return " ".join(words[:max_words])


def _field_counts(record: Dict[str, Any], field_name: str) -> Counter[str]:
    title = str(record.get("title", ""))
    lead = _lead_text(record.get("content", ""))
    if field_name == "title":
        return Counter(tokenize(title))
    if field_name == "lead":
        return Counter(tokenize(lead))
    if field_name == "title_lead":
        counts = Counter(tokenize(lead))
        for token, count in Counter(tokenize(title)).items():
            counts[token] += TITLE_WEIGHT * count
        return counts
    raise ValueError(f"unknown field BM25 source: {field_name}")


def build_field_bm25_index(
    field_name: str,
    *,
    entries_dir: Optional[Path] = None,
) -> FieldBM25Index:
    postings: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    page_ids: List[int] = []
    doc_lengths: List[float] = []

    for doc_idx, record in enumerate(iter_entries(entries_dir or ENTRIES_DIR)):
        page_ids.append(int(record["page_id"]))
        counts = _field_counts(record, field_name)
        doc_lengths.append(float(sum(counts.values())) or 1.0)
        for term, term_freq in counts.items():
            postings[term].append((doc_idx, float(term_freq)))

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

    return FieldBM25Index(
        name=field_name,
        page_ids=np.asarray(page_ids, dtype=np.int32),
        doc_lengths=np.asarray(doc_lengths, dtype=np.float32),
        doc_indices=np.asarray(flat_doc_indices, dtype=np.int32),
        term_freqs=np.asarray(flat_term_freqs, dtype=np.float32),
        terms=terms,
        avg_doc_length=avg_doc_length,
    )


def save_field_bm25_index(
    index: FieldBM25Index,
    artifacts_dir: Optional[Path] = None,
) -> None:
    root = artifacts_dir or ARTIFACTS_DIR
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / _artifact_name(index.name, "page_ids.npy"), index.page_ids)
    np.save(root / _artifact_name(index.name, "doc_lengths.npy"), index.doc_lengths)
    np.save(root / _artifact_name(index.name, "doc_indices.npy"), index.doc_indices)
    np.save(root / _artifact_name(index.name, "term_freqs.npy"), index.term_freqs)
    (root / _artifact_name(index.name, "terms.json")).write_text(
        json.dumps(
            {term: [offset, length, idf] for term, (offset, length, idf) in index.terms.items()},
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    (root / _artifact_name(index.name, "meta.json")).write_text(
        json.dumps(
            {
                "version": 1,
                "field": index.name,
                "num_docs": int(index.page_ids.shape[0]),
                "num_terms": len(index.terms),
                "num_postings": int(index.doc_indices.shape[0]),
                "avg_doc_length": index.avg_doc_length,
                "k1": index.k1,
                "b": index.b,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )


def field_bm25_artifacts_exist(
    field_name: str,
    artifacts_dir: Optional[Path] = None,
) -> bool:
    root = artifacts_dir or ARTIFACTS_DIR
    required = [
        "meta.json",
        "terms.json",
        "doc_indices.npy",
        "term_freqs.npy",
        "doc_lengths.npy",
        "page_ids.npy",
    ]
    return all((root / _artifact_name(field_name, suffix)).exists() for suffix in required)


def load_field_bm25_index(
    field_name: str,
    artifacts_dir: Optional[Path] = None,
) -> Optional[FieldBM25Index]:
    root = artifacts_dir or ARTIFACTS_DIR
    if not field_bm25_artifacts_exist(field_name, root):
        return None
    try:
        meta = json.loads((root / _artifact_name(field_name, "meta.json")).read_text(encoding="utf-8"))
        raw_terms = json.loads((root / _artifact_name(field_name, "terms.json")).read_text(encoding="utf-8"))
        terms = {
            term: (int(row[0]), int(row[1]), float(row[2]))
            for term, row in raw_terms.items()
        }
        return FieldBM25Index(
            name=field_name,
            page_ids=np.load(root / _artifact_name(field_name, "page_ids.npy")),
            doc_lengths=np.load(root / _artifact_name(field_name, "doc_lengths.npy")),
            doc_indices=np.load(root / _artifact_name(field_name, "doc_indices.npy")),
            term_freqs=np.load(root / _artifact_name(field_name, "term_freqs.npy")),
            terms=terms,
            avg_doc_length=float(meta["avg_doc_length"]),
            k1=float(meta.get("k1", BM25_K1)),
            b=float(meta.get("b", BM25_B)),
        )
    except Exception:
        return None


def load_field_bm25_indexes(
    artifacts_dir: Optional[Path] = None,
) -> Dict[str, FieldBM25Index]:
    indexes: Dict[str, FieldBM25Index] = {}
    for field_name in FIELD_BM25_NAMES:
        index = load_field_bm25_index(field_name, artifacts_dir)
        if index is not None:
            indexes[field_name] = index
    return indexes


def rank_field_bm25(
    index: FieldBM25Index,
    query: str,
    *,
    top_k: int = 100,
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

    candidate_count = min(top_k, int(index.page_ids.shape[0]))
    if candidate_count <= 0 or float(scores.max(initial=0.0)) <= 0.0:
        return []

    if candidate_count < index.page_ids.shape[0]:
        top_indices = np.argpartition(-scores, candidate_count - 1)[:candidate_count]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
    else:
        top_indices = np.argsort(-scores)

    return [
        (int(index.page_ids[int(idx)]), float(scores[int(idx)]))
        for idx in top_indices
        if scores[int(idx)] > 0.0
    ]


def rank_field_bm25_batch(
    index: FieldBM25Index,
    queries: Sequence[str],
    *,
    top_k: int = 100,
) -> List[List[Tuple[int, float]]]:
    return [rank_field_bm25(index, query, top_k=top_k) for query in queries]
