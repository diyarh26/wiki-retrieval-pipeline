"""Lightweight lexical retrieval artifacts for Section B.

The BM25 index is built offline and loaded by query-time retrieval.  It uses
only the standard library and NumPy so it fits the assignment import rules.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from utils import ARTIFACTS_DIR, ENTRIES_DIR, iter_entries

BM25_META_NAME = "bm25_meta.json"
BM25_TERMS_NAME = "bm25_terms.json"
BM25_DOCS_NAME = "bm25_doc_indices.npy"
BM25_TFS_NAME = "bm25_term_freqs.npy"
BM25_DOC_LENGTHS_NAME = "bm25_doc_lengths.npy"
BM25_PAGE_IDS_NAME = "bm25_page_ids.npy"

BM25_K1 = 1.2
BM25_B = 0.75
TITLE_WEIGHT = 3.0
MIN_TOKEN_LENGTH = 3

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


@dataclass
class BM25Index:
    page_ids: np.ndarray
    doc_lengths: np.ndarray
    doc_indices: np.ndarray
    term_freqs: np.ndarray
    terms: Dict[str, Tuple[int, int, float]]
    avg_doc_length: float
    k1: float = BM25_K1
    b: float = BM25_B


def tokenize(text: Any) -> List[str]:
    """Tokenize text for lexical retrieval."""
    return [
        token
        for token in _TOKEN_RE.findall(str(text).lower())
        if len(token) >= MIN_TOKEN_LENGTH and token not in _STOPWORDS
    ]


def expand_query(query: str) -> str:
    """Apply only generic lexical normalization useful for exact-match search.

    Decade expressions such as "1970s" are expanded into their explicit years,
    which helps exact lexical retrieval without relying on domain phrase lists.
    """
    query_text = str(query)
    lowered = query_text.lower()
    expanded = [query_text]

    for match in re.finditer(r"\b(\d{3})0s\b", lowered):
        base_year = int(match.group(1)) * 10
        expanded.append(" ".join(str(base_year + offset) for offset in range(10)))

    return " ".join(expanded)


def _weighted_counts(title: str, content: str) -> Counter[str]:
    counts = Counter(tokenize(content))
    for token, count in Counter(tokenize(title)).items():
        counts[token] += TITLE_WEIGHT * count
    return counts


def build_bm25_index(
    *,
    entries_dir: Optional[Path] = None,
) -> BM25Index:
    """Build a page-level BM25 inverted index in memory."""
    postings: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    page_ids: List[int] = []
    doc_lengths: List[float] = []

    for doc_idx, record in enumerate(iter_entries(entries_dir or ENTRIES_DIR)):
        page_ids.append(int(record["page_id"]))
        counts = _weighted_counts(
            str(record.get("title", "")),
            str(record.get("content", "") or ""),
        )
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

    return BM25Index(
        page_ids=np.asarray(page_ids, dtype=np.int32),
        doc_lengths=np.asarray(doc_lengths, dtype=np.float32),
        doc_indices=np.asarray(flat_doc_indices, dtype=np.int32),
        term_freqs=np.asarray(flat_term_freqs, dtype=np.float32),
        terms=terms,
        avg_doc_length=avg_doc_length,
    )


def save_bm25_index(index: BM25Index, artifacts_dir: Optional[Path] = None) -> None:
    """Persist a BM25Index under artifacts/."""
    root = artifacts_dir or ARTIFACTS_DIR
    root.mkdir(parents=True, exist_ok=True)

    np.save(root / BM25_PAGE_IDS_NAME, index.page_ids)
    np.save(root / BM25_DOC_LENGTHS_NAME, index.doc_lengths)
    np.save(root / BM25_DOCS_NAME, index.doc_indices)
    np.save(root / BM25_TFS_NAME, index.term_freqs)
    (root / BM25_TERMS_NAME).write_text(
        json.dumps(
            {term: [offset, length, idf] for term, (offset, length, idf) in index.terms.items()},
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    (root / BM25_META_NAME).write_text(
        json.dumps(
            {
                "version": 1,
                "num_docs": int(index.page_ids.shape[0]),
                "num_terms": len(index.terms),
                "num_postings": int(index.doc_indices.shape[0]),
                "avg_doc_length": index.avg_doc_length,
                "k1": index.k1,
                "b": index.b,
                "title_weight": TITLE_WEIGHT,
                "tokenizer": "lowercase alnum tokens, stopwords, min length 3",
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )


def bm25_artifacts_exist(artifacts_dir: Optional[Path] = None) -> bool:
    root = artifacts_dir or ARTIFACTS_DIR
    required = [
        BM25_META_NAME,
        BM25_TERMS_NAME,
        BM25_DOCS_NAME,
        BM25_TFS_NAME,
        BM25_DOC_LENGTHS_NAME,
        BM25_PAGE_IDS_NAME,
    ]
    return all((root / name).exists() for name in required)


def load_bm25_index(artifacts_dir: Optional[Path] = None) -> Optional[BM25Index]:
    """Load BM25 artifacts. Return None when artifacts are absent or invalid."""
    root = artifacts_dir or ARTIFACTS_DIR
    if not bm25_artifacts_exist(root):
        return None

    try:
        meta = json.loads((root / BM25_META_NAME).read_text(encoding="utf-8"))
        raw_terms = json.loads((root / BM25_TERMS_NAME).read_text(encoding="utf-8"))
        terms = {
            term: (int(row[0]), int(row[1]), float(row[2]))
            for term, row in raw_terms.items()
        }
        return BM25Index(
            page_ids=np.load(root / BM25_PAGE_IDS_NAME),
            doc_lengths=np.load(root / BM25_DOC_LENGTHS_NAME),
            doc_indices=np.load(root / BM25_DOCS_NAME),
            term_freqs=np.load(root / BM25_TFS_NAME),
            terms=terms,
            avg_doc_length=float(meta["avg_doc_length"]),
            k1=float(meta.get("k1", BM25_K1)),
            b=float(meta.get("b", BM25_B)),
        )
    except Exception:
        return None


def rank_bm25(
    index: BM25Index,
    query: str,
    *,
    top_k: int = 100,
) -> List[Tuple[int, float]]:
    """Return top page IDs by BM25 score for one query."""
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


def rank_bm25_batch(
    index: BM25Index,
    queries: Sequence[str],
    *,
    top_k: int = 100,
) -> List[List[Tuple[int, float]]]:
    return [rank_bm25(index, query, top_k=top_k) for query in queries]
