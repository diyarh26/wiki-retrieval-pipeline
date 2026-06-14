"""Build chunk-level BM25 artifacts."""
from __future__ import annotations

import sys
import time
from pathlib import Path

STUDENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDENT_ROOT))

from chunk_lexical import build_chunk_bm25_index, save_chunk_bm25_index
from utils import ARTIFACTS_DIR


def main() -> None:
    t0 = time.perf_counter()
    index = build_chunk_bm25_index()
    build_time = time.perf_counter() - t0
    t1 = time.perf_counter()
    save_chunk_bm25_index(index, ARTIFACTS_DIR)
    save_time = time.perf_counter() - t1
    print(f"chunk_bm25_terms={len(index.terms)}")
    print(f"chunk_bm25_postings={int(index.doc_indices.shape[0])}")
    print(f"build_time={build_time:.2f}s")
    print(f"save_time={save_time:.2f}s")


if __name__ == "__main__":
    main()
