"""Build offline BM25 lexical artifacts under artifacts/."""
from __future__ import annotations

import sys
import time
from pathlib import Path

STUDENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDENT_ROOT))

from lexical import build_bm25_index, save_bm25_index
from utils import ARTIFACTS_DIR


def main() -> None:
    t0 = time.perf_counter()
    index = build_bm25_index()
    build_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    save_bm25_index(index, ARTIFACTS_DIR)
    save_time = time.perf_counter() - t1

    print(f"bm25_docs={index.page_ids.shape[0]}")
    print(f"bm25_terms={len(index.terms)}")
    print(f"bm25_postings={index.doc_indices.shape[0]}")
    print(f"build_time={build_time:.2f}s")
    print(f"save_time={save_time:.2f}s")
    print(f"artifacts_dir={ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
