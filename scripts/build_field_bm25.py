"""Build title/lead BM25 artifacts for clean retrieval experiments."""
from __future__ import annotations

import sys
from pathlib import Path

STUDENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDENT_ROOT))

from field_lexical import FIELD_BM25_NAMES, build_field_bm25_index, save_field_bm25_index


def main() -> None:
    for field_name in FIELD_BM25_NAMES:
        index = build_field_bm25_index(field_name)
        save_field_bm25_index(index)
        print(
            f"{field_name}_bm25_docs={int(index.page_ids.shape[0])} "
            f"terms={len(index.terms)} postings={int(index.doc_indices.shape[0])}",
            flush=True,
        )


if __name__ == "__main__":
    main()
