"""Build page opening-signature artifacts for sibling expansion."""
from __future__ import annotations

import json
import sys
from pathlib import Path

STUDENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDENT_ROOT))

from retrieve import PAGE_SIGNATURES_NAME, _signature_from_content
from utils import ARTIFACTS_DIR, iter_entries


def main() -> None:
    page_ids = []
    signatures = []
    for record in iter_entries():
        page_ids.append(int(record["page_id"]))
        signatures.append(_signature_from_content(str(record.get("content", "") or "")))

    out_path = ARTIFACTS_DIR / PAGE_SIGNATURES_NAME
    out_path.write_text(
        json.dumps(
            {"page_ids": page_ids, "signatures": signatures},
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    print(f"page_signatures={len(page_ids)}")
    print(f"artifact={out_path}")


if __name__ == "__main__":
    main()
