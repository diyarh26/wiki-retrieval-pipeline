"""Build page semantic-feature artifacts for family-aware reranking."""
from __future__ import annotations

import sys
from pathlib import Path

STUDENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDENT_ROOT))

from page_features import PAGE_FEATURES_NAME, build_page_features, save_page_features, summarize_features
from utils import ARTIFACTS_DIR


def main() -> None:
    features = build_page_features()
    save_page_features(features, ARTIFACTS_DIR)
    counts = summarize_features(features.values())
    print(f"page_features={len(features)}")
    print("page_type_counts=" + " ".join(f"{key}:{counts[key]}" for key in sorted(counts)))
    print(f"artifact={ARTIFACTS_DIR / PAGE_FEATURES_NAME}")


if __name__ == "__main__":
    main()
