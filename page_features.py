"""Generic page metadata helpers.

This module intentionally avoids public-query or corpus-template rules.  It is
kept for offline diagnostics and artifact compatibility, but query-time
retrieval does not rely on synthetic family grouping.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from lexical import tokenize
from utils import ARTIFACTS_DIR, ENTRIES_DIR, iter_entries

PAGE_FEATURES_NAME = "page_features.json"
PAGE_TYPES = {
    "sports",
    "company",
    "city",
    "research",
    "diplomacy",
    "generic",
}

TYPE_HINTS: Dict[str, Tuple[Tuple[str, float], ...]] = {
    "sports": (
        ("basketball", 3.0),
        ("player", 1.0),
        ("team", 1.0),
        ("coach", 1.0),
        ("championship", 1.5),
        ("finals", 1.5),
        ("arena", 1.0),
    ),
    "company": (
        ("company", 2.0),
        ("founded", 1.0),
        ("headquartered", 2.0),
        ("chief executive", 2.0),
        ("factory", 1.0),
        ("revenue", 1.0),
        ("industry", 1.0),
    ),
    "city": (
        ("city", 2.0),
        ("population", 2.0),
        ("economy", 1.0),
        ("transport", 1.0),
        ("airport", 1.0),
        ("river", 0.5),
        ("coast", 0.5),
    ),
    "research": (
        ("research", 2.0),
        ("researcher", 2.0),
        ("laboratory", 1.5),
        ("institute", 1.5),
        ("published", 1.0),
        ("experiment", 1.0),
        ("method", 0.8),
    ),
    "diplomacy": (
        ("diplomatic", 2.0),
        ("agreement", 1.5),
        ("treaty", 2.5),
        ("minister", 1.5),
        ("negotiation", 1.0),
        ("negotiations", 1.0),
        ("accord", 1.0),
    ),
}


@dataclass(frozen=True)
class PageFeature:
    page_type: str
    canonical_entity: str
    family_key: str
    generic_penalty: float
    signals: Tuple[str, ...] = ()


def _clean_text(text: Any) -> str:
    return " ".join(str(text or "").split())


def _generic_penalty(title: str, content: str) -> float:
    """Small generic penalty for broad pages that often rank too easily."""
    lowered_title = title.lower()
    lowered_content = content.lower()
    token_count = len(tokenize(content))

    penalty = 0.0
    if token_count < 80:
        penalty += 0.20
    if " may refer to" in lowered_content or lowered_title.endswith("(disambiguation)"):
        penalty += 0.70
    if lowered_title.startswith("list of "):
        penalty += 0.30
    return min(1.0, penalty)


def _score_type(text: str, page_type: str) -> float:
    lowered = text.lower()
    return sum(weight for phrase, weight in TYPE_HINTS[page_type] if phrase in lowered)


def infer_page_type(title: str, content: str) -> str:
    """Infer a broad page type from generic vocabulary only."""
    text = f"{title} {content}"
    scores = {
        page_type: _score_type(text, page_type)
        for page_type in PAGE_TYPES
        if page_type != "generic"
    }
    best_type, best_score = max(scores.items(), key=lambda item: item[1])
    return best_type if best_score >= 2.0 else "generic"


def classify_query_type(query: str) -> str:
    """Classify a query by broad topic vocabulary, not public templates."""
    scores = {
        page_type: _score_type(str(query), page_type)
        for page_type in PAGE_TYPES
        if page_type != "generic"
    }
    best_type, best_score = max(scores.items(), key=lambda item: item[1])
    return best_type if best_score >= 1.25 else "generic"


def extract_page_feature(record: Dict[str, Any]) -> PageFeature:
    """Return generic per-page metadata without corpus-template extraction."""
    page_id = int(record.get("page_id", -1))
    title = _clean_text(record.get("title", ""))
    content = _clean_text(record.get("content", ""))
    return PageFeature(
        page_type=infer_page_type(title, content),
        canonical_entity=title,
        family_key=f"page:{page_id}",
        generic_penalty=_generic_penalty(title, content),
        signals=(),
    )


def build_page_features(
    *,
    entries_dir: Optional[Path] = None,
) -> Dict[int, PageFeature]:
    """Build generic features for every page in the corpus."""
    features: Dict[int, PageFeature] = {}
    for record in iter_entries(entries_dir or ENTRIES_DIR):
        features[int(record["page_id"])] = extract_page_feature(record)
    return features


def save_page_features(
    features: Dict[int, PageFeature],
    artifacts_dir: Optional[Path] = None,
) -> None:
    root = artifacts_dir or ARTIFACTS_DIR
    root.mkdir(parents=True, exist_ok=True)
    page_ids = sorted(features)
    payload = {
        "version": 2,
        "page_ids": page_ids,
        "page_type": [features[page_id].page_type for page_id in page_ids],
        "canonical_entity": [features[page_id].canonical_entity for page_id in page_ids],
        "family_key": [features[page_id].family_key for page_id in page_ids],
        "generic_penalty": [features[page_id].generic_penalty for page_id in page_ids],
        "signals": [list(features[page_id].signals) for page_id in page_ids],
    }
    (root / PAGE_FEATURES_NAME).write_text(
        json.dumps(payload, ensure_ascii=True),
        encoding="utf-8",
    )


def load_page_features(
    artifacts_dir: Optional[Path] = None,
) -> Optional[Tuple[Dict[int, PageFeature], Dict[str, List[int]]]]:
    """Load generic page features, returning None if absent or invalid."""
    root = artifacts_dir or ARTIFACTS_DIR
    path = root / PAGE_FEATURES_NAME
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        page_ids = [int(page_id) for page_id in payload["page_ids"]]
        page_types = list(payload["page_type"])
        canonical_entities = list(payload["canonical_entity"])
        family_keys = list(payload["family_key"])
        generic_penalties = [float(value) for value in payload["generic_penalty"]]
        raw_signals = payload.get("signals", [[] for _page_id in page_ids])
    except Exception:
        return None

    if not (
        len(page_ids)
        == len(page_types)
        == len(canonical_entities)
        == len(family_keys)
        == len(generic_penalties)
        == len(raw_signals)
    ):
        return None

    features: Dict[int, PageFeature] = {}
    family_to_pages: Dict[str, List[int]] = {}
    for page_id, page_type, canonical_entity, family_key, generic_penalty, signals in zip(
        page_ids,
        page_types,
        canonical_entities,
        family_keys,
        generic_penalties,
        raw_signals,
    ):
        feature = PageFeature(
            page_type="generic" if str(page_type) not in PAGE_TYPES else str(page_type),
            canonical_entity=str(canonical_entity),
            family_key=str(family_key),
            generic_penalty=float(generic_penalty),
            signals=tuple(str(signal) for signal in signals),
        )
        features[page_id] = feature
        family_to_pages.setdefault(feature.family_key, []).append(page_id)

    return features, family_to_pages


def summarize_features(features: Iterable[PageFeature]) -> Dict[str, int]:
    counts = {page_type: 0 for page_type in sorted(PAGE_TYPES)}
    for feature in features:
        counts[feature.page_type] = counts.get(feature.page_type, 0) + 1
    return counts
