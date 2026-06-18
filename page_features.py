"""Broad query-type classification for retrieval weighting."""
from __future__ import annotations

from typing import Dict, Tuple

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


def _score_type(text: str, page_type: str) -> float:
    lowered = text.lower()
    return sum(weight for phrase, weight in TYPE_HINTS[page_type] if phrase in lowered)


def classify_query_type(query: str) -> str:
    """Classify a query using broad topic vocabulary only."""
    scores = {
        page_type: _score_type(str(query), page_type)
        for page_type in PAGE_TYPES
        if page_type != "generic"
    }
    best_type, best_score = max(scores.items(), key=lambda item: item[1])
    return best_type if best_score >= 1.5 else "generic"
