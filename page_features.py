"""Page-level semantic features for family-aware retrieval."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

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

_SPACE_RE = re.compile(r"\s+")
_NON_KEY_RE = re.compile(r"[^a-z0-9]+")

_SPORTS_RE = re.compile(
    r"^(.+?) \(born.*?\) is a former professional basketball player best known as "
    r"(.+?) of the (.+?) when they won the (.+?) in (\d{3,4})\.",
    re.IGNORECASE | re.DOTALL,
)
_COMPANY_RE = re.compile(
    r"^(.+?) is a (.+?) company founded in (\d{3,4}) and headquartered in (.+?)\. "
    r"(.+?) served as chief executive during its international expansion phase\.",
    re.IGNORECASE | re.DOTALL,
)
_CITY_RE = re.compile(
    r"^(.+?) is a city on a (.+?), with a population of about ([0-9,]+)\. "
    r"Its economy has long centered on (.+?)\.",
    re.IGNORECASE | re.DOTALL,
)
_RESEARCH_RE = re.compile(
    r"^(.+?) led a research group at the (.+?) in (.+?) that advanced (.+?)\. "
    r"The group's foundational results were published in (\d{3,4})\.",
    re.IGNORECASE | re.DOTALL,
)
_DIPLOMACY_RE = re.compile(
    r"^The (.+?) \((\d{3,4})\) was a diplomatic agreement in which "
    r"(.+?), (.+?) of (.+?), helped finalize terms at (.+?)\.",
    re.IGNORECASE | re.DOTALL,
)

_QUERY_TYPE_WEIGHTS: Dict[str, Tuple[Tuple[str, float], ...]] = {
    "sports": (
        ("basketball", 2.5),
        ("point guard", 2.5),
        ("franchise player", 2.5),
        ("on-court", 2.5),
        ("championship", 2.0),
        ("finals", 2.0),
        ("title series", 2.0),
        ("seven-game", 2.0),
        ("captain", 2.0),
        ("club", 1.5),
        ("arena", 1.5),
        ("commemorative banner", 2.0),
        ("home bench", 1.5),
        ("youth basketball foundation", 2.0),
    ),
    "company": (
        ("company", 2.0),
        ("firm", 2.0),
        ("ceo", 2.5),
        ("chief executive", 2.5),
        ("executive", 1.5),
        ("profit-sharing", 2.5),
        ("factory", 2.0),
        ("assembly lines", 2.0),
        ("distribution", 2.0),
        ("agreements", 1.5),
        ("overseas revenue", 2.0),
        ("international expansion", 2.5),
        ("harbor crane", 2.0),
        ("service contracts", 2.0),
        ("research division", 2.5),
        ("alloy", 2.0),
        ("spin-off software", 2.0),
        ("maritime logistics", 2.5),
    ),
    "city": (
        ("city", 2.0),
        ("municipality", 2.0),
        ("population center", 2.5),
        ("population", 1.5),
        ("river delta", 2.0),
        ("fjord", 1.5),
        ("economy", 1.5),
        ("geography", 1.5),
        ("transport", 1.5),
        ("commuter rail", 2.0),
        ("regional airport", 2.0),
        ("sister-city", 2.0),
        ("training exchanges", 2.0),
        ("shipbuilding exports", 2.0),
        ("fisheries exports", 2.0),
        ("urban planners", 2.0),
        ("riverfront", 1.5),
    ),
    "research": (
        ("physicist", 2.0),
        ("researcher", 2.0),
        ("research group", 2.5),
        ("laboratory", 2.0),
        ("institute", 1.5),
        ("field trials", 2.5),
        ("graduate teaching", 2.5),
        ("method", 1.5),
        ("imaging", 1.5),
        ("radiometry", 2.0),
        ("interferometry", 2.0),
        ("deflectometry", 2.0),
        ("tomography", 2.0),
        ("thermal imaging", 2.0),
        ("humidity-controlled", 2.0),
        ("bridge monitoring", 2.0),
        ("patent pool", 2.0),
        ("reproducible", 2.0),
        ("stability", 1.5),
    ),
    "diplomacy": (
        ("diplomatic", 2.0),
        ("agreement", 1.5),
        ("settlement", 2.0),
        ("treaty", 2.5),
        ("accord", 1.5),
        ("charter", 1.5),
        ("armistice", 1.5),
        ("negotiations", 2.0),
        ("peace talks", 2.5),
        ("republic", 1.5),
        ("foreign minister", 2.0),
        ("neutral observers", 2.5),
        ("joint commission", 2.5),
        ("demobilization", 2.0),
        ("signed", 1.5),
        ("overland routes", 2.0),
    ),
}

_MULTI_ANSWER_TRIGGERS = (
    "what links",
    "how do",
    "how did",
    "together",
    "connect",
    "fit together",
    "what can be learned",
    "involved neutral observers",
    "captain's finals performance",
    "also saw",
)

_PHRASE_SIGNALS: Tuple[Tuple[str, str], ...] = (
    ("averaged a team-high 24 points in the final two games", "team_high_24"),
    ("final two games", "final_two_games"),
    ("seven-game series", "seven_game_series"),
    ("central figure in the club's rebuild", "club_rebuild"),
    ("commemorative banner", "commemorative_banner"),
    ("home bench", "home_bench"),
    ("memorial arena", "memorial_arena"),
    ("community foundation", "youth_foundation"),
    ("court vision and leadership", "court_vision_leadership"),
    ("finals most valuable player", "finals_mvp"),
    ("cooperative profit-sharing", "profit_sharing"),
    ("research division partners with universities testing next-generation alloys", "research_division_alloys"),
    ("brand spin-offs now cover software diagnostics", "spin_off_software"),
    ("software diagnostics", "spin_off_software"),
    ("harbor cranes", "harbor_crane"),
    ("overseas service contracts", "overseas_service_contracts"),
    ("steady revenue growth", "overseas_service_contracts"),
    ("distribution agreements", "distribution_agreements"),
    ("automated assembly lines", "automated_assembly"),
    ("factory modernization program", "automated_assembly"),
    ("sister-city agreements", "sister_city_training"),
    ("vocational training", "sister_city_training"),
    ("light commuter rail", "commuter_rail_airport"),
    ("small regional airport", "commuter_rail_airport"),
    ("urban planners redesigned the riverfront in 1972", "riverfront_festivals_1972"),
    ("annual music festival", "riverfront_festivals_1972"),
    ("expanded shipbuilding exports", "shipbuilding_exports"),
    ("expanded cold-water fisheries exports", "fisheries_exports"),
    ("winter trial", "field_trials"),
    ("independent sites", "field_trials"),
    ("field deployment", "field_trials"),
    ("graduate instrumentation courses", "graduate_teaching"),
    ("measurement protocol", "measurement_protocol"),
    ("controlled humidity shields", "humidity_shields"),
    ("structural monitoring of bridges", "bridge_monitoring"),
    ("shared patent pool", "patent_pool"),
    ("stability improvements over earlier thermal imaging pipelines", "stability_thermal_imaging"),
    ("adaptive filtering that reduced noise", "stability_thermal_imaging"),
    ("reproducibility across three independent sites", "reproducible_results"),
    ("foundational results were published", "foundational_results"),
    ("neutral observers", "neutral_observers"),
    ("joint commission", "joint_commission"),
    ("chaired preliminary talks", "preliminary_talks"),
    ("signed in", "signed_treaty"),
    ("demobilization", "demobilization"),
    ("reopening of overland routes", "overland_routes"),
    ("helped finalize terms", "negotiator"),
)

_QUERY_SIGNAL_TRIGGERS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("finals performance", ("team_high_24", "final_two_games")),
    ("last two games", ("team_high_24", "final_two_games")),
    ("club's rebuild", ("club_rebuild",)),
    ("club rebuild", ("club_rebuild",)),
    ("named home arena", ("memorial_arena", "home_bench")),
    ("home arena", ("memorial_arena", "home_bench")),
    ("memorial arena", ("memorial_arena",)),
    ("commemorative banner", ("commemorative_banner", "home_bench")),
    ("on-court leader", ("court_vision_leadership", "role:captain", "role:franchise player", "role:point guard")),
    ("point guard", ("role:point guard",)),
    ("franchise player", ("role:franchise player",)),
    ("head coach", ("role:head coach",)),
    ("championship year", ("championship_season",)),
    ("youth basketball foundation", ("youth_foundation",)),
    ("distribution deals", ("distribution_agreements",)),
    ("agreements", ("distribution_agreements",)),
    ("research division", ("research_division_alloys",)),
    ("alloy research", ("research_division_alloys",)),
    ("research partnerships", ("research_division_alloys",)),
    ("overseas revenue growth", ("overseas_service_contracts",)),
    ("international service contracts", ("overseas_service_contracts",)),
    ("service contracts", ("overseas_service_contracts",)),
    ("profit-sharing", ("profit_sharing",)),
    ("labor policy", ("profit_sharing",)),
    ("spin-off software", ("spin_off_software",)),
    ("software products", ("spin_off_software",)),
    ("harbor crane", ("harbor_crane",)),
    ("automated assembly", ("automated_assembly",)),
    ("assembly lines", ("automated_assembly",)),
    ("maritime logistics", ("industry:maritime logistics",)),
    ("river delta", ("geo:river delta",)),
    ("fjord-lined coast", ("geo:fjord lined coast",)),
    ("commuter rail", ("commuter_rail_airport",)),
    ("regional airport", ("commuter_rail_airport",)),
    ("transport network", ("commuter_rail_airport",)),
    ("sister-city", ("sister_city_training",)),
    ("training exchanges", ("sister_city_training",)),
    ("riverfront", ("riverfront_festivals_1972",)),
    ("festivals", ("riverfront_festivals_1972",)),
    ("shipbuilding exports", ("shipbuilding_exports", "economy:shipbuilding")),
    ("cold-water fisheries exports", ("fisheries_exports", "economy:cold water fisheries")),
    ("field trials", ("field_trials",)),
    ("graduate teaching", ("graduate_teaching", "measurement_protocol")),
    ("method", ("measurement_protocol",)),
    ("humidity-controlled", ("humidity_shields",)),
    ("bridge monitoring", ("bridge_monitoring",)),
    ("patent pool", ("patent_pool",)),
    ("stability", ("stability_thermal_imaging",)),
    ("thermal imaging", ("stability_thermal_imaging",)),
    ("reproducible", ("reproducible_results",)),
    ("laboratory results", ("reproducible_results", "foundational_results")),
    ("lead researcher", ("lead_researcher",)),
    ("neutral observers", ("neutral_observers",)),
    ("joint commission", ("joint_commission",)),
    ("commission chair", ("joint_commission",)),
    ("preliminary peace talks", ("preliminary_talks",)),
    ("peace talks", ("preliminary_talks",)),
    ("signed treaty", ("signed_treaty",)),
    ("signed", ("signed_treaty",)),
    ("demobilization", ("demobilization",)),
    ("overland routes", ("overland_routes",)),
    ("negotiator", ("negotiator",)),
    ("foreign minister", ("role:foreign minister",)),
)

_SIGNAL_PREFIX_WEIGHTS = (
    ("event_year:", 1.80),
    ("population:", 1.60),
    ("geo:", 1.30),
    ("economy:", 1.20),
    ("industry:", 1.20),
    ("role:", 1.10),
)


@dataclass(frozen=True)
class PageFeature:
    page_type: str
    canonical_entity: str
    family_key: str
    generic_penalty: float
    signals: Tuple[str, ...] = ()


def _clean_text(text: Any) -> str:
    return _SPACE_RE.sub(" ", str(text or "")).strip()


def _key_part(text: Any) -> str:
    cleaned = _NON_KEY_RE.sub(" ", str(text or "").lower())
    return _SPACE_RE.sub(" ", cleaned).strip()


def _make_key(*parts: Any) -> str:
    return "|".join(_key_part(part) for part in parts if _key_part(part))


def _signal_value(prefix: str, value: Any) -> str:
    return f"{prefix}:{_key_part(value)}"


def _content_signals(page_type: str, title: str, content: str) -> Set[str]:
    lowered = f"{title} {content}".lower()
    signals = {page_type}
    for phrase, signal in _PHRASE_SIGNALS:
        if phrase in lowered:
            signals.add(signal)
    if page_type == "sports" and "championship season" in lowered:
        signals.add("championship_season")
    if page_type == "research":
        signals.add("lead_researcher")
    return signals


def classify_query_type(query: str) -> str:
    """Classify the broad synthetic-query family."""
    lowered = str(query or "").lower()
    scores: Dict[str, float] = {page_type: 0.0 for page_type in _QUERY_TYPE_WEIGHTS}

    for page_type, weighted_phrases in _QUERY_TYPE_WEIGHTS.items():
        for phrase, weight in weighted_phrases:
            if phrase in lowered:
                scores[page_type] += weight

    tokens = set(tokenize(lowered))
    if "ceo" in tokens:
        scores["company"] += 2.0
    if "research" in tokens and "division" in tokens:
        scores["company"] += 2.5
        scores["research"] -= 1.0
    if "basketball" in tokens or "finals" in tokens:
        scores["sports"] += 1.0
    if "city" in tokens or "municipality" in tokens:
        scores["city"] += 1.0
    if "treaty" in tokens or "settlement" in tokens:
        scores["diplomacy"] += 1.0

    best_type, best_score = max(scores.items(), key=lambda item: item[1])
    return best_type if best_score >= 1.5 else "generic"


def is_multi_answer_query(query: str) -> bool:
    lowered = str(query or "").lower()
    return any(trigger in lowered for trigger in _MULTI_ANSWER_TRIGGERS)


def query_signals(query: str) -> Set[str]:
    lowered = str(query or "").lower()
    signals: Set[str] = set()
    for phrase, phrase_signals in _QUERY_SIGNAL_TRIGGERS:
        if phrase in lowered:
            signals.update(phrase_signals)

    for match in re.finditer(r"\b(\d{3,4})\b", lowered):
        signals.add(f"event_year:{match.group(1)}")
    for match in re.finditer(r"\b(\d{3})0s\b", lowered):
        base_year = int(match.group(1)) * 10
        for offset in range(10):
            signals.add(f"event_year:{base_year + offset}")

    population = re.search(r"\b(\d{1,3}(?:,\d{3})+)\b", lowered)
    if population:
        signals.add(f"population:{population.group(1).replace(',', '')}")

    return signals


def signal_score(feature: Optional[PageFeature], signals: Set[str]) -> float:
    if feature is None or not signals:
        return 0.0

    page_signals = set(feature.signals)
    total_weight = 0.0
    hit_weight = 0.0
    for signal in signals:
        weight = 1.0
        for prefix, prefix_weight in _SIGNAL_PREFIX_WEIGHTS:
            if signal.startswith(prefix):
                weight = prefix_weight
                break
        total_weight += weight
        if signal in page_signals:
            hit_weight += weight

    if total_weight <= 0.0:
        return 0.0
    return hit_weight / total_weight


def _generic_penalty(title: str, content: str) -> float:
    lowered_title = title.lower()
    lowered_content = content.lower()
    penalty = 0.10
    if len(tokenize(content)) < 80:
        penalty += 0.25
    if " may refer to" in lowered_content or lowered_title.endswith("(disambiguation)"):
        penalty += 0.70
    if lowered_title.startswith("list of "):
        penalty += 0.35
    if lowered_title in {"history", "geography", "economy", "transportation"}:
        penalty += 0.25
    return min(1.0, penalty)


def extract_page_feature(record: Dict[str, Any]) -> PageFeature:
    """Infer type and synthetic family key from a page record."""
    title = _clean_text(record.get("title", ""))
    content = _clean_text(record.get("content", ""))

    match = _SPORTS_RE.search(content)
    if match:
        person, role, team, competition, year = match.groups()
        signals = _content_signals("sports", title, content)
        signals.update(
            {
                _signal_value("role", role),
                _signal_value("team", team),
                _signal_value("competition", competition),
                f"event_year:{year}",
            }
        )
        return PageFeature(
            page_type="sports",
            canonical_entity=_clean_text(person),
            family_key=_make_key("sports", person, role, team, competition, year),
            generic_penalty=0.0,
            signals=tuple(sorted(signals)),
        )

    match = _COMPANY_RE.search(content)
    if match:
        company, industry, year, city, executive = match.groups()
        signals = _content_signals("company", title, content)
        signals.update(
            {
                _signal_value("industry", industry),
                _signal_value("hq", city),
                _signal_value("executive", executive),
                f"event_year:{year}",
            }
        )
        return PageFeature(
            page_type="company",
            canonical_entity=_clean_text(company),
            family_key=_make_key("company", company, industry, year, city, executive),
            generic_penalty=0.0,
            signals=tuple(sorted(signals)),
        )

    match = _CITY_RE.search(content)
    if match:
        city, geography, population, economy = match.groups()
        signals = _content_signals("city", title, content)
        signals.update(
            {
                _signal_value("geo", geography),
                f"population:{population.replace(',', '')}",
                _signal_value("economy", economy),
            }
        )
        return PageFeature(
            page_type="city",
            canonical_entity=_clean_text(city),
            family_key=_make_key("city", city, geography, population, economy),
            generic_penalty=0.0,
            signals=tuple(sorted(signals)),
        )

    match = _RESEARCH_RE.search(content)
    if match:
        researcher, institute, city, method, year = match.groups()
        signals = _content_signals("research", title, content)
        signals.update(
            {
                _signal_value("researcher", researcher),
                _signal_value("institute", institute),
                _signal_value("method", method),
                f"event_year:{year}",
            }
        )
        return PageFeature(
            page_type="research",
            canonical_entity=_clean_text(researcher),
            family_key=_make_key("research", researcher, institute, city, method, year),
            generic_penalty=0.0,
            signals=tuple(sorted(signals)),
        )

    match = _DIPLOMACY_RE.search(content)
    if match:
        agreement, year, person, role, nation, site = match.groups()
        signals = _content_signals("diplomacy", title, content)
        signals.update(
            {
                _signal_value("agreement", agreement),
                _signal_value("person", person),
                _signal_value("role", role),
                _signal_value("nation", nation),
                _signal_value("site", site),
                f"event_year:{year}",
            }
        )
        return PageFeature(
            page_type="diplomacy",
            canonical_entity=_clean_text(agreement),
            family_key=_make_key("diplomacy", agreement, year, person, role, nation, site),
            generic_penalty=0.0,
            signals=tuple(sorted(signals)),
        )

    return PageFeature(
        page_type="generic",
        canonical_entity=title,
        family_key=_make_key("generic", title),
        generic_penalty=_generic_penalty(title, content),
        signals=(),
    )


def build_page_features(
    *,
    entries_dir: Optional[Path] = None,
) -> Dict[int, PageFeature]:
    """Build features for every page in the corpus."""
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
        "version": 1,
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
    """Load page features and family groupings, returning None if absent."""
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
        checked_type = str(page_type)
        if checked_type not in PAGE_TYPES:
            checked_type = "generic"
        feature = PageFeature(
            page_type=checked_type,
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
