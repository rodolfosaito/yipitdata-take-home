"""Resolve messy article `company_name` strings to canonical company
metadata identities.

Resolution order (first hit wins), each tagged with a `match_method` so the
method used is auditable per-article:
  1. exact          - raw name matches a metadata key exactly (case-sensitive)
  2. exact_ci        - matches a metadata key case-insensitively
  3. alias          - matches a curated alias in config.COMPANY_ALIASES
  4. fuzzy           - best RapidFuzz whole-string ratio against metadata
                       keys clears config.FUZZY_MATCH_THRESHOLD
  5. unmatched       - none of the above; company_key falls back to the
                       raw name itself so the article/ARR observation is
                       still kept (never dropped), just without metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from rapidfuzz import fuzz

from . import config

MATCH_EXACT = "exact"
MATCH_EXACT_CI = "exact_ci"
MATCH_ALIAS = "alias"
MATCH_FUZZY = "fuzzy"
MATCH_UNMATCHED = "unmatched"


@dataclass(frozen=True)
class CompanyMatch:
    raw_name: str
    canonical_name: Optional[str]  # metadata key, or None if unmatched
    match_method: str
    match_score: Optional[float]  # 0-100 for fuzzy; 100 for exact/alias; None for unmatched


def _build_ci_lookup(metadata_keys: Iterable[str]) -> dict[str, str]:
    return {k.lower(): k for k in metadata_keys}


def _build_alias_ci_lookup() -> dict[str, str]:
    return {k.lower(): v for k, v in config.COMPANY_ALIASES.items()}


def resolve_company(raw_name: str, metadata_keys: Iterable[str]) -> CompanyMatch:
    """Resolve one raw company_name string against the set of metadata keys."""
    metadata_keys = list(metadata_keys)
    name = (raw_name or "").strip()

    if name in metadata_keys:
        return CompanyMatch(raw_name, name, MATCH_EXACT, 100.0)

    ci_lookup = _build_ci_lookup(metadata_keys)
    ci_hit = ci_lookup.get(name.lower())
    if ci_hit is not None:
        return CompanyMatch(raw_name, ci_hit, MATCH_EXACT_CI, 100.0)

    alias_hit = config.COMPANY_ALIASES.get(name)
    if alias_hit is None:
        alias_hit = _build_alias_ci_lookup().get(name.lower())
    if alias_hit is not None and alias_hit in metadata_keys:
        return CompanyMatch(raw_name, alias_hit, MATCH_ALIAS, 100.0)

    best_key, best_score = None, -1.0
    for key in metadata_keys:
        score = fuzz.ratio(name, key)
        if score > best_score:
            best_key, best_score = key, score

    if best_key is not None and best_score >= config.FUZZY_MATCH_THRESHOLD:
        return CompanyMatch(raw_name, best_key, MATCH_FUZZY, best_score)

    return CompanyMatch(raw_name, None, MATCH_UNMATCHED, best_score if best_key else None)
