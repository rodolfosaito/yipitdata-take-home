"""Reusable cleaning functions for the messy `revenue`, `published_date`, and
`category` columns in tech_news.csv.

Every function here is pure (string/scalar in, typed value out) so it can be
unit tested directly against the real messy values found in the source file,
and reused by both the silver-layer builder and any future batch.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

from . import config

# ---------------------------------------------------------------------------
# Revenue / ARR cleaning
# ---------------------------------------------------------------------------

# Statuses:
#   parsed        -> a numeric ARR value was successfully extracted
#   missing        -> blank/null/"N/A": no value was ever provided
#   not_disclosed  -> the source explicitly states the figure is undisclosed
#   unparseable    -> a non-empty value was provided but could not be parsed
ARR_STATUS_PARSED = "parsed"
ARR_STATUS_MISSING = "missing"
ARR_STATUS_NOT_DISCLOSED = "not_disclosed"
ARR_STATUS_UNPARSEABLE = "unparseable"

_MULTIPLIER_WORD_RE = re.compile(r"\b(billion|million)\b", re.IGNORECASE)
_MULTIPLIER_LETTER_RE = re.compile(r"(?<=[0-9])\s*([MB])\b", re.IGNORECASE)
_CURRENCY_WORD_RE = re.compile(r"\b(USD|EUR|GBP|JPY)\b", re.IGNORECASE)
_RANGE_SPLIT_RE = re.compile(r"\s-\s")


@dataclass(frozen=True)
class RevenueParseResult:
    """Result of parsing one raw `revenue` string."""

    arr_usd: Optional[int]
    arr_status: str
    currency_detected: Optional[str]
    raw_value: str


def _parse_single_amount(text: str) -> Optional[tuple[float, str]]:
    """Parse a single (non-range) amount string, e.g. "$0.135B", "5.2 billion",
    "500M USD", "£244,094,488". Returns (amount_in_native_currency, currency_code)
    or None if the string cannot be parsed as a number.
    """
    s = text.strip()
    if not s:
        return None

    currency = "USD"  # default when no symbol/word is present (e.g. "5.2B")
    for symbol, code in config.CURRENCY_SYMBOLS.items():
        if symbol in s:
            currency = code
            s = s.replace(symbol, "")
            break

    m = _CURRENCY_WORD_RE.search(s)
    if m:
        currency = m.group(1).upper()
        s = s[: m.start()] + s[m.end() :]

    multiplier = 1.0
    m = _MULTIPLIER_WORD_RE.search(s)
    if m:
        multiplier = 1e9 if m.group(1).lower() == "billion" else 1e6
        s = s[: m.start()] + s[m.end() :]
    else:
        m = _MULTIPLIER_LETTER_RE.search(s)
        if m:
            multiplier = 1e9 if m.group(1).upper() == "B" else 1e6
            s = s[: m.start()] + s[m.end() :]

    s = s.strip().rstrip(".").strip().replace(",", "")
    if not s:
        return None
    try:
        value = float(s)
    except ValueError:
        return None
    return value * multiplier, currency


def parse_revenue(raw_value) -> RevenueParseResult:
    """Clean one raw `revenue` cell into a normalized USD ARR observation.

    Handles: USD/EUR/GBP/JPY, "$5.2B"/"5.2 billion"/"$5,200,000,000"/"500M USD"
    style amounts, "$10M - $20M" ranges (midpoint), and missing/N/A/"Not
    disclosed" values. Never returns a numeric value for missing or
    undisclosed input -- callers must check `arr_status`.
    """
    raw_str = "" if raw_value is None else str(raw_value)
    s = raw_str.strip()

    if s == "" or s.upper() == "N/A" or s.lower() == "nan":
        return RevenueParseResult(None, ARR_STATUS_MISSING, None, raw_str)
    if s.lower() == "not disclosed":
        return RevenueParseResult(None, ARR_STATUS_NOT_DISCLOSED, None, raw_str)

    if _RANGE_SPLIT_RE.search(s):
        left_str, right_str = _RANGE_SPLIT_RE.split(s, maxsplit=1)
        left = _parse_single_amount(left_str)
        right = _parse_single_amount(right_str)
        if left is None or right is None:
            return RevenueParseResult(None, ARR_STATUS_UNPARSEABLE, None, raw_str)
        (low_val, low_cur), (high_val, high_cur) = left, right
        # Assumption: both sides of a range share one currency (always true
        # in the observed data); we use the left side's currency for the
        # detected-currency column.
        low_usd = low_val * config.CURRENCY_TO_USD.get(low_cur, 1.0)
        high_usd = high_val * config.CURRENCY_TO_USD.get(high_cur, 1.0)
        midpoint = (low_usd + high_usd) / 2.0
        return RevenueParseResult(int(round(midpoint)), ARR_STATUS_PARSED, low_cur, raw_str)

    parsed = _parse_single_amount(s)
    if parsed is None:
        return RevenueParseResult(None, ARR_STATUS_UNPARSEABLE, None, raw_str)
    value, currency = parsed
    usd = value * config.CURRENCY_TO_USD.get(currency, 1.0)
    return RevenueParseResult(int(round(usd)), ARR_STATUS_PARSED, currency, raw_str)


# ---------------------------------------------------------------------------
# Date normalization
# ---------------------------------------------------------------------------

DATE_STATUS_PARSED = "parsed"
DATE_STATUS_MISSING = "missing"
DATE_STATUS_UNPARSEABLE = "unparseable"

# Formats with no day/month ambiguity: the month is either explicit (ISO,
# zero-padded numeric position) or spelled out as a word.
_UNAMBIGUOUS_FORMATS = [
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d",
    "%d %b %Y",  # "21 Feb 2020"
    "%B %d, %Y",  # "October 19, 2022"
]

_SLASH_NUMERIC_RE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")
_DASH_NUMERIC_RE = re.compile(r"^(\d{1,2})-(\d{1,2})-(\d{4})$")


def _try_unambiguous_formats(s: str) -> Optional[date]:
    for fmt in _UNAMBIGUOUS_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _resolve_ambiguous_numeric_date(a: int, b: int, year: int) -> Optional[date]:
    """Resolve a numeric a/b/year date where `a` and `b` could each be the
    day or the month.

    Rule (documented in the assignment and applied consistently to both
    "/" and "-" separated numeric dates, since both formats appear with a
    mix of day-first and month-first values in this dataset):
      - If exactly one of a/b is > 12, that component MUST be the day
        (the other is the month).
      - Otherwise (both <= 12, genuinely ambiguous -- e.g. "01/08/2021")
        we default to US-style month-first (a=month, b=day). This is a
        documented assumption, not a detected fact: some of those rows
        could actually be day-first. See DATA_ARCHITECTURE.md.
    """
    if a > 12 and b <= 12:
        month, day = b, a
    elif b > 12 and a <= 12:
        month, day = a, b
    elif a <= 12 and b <= 12:
        month, day = a, b  # ambiguous -> assume US-style MM/DD
    else:
        return None  # both > 12: not a valid date
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_published_date(raw_value) -> tuple[Optional[date], str]:
    """Clean one raw `published_date` cell.

    Returns (date_or_None, status) where status is one of parsed/missing/
    unparseable. Handles ISO (with and without time), "DD Mon YYYY",
    "Month DD, YYYY", and ambiguous numeric "MM/DD/YYYY" or "DD-MM-YYYY"
    style dates (see `_resolve_ambiguous_numeric_date` for the ambiguity
    rule).
    """
    raw_str = "" if raw_value is None else str(raw_value)
    s = raw_str.strip()
    if not s:
        return None, DATE_STATUS_MISSING

    parsed = _try_unambiguous_formats(s)
    if parsed is not None:
        return parsed, DATE_STATUS_PARSED

    m = _SLASH_NUMERIC_RE.match(s) or _DASH_NUMERIC_RE.match(s)
    if m:
        a, b, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        parsed = _resolve_ambiguous_numeric_date(a, b, year)
        if parsed is not None:
            return parsed, DATE_STATUS_PARSED

    return None, DATE_STATUS_UNPARSEABLE


def date_parts(d: Optional[date]) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """Return (year, quarter, month) for filtering/analysis, or (None, None, None)."""
    if d is None:
        return None, None, None
    quarter = (d.month - 1) // 3 + 1
    return d.year, quarter, d.month


# ---------------------------------------------------------------------------
# Category standardization
# ---------------------------------------------------------------------------


def standardize_category(raw_value) -> str:
    """Map a raw `category` (or company `industry`) string to the canonical
    taxonomy defined in config.CATEGORY_TAXONOMY. Unrecognized/blank values
    map to config.UNKNOWN_CATEGORY rather than being dropped or raising.
    """
    if raw_value is None:
        return config.UNKNOWN_CATEGORY
    key = str(raw_value).strip().lower()
    return config.CATEGORY_RAW_TO_STD.get(key, config.UNKNOWN_CATEGORY)
