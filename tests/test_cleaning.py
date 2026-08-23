"""Unit tests for pipeline.cleaning, using the real messy values observed in
tech_news.csv (not synthetic ones) as fixtures.
"""
from datetime import date

import pytest

from pipeline import cleaning


# ---------------------------------------------------------------------------
# Revenue / ARR parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected_usd",
    [
        ("$1,010,000,000", 1_010_000_000),
        ("5.2 billion", 5_200_000_000),
        ("$0.135B", 135_000_000),
        ("£244,094,488", round(244_094_488 * 1.27)),
        ("¥12,300,000,000,000", round(12_300_000_000_000 / 150)),
        ("€1,700,000,000", round(1_700_000_000 * 1.1)),
        ("500M USD", 500_000_000),
        ("75000.0M USD", 75_000_000_000),
        ("$980.0M", 980_000_000),
        ("$1.480 billion", 1_480_000_000),
        ("$32.000B", 32_000_000_000),
    ],
)
def test_parse_revenue_single_amounts(raw, expected_usd):
    result = cleaning.parse_revenue(raw)
    assert result.arr_status == cleaning.ARR_STATUS_PARSED
    assert result.arr_usd == expected_usd


@pytest.mark.parametrize(
    "raw, expected_usd",
    [
        ("$0.968B - $1.070B", round((0.968e9 + 1.070e9) / 2)),
        ("$16910.0M - $18690.0M", round((16910.0e6 + 18690.0e6) / 2)),
        ("$10M - $20M", 15_000_000),
        ("$90.2M - $99.8M", round((90.2e6 + 99.8e6) / 2)),
    ],
)
def test_parse_revenue_ranges_take_midpoint(raw, expected_usd):
    result = cleaning.parse_revenue(raw)
    assert result.arr_status == cleaning.ARR_STATUS_PARSED
    assert result.arr_usd == expected_usd


@pytest.mark.parametrize(
    "raw, expected_status",
    [
        ("", cleaning.ARR_STATUS_MISSING),
        (None, cleaning.ARR_STATUS_MISSING),
        ("N/A", cleaning.ARR_STATUS_MISSING),
        ("n/a", cleaning.ARR_STATUS_MISSING),
        ("Not disclosed", cleaning.ARR_STATUS_NOT_DISCLOSED),
    ],
)
def test_parse_revenue_missing_and_not_disclosed_never_become_numeric(raw, expected_status):
    result = cleaning.parse_revenue(raw)
    assert result.arr_status == expected_status
    assert result.arr_usd is None


def test_parse_revenue_unparseable_garbage_does_not_raise():
    result = cleaning.parse_revenue("banana")
    assert result.arr_status == cleaning.ARR_STATUS_UNPARSEABLE
    assert result.arr_usd is None


def test_parse_revenue_currency_detected():
    assert cleaning.parse_revenue("£244,094,488").currency_detected == "GBP"
    assert cleaning.parse_revenue("¥12,300,000,000,000").currency_detected == "JPY"
    assert cleaning.parse_revenue("€1,700,000,000").currency_detected == "EUR"
    assert cleaning.parse_revenue("$5,200,000,000").currency_detected == "USD"
    assert cleaning.parse_revenue("5.2B").currency_detected == "USD"  # default when no symbol


def test_parse_revenue_returns_python_int_not_float():
    result = cleaning.parse_revenue("$5,200,000,000")
    assert isinstance(result.arr_usd, int)


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2020-01-14", date(2020, 1, 14)),
        ("2020-01-14T00:00:00Z", date(2020, 1, 14)),
        ("21 Feb 2020", date(2020, 2, 21)),
        ("October 19, 2022", date(2022, 10, 19)),
    ],
)
def test_parse_published_date_unambiguous_formats(raw, expected):
    parsed, status = cleaning.parse_published_date(raw)
    assert status == cleaning.DATE_STATUS_PARSED
    assert parsed == expected


def test_parse_published_date_slash_unambiguous_day_gt_12():
    # 14 can't be a month -> must be day-first despite being a "MM/DD" shaped string
    parsed, status = cleaning.parse_published_date("14/02/2020")
    assert status == cleaning.DATE_STATUS_PARSED
    assert parsed == date(2020, 2, 14)


def test_parse_published_date_slash_ambiguous_assumes_us_month_first():
    # both components <= 12: documented assumption is US-style MM/DD/YYYY
    parsed, status = cleaning.parse_published_date("01/08/2021")
    assert status == cleaning.DATE_STATUS_PARSED
    assert parsed == date(2021, 1, 8)


def test_parse_published_date_dash_format_applies_same_ambiguity_rule():
    # day-first because 23 can't be a month
    parsed, status = cleaning.parse_published_date("23-08-2023")
    assert status == cleaning.DATE_STATUS_PARSED
    assert parsed == date(2023, 8, 23)

    # month-first because 21 can't be a month but IS a valid day and the
    # first component (06) is a valid month -> month=06, day=21
    parsed, status = cleaning.parse_published_date("06-21-2024")
    assert status == cleaning.DATE_STATUS_PARSED
    assert parsed == date(2024, 6, 21)


@pytest.mark.parametrize("raw", ["", None])
def test_parse_published_date_missing(raw):
    parsed, status = cleaning.parse_published_date(raw)
    assert parsed is None
    assert status == cleaning.DATE_STATUS_MISSING


def test_parse_published_date_unparseable_garbage():
    parsed, status = cleaning.parse_published_date("not a date")
    assert parsed is None
    assert status == cleaning.DATE_STATUS_UNPARSEABLE


def test_date_parts_extracts_year_quarter_month():
    year, quarter, month = cleaning.date_parts(date(2023, 12, 13))
    assert (year, quarter, month) == (2023, 4, 12)


def test_date_parts_none_for_missing_date():
    assert cleaning.date_parts(None) == (None, None, None)


# ---------------------------------------------------------------------------
# Category standardization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("AI & ML", "AI_ML"),
        ("AI/ML", "AI_ML"),
        ("Artificial Intelligence", "AI_ML"),
        ("Machine Learning", "AI_ML"),
        ("Analytics", "DATA_ANALYTICS"),
        ("Big Data", "DATA_ANALYTICS"),
        ("Data Analytics", "DATA_ANALYTICS"),
        ("Cloud", "CLOUD"),
        ("Cloud Computing", "CLOUD"),
        ("Cloud Services", "CLOUD"),
        ("Cybersecurity", "SECURITY"),
        ("InfoSec", "SECURITY"),
        ("Security", "SECURITY"),
        ("Finance", "FINTECH"),
        ("Financial Technology", "FINTECH"),
        ("FinTech", "FINTECH"),
        ("Enterprise Software", "SOFTWARE"),
        ("SaaS", "SOFTWARE"),
        ("Software", "SOFTWARE"),
    ],
)
def test_standardize_category_covers_all_observed_raw_values(raw, expected):
    assert cleaning.standardize_category(raw) == expected


def test_standardize_category_unknown_maps_to_other():
    assert cleaning.standardize_category("Quantum Blockchain") == "OTHER"


def test_standardize_category_is_case_insensitive():
    assert cleaning.standardize_category("ai/ml") == "AI_ML"


def test_standardize_category_none():
    from pipeline import config

    assert cleaning.standardize_category(None) == config.UNKNOWN_CATEGORY
