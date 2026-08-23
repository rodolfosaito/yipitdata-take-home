"""Tests for gold-layer table construction and the ai_articles_enriched
export filter, using small synthetic silver-shaped DataFrames so they run
fast and independently of the real source files.
"""
import pandas as pd
import pytest

from pipeline import config, exports, gold


def _silver_company_metadata():
    return pd.DataFrame(
        [
            {
                "company_name": "Acme",
                "founded_year": 2000,
                "headquarters": "NYC",
                "employee_count": 5000,
                "industry": "AI/ML",
                "industry_std": "AI_ML",
                "is_public": True,
                "stock_ticker": "ACME",
                "company_size_category": "Small",
                "_source_row_number": 1,
                "_source_file": "company_metadata.json",
                "_loaded_at": "2024-01-01T00:00:00Z",
            }
        ]
    )


def _silver_articles():
    return pd.DataFrame(
        [
            {
                # matched company, qualifies for AI export
                "article_id": "ART0001",
                "title": "Acme raises huge round",
                "company_name_raw": "Acme",
                "company_key": "Acme",
                "company_matched": True,
                "company_match_method": "exact",
                "company_match_score": 100.0,
                "published_date_raw": "2023-01-01",
                "published_date_clean": pd.Timestamp("2023-01-01"),
                "date_status": "parsed",
                "pub_year": 2023,
                "pub_quarter": 1,
                "pub_month": 1,
                "category_raw": "AI/ML",
                "category_std": "AI_ML",
                "revenue_raw": "$100M",
                "arr_usd": 100_000_000,
                "arr_status": "parsed",
                "currency_detected": "USD",
                "summary": "Big round.",
                "url": "https://example.com/1",
                "author": "A",
                "word_count": "100",
                "_source_row_number": 1,
                "_source_file": "tech_news.csv",
                "_loaded_at": "2024-01-01T00:00:00Z",
            },
            {
                # unmatched company, ARR present but category isn't AI -> excluded
                "article_id": "ART0002",
                "title": "Widgets Co ships a widget",
                "company_name_raw": "Widgets Co",
                "company_key": "Widgets Co",
                "company_matched": False,
                "company_match_method": "unmatched",
                "company_match_score": 20.0,
                "published_date_raw": "2023-02-01",
                "published_date_clean": pd.Timestamp("2023-02-01"),
                "date_status": "parsed",
                "pub_year": 2023,
                "pub_quarter": 1,
                "pub_month": 2,
                "category_raw": "Software",
                "category_std": "SOFTWARE",
                "revenue_raw": "$200M",
                "arr_usd": 200_000_000,
                "arr_status": "parsed",
                "currency_detected": "USD",
                "summary": "Ships widgets.",
                "url": "https://example.com/2",
                "author": "B",
                "word_count": "80",
                "_source_row_number": 2,
                "_source_file": "tech_news.csv",
                "_loaded_at": "2024-01-01T00:00:00Z",
            },
            {
                # AI category but ARR below $50M threshold -> excluded
                "article_id": "ART0003",
                "title": "Acme ships a small feature",
                "company_name_raw": "Acme",
                "company_key": "Acme",
                "company_matched": True,
                "company_match_method": "exact",
                "company_match_score": 100.0,
                "published_date_raw": "2023-03-01",
                "published_date_clean": pd.Timestamp("2023-03-01"),
                "date_status": "parsed",
                "pub_year": 2023,
                "pub_quarter": 1,
                "pub_month": 3,
                "category_raw": "AI/ML",
                "category_std": "AI_ML",
                "revenue_raw": "$10M",
                "arr_usd": 10_000_000,
                "arr_status": "parsed",
                "currency_detected": "USD",
                "summary": "Small feature.",
                "url": "https://example.com/3",
                "author": "C",
                "word_count": "60",
                "_source_row_number": 3,
                "_source_file": "tech_news.csv",
                "_loaded_at": "2024-01-01T00:00:00Z",
            },
            {
                # AI category, big ARR, but wrong year -> excluded
                "article_id": "ART0004",
                "title": "Acme raises another round",
                "company_name_raw": "Acme",
                "company_key": "Acme",
                "company_matched": True,
                "company_match_method": "exact",
                "company_match_score": 100.0,
                "published_date_raw": "2021-01-01",
                "published_date_clean": pd.Timestamp("2021-01-01"),
                "date_status": "parsed",
                "pub_year": 2021,
                "pub_quarter": 1,
                "pub_month": 1,
                "category_raw": "AI/ML",
                "category_std": "AI_ML",
                "revenue_raw": "$500M",
                "arr_usd": 500_000_000,
                "arr_status": "parsed",
                "currency_detected": "USD",
                "summary": "Another round.",
                "url": "https://example.com/4",
                "author": "D",
                "word_count": "70",
                "_source_row_number": 4,
                "_source_file": "tech_news.csv",
                "_loaded_at": "2024-01-01T00:00:00Z",
            },
            {
                # AI category, big number, but ARR is "not_disclosed" -> must
                # never qualify even though some downstream number might exist
                "article_id": "ART0005",
                "title": "Acme reportedly worth a lot",
                "company_name_raw": "Acme",
                "company_key": "Acme",
                "company_matched": True,
                "company_match_method": "exact",
                "company_match_score": 100.0,
                "published_date_raw": "2023-05-01",
                "published_date_clean": pd.Timestamp("2023-05-01"),
                "date_status": "parsed",
                "pub_year": 2023,
                "pub_quarter": 2,
                "pub_month": 5,
                "category_raw": "AI/ML",
                "category_std": "AI_ML",
                "revenue_raw": "Not disclosed",
                "arr_usd": None,
                "arr_status": "not_disclosed",
                "currency_detected": None,
                "summary": "Undisclosed.",
                "url": "https://example.com/5",
                "author": "E",
                "word_count": "50",
                "_source_row_number": 5,
                "_source_file": "tech_news.csv",
                "_loaded_at": "2024-01-01T00:00:00Z",
            },
        ]
    )


@pytest.fixture
def silver_pair():
    return _silver_company_metadata(), _silver_articles()


def test_dim_company_keeps_unmatched_companies_with_null_metadata(silver_pair):
    silver_meta, silver_articles = silver_pair
    dim_company = gold.build_dim_company(silver_meta, silver_articles)
    widgets = dim_company[dim_company["company_key"] == "Widgets Co"].iloc[0]
    assert widgets["metadata_matched"] == False  # noqa: E712
    assert pd.isna(widgets["founded_year"])
    assert pd.isna(widgets["industry"])

    acme = dim_company[dim_company["company_key"] == "Acme"].iloc[0]
    assert acme["metadata_matched"] == True  # noqa: E712
    assert acme["founded_year"] == 2000


def test_fact_arr_observations_grain_is_one_row_per_article(silver_pair):
    silver_meta, silver_articles = silver_pair
    dim_company = gold.build_dim_company(silver_meta, silver_articles)
    fact = gold.build_fact_arr_observations(silver_articles, dim_company)
    assert len(fact) == len(silver_articles)
    assert fact["article_id"].is_unique


def test_fact_arr_observations_never_drops_unmatched_company_rows(silver_pair):
    silver_meta, silver_articles = silver_pair
    dim_company = gold.build_dim_company(silver_meta, silver_articles)
    fact = gold.build_fact_arr_observations(silver_articles, dim_company)
    assert "ART0002" in set(fact["article_id"])  # the unmatched-company row


def test_fact_arr_observations_computes_company_age(silver_pair):
    silver_meta, silver_articles = silver_pair
    dim_company = gold.build_dim_company(silver_meta, silver_articles)
    fact = gold.build_fact_arr_observations(silver_articles, dim_company)
    row = fact[fact["article_id"] == "ART0001"].iloc[0]
    assert row["company_age"] == 2023 - 2000  # published 2023, founded 2000


def test_fact_arr_observations_company_age_null_when_unmatched(silver_pair):
    silver_meta, silver_articles = silver_pair
    dim_company = gold.build_dim_company(silver_meta, silver_articles)
    fact = gold.build_fact_arr_observations(silver_articles, dim_company)
    row = fact[fact["article_id"] == "ART0002"].iloc[0]
    assert pd.isna(row["company_age"])


def test_ai_articles_enriched_filter_ai_year_and_arr_threshold(silver_pair):
    silver_meta, silver_articles = silver_pair
    dim_company = gold.build_dim_company(silver_meta, silver_articles)
    fact = gold.build_fact_arr_observations(silver_articles, dim_company)
    result = exports.build_ai_articles_enriched(silver_articles, dim_company, fact)
    # Only ART0001 satisfies: AI category, 2023 (in [2022,2024]), ARR > $50M, parsed.
    assert list(result["article_id"]) == ["ART0001"]


def test_ai_articles_enriched_excludes_not_disclosed_even_if_ai_and_in_range(silver_pair):
    silver_meta, silver_articles = silver_pair
    dim_company = gold.build_dim_company(silver_meta, silver_articles)
    fact = gold.build_fact_arr_observations(silver_articles, dim_company)
    result = exports.build_ai_articles_enriched(silver_articles, dim_company, fact)
    assert "ART0005" not in set(result["article_id"])


def test_latest_arr_per_company_picks_most_recent_parsed_observation():
    fact = pd.DataFrame(
        [
            {"article_id": "A1", "company_key": "Acme", "published_date": pd.Timestamp("2022-01-01"), "arr_usd": 10, "arr_status": "parsed", "pub_year": 2022, "pub_quarter": 1},
            {"article_id": "A2", "company_key": "Acme", "published_date": pd.Timestamp("2023-06-01"), "arr_usd": 20, "arr_status": "parsed", "pub_year": 2023, "pub_quarter": 2},
            {"article_id": "A3", "company_key": "Acme", "published_date": pd.Timestamp("2024-01-01"), "arr_usd": None, "arr_status": "missing", "pub_year": 2024, "pub_quarter": 1},
        ]
    )
    latest = gold.build_latest_arr_per_company(fact)
    assert len(latest) == 1
    assert latest.iloc[0]["article_id"] == "A2"
    assert latest.iloc[0]["latest_arr_usd"] == 20


def test_quarterly_arr_one_row_per_company_year_quarter():
    fact = pd.DataFrame(
        [
            {"article_id": "A1", "company_key": "Acme", "published_date": pd.Timestamp("2023-01-15"), "arr_usd": 10, "arr_status": "parsed", "pub_year": 2023, "pub_quarter": 1},
            {"article_id": "A2", "company_key": "Acme", "published_date": pd.Timestamp("2023-02-15"), "arr_usd": 15, "arr_status": "parsed", "pub_year": 2023, "pub_quarter": 1},
            {"article_id": "A3", "company_key": "Acme", "published_date": pd.Timestamp("2023-04-01"), "arr_usd": 30, "arr_status": "parsed", "pub_year": 2023, "pub_quarter": 2},
        ]
    )
    quarterly = gold.build_quarterly_arr(fact)
    assert len(quarterly) == 2
    q1 = quarterly[(quarterly["year"] == 2023) & (quarterly["quarter"] == 1)].iloc[0]
    assert q1["quarterly_arr_usd"] == 15  # the later of the two Q1 observations
