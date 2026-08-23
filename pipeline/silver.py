"""Silver layer: typed, cleaned, resolved columns -- one row per bronze row.

silver_articles keeps the original raw strings alongside every derived/
cleaned column, so a failed parse or an unmatched company can always be
traced back to exactly what the source contained.
"""
from __future__ import annotations

import pandas as pd

from . import cleaning, config
from .company_resolution import resolve_company


def build_silver_company_metadata(bronze_metadata: pd.DataFrame) -> pd.DataFrame:
    df = bronze_metadata.copy()
    df["industry_std"] = df["industry"].apply(cleaning.standardize_category)
    df["company_size_category"] = df["employee_count"].apply(_size_category)
    df["founded_year"] = pd.array(df["founded_year"], dtype="Int64")
    df["employee_count"] = pd.array(df["employee_count"], dtype="Int64")
    return df[
        [
            "company_name_raw",
            "founded_year",
            "headquarters",
            "employee_count",
            "industry",
            "industry_std",
            "is_public",
            "stock_ticker",
            "company_size_category",
            "_source_row_number",
            "_source_file",
            "_loaded_at",
        ]
    ].rename(columns={"company_name_raw": "company_name"})


def _size_category(employee_count) -> str | None:
    if employee_count is None or (isinstance(employee_count, float) and pd.isna(employee_count)):
        return None
    n = int(employee_count)
    if n < config.SIZE_SMALL_MAX:
        return "Small"
    if n <= config.SIZE_MEDIUM_MAX:
        return "Medium"
    return "Large"


def build_silver_articles(bronze_articles: pd.DataFrame, metadata_keys: list[str]) -> pd.DataFrame:
    df = bronze_articles.copy()

    # --- revenue / ARR ---
    revenue_results = df["revenue"].apply(cleaning.parse_revenue)
    # Int64 (nullable) keeps ARR as whole-dollar integers per the spec while
    # still allowing NA for missing/not_disclosed/unparseable rows -- a plain
    # numpy int column can't hold NA and float64 would print "310000000.0".
    df["arr_usd"] = pd.array([r.arr_usd for r in revenue_results], dtype="Int64")
    df["arr_status"] = [r.arr_status for r in revenue_results]
    df["currency_detected"] = [r.currency_detected for r in revenue_results]

    # --- published_date ---
    date_results = df["published_date"].apply(cleaning.parse_published_date)
    parsed_dates = [d for d, _ in date_results]
    df["published_date_clean"] = pd.to_datetime(pd.Series(parsed_dates))
    df["date_status"] = [status for _, status in date_results]
    year_quarter_month = [cleaning.date_parts(d) for d in parsed_dates]
    df["pub_year"] = pd.array([y for y, _, _ in year_quarter_month], dtype="Int64")
    df["pub_quarter"] = pd.array([q for _, q, _ in year_quarter_month], dtype="Int64")
    df["pub_month"] = pd.array([m for _, _, m in year_quarter_month], dtype="Int64")

    # --- category ---
    df["category_std"] = df["category"].apply(cleaning.standardize_category)

    # --- company resolution ---
    matches = df["company_name"].apply(lambda name: resolve_company(name, metadata_keys))
    df["company_key"] = [m.canonical_name if m.canonical_name else m.raw_name.strip() for m in matches]
    df["company_matched"] = [m.canonical_name is not None for m in matches]
    df["company_match_method"] = [m.match_method for m in matches]
    df["company_match_score"] = [m.match_score for m in matches]

    df = df.rename(
        columns={
            "company_name": "company_name_raw",
            "revenue": "revenue_raw",
            "published_date": "published_date_raw",
            "category": "category_raw",
        }
    )

    return df[
        [
            "article_id",
            "title",
            "company_name_raw",
            "company_key",
            "company_matched",
            "company_match_method",
            "company_match_score",
            "published_date_raw",
            "published_date_clean",
            "date_status",
            "pub_year",
            "pub_quarter",
            "pub_month",
            "category_raw",
            "category_std",
            "revenue_raw",
            "arr_usd",
            "arr_status",
            "currency_detected",
            "summary",
            "url",
            "author",
            "word_count",
            "_source_row_number",
            "_source_file",
            "_loaded_at",
        ]
    ]
