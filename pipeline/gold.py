"""Gold layer: warehouse-style dimensional model.

dim_company             -- one row per resolved company identity
fact_arr_observations   -- one row per article_id (the fact grain AND the
                            idempotency key: re-running the pipeline
                            recomputes this table from source and replaces
                            it wholesale / upserts on article_id, it never
                            appends duplicates)
gold_latest_arr_per_company -- derived view: most recent parsed ARR per company
gold_quarterly_arr          -- derived view: one representative ARR per
                                company per calendar quarter

`company_key` is a natural key (the canonical company name, or the raw
article company_name string when no metadata match exists) rather than a
generated surrogate integer -- this keeps every table's contents
deterministic and diffable across re-runs, which matters for idempotency.
"""
from __future__ import annotations

import pandas as pd


def build_dim_company(silver_company_metadata: pd.DataFrame, silver_articles: pd.DataFrame) -> pd.DataFrame:
    metadata_df = silver_company_metadata.set_index("company_name")
    article_keys = set(silver_articles["company_key"].dropna().unique())
    all_keys = sorted(article_keys | set(metadata_df.index))

    rows = []
    for key in all_keys:
        if key in metadata_df.index:
            meta = metadata_df.loc[key]
            rows.append(
                {
                    "company_key": key,
                    "company_name": key,
                    "metadata_matched": True,
                    "founded_year": meta["founded_year"],
                    "headquarters": meta["headquarters"],
                    "employee_count": meta["employee_count"],
                    "industry": meta["industry"],
                    "industry_std": meta["industry_std"],
                    "is_public": meta["is_public"],
                    "stock_ticker": meta["stock_ticker"],
                    "company_size_category": meta["company_size_category"],
                }
            )
        else:
            rows.append(
                {
                    "company_key": key,
                    "company_name": key,
                    "metadata_matched": False,
                    "founded_year": None,
                    "headquarters": None,
                    "employee_count": None,
                    "industry": None,
                    "industry_std": None,
                    "is_public": None,
                    "stock_ticker": None,
                    "company_size_category": None,
                }
            )
    result = pd.DataFrame(rows)
    result["founded_year"] = pd.array(result["founded_year"], dtype="Int64")
    result["employee_count"] = pd.array(result["employee_count"], dtype="Int64")
    return result


def _company_age(pub_year, founded_year):
    if pub_year is None or founded_year is None or pd.isna(pub_year) or pd.isna(founded_year):
        return None
    return int(pub_year) - int(founded_year)


def build_fact_arr_observations(silver_articles: pd.DataFrame, dim_company: pd.DataFrame) -> pd.DataFrame:
    df = silver_articles.copy()
    founded_by_key = dim_company.set_index("company_key")["founded_year"]
    df["founded_year"] = df["company_key"].map(founded_by_key)
    df["company_age"] = pd.array(
        [_company_age(y, f) for y, f in zip(df["pub_year"], df["founded_year"])], dtype="Int64"
    )

    fact = df[
        [
            "article_id",
            "company_key",
            "company_name_raw",
            "company_matched",
            "company_match_method",
            "company_match_score",
            "published_date_clean",
            "date_status",
            "pub_year",
            "pub_quarter",
            "pub_month",
            "category_std",
            "category_raw",
            "revenue_raw",
            "arr_usd",
            "arr_status",
            "currency_detected",
            "company_age",
            "_source_row_number",
            "_loaded_at",
        ]
    ].rename(columns={"published_date_clean": "published_date"})

    # Idempotency: grain is one row per article_id. Sorting gives a stable,
    # diffable file across re-runs; duplicate article_id is a hard invariant
    # violation (would indicate a bronze/source problem) rather than
    # something to silently dedupe.
    assert fact["article_id"].is_unique, "fact_arr_observations must be one row per article_id"
    return fact.sort_values("article_id").reset_index(drop=True)


def build_latest_arr_per_company(fact_arr_observations: pd.DataFrame) -> pd.DataFrame:
    """One row per company_key: the most recent *parsed* ARR observation."""
    parsed = fact_arr_observations[fact_arr_observations["arr_status"] == "parsed"].copy()
    parsed = parsed.sort_values(["company_key", "published_date"])
    latest = parsed.groupby("company_key", as_index=False).tail(1)
    return latest[
        ["company_key", "published_date", "arr_usd", "article_id", "arr_status"]
    ].rename(columns={"published_date": "latest_published_date", "arr_usd": "latest_arr_usd"}).sort_values(
        "company_key"
    ).reset_index(drop=True)


def build_quarterly_arr(fact_arr_observations: pd.DataFrame) -> pd.DataFrame:
    """One row per (company_key, year, quarter): the most recent parsed ARR
    observation reported within that quarter. ARR is a point-in-time
    reported figure (not a flow to sum), so "most recent in the quarter" is
    used as the representative value rather than an average or a sum.
    """
    parsed = fact_arr_observations[fact_arr_observations["arr_status"] == "parsed"].copy()
    parsed = parsed.dropna(subset=["pub_year", "pub_quarter"])
    parsed = parsed.sort_values(["company_key", "pub_year", "pub_quarter", "published_date"])
    quarterly = parsed.groupby(["company_key", "pub_year", "pub_quarter"], as_index=False).tail(1)
    return quarterly[
        ["company_key", "pub_year", "pub_quarter", "published_date", "arr_usd", "article_id"]
    ].rename(columns={"pub_year": "year", "pub_quarter": "quarter", "arr_usd": "quarterly_arr_usd"}).sort_values(
        ["company_key", "year", "quarter"]
    ).reset_index(drop=True)


def build_unmatched_companies(dim_company: pd.DataFrame, silver_articles: pd.DataFrame) -> pd.DataFrame:
    """Companies referenced by articles that have no metadata match, with
    counts and the best fuzzy score seen, for triage."""
    unmatched_keys = set(dim_company.loc[~dim_company["metadata_matched"], "company_key"])
    subset = silver_articles[silver_articles["company_key"].isin(unmatched_keys)]
    agg = (
        subset.groupby("company_name_raw")
        .agg(
            article_count=("article_id", "count"),
            best_fuzzy_score=("company_match_score", "max"),
            match_method=("company_match_method", "first"),
        )
        .reset_index()
        .sort_values("company_name_raw")
        .reset_index(drop=True)
    )
    return agg
