"""CSV export helpers: the AI-article dataset and all warehouse table CSVs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def build_ai_articles_enriched(
    silver_articles: pd.DataFrame,
    dim_company: pd.DataFrame,
    fact_arr_observations: pd.DataFrame,
    embeddings: np.ndarray | None = None,
    embedding_article_ids: list[str] | None = None,
    top_similar: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Articles where (category is AI/ML OR company industry is AI/ML) AND
    published in [2022, 2024] AND arr_usd > $50M (only successfully parsed
    ARR values can qualify -- missing/unparseable ARR never passes here).
    """
    df = silver_articles.merge(dim_company, on="company_key", how="left", suffixes=("", "_dim"))
    df = df.merge(fact_arr_observations[["article_id", "company_age"]], on="article_id", how="left")

    is_ai_category = df["category_std"] == "AI_ML"
    is_ai_industry = df["industry_std"] == "AI_ML"
    in_year_range = df["pub_year"].between(config.AI_ARTICLE_YEAR_MIN, config.AI_ARTICLE_YEAR_MAX)
    has_qualifying_arr = (df["arr_status"] == "parsed") & (df["arr_usd"] > config.AI_ARTICLE_ARR_MIN_USD)

    mask = (is_ai_category | is_ai_industry) & in_year_range & has_qualifying_arr
    result = df.loc[
        mask,
        [
            "article_id",
            "title",
            "company_key",
            "published_date_clean",
            "category_std",
            "arr_usd",
            "summary",
            "url",
            "industry",
            "founded_year",
            "headquarters",
            "employee_count",
            "is_public",
            "stock_ticker",
            "company_age",
            "company_size_category",
        ],
    ].rename(
        columns={
            "company_key": "company_name",
            "published_date_clean": "published_date",
            "category_std": "category",
        }
    )

    if embeddings is not None and embedding_article_ids is not None:
        emb_by_id = {aid: emb for aid, emb in zip(embedding_article_ids, embeddings)}
        result["embedding"] = [
            json.dumps([round(float(x), 6) for x in emb_by_id[aid]]) if aid in emb_by_id else None
            for aid in result["article_id"]
        ]
    if top_similar is not None:
        result["top_similar_articles"] = [
            json.dumps(top_similar.get(aid, [])) for aid in result["article_id"]
        ]

    return result.sort_values("article_id").reset_index(drop=True)


def write_csv(df: pd.DataFrame, filename: str, output_dir=None) -> Path:
    output_dir = Path(output_dir or config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    df.to_csv(path, index=False)
    return path
