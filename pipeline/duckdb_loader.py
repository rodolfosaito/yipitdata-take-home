"""DuckDB warehouse: loads the gold tables + article embeddings, and
provides idempotent upserts plus a hybrid (SQL filter + vector similarity)
search function.

Idempotency: every load function upserts on the table's declared key
(DELETE the incoming keys, then INSERT) instead of appending, so re-running
the pipeline against the same or a refreshed source never produces
duplicate rows -- see tests/test_idempotency.py for a re-run proof.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from . import config


def get_connection(db_path=None) -> duckdb.DuckDBPyConnection:
    db_path = Path(db_path or config.DUCKDB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))


def _upsert(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame, key_cols: list[str]) -> None:
    con.register("_incoming", df)
    exists = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [table]
    ).fetchone()[0]
    if exists:
        existing_cols = [
            r[0]
            for r in con.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = ? ORDER BY ordinal_position",
                [table],
            ).fetchall()
        ]
        if existing_cols != list(df.columns):
            # Column set changed since the last run (e.g. a pipeline code
            # change added/removed a column). For this local exercise we
            # just recreate the table from the freshly-computed data rather
            # than attempt an in-place ALTER TABLE; see DATA_ARCHITECTURE.md
            # for how a production system would evolve the schema instead.
            con.execute(f"DROP TABLE {table}")
            exists = False
    if not exists:
        con.execute(f"CREATE TABLE {table} AS SELECT * FROM _incoming")
    else:
        key_predicate = " AND ".join(f"t.{c} = i.{c}" for c in key_cols)
        con.execute(
            f"DELETE FROM {table} t WHERE EXISTS "
            f"(SELECT 1 FROM _incoming i WHERE {key_predicate})"
        )
        con.execute(f"INSERT INTO {table} SELECT * FROM _incoming")
    con.unregister("_incoming")


def load_dim_company(con, dim_company: pd.DataFrame) -> None:
    _upsert(con, "dim_company", dim_company, ["company_key"])


def load_fact_arr_observations(con, fact_arr_observations: pd.DataFrame) -> None:
    _upsert(con, "fact_arr_observations", fact_arr_observations, ["article_id"])


def load_ai_articles_enriched(con, ai_articles_enriched: pd.DataFrame) -> None:
    _upsert(con, "ai_articles_enriched", ai_articles_enriched, ["article_id"])


def load_article_embeddings(con, article_ids: list[str], embeddings: np.ndarray) -> None:
    df = pd.DataFrame({"article_id": article_ids, "embedding": [row.tolist() for row in embeddings]})
    _upsert(con, "article_embeddings", df, ["article_id"])


def hybrid_search(
    con: duckdb.DuckDBPyConnection,
    query_embedding: np.ndarray,
    top_k: int = 5,
    category_std: str | None = None,
    industry_std: str | None = None,
    min_arr_usd: float | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
) -> pd.DataFrame:
    """SQL filters (category/industry/ARR/year range) over
    fact_arr_observations + dim_company, ranked by cosine similarity to
    `query_embedding` against article_embeddings. Requires dim_company,
    fact_arr_observations, and article_embeddings to already be loaded.
    """
    con.register("_query_vec", pd.DataFrame({"v": [query_embedding.tolist()]}))

    where = []
    params: list = []
    if category_std:
        where.append("f.category_std = ?")
        params.append(category_std)
    if industry_std:
        where.append("d.industry_std = ?")
        params.append(industry_std)
    if min_arr_usd is not None:
        where.append("f.arr_usd > ?")
        params.append(min_arr_usd)
    if year_min is not None:
        where.append("f.pub_year >= ?")
        params.append(year_min)
    if year_max is not None:
        where.append("f.pub_year <= ?")
        params.append(year_max)
    where_clause = f"WHERE {' AND '.join(where)}" if where else ""

    sql = f"""
        SELECT
            f.article_id,
            f.company_key,
            d.industry_std,
            f.category_std,
            f.arr_usd,
            f.pub_year,
            list_cosine_similarity(e.embedding, (SELECT v FROM _query_vec)) AS similarity
        FROM fact_arr_observations f
        JOIN dim_company d ON f.company_key = d.company_key
        JOIN article_embeddings e ON f.article_id = e.article_id
        {where_clause}
        ORDER BY similarity DESC
        LIMIT {int(top_k)}
    """
    result = con.execute(sql, params).fetchdf()
    con.unregister("_query_vec")
    return result
