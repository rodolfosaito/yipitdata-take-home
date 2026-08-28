"""DuckDB warehouse: loads every layer (bronze/silver/gold + embeddings) as
persisted, independently re-loadable checkpoints, and provides a hybrid
(SQL filter + vector similarity) search function.

Idempotency: every load function upserts on the table's declared key
(DELETE the incoming keys, then INSERT) instead of appending, so re-running
the pipeline against the same or a refreshed source never produces
duplicate rows -- see tests/test_idempotency.py for a re-run proof.

Checkpointing: `save_checkpoint`/`load_checkpoint`/`checkpoint_exists` let
`pipeline/run_pipeline.py` run any single stage (bronze/silver/gold/
embeddings/exports) standalone, by reading the previous stage's persisted
DuckDB table instead of requiring an in-memory DataFrame from the same
process. `check_freshness` is a lightweight (mtime+size, not a content
hash) staleness check: it warns -- or, with `strict=True`, raises -- when a
checkpoint was built from a different `tech_news.csv`/`company_metadata.json`
than what's currently on disk. This is deliberately cheap, not a real
data-versioning system.
"""
from __future__ import annotations

from datetime import datetime, timezone
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


# ---------------------------------------------------------------------------
# Checkpoint API: every pipeline stage's output is a table any later stage
# (in this process or a fresh one) can read back with `load_checkpoint`.
# ---------------------------------------------------------------------------


def _ensure_meta_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        "CREATE TABLE IF NOT EXISTS _checkpoint_meta ("
        "stage VARCHAR PRIMARY KEY, source_signature VARCHAR, "
        "row_count BIGINT, checkpointed_at VARCHAR)"
    )


def file_signature(path) -> str:
    """A cheap (mtime_ns, size) fingerprint for a source file -- enough to
    detect "this file was touched since the checkpoint was built", not a
    content hash."""
    stat = Path(path).stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def combined_source_signature() -> str:
    """Fingerprint of both raw source files together. Every checkpoint in
    this pipeline (bronze through gold) is ultimately derived from both
    tech_news.csv and company_metadata.json, so all of them are compared
    against this same combined signature rather than tracking per-hop
    lineage through intermediate tables."""
    return f"{file_signature(config.TECH_NEWS_CSV)}|{file_signature(config.COMPANY_METADATA_JSON)}"


def save_checkpoint(
    con: duckdb.DuckDBPyConnection,
    table: str,
    df: pd.DataFrame,
    key_cols: list[str],
    source_signature: str | None = None,
) -> None:
    """Upsert `df` into `table` keyed on `key_cols`. If `source_signature`
    is given, records it in `_checkpoint_meta` for later freshness checks."""
    _upsert(con, table, df, key_cols)
    if source_signature is not None:
        _ensure_meta_table(con)
        con.execute("DELETE FROM _checkpoint_meta WHERE stage = ?", [table])
        con.execute(
            "INSERT INTO _checkpoint_meta VALUES (?, ?, ?, ?)",
            [table, source_signature, len(df), datetime.now(timezone.utc).isoformat()],
        )


def load_checkpoint(con: duckdb.DuckDBPyConnection, table: str) -> pd.DataFrame:
    return con.execute(f"SELECT * FROM {table}").fetchdf()


def checkpoint_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    return (
        con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [table]
        ).fetchone()[0]
        > 0
    )


def check_freshness(
    con: duckdb.DuckDBPyConnection, table: str, expected_signature: str, strict: bool = False
) -> None:
    """Warn (or, if strict, raise) when `table`'s checkpoint was built from
    a source signature that no longer matches `expected_signature` -- i.e.
    tech_news.csv/company_metadata.json changed since this checkpoint was
    last computed."""
    _ensure_meta_table(con)
    row = con.execute(
        "SELECT source_signature, checkpointed_at FROM _checkpoint_meta WHERE stage = ?", [table]
    ).fetchone()
    if row is None or row[0] == expected_signature:
        return
    message = (
        f"WARNING: checkpoint '{table}' (built {row[1]}) was computed from a different "
        f"version of tech_news.csv/company_metadata.json than what's on disk now. "
        f"Re-run the stage(s) that produce it (or `--stage all`) to refresh it."
    )
    if strict:
        raise SystemExit(f"{message}\n(Failing because --strict-checkpoints was set.)")
    print(message)


def load_dim_company(con, dim_company: pd.DataFrame, source_signature: str | None = None) -> None:
    save_checkpoint(con, "dim_company", dim_company, ["company_key"], source_signature)


def load_fact_arr_observations(
    con, fact_arr_observations: pd.DataFrame, source_signature: str | None = None
) -> None:
    save_checkpoint(con, "fact_arr_observations", fact_arr_observations, ["article_id"], source_signature)


def load_ai_articles_enriched(
    con, ai_articles_enriched: pd.DataFrame, source_signature: str | None = None
) -> None:
    save_checkpoint(con, "ai_articles_enriched", ai_articles_enriched, ["article_id"], source_signature)


def load_article_embeddings(
    con, article_ids: list[str], embeddings: np.ndarray, source_signature: str | None = None
) -> None:
    df = pd.DataFrame({"article_id": article_ids, "embedding": [row.tolist() for row in embeddings]})
    save_checkpoint(con, "article_embeddings", df, ["article_id"], source_signature)


def load_bronze_articles_checkpoint(con, df: pd.DataFrame, source_signature: str | None = None) -> None:
    save_checkpoint(con, "bronze_articles", df, ["article_id"], source_signature)


def load_bronze_metadata_checkpoint(con, df: pd.DataFrame, source_signature: str | None = None) -> None:
    save_checkpoint(con, "bronze_company_metadata", df, ["company_name_raw"], source_signature)


def load_silver_articles_checkpoint(con, df: pd.DataFrame, source_signature: str | None = None) -> None:
    save_checkpoint(con, "silver_articles", df, ["article_id"], source_signature)


def load_silver_metadata_checkpoint(con, df: pd.DataFrame, source_signature: str | None = None) -> None:
    save_checkpoint(con, "silver_company_metadata", df, ["company_name"], source_signature)


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
