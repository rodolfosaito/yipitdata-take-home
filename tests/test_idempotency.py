"""Proves the DuckDB loader upserts on the declared key instead of
appending: re-running a load with the same data doesn't duplicate rows, a
corrected value overwrites in place, and a genuinely new row is added
without disturbing existing ones. This is the re-run/backfill guarantee for
fact_arr_observations (grain = one row per article_id).
"""
import pandas as pd

from pipeline import duckdb_loader


def _fact_df(rows):
    return pd.DataFrame(rows, columns=["article_id", "company_key", "arr_usd"])


def test_reloading_identical_data_does_not_duplicate_rows(tmp_path):
    db_path = tmp_path / "test.duckdb"
    con = duckdb_loader.get_connection(db_path)
    df = _fact_df([("ART0001", "Acme", 100), ("ART0002", "Acme", 200)])

    duckdb_loader.load_fact_arr_observations(con, df)
    duckdb_loader.load_fact_arr_observations(con, df)  # re-run, same data

    count = con.execute("SELECT count(*) FROM fact_arr_observations").fetchone()[0]
    assert count == 2
    con.close()


def test_reload_with_corrected_value_overwrites_not_appends(tmp_path):
    db_path = tmp_path / "test.duckdb"
    con = duckdb_loader.get_connection(db_path)
    v1 = _fact_df([("ART0001", "Acme", 100)])
    duckdb_loader.load_fact_arr_observations(con, v1)

    v2_corrected = _fact_df([("ART0001", "Acme", 999)])  # simulates a backfill correction
    duckdb_loader.load_fact_arr_observations(con, v2_corrected)

    rows = con.execute("SELECT article_id, arr_usd FROM fact_arr_observations").fetchall()
    assert rows == [("ART0001", 999)]  # one row, corrected value -- not two rows
    con.close()


def test_reload_with_a_new_article_adds_exactly_one_row(tmp_path):
    db_path = tmp_path / "test.duckdb"
    con = duckdb_loader.get_connection(db_path)
    v1 = _fact_df([("ART0001", "Acme", 100)])
    duckdb_loader.load_fact_arr_observations(con, v1)

    v2_with_new_article = _fact_df([("ART0001", "Acme", 100), ("ART0002", "Acme", 200)])
    duckdb_loader.load_fact_arr_observations(con, v2_with_new_article)

    count = con.execute("SELECT count(*) FROM fact_arr_observations").fetchone()[0]
    assert count == 2
    con.close()


def test_full_pipeline_rerun_produces_stable_row_counts():
    """End-to-end proof against the real source files: running the gold-layer
    build twice from scratch yields identical row counts and no duplicate
    article_ids -- the pipeline is safe to re-run on an unchanged source.
    """
    from pipeline import bronze, gold, silver

    def build_fact():
        bronze_articles = bronze.load_bronze_articles()
        bronze_metadata = bronze.load_bronze_company_metadata()
        metadata_keys = bronze_metadata["company_name_raw"].tolist()
        silver_meta = silver.build_silver_company_metadata(bronze_metadata)
        silver_articles = silver.build_silver_articles(bronze_articles, metadata_keys)
        dim_company = gold.build_dim_company(silver_meta, silver_articles)
        return gold.build_fact_arr_observations(silver_articles, dim_company)

    fact_run1 = build_fact()
    fact_run2 = build_fact()

    assert len(fact_run1) == len(fact_run2)
    assert fact_run1["article_id"].is_unique
    assert set(fact_run1["article_id"]) == set(fact_run2["article_id"])
    # Deterministic: same content byte-for-byte across runs (modulo the
    # _loaded_at lineage timestamp, which is expected to change per run).
    cols_to_compare = [c for c in fact_run1.columns if c != "_loaded_at"]
    pd.testing.assert_frame_equal(
        fact_run1[cols_to_compare].reset_index(drop=True),
        fact_run2[cols_to_compare].reset_index(drop=True),
    )
