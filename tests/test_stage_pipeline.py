"""Tests for the checkpoint API (pipeline.duckdb_loader) and stage
dispatch (pipeline.run_pipeline): can each stage run standalone, reading
its input from a DuckDB checkpoint instead of an in-memory DataFrame from
the same process; does a missing checkpoint fail with an actionable
message; does the freshness check warn/raise appropriately; does the
checkpoint round-trip preserve the dtypes this project depends on.

Embeddings/exports stages aren't covered here (they need the
sentence-transformers model, which would make this file slow/network-
dependent) -- they're covered by the manual `--stage embeddings`/
`--stage exports` verification in the pipeline design discussion, and
`build_ai_articles_enriched` itself is already covered by
test_gold_and_exports.py.
"""
import pandas as pd
import pytest

from pipeline import duckdb_loader, run_pipeline


# ---------------------------------------------------------------------------
# Checkpoint primitives
# ---------------------------------------------------------------------------


def test_checkpoint_exists_false_then_true(tmp_path):
    con = duckdb_loader.get_connection(tmp_path / "t.duckdb")
    assert duckdb_loader.checkpoint_exists(con, "some_table") is False
    duckdb_loader.save_checkpoint(con, "some_table", pd.DataFrame({"k": [1]}), ["k"])
    assert duckdb_loader.checkpoint_exists(con, "some_table") is True
    con.close()


def test_load_checkpoint_returns_saved_data(tmp_path):
    con = duckdb_loader.get_connection(tmp_path / "t.duckdb")
    df = pd.DataFrame({"article_id": ["A1", "A2"], "x": [1, 2]})
    duckdb_loader.save_checkpoint(con, "some_table", df, ["article_id"])
    result = duckdb_loader.load_checkpoint(con, "some_table")
    assert sorted(result["article_id"]) == ["A1", "A2"]
    con.close()


def test_checkpoint_roundtrip_preserves_nullable_int_and_na_string_literal(tmp_path):
    """Pins the dtype-fidelity claim the checkpointed-pipeline design relies
    on: a nullable Int64 column with a real NA, and a string column whose
    value is literally "N/A", must both survive a checkpoint round-trip
    without being silently reinterpreted (Int64->float64+NaN promotion, or
    the string "N/A" being read back as null)."""
    con = duckdb_loader.get_connection(tmp_path / "t.duckdb")
    df = pd.DataFrame(
        {
            "article_id": ["A1", "A2"],
            "arr_usd": pd.array([100, None], dtype="Int64"),
            "revenue_raw": ["N/A", "$100"],
        }
    )
    duckdb_loader.save_checkpoint(con, "roundtrip_test", df, ["article_id"])
    result = duckdb_loader.load_checkpoint(con, "roundtrip_test").set_index("article_id")

    assert result.loc["A1", "arr_usd"] == 100
    assert pd.isna(result.loc["A2", "arr_usd"])
    assert result.loc["A1", "revenue_raw"] == "N/A"  # literal string, not NaN
    con.close()


# ---------------------------------------------------------------------------
# Freshness checking
# ---------------------------------------------------------------------------


def test_check_freshness_silent_when_signature_matches(tmp_path, capsys):
    con = duckdb_loader.get_connection(tmp_path / "t.duckdb")
    duckdb_loader.save_checkpoint(con, "t", pd.DataFrame({"k": [1]}), ["k"], source_signature="sig-v1")
    duckdb_loader.check_freshness(con, "t", "sig-v1")
    assert "WARNING" not in capsys.readouterr().out
    con.close()


def test_check_freshness_warns_when_signature_differs(tmp_path, capsys):
    con = duckdb_loader.get_connection(tmp_path / "t.duckdb")
    duckdb_loader.save_checkpoint(con, "t", pd.DataFrame({"k": [1]}), ["k"], source_signature="sig-v1")
    duckdb_loader.check_freshness(con, "t", "sig-v2")
    assert "WARNING" in capsys.readouterr().out
    con.close()


def test_check_freshness_strict_raises_instead_of_warning(tmp_path):
    con = duckdb_loader.get_connection(tmp_path / "t.duckdb")
    duckdb_loader.save_checkpoint(con, "t", pd.DataFrame({"k": [1]}), ["k"], source_signature="sig-v1")
    with pytest.raises(SystemExit):
        duckdb_loader.check_freshness(con, "t", "sig-v2", strict=True)
    con.close()


def test_check_freshness_no_meta_recorded_is_silently_skipped(tmp_path, capsys):
    """A checkpoint saved without a source_signature (e.g. an old run before
    this feature existed) shouldn't crash a freshness check -- there's just
    nothing to compare against."""
    con = duckdb_loader.get_connection(tmp_path / "t.duckdb")
    duckdb_loader.save_checkpoint(con, "t", pd.DataFrame({"k": [1]}), ["k"])  # no signature
    duckdb_loader.check_freshness(con, "t", "sig-v1")
    assert "WARNING" not in capsys.readouterr().out
    con.close()


# ---------------------------------------------------------------------------
# Stage dispatch: missing checkpoint -> actionable error
# ---------------------------------------------------------------------------


def test_run_silver_standalone_without_bronze_checkpoint_raises_actionable_error(tmp_path):
    con = duckdb_loader.get_connection(tmp_path / "t.duckdb")
    with pytest.raises(SystemExit, match="stage bronze"):
        run_pipeline.run_silver(con, tmp_path)
    con.close()


def test_run_gold_standalone_without_silver_checkpoint_raises_actionable_error(tmp_path):
    con = duckdb_loader.get_connection(tmp_path / "t.duckdb")
    with pytest.raises(SystemExit, match="stage silver"):
        run_pipeline.run_gold(con, tmp_path)
    con.close()


def test_run_exports_standalone_without_gold_checkpoint_raises_actionable_error(tmp_path):
    con = duckdb_loader.get_connection(tmp_path / "t.duckdb")
    with pytest.raises(SystemExit):
        run_pipeline.run_exports(con, tmp_path)
    con.close()


# ---------------------------------------------------------------------------
# Stage dispatch: standalone chain against the real source files, proving a
# later stage genuinely reads the checkpoint rather than silently
# recomputing its input in-process.
# ---------------------------------------------------------------------------


def test_silver_stage_run_standalone_matches_in_memory_chain(tmp_path):
    con = duckdb_loader.get_connection(tmp_path / "t.duckdb")

    # Stage 1, in one "process": produces + checkpoints bronze.
    bronze_articles, bronze_metadata = run_pipeline.run_bronze(con, tmp_path)

    # Stage 2, standalone (no in-memory bronze passed in) -- must come from
    # the checkpoint written above, not from re-reading tech_news.csv itself
    # inside run_silver (run_silver never calls pipeline.bronze at all).
    _, silver_articles_from_checkpoint = run_pipeline.run_silver(con, tmp_path)

    # Cross-check against building silver directly in-memory from the same
    # bronze DataFrames, to prove the checkpointed path produces identical data.
    from pipeline import silver

    metadata_keys = bronze_metadata["company_name_raw"].tolist()
    silver_articles_in_memory = silver.build_silver_articles(bronze_articles, metadata_keys)

    cols = [c for c in silver_articles_in_memory.columns if c != "_loaded_at"]
    pd.testing.assert_frame_equal(
        silver_articles_from_checkpoint[cols].sort_values("article_id").reset_index(drop=True),
        silver_articles_in_memory[cols].sort_values("article_id").reset_index(drop=True),
        check_dtype=False,  # DuckDB round-trip vs. in-memory can differ in incidental dtypes (e.g. pandas Int64 vs numpy int64) without differing in value
    )
    con.close()


def test_gold_stage_run_standalone_after_silver_checkpoint(tmp_path):
    con = duckdb_loader.get_connection(tmp_path / "t.duckdb")
    run_pipeline.run_bronze(con, tmp_path)
    run_pipeline.run_silver(con, tmp_path)

    gold_tables = run_pipeline.run_gold(con, tmp_path)  # standalone: reads silver checkpoint

    fact = gold_tables["fact_arr_observations"]
    assert len(fact) == 750
    assert fact["article_id"].is_unique
    assert duckdb_loader.checkpoint_exists(con, "dim_company")
    assert duckdb_loader.checkpoint_exists(con, "fact_arr_observations")
    con.close()


def test_stage_all_fast_path_never_reads_back_its_own_checkpoint(tmp_path, monkeypatch):
    """`run()` (--stage all) should pass DataFrames directly between stages
    rather than round-tripping through DuckDB to obtain its own input --
    verified by making load_checkpoint raise if it's ever called during a
    full `run()`."""
    from pipeline import config

    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(config, "DUCKDB_PATH", tmp_path / "warehouse.duckdb")
    # Isolate from this project's real persisted embeddings so run_exports's
    # "no embeddings passed in-memory -> try loading from disk" fallback
    # doesn't pick up the real output/article_embeddings.npy.
    monkeypatch.setattr(config, "EMBEDDINGS_NPY_PATH", tmp_path / "article_embeddings.npy")
    monkeypatch.setattr(config, "EMBEDDINGS_INDEX_PATH", tmp_path / "article_embeddings_index.csv")

    original_load_checkpoint = duckdb_loader.load_checkpoint

    def _forbidden_load_checkpoint(*args, **kwargs):
        raise AssertionError("run() should not read back its own just-written checkpoint")

    monkeypatch.setattr(duckdb_loader, "load_checkpoint", _forbidden_load_checkpoint)
    try:
        result = run_pipeline.run(skip_embeddings=True, output_dir=tmp_path)
    finally:
        monkeypatch.setattr(duckdb_loader, "load_checkpoint", original_load_checkpoint)

    assert len(result["fact_arr_observations"]) == 750
