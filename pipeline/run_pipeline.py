"""Pipeline orchestrator: bronze -> silver -> gold -> embeddings -> exports.

Each stage is independently runnable. `--stage all` (the default) chains
every stage in one process, passing DataFrames directly from one stage
function to the next (the fast path -- no round-trip through DuckDB to
obtain its own input) while still checkpointing each stage's output to
`output/warehouse.duckdb` as a side effect, so a later standalone run has
something to read. `--stage <name>` runs exactly one stage, reading its
required input(s) from the checkpoint(s) left by the previous stage instead
of recomputing them -- see DATA_ARCHITECTURE.md for why this is safe
(pure, side-effect-free `build_*` functions) and what it doesn't guarantee
(the freshness check is mtime/size-based, not a content hash).

Usage:
    python -m pipeline.run_pipeline                       # everything (equivalent to --stage all)
    python -m pipeline.run_pipeline --skip-embeddings      # everything except semantic search outputs
    python -m pipeline.run_pipeline --stage bronze         # just bronze: read source files, checkpoint + export
    python -m pipeline.run_pipeline --stage silver         # just silver: requires a bronze checkpoint to already exist
    python -m pipeline.run_pipeline --stage gold           # just gold: requires a silver checkpoint
    python -m pipeline.run_pipeline --stage embeddings     # just embeddings: requires a silver checkpoint
    python -m pipeline.run_pipeline --stage exports        # just ai_articles_enriched.csv: requires silver+gold checkpoints
    python -m pipeline.run_pipeline --stage silver --strict-checkpoints   # hard-fail instead of warn on a stale upstream checkpoint
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from . import bronze, config, duckdb_loader, embeddings as emb, exports, gold, silver


def _require_checkpoint(con, table: str, upstream_stage: str) -> None:
    if not duckdb_loader.checkpoint_exists(con, table):
        raise SystemExit(
            f"This stage requires the '{table}' checkpoint, which doesn't exist yet.\n"
            f"Run `python -m pipeline.run_pipeline --stage {upstream_stage}` first, "
            f"or `python -m pipeline.run_pipeline --stage all`."
        )


def run_bronze(con, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("[bronze] Loading raw source files...")
    bronze_articles = bronze.load_bronze_articles()
    bronze_metadata = bronze.load_bronze_company_metadata()
    print(f"         {len(bronze_articles)} article rows, {len(bronze_metadata)} company metadata rows")

    sig = duckdb_loader.combined_source_signature()
    duckdb_loader.load_bronze_articles_checkpoint(con, bronze_articles, sig)
    duckdb_loader.load_bronze_metadata_checkpoint(con, bronze_metadata, sig)
    exports.write_csv(bronze_articles, "bronze_articles.csv", output_dir)
    exports.write_csv(bronze_metadata, "bronze_company_metadata.csv", output_dir)
    return bronze_articles, bronze_metadata


def run_silver(
    con,
    output_dir: Path,
    bronze_articles: pd.DataFrame | None = None,
    bronze_metadata: pd.DataFrame | None = None,
    strict: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if bronze_articles is None or bronze_metadata is None:
        _require_checkpoint(con, "bronze_articles", "bronze")
        _require_checkpoint(con, "bronze_company_metadata", "bronze")
        sig = duckdb_loader.combined_source_signature()
        duckdb_loader.check_freshness(con, "bronze_articles", sig, strict=strict)
        duckdb_loader.check_freshness(con, "bronze_company_metadata", sig, strict=strict)
        print("[silver] Reading bronze checkpoint from DuckDB...")
        bronze_articles = duckdb_loader.load_checkpoint(con, "bronze_articles")
        bronze_metadata = duckdb_loader.load_checkpoint(con, "bronze_company_metadata")

    print("[silver] Cleaning + resolving...")
    metadata_keys = bronze_metadata["company_name_raw"].tolist()
    silver_company_metadata = silver.build_silver_company_metadata(bronze_metadata)
    silver_articles = silver.build_silver_articles(bronze_articles, metadata_keys)

    n_arr_parsed = (silver_articles["arr_status"] == "parsed").sum()
    n_arr_missing = (silver_articles["arr_status"] == "missing").sum()
    n_arr_not_disclosed = (silver_articles["arr_status"] == "not_disclosed").sum()
    n_arr_unparseable = (silver_articles["arr_status"] == "unparseable").sum()
    n_date_unparseable = (silver_articles["date_status"] == "unparseable").sum()
    n_unmatched = (~silver_articles["company_matched"]).sum()
    print(
        f"         arr_status: parsed={n_arr_parsed} missing={n_arr_missing} "
        f"not_disclosed={n_arr_not_disclosed} unparseable={n_arr_unparseable}"
    )
    print(f"         date_status unparseable={n_date_unparseable}")
    print(f"         company resolution: unmatched={n_unmatched} / {len(silver_articles)}")

    sig = duckdb_loader.combined_source_signature()
    duckdb_loader.load_silver_articles_checkpoint(con, silver_articles, sig)
    duckdb_loader.load_silver_metadata_checkpoint(con, silver_company_metadata, sig)
    exports.write_csv(silver_articles, "silver_articles.csv", output_dir)
    exports.write_csv(silver_company_metadata, "silver_company_metadata.csv", output_dir)
    return silver_company_metadata, silver_articles


def run_gold(
    con,
    output_dir: Path,
    silver_company_metadata: pd.DataFrame | None = None,
    silver_articles: pd.DataFrame | None = None,
    strict: bool = False,
) -> dict[str, pd.DataFrame]:
    if silver_company_metadata is None or silver_articles is None:
        _require_checkpoint(con, "silver_articles", "silver")
        _require_checkpoint(con, "silver_company_metadata", "silver")
        sig = duckdb_loader.combined_source_signature()
        duckdb_loader.check_freshness(con, "silver_articles", sig, strict=strict)
        duckdb_loader.check_freshness(con, "silver_company_metadata", sig, strict=strict)
        print("[gold] Reading silver checkpoint from DuckDB...")
        silver_articles = duckdb_loader.load_checkpoint(con, "silver_articles")
        silver_company_metadata = duckdb_loader.load_checkpoint(con, "silver_company_metadata")

    print("[gold] Building dim_company, fact_arr_observations, views...")
    dim_company = gold.build_dim_company(silver_company_metadata, silver_articles)
    fact_arr_observations = gold.build_fact_arr_observations(silver_articles, dim_company)
    latest_arr_per_company = gold.build_latest_arr_per_company(fact_arr_observations)
    quarterly_arr = gold.build_quarterly_arr(fact_arr_observations)
    unmatched_companies = gold.build_unmatched_companies(dim_company, silver_articles)

    sig = duckdb_loader.combined_source_signature()
    duckdb_loader.load_dim_company(con, dim_company, sig)
    duckdb_loader.load_fact_arr_observations(con, fact_arr_observations, sig)
    exports.write_csv(dim_company, "dim_company.csv", output_dir)
    exports.write_csv(fact_arr_observations, "fact_arr_observations.csv", output_dir)
    exports.write_csv(latest_arr_per_company, "gold_latest_arr_per_company.csv", output_dir)
    exports.write_csv(quarterly_arr, "gold_quarterly_arr.csv", output_dir)
    exports.write_csv(unmatched_companies, "unmatched_companies.csv", output_dir)

    return {
        "dim_company": dim_company,
        "fact_arr_observations": fact_arr_observations,
        "latest_arr_per_company": latest_arr_per_company,
        "quarterly_arr": quarterly_arr,
        "unmatched_companies": unmatched_companies,
    }


def run_embeddings(
    con,
    output_dir: Path,
    silver_articles: pd.DataFrame | None = None,
    strict: bool = False,
) -> tuple[object, list[str] | None, dict[str, list[str]] | None]:
    if silver_articles is None:
        _require_checkpoint(con, "silver_articles", "silver")
        sig = duckdb_loader.combined_source_signature()
        duckdb_loader.check_freshness(con, "silver_articles", sig, strict=strict)
        print("[embeddings] Reading silver checkpoint from DuckDB...")
        silver_articles = duckdb_loader.load_checkpoint(con, "silver_articles")

    print("[embeddings] Generating article embeddings (sentence-transformers, first run downloads the model)...")
    embeddings_arr, embedding_ids = emb.generate_article_embeddings(
        silver_articles[["article_id", "title", "summary"]]
    )
    emb.save_embeddings(embeddings_arr, embedding_ids)
    top_similar = emb.compute_top_similar_articles(embeddings_arr, embedding_ids)
    print(f"             {embeddings_arr.shape[0]} embeddings x {embeddings_arr.shape[1]} dims saved")

    sig = duckdb_loader.combined_source_signature()
    duckdb_loader.load_article_embeddings(con, embedding_ids, embeddings_arr, sig)
    return embeddings_arr, embedding_ids, top_similar


def run_exports(
    con,
    output_dir: Path,
    silver_articles: pd.DataFrame | None = None,
    dim_company: pd.DataFrame | None = None,
    fact_arr_observations: pd.DataFrame | None = None,
    embeddings_arr=None,
    embedding_ids: list[str] | None = None,
    top_similar: dict[str, list[str]] | None = None,
    strict: bool = False,
) -> pd.DataFrame:
    if silver_articles is None or dim_company is None or fact_arr_observations is None:
        _require_checkpoint(con, "silver_articles", "silver")
        _require_checkpoint(con, "dim_company", "gold")
        _require_checkpoint(con, "fact_arr_observations", "gold")
        sig = duckdb_loader.combined_source_signature()
        duckdb_loader.check_freshness(con, "silver_articles", sig, strict=strict)
        duckdb_loader.check_freshness(con, "dim_company", sig, strict=strict)
        duckdb_loader.check_freshness(con, "fact_arr_observations", sig, strict=strict)
        print("[exports] Reading silver + gold checkpoints from DuckDB...")
        silver_articles = duckdb_loader.load_checkpoint(con, "silver_articles")
        dim_company = duckdb_loader.load_checkpoint(con, "dim_company")
        fact_arr_observations = duckdb_loader.load_checkpoint(con, "fact_arr_observations")

    if embeddings_arr is None and embedding_ids is None:
        try:
            embeddings_arr, embedding_ids = emb.load_embeddings()
            top_similar = emb.compute_top_similar_articles(embeddings_arr, embedding_ids)
        except FileNotFoundError:
            print(
                "[exports] No persisted embeddings found (run --stage embeddings first for "
                "the embedding/top_similar_articles columns) -- exporting without them."
            )

    print("[exports] Building ai_articles_enriched.csv...")
    ai_articles_enriched = exports.build_ai_articles_enriched(
        silver_articles,
        dim_company,
        fact_arr_observations,
        embeddings=embeddings_arr,
        embedding_article_ids=embedding_ids,
        top_similar=top_similar,
    )
    print(f"          {len(ai_articles_enriched)} qualifying AI articles")

    sig = duckdb_loader.combined_source_signature()
    exports.write_csv(ai_articles_enriched, "ai_articles_enriched.csv", output_dir)
    duckdb_loader.load_ai_articles_enriched(
        con, ai_articles_enriched.drop(columns=["embedding"], errors="ignore"), sig
    )
    return ai_articles_enriched


def run(skip_embeddings: bool = False, output_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Run every stage in one process (the `--stage all` fast path): each
    stage's output is passed directly to the next in memory (no checkpoint
    read-back) while still being checkpointed as a side effect."""
    output_dir = Path(output_dir or config.OUTPUT_DIR)
    t0 = time.time()
    con = duckdb_loader.get_connection()

    bronze_articles, bronze_metadata = run_bronze(con, output_dir)
    silver_company_metadata, silver_articles = run_silver(con, output_dir, bronze_articles, bronze_metadata)
    gold_tables = run_gold(con, output_dir, silver_company_metadata, silver_articles)

    embeddings_arr, embedding_ids, top_similar = None, None, None
    if not skip_embeddings:
        embeddings_arr, embedding_ids, top_similar = run_embeddings(con, output_dir, silver_articles)
    else:
        print("[embeddings] Skipping (--skip-embeddings)")

    ai_articles_enriched = run_exports(
        con,
        output_dir,
        silver_articles,
        gold_tables["dim_company"],
        gold_tables["fact_arr_observations"],
        embeddings_arr,
        embedding_ids,
        top_similar,
    )

    con.close()
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Outputs written to {output_dir}")

    return {
        "bronze_articles": bronze_articles,
        "bronze_metadata": bronze_metadata,
        "silver_articles": silver_articles,
        "silver_company_metadata": silver_company_metadata,
        "ai_articles_enriched": ai_articles_enriched,
        **gold_tables,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--stage",
        choices=["bronze", "silver", "gold", "embeddings", "exports", "all"],
        default="all",
        help="Run a single stage standalone (reading its input from the previous stage's DuckDB "
        "checkpoint) instead of the full pipeline.",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="(--stage all only) Skip sentence-transformers embedding generation.",
    )
    parser.add_argument(
        "--strict-checkpoints",
        action="store_true",
        help="Fail instead of warn when a standalone stage reads a checkpoint that looks stale "
        "relative to the current tech_news.csv/company_metadata.json.",
    )
    args = parser.parse_args()

    if args.stage == "all":
        run(skip_embeddings=args.skip_embeddings)
        return 0

    output_dir = Path(config.OUTPUT_DIR)
    con = duckdb_loader.get_connection()
    t0 = time.time()
    if args.stage == "bronze":
        run_bronze(con, output_dir)
    elif args.stage == "silver":
        run_silver(con, output_dir, strict=args.strict_checkpoints)
    elif args.stage == "gold":
        run_gold(con, output_dir, strict=args.strict_checkpoints)
    elif args.stage == "embeddings":
        run_embeddings(con, output_dir, strict=args.strict_checkpoints)
    elif args.stage == "exports":
        run_exports(con, output_dir, strict=args.strict_checkpoints)
    con.close()
    print(f"\nDone in {time.time() - t0:.1f}s. Outputs written to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
