"""Pipeline orchestrator: bronze -> silver -> gold -> exports (+ optional
embeddings / DuckDB load).

Usage:
    python -m pipeline.run_pipeline
    python -m pipeline.run_pipeline --skip-embeddings   # faster, no semantic search outputs
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from . import bronze, config, duckdb_loader, embeddings as emb, exports, gold, silver


def run(skip_embeddings: bool = False, output_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir or config.OUTPUT_DIR)
    t0 = time.time()

    print("[1/6] Loading bronze layer...")
    bronze_articles = bronze.load_bronze_articles()
    bronze_metadata = bronze.load_bronze_company_metadata()
    print(f"      {len(bronze_articles)} article rows, {len(bronze_metadata)} company metadata rows")

    print("[2/6] Building silver layer...")
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
        f"      arr_status: parsed={n_arr_parsed} missing={n_arr_missing} "
        f"not_disclosed={n_arr_not_disclosed} unparseable={n_arr_unparseable}"
    )
    print(f"      date_status unparseable={n_date_unparseable}")
    print(f"      company resolution: unmatched={n_unmatched} / {len(silver_articles)}")

    print("[3/6] Building gold layer (dim_company, fact_arr_observations, views)...")
    dim_company = gold.build_dim_company(silver_company_metadata, silver_articles)
    fact_arr_observations = gold.build_fact_arr_observations(silver_articles, dim_company)
    latest_arr_per_company = gold.build_latest_arr_per_company(fact_arr_observations)
    quarterly_arr = gold.build_quarterly_arr(fact_arr_observations)
    unmatched_companies = gold.build_unmatched_companies(dim_company, silver_articles)

    embeddings_arr, embedding_ids, top_similar = None, None, None
    if not skip_embeddings:
        print("[4/6] Generating article embeddings (sentence-transformers, first run downloads the model)...")
        embeddings_arr, embedding_ids = emb.generate_article_embeddings(
            silver_articles[["article_id", "title", "summary"]]
        )
        emb.save_embeddings(embeddings_arr, embedding_ids)
        top_similar = emb.compute_top_similar_articles(embeddings_arr, embedding_ids)
        print(f"      {embeddings_arr.shape[0]} embeddings x {embeddings_arr.shape[1]} dims saved")
    else:
        print("[4/6] Skipping embeddings (--skip-embeddings)")

    print("[5/6] Building ai_articles_enriched.csv...")
    ai_articles_enriched = exports.build_ai_articles_enriched(
        silver_articles,
        dim_company,
        fact_arr_observations,
        embeddings=embeddings_arr,
        embedding_article_ids=embedding_ids,
        top_similar=top_similar,
    )
    print(f"      {len(ai_articles_enriched)} qualifying AI articles")

    print("[6/6] Writing CSV outputs...")
    exports.write_csv(bronze_articles, "bronze_articles.csv", output_dir)
    exports.write_csv(bronze_metadata, "bronze_metadata.csv", output_dir)
    exports.write_csv(dim_company, "dim_company.csv", output_dir)
    exports.write_csv(fact_arr_observations, "fact_arr_observations.csv", output_dir)
    exports.write_csv(latest_arr_per_company, "gold_latest_arr_per_company.csv", output_dir)
    exports.write_csv(quarterly_arr, "gold_quarterly_arr.csv", output_dir)
    exports.write_csv(unmatched_companies, "unmatched_companies.csv", output_dir)
    exports.write_csv(ai_articles_enriched, "ai_articles_enriched.csv", output_dir)
    # Supplementary lineage exports (not strictly required, but requested:
    # "include enough...data to inspect lineage, debug failed parses").
    exports.write_csv(silver_articles, "silver_articles.csv", output_dir)
    exports.write_csv(silver_company_metadata, "silver_company_metadata.csv", output_dir)

    print("      Loading gold tables + embeddings into DuckDB...")
    con = duckdb_loader.get_connection()
    duckdb_loader.load_dim_company(con, dim_company)
    duckdb_loader.load_fact_arr_observations(con, fact_arr_observations)
    duckdb_loader.load_ai_articles_enriched(
        con, ai_articles_enriched.drop(columns=["embedding"], errors="ignore")
    )
    if embeddings_arr is not None:
        duckdb_loader.load_article_embeddings(con, embedding_ids, embeddings_arr)
    con.close()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Outputs written to {output_dir}")

    return {
        "bronze_articles": bronze_articles,
        "bronze_metadata": bronze_metadata,
        "silver_articles": silver_articles,
        "silver_company_metadata": silver_company_metadata,
        "dim_company": dim_company,
        "fact_arr_observations": fact_arr_observations,
        "latest_arr_per_company": latest_arr_per_company,
        "quarterly_arr": quarterly_arr,
        "unmatched_companies": unmatched_companies,
        "ai_articles_enriched": ai_articles_enriched,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip sentence-transformers embedding generation (faster; skips semantic search outputs)",
    )
    args = parser.parse_args()
    run(skip_embeddings=args.skip_embeddings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
