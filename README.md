# YipitData Data Engineering Take-Home

Local Python pipeline that turns `tech_news.csv` (750 articles) +
`company_metadata.json` (21 companies) into a bronze/silver/gold warehouse
model of company ARR observations, a filtered AI-article export, and a
semantic-search layer over articles.

See [DATA_ARCHITECTURE.md](DATA_ARCHITECTURE.md) for the data model, grain,
lineage, and design trade-offs. This file covers installation and usage.

## System requirements

- Python 3.11 (developed/tested against 3.11.1).
- **This machine's Python is invoked as `python`, not `python3`** — every
  command below uses `python`. If your `python` resolves to Python 2 (rare
  today, but some Linux distros still do this), substitute `python3`.
- ~500 MB free disk for the virtual environment + the sentence-transformers
  model (`all-MiniLM-L6-v2`, ~90 MB, downloaded once from Hugging Face on
  first run and cached under `~/.cache/huggingface`).
- Internet access is only required the *first* time you run the pipeline
  with embeddings enabled (to download the model). Everything else runs
  fully offline.

## Libraries used

Everything below is pinned in [requirements.txt](requirements.txt) and
installed into `.venv` (see Installation). Versions shown are what's
actually installed in the tested `.venv`.

| Library | Version | What it's used for |
|---|---|---|
| [pandas](https://pandas.pydata.org/) | 3.0.5 | The core data structure for every layer (bronze/silver/gold DataFrames), CSV read/write, joins, groupby aggregations for the derived views. |
| [numpy](https://numpy.org/) | 2.4.6 | Backs pandas; used directly for embedding vector math (L2 normalization, cosine similarity, `argsort` for top-k lookups). |
| [duckdb](https://duckdb.org/) | 1.5.5 | The optional local warehouse database. Stores the gold tables + article embeddings, does the upsert loads, and runs `hybrid_search` (SQL filters combined with its native `list_cosine_similarity` for vector ranking). |
| [rapidfuzz](https://github.com/rapidfuzz/RapidFuzz) | 3.14.5 | Fuzzy string matching (`fuzz.ratio`) — the last-resort step in company-name resolution, after exact/alias matching fails. |
| [sentence-transformers](https://www.sbert.net/) | 6.0.0 | Loads `all-MiniLM-L6-v2` and encodes article `title + summary` text into 384-dim embeddings for semantic search. Pulls in **PyTorch** and Hugging Face's **transformers**/**huggingface-hub** as transitive dependencies (the actual model runtime + model download/cache); those aren't imported directly by this project's code. |
| [pytest](https://docs.pytest.org/) | 9.1.1 | Test runner for the 117 unit/integration tests in `tests/`. |

Everything else (`json`, `re`, `datetime`, `dataclasses`, `pathlib`,
`argparse`) is Python 3.11 standard library — no extra install needed for
those.

## Installation

Run these from the project root (the folder containing this README).

```bash
python -m venv .venv
```

Activate it — Windows (PowerShell):

```bash
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

> This project was built and run inside an isolated `.venv` deliberately —
> the machine it was developed on has other unrelated Python projects
> installed globally (e.g. `databricks-sql-connector`, `numpy`/`pandas` at
> different pinned versions), and this pipeline needs `numpy>=2` for
> current `sentence-transformers`/`pandas`. Installing into a venv avoids
> touching/upgrading packages other projects on the same machine depend on.
> Always activate `.venv` before running anything below.

## How to run the pipeline

Full run (cleans + models the data, builds embeddings, writes all CSVs,
loads DuckDB):

```bash
python -m pipeline.run_pipeline
```

Faster run that skips embedding generation (useful while iterating on the
cleaning/modeling logic — skips the semantic-search-related outputs):

```bash
python -m pipeline.run_pipeline --skip-embeddings
```

Typical full-run console output:

```
[bronze] Loading raw source files...
         750 article rows, 21 company metadata rows
[silver] Cleaning + resolving...
         arr_status: parsed=558 missing=107 not_disclosed=85 unparseable=0
         date_status unparseable=0
         company resolution: unmatched=28 / 750
[gold] Building dim_company, fact_arr_observations, views...
[embeddings] Generating article embeddings (sentence-transformers, first run downloads the model)...
             750 embeddings x 384 dims saved
[exports] Building ai_articles_enriched.csv...
          121 qualifying AI articles

Done in ~20-60s (first run downloads the embedding model; subsequent runs
are the faster end of that range). Outputs written to .../output
```

The pipeline is safe to re-run at any time: CSVs are fully recomputed and
overwritten each run, and the DuckDB loader upserts on each table's key
(`article_id` for `fact_arr_observations`/`ai_articles_enriched`/
`article_embeddings`, `company_key` for `dim_company`) rather than
appending — see `tests/test_idempotency.py` and
[DATA_ARCHITECTURE.md §7](DATA_ARCHITECTURE.md#7-re-running-the-pipeline--idempotency).

### Running a single stage

Every layer (bronze/silver/gold/embeddings/exports) can also be run on its
own with `--stage`, reading its input from the DuckDB checkpoint the
previous stage left in `output/warehouse.duckdb` instead of recomputing it
in-process:

```bash
python -m pipeline.run_pipeline --stage bronze       # read source files, checkpoint + export bronze only
python -m pipeline.run_pipeline --stage silver        # requires a bronze checkpoint; reads it, doesn't touch tech_news.csv directly
python -m pipeline.run_pipeline --stage gold           # requires a silver checkpoint
python -m pipeline.run_pipeline --stage embeddings      # requires a silver checkpoint
python -m pipeline.run_pipeline --stage exports          # requires silver + gold checkpoints
```

Running a stage before its dependency has been checkpointed fails with an
actionable message telling you which stage to run first, instead of a
confusing downstream error. If you run a single stage against a stale
upstream checkpoint (e.g. `tech_news.csv` changed since `--stage bronze`
last ran), you'll get a non-fatal warning by default; pass
`--strict-checkpoints` to turn that into a hard failure instead. See
[DATA_ARCHITECTURE.md §8](DATA_ARCHITECTURE.md#8-running-pipeline-stages-independently)
for the full design and trade-offs.

## Regenerating all required CSV outputs

`python -m pipeline.run_pipeline` (no flags) regenerates everything into
`output/`:

| File | What it is |
|---|---|
| `ai_articles_enriched.csv` | **Required.** Filtered/enriched AI-article export (see filter definition below) |
| `dim_company.csv` | **Required warehouse table.** One row per resolved company identity |
| `fact_arr_observations.csv` | **Required warehouse table.** One row per article; the ARR-observation fact table |
| `gold_latest_arr_per_company.csv` | Derived view: most recent parsed ARR per company |
| `gold_quarterly_arr.csv` | Derived view: one representative ARR per company per quarter |
| `unmatched_companies.csv` | Companies referenced by articles with no metadata match, for triage |
| `bronze_articles.csv` | Supplementary: `tech_news.csv` read as-is (untouched raw strings) + lineage columns |
| `bronze_company_metadata.csv` | Supplementary: `company_metadata.json` flattened as-is + lineage columns |
| `silver_articles.csv` | Supplementary: every article with raw *and* cleaned columns side by side (lineage/debugging) |
| `silver_company_metadata.csv` | Supplementary: cleaned company metadata |
| `warehouse.duckdb` | Optional persisted DuckDB database: every layer (bronze/silver/gold) + embeddings, each as an independently re-loadable checkpoint (see [Running a single stage](#running-a-single-stage)) |
| `article_embeddings.npy` / `article_embeddings_index.csv` | Persisted embeddings (skipped with `--skip-embeddings`) |

`ai_articles_enriched.csv` includes articles where **(article category
maps to `AI_ML` OR the company's metadata industry maps to `AI_ML`) AND
published_date year is in [2022, 2024] AND `arr_usd` > $50,000,000** — only
rows with a successfully *parsed* ARR value can qualify; missing/
not-disclosed/unparseable ARR never passes this filter.

## Running the tests

```bash
python -m pytest -q
```

130 tests, all fast (~4s, no network/model download needed — the unit tests
exercise the cleaning/resolution/gold-layer functions directly with real
messy values pulled from `tech_news.csv`, plus DuckDB-backed idempotency
and stage-checkpoint tests using `tmp_path`). Confirmed passing:

```
130 passed in 4.60s
```

## Example usage

### Query the warehouse tables directly from the CSVs

```python
import pandas as pd

fact = pd.read_csv("output/fact_arr_observations.csv")
dim = pd.read_csv("output/dim_company.csv")

# ARR observations for one company over time (the primary use case)
snowflake = fact[fact["company_key"] == "Snowflake"].sort_values("published_date")
print(snowflake[["article_id", "published_date", "arr_usd", "arr_status"]])

# Find the source article for a given ARR observation
row = fact[fact["article_id"] == "ART0002"].iloc[0]
print(row["revenue_raw"], "->", row["arr_usd"], "USD, via", row["company_match_method"])

# Filter articles by date / category / industry / ARR threshold without
# creating new modeled records -- this is just a query over the existing tables
big_ai_2023 = fact.merge(dim, on="company_key").query(
    "category_std == 'AI_ML' and pub_year == 2023 and arr_usd > 500_000_000"
)
```

### Or query the DuckDB file

```python
import duckdb

con = duckdb.connect("output/warehouse.duckdb")
con.sql("""
    SELECT company_key, published_date, arr_usd
    FROM fact_arr_observations
    WHERE company_key = 'Snowflake' AND arr_status = 'parsed'
    ORDER BY published_date
""").show()
```

### Semantic search

```python
from pipeline import embeddings as emb

vectors, article_ids = emb.load_embeddings()  # loads the persisted .npy/.csv
results = emb.find_similar_articles(
    "OpenAI raises a massive funding round for AI research",
    vectors, article_ids, top_k=5,
)
for article_id, score in results:
    print(article_id, round(score, 4))
```

### Hybrid search (SQL filters + vector similarity, in DuckDB)

```python
from pipeline import duckdb_loader

con = duckdb_loader.get_connection()
query_vector = vectors[0]  # or embed a fresh query with the model directly

results = duckdb_loader.hybrid_search(
    con,
    query_vector,
    top_k=5,
    category_std="AI_ML",
    min_arr_usd=50_000_000,
    year_min=2022,
    year_max=2024,
)
print(results)
```

## Project layout

```
pipeline/
  config.py              constants: currency rates, category taxonomy,
                          company aliases, thresholds -- the single place
                          every business rule lives
  cleaning.py             revenue / date / category cleaning functions
  company_resolution.py   company name -> metadata identity resolution
  bronze.py                raw ingestion + lineage columns
  silver.py                typed/cleaned/resolved columns
  gold.py                   dim_company, fact_arr_observations, derived views
  exports.py                ai_articles_enriched.csv + CSV writer
  embeddings.py             sentence-transformers embeddings, similarity search
  duckdb_loader.py          DuckDB checkpoint/upsert loaders, hybrid_search
  run_pipeline.py            per-stage functions + --stage CLI entrypoint
tests/
  test_cleaning.py             revenue/date/category, real messy fixtures
  test_company_resolution.py   exact/alias/case/fuzzy/unmatched, real names
  test_gold_and_exports.py     grain, lineage, AI-article filter correctness
  test_idempotency.py          DuckDB upsert + full-pipeline re-run proof
  test_stage_pipeline.py       checkpoint round-trip, --stage dispatch, staleness
output/                    generated CSVs + warehouse.duckdb (gitignored contents
                            regenerate via `python -m pipeline.run_pipeline`)
DATA_ARCHITECTURE.md       data model, lineage, and design trade-offs
```
