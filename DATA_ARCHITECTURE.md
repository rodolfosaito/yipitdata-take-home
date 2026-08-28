# Data Architecture Document

YipitData Data Engineering take-home. This document explains the chosen data
model, how lineage is preserved end to end, how invalid/missing data is
handled, how to derive the required views, and how this would run in a real
(non-local, recurring-batch) setting. See [README.md](README.md) for how to
install and run everything.

## 1. Layered model: bronze -> silver -> gold

```
tech_news.csv ────┐
                   ├─▶ bronze_articles ──▶ silver_articles ──┐
company_metadata.json ┐                                       ├─▶ dim_company
                       └─▶ bronze_company_metadata ─▶ silver_company_metadata ┘
                                                                │
                                                                ▼
                                                     fact_arr_observations
                                                                │
                                                    ┌───────────┴────────────┐
                                                    ▼                        ▼
                                     gold_latest_arr_per_company   gold_quarterly_arr

silver_articles + dim_company + embeddings ──▶ ai_articles_enriched.csv
```

**Bronze** (`pipeline/bronze.py`) is the untouched source: every column read
as a string, no parsing, no joins. Each row gets `_source_row_number`,
`_source_file`, and `_loaded_at` lineage columns. Nothing is dropped or
reinterpreted here — this is the layer you'd point at raw S3/landing-zone
files in production and it never changes shape based on downstream logic.
Exported as `bronze_articles.csv` / `bronze_company_metadata.csv` so the
very first hop of lineage is inspectable without re-running any code.

**Silver** (`pipeline/silver.py`) is one row per bronze row, typed and
cleaned, with the *original raw string preserved alongside* every derived
column (`revenue_raw` next to `arr_usd`/`arr_status`, `published_date_raw`
next to `published_date_clean`/`date_status`, `category_raw` next to
`category_std`). Company names are resolved to a canonical identity here
too (`company_key`, `company_matched`, `company_match_method`,
`company_match_score`). Nothing is filtered out at this layer — every bronze
row produces exactly one silver row, even if every derived field on it ends
up null.

**Gold** (`pipeline/gold.py`) is the warehouse-style dimensional model
described below, plus derived views. This is what the query patterns in the
assignment run against.

Why bronze/silver/gold instead of just cleaning in place: it makes every
"why does this ARR number look wrong" question answerable by walking
backwards one layer at a time without re-parsing anything, and it means a
new batch only ever adds rows to bronze — silver and gold are always fully
rebuilt/reconciled from bronze, not hand-patched.

## 2. Table grain, keys, and relationships

| Table | Grain | Key | Notes |
|---|---|---|---|
| `dim_company` | one row per resolved company identity | `company_key` (natural key: the canonical metadata company name, or the raw `company_name` string if unmatched) | 21 metadata companies + N unmatched raw names seen in articles |
| `fact_arr_observations` | **one row per `article_id`** | `article_id` (also the idempotency/upsert key) | FK `company_key` → `dim_company.company_key`. Every article is represented, whether or not it carried a usable ARR value |
| `gold_latest_arr_per_company` | one row per `company_key` | `company_key` | derived from `fact_arr_observations` |
| `gold_quarterly_arr` | one row per (`company_key`, `year`, `quarter`) | composite | derived from `fact_arr_observations` |
| `ai_articles_enriched` | one row per qualifying `article_id` | `article_id` | filtered + enriched export, not a warehouse table in its own right |
| `article_embeddings` (DuckDB only) | one row per `article_id` | `article_id` | 384-dim vector per article |

`company_key` is a **natural key**, not a generated surrogate integer. That
keeps every table's contents deterministic and diffable run-over-run (the
same input always produces the same key), which is what the idempotency
tests in `tests/test_idempotency.py` rely on. A production version at real
scale would likely add a surrogate `company_id` for join performance and
SCD2 history, while keeping `company_key` as the natural/business key.

**Why `fact_arr_observations` grain = one row per article, not one row per
"ARR observation"**: every article makes at most one ARR claim (the
`revenue` column), so "one row per article" and "one row per ARR
observation" are the same grain here. Making `article_id` the grain (and
the upsert key) is also what makes the fact table trivially idempotent: a
re-run recomputes the row for a given article from scratch and replaces it,
it never appends.

## 3. Lineage: from an ARR number back to its source article

Every fact/gold row carries `article_id`. `fact_arr_observations` additionally
carries `_source_row_number` and `_loaded_at` straight from bronze, and the
full `revenue_raw` string is preserved on the same row as the parsed
`arr_usd`. So the chain is:

```
gold_latest_arr_per_company.article_id
  -> fact_arr_observations (article_id, revenue_raw, arr_usd, arr_status, currency_detected)
    -> silver_articles.csv (article_id, revenue_raw, published_date_raw, category_raw, company_match_method, ...)
      -> tech_news.csv row _source_row_number
```

`silver_articles.csv` and `silver_company_metadata.csv` are exported
alongside the gold tables specifically so this chain can be walked from a
CSV viewer without touching code — every failed/ambiguous parse is visible
with its original raw value right next to it.

## 4. Handling invalid / missing ARR

`revenue` is cleaned into two columns: `arr_usd` (nullable integer) and
`arr_status`, one of:

- `parsed` — a numeric ARR value was extracted; only these rows have a
  non-null `arr_usd`.
- `missing` — the cell was blank or `"N/A"`: no figure was ever provided.
- `not_disclosed` — the source explicitly states the company declined to
  disclose a figure. Kept distinct from `missing` because it's a different
  fact (an active non-disclosure vs. an absent field) and it's useful to be
  able to count/report on it separately.
- `unparseable` — a non-empty value was present but couldn't be parsed
  (defensive; none of the 750 rows hit this in practice, but the pipeline
  doesn't assume future batches will be as clean).

**None of `missing`/`not_disclosed`/`unparseable` ever produce a numeric
`arr_usd`** (never `0`, never silently dropped) — the row stays in
`fact_arr_observations` with `arr_usd = NULL`. All derived views
(`gold_latest_arr_per_company`, `gold_quarterly_arr`, the `ai_articles_enriched`
ARR-threshold filter) explicitly filter to `arr_status = 'parsed'` before
computing anything, so an undisclosed or unparseable row can never be
mistaken for a $0 ARR observation or silently pass a `> $50M` filter.

Currency conversion (`pipeline/config.py::CURRENCY_TO_USD`): EUR ×1.1,
GBP ×1.27, JPY ÷150, fixed rates as specified in the assignment. A range
value (`"$10M - $20M"`) is parsed as the midpoint of its two parsed
endpoints, assuming (as observed in the data) both sides share one
currency.

## 5. Handling unmatched companies

Company resolution (`pipeline/company_resolution.py`) tries, per raw
`company_name`, in order: **exact** match → **case-insensitive exact** →
curated **alias** dict → whole-string **RapidFuzz `ratio`** ≥ 88 → else
**unmatched**. The method and (for fuzzy) the score are stored per article
(`company_match_method`, `company_match_score`) so every resolution is
auditable.

Whole-string `ratio` (not `partial_ratio`/substring matching) is used
deliberately: `partial_ratio` would happily match `"The Boring Company /
SpaceX"` against `"SpaceX"` because the substring is present, which is
exactly the kind of false positive that silently mis-attributes an
observation to the wrong company. `ratio` scores that pair at ~36/100,
correctly below threshold.

**Unmatched companies are never dropped.** `dim_company` includes them as
rows with `company_key` = their raw name, `metadata_matched = False`, and
every metadata field null. `fact_arr_observations` still has a row for
every one of their articles (with `company_age` also null, since it needs
`founded_year`). `unmatched_companies.csv` is a dedicated triage export:
one row per unmatched raw name with its article count and best fuzzy score
seen, so a human can decide whether to add an alias or accept the gap.

In this dataset, 28 of 750 articles (6 raw company names: `Cohere`,
`Mistral AI`, `Perplexity AI`, `Hugging Face`, `xAI`, `The Boring Company /
SpaceX`) end up unmatched — all expected, since `company_metadata.json`
only has 21 companies and these are real companies simply absent from it,
not typos of the 21 that are.

## 6. Deriving latest-ARR and quarterly-ARR

Both are plain aggregations over `fact_arr_observations`, filtered to
`arr_status = 'parsed'` first (undisclosed/missing/unparseable rows never
participate):

- **Latest ARR per company** (`gold_latest_arr_per_company`): for each
  `company_key`, the row with the maximum `published_date`. Implemented as
  a sort + `groupby(...).tail(1)`; in SQL/DuckDB this is a
  `QUALIFY row_number() OVER (PARTITION BY company_key ORDER BY
  published_date DESC) = 1`.
- **Quarterly ARR** (`gold_quarterly_arr`): for each (`company_key`, `year`,
  `quarter`), the most recently reported parsed observation *within* that
  quarter. ARR is a point-in-time reported figure, not a flow to sum or
  average across a quarter's articles — if a company had two ARR mentions
  in the same quarter, the later one is treated as the more current figure.
  This is a documented assumption, not the only defensible one (an
  alternative would be to average multiple observations in a quarter, or
  keep every observation and let the consumer choose); "most recent wins"
  was chosen because it matches how the latest-ARR view already works and
  is the simplest to explain and defend.

Both views carry `article_id` so their numbers trace back to a specific
source row.

## 7. Re-running the pipeline / idempotency

`fact_arr_observations`'s grain (`article_id`) is also its upsert key. The
DuckDB loader (`pipeline/duckdb_loader.py::_upsert`) does `DELETE ... WHERE
article_id IN (incoming) THEN INSERT incoming` for every gold table, never
a bare append. The CSV exports are simpler still: each run fully recomputes
each table from bronze and overwrites the file, so the file's content is a
pure function of the source data (not of how many times the pipeline has
run). `tests/test_idempotency.py` proves both:

- Loading the same batch into DuckDB twice leaves row counts unchanged.
- Loading a batch with one corrected value overwrites in place (no
  duplicate row for that `article_id`).
- Loading a batch with one new `article_id` added increases the row count
  by exactly one.
- Rebuilding `fact_arr_observations` from the real source files twice
  produces byte-identical output (aside from the `_loaded_at` timestamp).

## 8. Running pipeline stages independently

Each layer's build functions (`silver.build_silver_articles`,
`gold.build_dim_company`, etc.) have always been pure — a DataFrame in, a
DataFrame out, no disk I/O. What made the *pipeline as a whole* only
runnable end-to-end wasn't those functions, it was `run_pipeline.py`:
everything flowed through Python variables inside one `run()` call, and
while bronze/silver were exported to CSV for audit, nothing downstream ever
read those CSVs back in — so even "just run silver" had no bronze data to
start from without recomputing it.

`pipeline/duckdb_loader.py` now checkpoints every layer (bronze, silver,
gold) as its own DuckDB table, reusing the same idempotent `_upsert` this
project already relies on for gold (§7) — `save_checkpoint`/
`load_checkpoint`/`checkpoint_exists`. `run_pipeline.py` exposes one stage
function per layer (`run_bronze`, `run_silver`, `run_gold`, `run_embeddings`,
`run_exports`), each shaped as **read input → compute (the same unchanged
pure `build_*` call) → checkpoint → CSV export**:

- `python -m pipeline.run_pipeline --stage silver` reads the
  `bronze_articles`/`bronze_company_metadata` checkpoints, builds silver,
  and writes a new `silver_articles`/`silver_company_metadata` checkpoint —
  no bronze recomputation needed, and no in-memory object from a prior
  process required.
- `python -m pipeline.run_pipeline` (`--stage all`, the default) still runs
  every stage in one process, passing each stage's output directly to the
  next as a function argument — it never round-trips through its own
  checkpoint to obtain input it just computed, so the common case is exactly
  as fast as before this change. Checkpointing happens as a side effect, for
  a *future* standalone run to use.
- Running a stage before its dependency has ever been checkpointed fails
  with an actionable message (`"Run --stage bronze first, or --stage all"`)
  rather than a confusing downstream error.
- A lightweight `_checkpoint_meta` table records the `(mtime, size)`
  signature of `tech_news.csv`/`company_metadata.json` at the time each
  checkpoint was built. A standalone stage run compares that against the
  current source files and prints a warning (or, with
  `--strict-checkpoints`, raises) if they've diverged — e.g. running
  `--stage silver` days after `--stage bronze`, against a source file that
  changed in between. This is a cheap mtime/size check, not a content hash
  or a real data-versioning system, and is documented as such — it catches
  "you forgot to re-run bronze," not "bronze ran against corrupted input."

DuckDB (not a CSV round-trip) is the checkpoint format specifically because
it preserves dtypes exactly — verified for this project's two concrete
gotchas (`Int64` columns with `pd.NA` don't get silently promoted to
`float64`+`NaN`, and a cell whose literal value is the string `"N/A"`
doesn't get reinterpreted as null on read-back) — see
`tests/test_stage_pipeline.py::test_checkpoint_roundtrip_preserves_nullable_int_and_na_string_literal`.
CSVs remain the audit/deliverable exports; they are not, and never were,
what any stage reads to obtain its input.

This is deliberately scoped to CLI-level stage separation, not a scheduler:
no Airflow/Dagster DAG, no separate metadata database, no task retries —
those stay future work (§9's Orchestration bullet, below), since this
pipeline's actual runtime (~20s including a cold model download; sub-second
for everything except embeddings) doesn't need them to demonstrate the
thing being asked for, which is that the layer boundaries are real and
independently runnable, not just files organized into separate modules.

## 9. Backfills, new batches, and schema changes in a production setting

This exercise runs entirely as a local batch: read two files, produce
some CSVs and a DuckDB file. Adapting the same model to run reliably and
recur in production:

- **New batches**: land new `tech_news.csv`-shaped files (e.g. daily) into
  an append-only landing zone/bronze store (e.g. partitioned by load date
  in S3 + a bronze external/Iceberg table), keyed by `article_id`. Silver
  and gold are re-derived incrementally: only bronze rows newer than the
  last successful run (or, more robustly, any bronze row whose derived
  silver/gold row doesn't yet exist or is stale) need reprocessing.
  `article_id` uniqueness would need to be enforced/deduped at bronze
  ingestion time if the same article could plausibly appear in two batches.
- **Backfills / corrections**: because `fact_arr_observations` upserts on
  `article_id`, replaying an old batch (e.g. a corrected `revenue` value
  for an already-loaded article) is safe — it overwrites that one row
  in place rather than duplicating it. In a warehouse like Snowflake/
  BigQuery/Databricks this is a `MERGE INTO ... ON article_id`; in dbt,
  an incremental model with `unique_key = article_id`.
- **Company metadata changes**: `dim_company` would become a slowly
  changing dimension (SCD Type 2: `valid_from`/`valid_to`/`is_current`)
  instead of being overwritten wholesale, so that `company_age` and
  `company_size_category` computed against an article's `published_date`
  use the metadata that was true *at that time*, not today's metadata.
  This exercise treats metadata as a single current snapshot since the
  provided file has no history.
- **Schema changes**: adding a column (e.g. a new metadata field) is
  additive and low-risk with `MERGE`/incremental models. Removing or
  retyping a column is handled by versioning the table build (a new model
  version, migrated forward) rather than mutating a live table in place —
  this exercise's local DuckDB loader takes the simplest safe option
  available for one-off local runs (drop and rebuild the table when the
  incoming column set differs from what's stored) which would be too
  blunt for a live warehouse table with downstream consumers.
- **FX rates**: this exercise uses one fixed EUR/GBP/JPY rate for the whole
  dataset (as specified). Production would join each observation's
  `published_date` against a dated FX rate table instead of a constant, so
  the same original amount converts differently depending on when it was
  reported.
- **Orchestration**: bronze → silver → gold → exports would become
  discrete tasks in Airflow/Dagster with per-layer freshness checks (e.g.
  "gold ran only after silver's row count for this batch stabilized"), and
  the CSV/DuckDB outputs would become external tables or a real warehouse
  schema rather than files on disk. §8 above already makes those four
  layers independently runnable and checkpointed at the CLI level — a real
  scheduler would call the same `run_bronze`/`run_silver`/`run_gold`/
  `run_exports` functions as separate tasks rather than needing new seams
  cut into the pipeline first.

## 10. Key assumptions and trade-offs (for the live review)

- **Date ambiguity rule**: for numeric `MM/DD/YYYY`-or-`DD/MM/YYYY`-shaped
  dates (both `/` and `-` separated — both separators are observed with a
  genuine mix of day-first and month-first values in this dataset), if
  exactly one of the two numeric components is >12 it must be the day; if
  both are ≤12 the date is genuinely ambiguous and we default to
  US-style month-first. This default is a guess for the ambiguous subset,
  not a detected fact — see `pipeline/cleaning.py::_resolve_ambiguous_numeric_date`.
- **Category taxonomy**: 19 raw values collapsed into 6 canonical buckets
  (`AI_ML`, `DATA_ANALYTICS`, `CLOUD`, `SECURITY`, `FINTECH`, `SOFTWARE`),
  chosen by grouping obviously-synonymous raw strings. The same mapping is
  reused for company `industry` so the AI-article filter can check either
  field consistently. Anything unrecognized maps to `OTHER` rather than
  raising or being dropped.
- **Fuzzy-match threshold (88/100, RapidFuzz whole-string `ratio`)**: picked
  to catch case/punctuation variants and near-miss typos while rejecting
  compound/joint-entity names that merely contain a real company name (e.g.
  "The Boring Company / SpaceX"). At 88, every alias in the data that isn't
  already caught by the exact/alias paths correctly falls through to
  fuzzy or unmatched with no false positives observed; this was validated
  against the real 46 distinct `company_name` values, not chosen blind.
- **`company_key` as a natural key**: simplifies idempotency for this
  exercise; a production system at scale would add a surrogate key (see
  §9).
- **`ai_articles_enriched.company_name`**: populated from the *resolved*
  `company_key` (canonical name when matched, raw name when not), not the
  raw `company_name` column, since this export is meant to be analysis-
  ready. The raw string is still available via `fact_arr_observations`/
  `silver_articles.csv` for lineage.
- **Quarterly ARR = latest observation in the quarter**, not sum/average —
  see §6.
- **Embedding storage**: the canonical/reusable form is `article_embeddings.npy`
  + `article_embeddings_index.csv` (row order = `article_id` order) and the
  DuckDB `article_embeddings` table (`DOUBLE[]` column, queried with
  DuckDB's native `list_cosine_similarity`). The `embedding` column inside
  `ai_articles_enriched.csv` is a JSON-serialized, rounded copy for
  spreadsheet-level inspection only — don't treat it as the source of
  truth for re-computation.
