"""Bronze layer: raw ingestion with lineage columns, no cleaning.

Bronze tables are the untouched source data (every original column kept as
the raw string it was read as) plus lineage metadata: which source row it
came from and when it was loaded. This is the audit trail every downstream
layer can be traced back to.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from . import config


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_bronze_articles(csv_path=None) -> pd.DataFrame:
    """Read tech_news.csv as-is (all columns as strings) and add lineage columns."""
    csv_path = csv_path or config.TECH_NEWS_CSV
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    loaded_at = _now_iso()
    df.insert(0, "_source_row_number", range(1, len(df) + 1))
    df["_source_file"] = str(csv_path.name if hasattr(csv_path, "name") else csv_path)
    df["_loaded_at"] = loaded_at
    return df


def load_bronze_company_metadata(json_path=None) -> pd.DataFrame:
    """Read company_metadata.json (dict keyed by company name) into a flat table."""
    json_path = json_path or config.COMPANY_METADATA_JSON
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    loaded_at = _now_iso()
    rows = []
    for idx, (company_name, fields) in enumerate(raw.items(), start=1):
        row = {"_source_row_number": idx, "company_name_raw": company_name}
        row.update(fields)
        row["_source_file"] = str(json_path.name if hasattr(json_path, "name") else json_path)
        row["_loaded_at"] = loaded_at
        rows.append(row)
    return pd.DataFrame(rows)
