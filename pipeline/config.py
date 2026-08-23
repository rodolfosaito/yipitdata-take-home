"""Central configuration: constants, taxonomies, and thresholds.

Keeping every "business rule" (currency rates, category taxonomy, company
aliases, size thresholds, fuzzy-match threshold) in one module makes the
assumptions easy to find, cite in the live review, and change in one place.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Currency conversion to USD (fixed rates for this exercise; a production
# pipeline would pull a dated FX rate per published_date instead of one
# static multiplier -- see DATA_ARCHITECTURE.md).
# ---------------------------------------------------------------------------
CURRENCY_TO_USD = {
    "USD": 1.0,
    "EUR": 1.1,
    "GBP": 1.27,
    "JPY": 1.0 / 150.0,
}

CURRENCY_SYMBOLS = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
}

# ---------------------------------------------------------------------------
# Category taxonomy: raw `category` (article) and `industry` (company
# metadata) strings both get folded into the same small canonical set, since
# both describe "what kind of tech company is this" and the AI-article
# export needs to check either field. Discovered raw values are listed in
# full here (see README / Data Architecture doc for how they were found).
# ---------------------------------------------------------------------------
CATEGORY_TAXONOMY = {
    # canonical -> raw values that map to it
    "AI_ML": [
        "AI & ML",
        "AI/ML",
        "Artificial Intelligence",
        "Machine Learning",
    ],
    "DATA_ANALYTICS": [
        "Analytics",
        "Big Data",
        "Data Analytics",
    ],
    "CLOUD": [
        "Cloud",
        "Cloud Computing",
        "Cloud Services",
    ],
    "SECURITY": [
        "Cybersecurity",
        "InfoSec",
        "Security",
    ],
    "FINTECH": [
        "Finance",
        "Financial Technology",
        "FinTech",
    ],
    "SOFTWARE": [
        "Enterprise Software",
        "SaaS",
        "Software",
    ],
}

# Flattened raw -> canonical lookup (case-insensitive keys).
CATEGORY_RAW_TO_STD = {
    raw.strip().lower(): canonical
    for canonical, raw_values in CATEGORY_TAXONOMY.items()
    for raw in raw_values
}

UNKNOWN_CATEGORY = "OTHER"

# ---------------------------------------------------------------------------
# Company alias resolution
# ---------------------------------------------------------------------------
# Static alias map: raw company_name string (as it appears in tech_news.csv)
# -> canonical company_name key (as it appears in company_metadata.json).
# Seeded from the pairs called out in the assignment brief, then extended
# with every other alias/case-variant found by scanning the 46 distinct
# company_name values in tech_news.csv against the 21 metadata keys.
COMPANY_ALIASES = {
    "AWS": "Amazon Web Services",
    "Amazon Web Services (AWS)": "Amazon Web Services",
    "Azure": "Microsoft",
    "Microsoft Azure": "Microsoft",
    "Open AI": "OpenAI",
    "OpenAI Inc.": "OpenAI",
    "Databricks Inc.": "Databricks",
    "Snowflake Inc.": "Snowflake",
    "Stripe Inc.": "Stripe",
    "NVIDIA Corporation": "NVIDIA",
    "Nvidia": "NVIDIA",
    "Data Robot": "DataRobot",
    "Mongo DB": "MongoDB",
    "Facebook AI Research": "Meta AI",
    "Meta AI Research": "Meta AI",
    "DeepMind": "Google DeepMind",
    "Google Deepmind": "Google DeepMind",
    "CloudFlare": "Cloudflare",
    "Palantir Technologies": "Palantir",
}

# Company names observed in the data that are known, on inspection, to have
# no reasonable metadata match (separate legal entity, or simply absent from
# company_metadata.json). Listed explicitly so "unmatched" is a documented
# expectation rather than a silent gap. They still flow through fuzzy
# matching below (in case of coincidence) but are expected to fail it.
EXPECTED_UNMATCHED_COMPANIES = {
    "Cohere",
    "Mistral AI",
    "Perplexity AI",
    "Hugging Face",
    "xAI",
    "The Boring Company / SpaceX",
}

# Minimum RapidFuzz `ratio` (0-100, whole-string similarity, NOT partial/
# substring) score for an unresolved company_name to be accepted as a fuzzy
# match against a metadata company key. Whole-string ratio (not
# partial_ratio) is used deliberately so that e.g. "The Boring Company /
# SpaceX" does NOT fuzzy-match "SpaceX" -- partial/substring matching would
# produce false positives on names that merely contain a known company name.
FUZZY_MATCH_THRESHOLD = 88

# ---------------------------------------------------------------------------
# Company size thresholds (from metadata employee_count)
# ---------------------------------------------------------------------------
SIZE_SMALL_MAX = 10_000  # < this => Small
SIZE_MEDIUM_MAX = 30_000  # <= this (and >= SMALL_MAX) => Medium; above => Large

# ---------------------------------------------------------------------------
# AI article export filter
# ---------------------------------------------------------------------------
AI_ARTICLE_YEAR_MIN = 2022
AI_ARTICLE_YEAR_MAX = 2024
AI_ARTICLE_ARR_MIN_USD = 50_000_000  # strictly greater than

# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_SIMILAR_ARTICLES_K = 3

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TECH_NEWS_CSV = PROJECT_ROOT / "tech_news.csv"
COMPANY_METADATA_JSON = PROJECT_ROOT / "company_metadata.json"
OUTPUT_DIR = PROJECT_ROOT / "output"
DUCKDB_PATH = OUTPUT_DIR / "warehouse.duckdb"
EMBEDDINGS_NPY_PATH = OUTPUT_DIR / "article_embeddings.npy"
EMBEDDINGS_INDEX_PATH = OUTPUT_DIR / "article_embeddings_index.csv"
