"""Unit tests for pipeline.company_resolution, using the real company_name
values observed in tech_news.csv against the real company_metadata.json keys.
"""
import json

import pytest

from pipeline import company_resolution as cr
from pipeline import config


@pytest.fixture(scope="module")
def metadata_keys():
    with open(config.COMPANY_METADATA_JSON, "r", encoding="utf-8") as f:
        return list(json.load(f).keys())


@pytest.mark.parametrize(
    "raw_name",
    [
        "Airbnb",
        "Anthropic",
        "Confluent",
        "Elastic",
        "Microsoft",
        "NVIDIA",
        "OpenAI",
        "Palantir",
        "Scale AI",
        "Snowflake",
        "SpaceX",
        "Stripe",
        "Tesla",
        "Uber",
        "MongoDB",
        "Databricks",
        "DataRobot",
        "Cloudflare",
        "Meta AI",
        "Google DeepMind",
        "Amazon Web Services",
    ],
)
def test_exact_match(raw_name, metadata_keys):
    result = cr.resolve_company(raw_name, metadata_keys)
    assert result.canonical_name == raw_name
    assert result.match_method == cr.MATCH_EXACT


@pytest.mark.parametrize(
    "raw_name, expected_canonical",
    [
        ("AWS", "Amazon Web Services"),
        ("Amazon Web Services (AWS)", "Amazon Web Services"),
        ("Azure", "Microsoft"),
        ("Microsoft Azure", "Microsoft"),
        ("Open AI", "OpenAI"),
        ("OpenAI Inc.", "OpenAI"),
        ("Databricks Inc.", "Databricks"),
        ("Snowflake Inc.", "Snowflake"),
        ("Stripe Inc.", "Stripe"),
        ("NVIDIA Corporation", "NVIDIA"),
        ("Data Robot", "DataRobot"),
        ("Mongo DB", "MongoDB"),
        ("Facebook AI Research", "Meta AI"),
        ("Meta AI Research", "Meta AI"),
        ("DeepMind", "Google DeepMind"),
    ],
)
def test_alias_match(raw_name, expected_canonical, metadata_keys):
    result = cr.resolve_company(raw_name, metadata_keys)
    assert result.canonical_name == expected_canonical
    assert result.match_method == cr.MATCH_ALIAS


@pytest.mark.parametrize(
    "raw_name, expected_canonical",
    [
        ("CloudFlare", "Cloudflare"),
        ("Nvidia", "NVIDIA"),
        ("Google Deepmind", "Google DeepMind"),
    ],
)
def test_case_variants_resolve_case_insensitively(raw_name, expected_canonical, metadata_keys):
    result = cr.resolve_company(raw_name, metadata_keys)
    assert result.canonical_name == expected_canonical
    # These are covered by the case-insensitive exact check, ahead of fuzzy.
    assert result.match_method == cr.MATCH_EXACT_CI


@pytest.mark.parametrize(
    "raw_name",
    [
        "Cohere",
        "Mistral AI",
        "Perplexity AI",
        "Hugging Face",
        "xAI",
        "The Boring Company / SpaceX",
    ],
)
def test_expected_unmatched_companies_stay_unmatched(raw_name, metadata_keys):
    result = cr.resolve_company(raw_name, metadata_keys)
    assert result.canonical_name is None
    assert result.match_method == cr.MATCH_UNMATCHED


def test_fuzzy_match_catches_near_miss_typo(metadata_keys):
    # a plausible near-miss that isn't in the alias dict and isn't an exact
    # case-insensitive match, but should clear the fuzzy threshold
    result = cr.resolve_company("Snowflak", metadata_keys)
    assert result.canonical_name == "Snowflake"
    assert result.match_method == cr.MATCH_FUZZY


def test_unmatched_company_still_returns_a_score_for_triage(metadata_keys):
    result = cr.resolve_company("xAI", metadata_keys)
    assert result.match_score is not None
    assert result.match_score < config.FUZZY_MATCH_THRESHOLD
