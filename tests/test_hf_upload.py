from pathlib import Path

import pytest

from src.pipeline.upload_hf_dataset import build_allow_patterns, resolve_repo_id


def test_resolve_repo_id_prefers_cli_value():
    cfg = {"huggingface": {"repo_id": "owner/from-config"}}

    assert resolve_repo_id("owner/from-cli", cfg) == "owner/from-cli"


def test_resolve_repo_id_reads_config_without_hardcoding():
    cfg = {"huggingface": {"repo_id": "owner/from-config"}}

    assert resolve_repo_id("", cfg) == "owner/from-config"


def test_resolve_repo_id_requires_value():
    with pytest.raises(ValueError):
        resolve_repo_id("", {})


def test_intraday_upload_patterns_do_not_include_raw_data():
    patterns = build_allow_patterns("intraday")

    assert "processed/hourly_panel.parquet" in patterns
    assert "raw/intraday_prices/raw_intraday_prices.parquet" not in patterns
    assert all(not p.startswith("raw/") for p in patterns)

