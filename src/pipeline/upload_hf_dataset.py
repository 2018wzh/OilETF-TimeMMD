from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from huggingface_hub import HfApi

from src.pipeline.config import build_paths


DAILY_PATTERNS = [
    "processed/daily_panel.parquet",
    "numerical/OilETF/OilETF.csv",
    "textual/OilETF/OilETF_search.csv",
    "textual/OilETF/OilETF_report.csv",
    "processed/samples_H60_F1.parquet",
    "processed/samples_H120_F5.parquet",
    "metadata/OilETF_dataset_card.json",
    "README.md",
]

INTRADAY_PATTERNS = [
    "processed/hourly_panel.parquet",
    "numerical/OilETF/OilETF_intraday.csv",
    "textual/OilETF/OilETF_intraday_search.csv",
    "processed/intraday_samples_H60_F1.parquet",
    "processed/intraday_samples_H120_F7.parquet",
    "metadata/OilETF_intraday_dataset_card.json",
    "README.md",
]


def resolve_repo_id(repo_id: str | None, cfg: Dict[str, Any]) -> str:
    value = (repo_id or "").strip() or str(cfg.get("huggingface", {}).get("repo_id") or "").strip()
    if not value:
        raise ValueError("Missing Hugging Face dataset repo id. Pass --hf-repo-id or set huggingface.repo_id in config.")
    return value


def build_allow_patterns(scope: str) -> List[str]:
    if scope == "daily":
        return DAILY_PATTERNS
    if scope == "intraday":
        return INTRADAY_PATTERNS
    if scope == "all":
        return DAILY_PATTERNS + [p for p in INTRADAY_PATTERNS if p not in DAILY_PATTERNS]
    raise ValueError(f"Unsupported upload scope: {scope}")


def upload_dataset(cfg: Dict[str, Any], repo_id: str | None, scope: str = "all", private: bool | None = None) -> str:
    paths = build_paths(cfg)
    resolved_repo_id = resolve_repo_id(repo_id, cfg)
    hf_cfg = cfg.get("huggingface", {})
    api = HfApi()
    api.create_repo(
        repo_id=resolved_repo_id,
        repo_type="dataset",
        private=bool(hf_cfg.get("private", False) if private is None else private),
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(paths.root / "data"),
        repo_id=resolved_repo_id,
        repo_type="dataset",
        allow_patterns=build_allow_patterns(scope),
        ignore_patterns=["raw/**", ".env", "**/.env"],
        commit_message=f"Upload OilETF-TimeMMD {scope} dataset",
    )
    return f"https://huggingface.co/datasets/{resolved_repo_id}"
