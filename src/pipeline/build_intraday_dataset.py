from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.collect.intraday_prices import fetch_intraday_hour_bars
from src.datasets.build_intraday_samples import build_intraday_samples
from src.features.build_intraday_features import build_intraday_panel
from src.pipeline.config import build_paths, load_config, normalize_config
from src.timemmd.build_intraday_timemmd import build_intraday_numerical, build_intraday_search_text


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _range(cfg: Dict[str, Any]) -> tuple[str, str]:
    intraday = cfg.get("intraday", {})
    end = pd.to_datetime(cfg["date_range"]["end"]).date()
    years = int(intraday.get("lookback_years", 2))
    start = (pd.Timestamp(end) - pd.DateOffset(years=years)).date()
    return str(start), str(end)


def _write_empty_events(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        columns=[
            "event_id",
            "event_type",
            "source",
            "title",
            "summary",
            "published_at_utc",
            "available_at_utc",
            "affected_symbols",
            "sentiment",
            "relevance",
            "provider",
            "url_hash",
        ]
    ).to_parquet(path, index=False)


def _intraday_summary(panel: pd.DataFrame, sample_paths: list[Path]) -> Dict[str, Any]:
    bar_end = pd.to_datetime(panel["bar_end_utc"], utc=True)
    day_counts = bar_end.dt.date.value_counts()
    split_counts: Dict[str, Dict[str, int]] = {}
    for path in sample_paths:
        samples = pd.read_parquet(path, columns=["split"])
        split_counts[path.name] = {str(k): int(v) for k, v in samples["split"].value_counts().to_dict().items()}

    return {
        "row_counts": {
            "hourly_panel": int(len(panel)),
            "intraday_samples": {path.name: int(pd.read_parquet(path, columns=["sample_id"]).shape[0]) for path in sample_paths},
        },
        "actual_date_range_utc": {
            "start": bar_end.min().isoformat(),
            "end": bar_end.max().isoformat(),
        },
        "max_bars_per_trading_day": int(day_counts.max()) if not day_counts.empty else 0,
        "sample_splits": split_counts,
    }


def build(cfg_path: Path) -> Dict[str, Any]:
    cfg = normalize_config(load_config(cfg_path))
    paths = build_paths(cfg)
    intraday = cfg.get("intraday", {})
    symbols = intraday.get("symbols", [cfg.get("target_symbol", "USO")])
    start, end = _range(cfg)

    raw_intraday_path = paths.raw / "intraday_prices" / "raw_intraday_prices.parquet"
    events_path = paths.raw / "events" / "raw_events.parquet"
    panel_path = paths.processed / "hourly_panel.parquet"
    numerical_path = paths.timemmd_numerical / "OilETF_intraday.csv"
    search_path = paths.timemmd_textual / "OilETF_intraday_search.csv"

    fetch_intraday_hour_bars(symbols, start, end, raw_intraday_path, period=str(intraday.get("yfinance_period", "730d")))
    _write_empty_events(events_path)
    panel = build_intraday_panel(raw_intraday_path, events_path, panel_path)
    numerical = build_intraday_numerical(panel, numerical_path)
    build_intraday_search_text(panel, search_path)
    sample_paths = build_intraday_samples(
        numerical,
        intraday.get("sample_settings", [{"H": 60, "F": 1}, {"H": 120, "F": 7}]),
        symbols[0],
        paths.processed,
        x_numeric_path=_relative(numerical_path, paths.root),
        x_text_embedding_path=_relative(search_path, paths.root),
    )

    metadata = {
        "dataset_name": f"{cfg.get('outputs', {}).get('dataset_name', 'OilETF')}-intraday",
        "frequency": "1h regular session",
        "symbols": symbols,
        "requested_date_range": {"start": start, "end": end},
        "sources": {"prices": "yfinance 1h bars", "news": "not included in intraday v1"},
        "license_note": "yfinance intraday data is suitable for research demos; verify redistribution terms before publishing raw data.",
        "outputs": {
            "raw_intraday_prices": _relative(raw_intraday_path, paths.root),
            "events": _relative(events_path, paths.root),
            "hourly_panel": _relative(panel_path, paths.root),
            "numerical": _relative(numerical_path, paths.root),
            "search_text": _relative(search_path, paths.root),
            "samples": [_relative(p, paths.root) for p in sample_paths],
        },
    }
    metadata.update(_intraday_summary(panel, sample_paths))
    metadata_path = paths.metadata / "OilETF_intraday_dataset_card.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata["outputs"] | {"metadata": _relative(metadata_path, paths.root)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OilETF-TimeMMD intraday dataset pipeline")
    parser.add_argument("--config", type=str, default="configs/data_config.yaml")
    args = parser.parse_args()
    result = build(Path(args.config))
    print("Intraday build completed")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
