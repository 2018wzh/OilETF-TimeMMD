from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict
from tqdm.auto import tqdm

import pandas as pd

from src.collect.intraday_news import build_intraday_events
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


def _intraday_summary(panel: pd.DataFrame, events: pd.DataFrame, sample_paths: list[Path]) -> Dict[str, Any]:
    bar_end = pd.to_datetime(panel["bar_end_utc"], utc=True)
    day_counts = bar_end.dt.date.value_counts()
    split_counts: Dict[str, Dict[str, int]] = {}
    for path in sample_paths:
        samples = pd.read_parquet(path, columns=["split"])
        split_counts[path.name] = {str(k): int(v) for k, v in samples["split"].value_counts().to_dict().items()}

    return {
        "row_counts": {
            "hourly_panel": int(len(panel)),
            "events": int(len(events)),
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
    pbar = tqdm(total=9, desc="intraday: start", unit="stage")
    try:
        cfg = normalize_config(load_config(cfg_path))
        pbar.update(1)
        pbar.set_description("intraday: load config")

        paths = build_paths(cfg)
        intraday = cfg.get("intraday", {})
        symbols = intraday.get("symbols") or cfg.get("symbols") or [cfg.get("target_symbol", "USO")]
        symbols = list(dict.fromkeys(symbols))  # de-dupe, preserve order
        if len(symbols) == 0:
            symbols = ["USO"]
        pbar.update(1)
        pbar.set_description(f"intraday: resolve date range ({len(symbols)} symbols)")
        start, end = _range(cfg)

        raw_intraday_path = paths.raw / "intraday_prices" / "raw_intraday_prices.parquet"
        events_path = paths.raw / "events" / "raw_events.parquet"
        panel_path = paths.processed / "hourly_panel.parquet"
        numerical_path = paths.timemmd_numerical / "OilETF_intraday.csv"
        search_path = paths.timemmd_textual / "OilETF_intraday_search.csv"
        pbar.set_description(f"intraday: fetch {len(symbols)} symbols 1h bars")

        fetch_intraday_hour_bars(symbols, start, end, raw_intraday_path, period=str(intraday.get("yfinance_period", "730d")))
        pbar.update(1)
        pbar.set_description(f"intraday: fetch events for {len(symbols)} symbols")
        intraday_collection_cfg = dict(cfg.get("collection", {}))
        # merge news sources: intraday overrides, otherwise inherit from daily collection
        _intra_sources = intraday.get("news_sources", ["finnhub"])
        if isinstance(_intra_sources, str):
            _intra_sources = [_intra_sources]
        intraday_collection_cfg["news_sources"] = _intra_sources
        # carry finnhub + gdelt sub-configs from daily collection
        if "finnhub" in intraday:
            intraday_collection_cfg["finnhub"] = intraday["finnhub"]
        if "finnhub" in cfg.get("collection", {}):
            intraday_collection_cfg.setdefault("finnhub", cfg["collection"]["finnhub"])
        if "gdelt" in cfg.get("collection", {}):
            intraday_collection_cfg.setdefault("gdelt", {})
            intraday_collection_cfg["gdelt"] = {
                **cfg["collection"]["gdelt"],
                **intraday_collection_cfg.get("gdelt", {}),
            }
        news_cache_path = paths.raw / "news" / "raw_news.csv"
        events, _news_miss = build_intraday_events(
            symbols,
            start,
            end,
            events_path,
            collection_cfg=intraday_collection_cfg,
            per_symbol_limit=int(cfg.get("collection", {}).get("news_per_symbol_limit", 200)),
            news_cache_path=news_cache_path,
        )
        # Summarise event coverage per symbol
        if not events.empty:
            ev_sym_counts = (
                events["affected_symbols"].str.split(";").explode().str.strip().str.upper().value_counts().to_dict()
            )
            event_date_min = str(pd.to_datetime(events["published_at_utc"]).min().date())
            event_date_max = str(pd.to_datetime(events["published_at_utc"]).max().date())
            _news_miss.append(
                f"intraday events: {len(events)} total, {event_date_min}..{event_date_max}; per-symbol: {ev_sym_counts}"
            )
        pbar.update(1)
        pbar.set_description("intraday: build panel")
        panel = build_intraday_panel(raw_intraday_path, events_path, panel_path)
        pbar.update(1)
        pbar.set_description("intraday: build numerical")
        numerical = build_intraday_numerical(panel, numerical_path)
        pbar.update(1)
        pbar.set_description("intraday: build search text")
        build_intraday_search_text(panel, search_path)
        pbar.update(1)
        pbar.set_description("intraday: build samples")
        news_sources = intraday_collection_cfg.get("news_sources", ["finnhub"])
        source_tokens: list[str] = []
        if any(str(s).lower() == "finnhub" for s in news_sources):
            source_tokens.append("finnhub.company-news")
        if any(str(s).lower() == "gdelt" for s in news_sources):
            source_tokens.append("gdelt.doc")
        if not source_tokens:
            source_tokens.append("news disabled")
        sample_paths = build_intraday_samples(
            numerical,
            intraday.get("sample_settings", [{"H": 60, "F": 1}, {"H": 120, "F": 7}]),
            paths.processed,
            x_numeric_path=_relative(numerical_path, paths.root),
            x_text_embedding_path=_relative(search_path, paths.root),
        )
        pbar.update(1)
        pbar.set_description("intraday: write metadata")

        metadata = {
            "dataset_name": f"{cfg.get('outputs', {}).get('dataset_name', 'OilETF')}-intraday",
            "frequency": "1h regular session",
            "symbols": symbols,
            "requested_date_range": {"start": start, "end": end},
            "sources": {"prices": "yfinance 1h bars", "news": " + ".join(source_tokens)},
            "license_note": (
                "yfinance intraday data is suitable for research demos; verify redistribution terms before publishing raw data. "
                "Finnhub free-tier company-news only returns ~30 days of recent articles regardless of from/to dates."
            ),
            "news_warnings": [m for m in _news_miss if "finnhub collected" in m or "intraday events" in m],
            "outputs": {
                "raw_intraday_prices": _relative(raw_intraday_path, paths.root),
                "events": _relative(events_path, paths.root),
                "hourly_panel": _relative(panel_path, paths.root),
                "numerical": _relative(numerical_path, paths.root),
                "search_text": _relative(search_path, paths.root),
                "samples": [_relative(p, paths.root) for p in sample_paths],
            },
        }
        metadata.update(_intraday_summary(panel, events, sample_paths))
        metadata_path = paths.metadata / "OilETF_intraday_dataset_card.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        pbar.update(1)
        pbar.set_description("intraday: done")
        return metadata["outputs"] | {"metadata": _relative(metadata_path, paths.root)}
    finally:
        pbar.close()


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
