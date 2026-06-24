from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
from tqdm.auto import tqdm

import pandas as pd

from src.pipeline.config import build_paths, load_config, normalize_config
from src.collect import prices as collect_prices
from src.collect import calendar as collect_calendar
from src.collect import news as collect_news
from src.collect import reports as collect_reports
from src.features.build_features import build_daily_panel, write_panel
from src.timemmd.build_timemmd import build_numerical_df, build_search_text, build_report_text
from src.datasets.build_samples import build_samples
from src.visuals.candles import generate_candles
from src.metadata.build_metadata import build_dataset_card, write_quality_report


def _concat_dedup(new_df: pd.DataFrame, path: Path, key_cols: List[str]) -> pd.DataFrame:
    if path.exists():
        existing = pd.read_csv(path)
        all_df = pd.concat([existing, new_df], ignore_index=True)
        all_df = all_df.drop_duplicates(subset=key_cols, keep="last").sort_values(key_cols).reset_index(drop=True)
        return all_df
    return new_df


def _ensure_series_dates(cfg: Dict[str, Any]):
    start = cfg["date_range"]["start"]
    end = cfg["date_range"]["end"]
    return str(start), str(end)


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def build(cfg_path: Path, incremental: bool = False, skip_visuals: bool = False) -> Dict[str, Any]:
    pbar = tqdm(total=16, desc="daily: start", unit="stage")
    try:
        cfg = normalize_config(load_config(cfg_path))
        pbar.update(1)
        pbar.set_description("daily: build paths")
        paths = build_paths(cfg)
        start_str, end_str = _ensure_series_dates(cfg)
        pbar.update(1)
        pbar.set_description("daily: resolve date range")

        raw_prices_path = paths.raw / "prices" / "raw_prices.csv"
        raw_macro_path = paths.raw / "oil_macro" / "raw_oil_macro.csv"
        raw_news_path = paths.raw / "news" / "raw_news.csv"
        raw_reports_path = paths.raw / "reports" / "raw_reports.csv"
        raw_calendar_path = paths.raw / "calendar" / "calendar.csv"

        if incremental and raw_prices_path.exists():
            existing = pd.read_csv(raw_prices_path)
            if not existing.empty:
                max_date = pd.to_datetime(existing["timestamp"]).max().date()
                start_collect = str(max_date + pd.Timedelta(days=1))
                cfg["date_range"]["start"] = start_collect
            else:
                start_collect = start_str
        else:
            start_collect = start_str

        if pd.to_datetime(start_collect) > pd.to_datetime(end_str):
            start_collect = end_str
        pbar.update(1)
        pbar.set_description("daily: fetch prices")

        prices = collect_prices.fetch_prices(cfg["symbols"], start_collect, end_str)
        pbar.update(1)
        pbar.set_description("daily: persist prices")
        if incremental:
            raw_prices = _concat_dedup(prices, raw_prices_path, ["symbol", "timestamp"])
        elif prices.empty and raw_prices_path.exists():
            raw_prices = pd.read_csv(raw_prices_path)
        else:
            raw_prices = prices
        raw_prices.to_csv(raw_prices_path, index=False)
        pbar.update(1)
        pbar.set_description("daily: fetch macro")

        macro = collect_prices.fetch_macro_data(start_collect, end_str, cfg["macro"])
        pbar.update(1)
        pbar.set_description("daily: persist macro")
        if incremental and raw_macro_path.exists():
            existing_macro = pd.read_csv(raw_macro_path)
            raw_macro = pd.concat([existing_macro, macro], ignore_index=True)
            raw_macro = raw_macro.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
        elif macro.empty and raw_macro_path.exists():
            raw_macro = pd.read_csv(raw_macro_path)
        else:
            raw_macro = macro
        raw_macro.to_csv(raw_macro_path, index=False)
        pbar.update(1)
        pbar.set_description(f"daily: fetch news ({len(cfg['symbols'])} symbols)")

        if cfg["collection"].get("include_news", True):
            news, news_miss = collect_news.fetch_news_with_cache(
                cfg["symbols"],
                start_collect,
                end_str,
                cache_path=raw_news_path,
                per_symbol_limit=cfg["collection"]["news_per_symbol_limit"],
                collection_cfg=cfg["collection"],
            )
            news["published_at_utc"] = pd.to_datetime(news["published_at_utc"], utc=True, errors="coerce").astype(str)
            if not news.empty:
                tmp_ts = pd.to_datetime(news["published_at_utc"], utc=True)
                sym_counts = news["target_symbol"].value_counts().to_dict()
                news_miss.append(
                    f"daily news: {len(news)} articles, {tmp_ts.min().date()}..{tmp_ts.max().date()}; "
                    f"per-symbol: {sym_counts}"
                )
            raw_news = news
        else:
            news_miss = ["news disabled by config"]
            raw_news = pd.DataFrame(
                columns=[
                    "article_id",
                    "published_at_utc",
                    "source",
                    "title",
                    "summary",
                    "url_hash",
                    "tickers",
                    "entities",
                    "sentiment",
                    "relevance",
                    "provider",
                    "target_symbol",
                ]
            )
        raw_news.to_csv(raw_news_path, index=False)
        pbar.update(1)
        pbar.set_description("daily: fetch reports")

        if cfg["collection"].get("include_reports", True):
            reports, miss = collect_reports.collect_reports(start_str, end_str, cfg["collection"]["max_report_items_per_source"])
        else:
            reports = pd.DataFrame(
                columns=[
                    "published_at_utc",
                    "source",
                    "title",
                    "fact",
                    "preds",
                    "url",
                    "url_hash",
                    "start_date",
                    "end_date",
                    "provider",
                ]
            )
            miss = []
        reports.to_csv(raw_reports_path, index=False)
        pbar.update(1)
        pbar.set_description("daily: build calendar")

        calendar = collect_calendar.build_trading_calendar(start_collect, end_str, cfg["calendar"]["source_symbol"])
        calendar.to_csv(raw_calendar_path, index=False)
        pbar.update(1)
        pbar.set_description("daily: build panel")

        daily_panel = build_daily_panel(
            raw_prices_path=raw_prices_path,
            raw_macro_path=raw_macro_path,
            raw_news_path=raw_news_path,
            calendar_path=raw_calendar_path,
            symbols=cfg["symbols"],
            cfg=cfg,
        )
        panel_path = paths.processed / "daily_panel.parquet"
        write_panel(daily_panel, panel_path)
        pbar.update(1)
        pbar.set_description("daily: build timmmd features")

        timemmd_num_path = paths.timemmd_numerical / "OilETF.csv"
        numerical = build_numerical_df(daily_panel, timemmd_num_path)
        timemmd_search_path = paths.timemmd_textual / "OilETF_search.csv"
        timemmd_report_path = paths.timemmd_textual / "OilETF_report.csv"
        build_search_text(
            daily_panel=daily_panel,
            raw_news=raw_news,
            out_path=timemmd_search_path,
            trading_dates=pd.to_datetime(daily_panel["end_date"]),
            cutoff_hour=cfg.get("collection", {}).get("news_cutoff_hour_et", 16),
        )
        build_report_text(reports, timemmd_report_path, miss, daily_panel=daily_panel)
        pbar.update(1)
        pbar.set_description("daily: build samples")

        sample_paths = build_samples(
            numerical=numerical,
            settings=cfg["sample_settings"],
            symbol=cfg.get("target_symbol", "USO"),
            output_dir=paths.processed,
            x_numeric_path=_relative_path(timemmd_num_path, paths.root),
            x_text_embedding_path=_relative_path(timemmd_search_path, paths.root),
            x_image_dir=_relative_path(paths.images, paths.root),
        )

        if not skip_visuals:
            generate_candles(raw_prices_path, cfg["symbols"], cfg["windows"]["H"], paths.images)
        pbar.update(1)
        pbar.set_description("daily: calc split stats")

        split_counts = {}
        for sp in sample_paths:
            if Path(sp).exists():
                df = pd.read_parquet(sp)
                split_counts[sp.name] = df["split"].value_counts().to_dict()
        pbar.update(1)
        pbar.set_description("daily: write metadata")

        data_dict = {
            "config": cfg_path,
            "raw_prices": raw_prices_path,
            "raw_macro": raw_macro_path,
            "raw_news": raw_news_path,
            "raw_reports": raw_reports_path,
            "raw_calendar": raw_calendar_path,
            "daily_panel": panel_path,
            "numerical": timemmd_num_path,
            "search_text": timemmd_search_path,
            "report_text": timemmd_report_path,
        }
        for p in sample_paths:
            data_dict[f"sample_{p.name}"] = p

        card = build_dataset_card(
            cfg=cfg,
            raw_paths=data_dict,
            outputs={
                "numerical_columns": numerical.columns.tolist(),
                "numerical": timemmd_num_path,
                "search": timemmd_search_path,
                "report": timemmd_report_path,
                "samples": sample_paths[0] if sample_paths else panel_path,
                "panel": panel_path,
                "calendar": raw_calendar_path,
                "raw_prices": raw_prices_path,
            },
            split_counts=split_counts,
            paths={"config": cfg_path},
            provenance=miss + news_miss,
        )

        metadata_path = paths.metadata / "OilETF_dataset_card.json"
        metadata_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
        pbar.update(1)
        pbar.set_description("daily: write qc")

        qc_path = paths.outputs / "data_qc_report.md"
        qc = write_quality_report(
            cfg,
            numerical,
            sample_paths,
            qc_path,
            raw_calendar_path,
            search_path=timemmd_search_path,
            report_path=timemmd_report_path,
            project_root=paths.root,
        )
        pbar.update(1)
        pbar.set_description("daily: done")
        return {
            "config": str(cfg_path),
            "calendar": str(raw_calendar_path),
            "panel": str(panel_path),
            "numerical": str(timemmd_num_path),
            "samples": [str(x) for x in sample_paths],
            "images": str(paths.images),
            "metadata": str(metadata_path),
            "qc_report": str(qc_path),
            "qc": qc,
        }
    finally:
        pbar.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OilETF-TimeMMD dataset pipeline")
    parser.add_argument("--config", type=str, default="configs/data_config.yaml")
    parser.add_argument("--incremental", action="store_true", help="append new trading dates only for raw data")
    parser.add_argument("--skip-visuals", action="store_true", help="skip candlestick image rendering")
    args = parser.parse_args()
    result = build(Path(args.config), incremental=args.incremental, skip_visuals=args.skip_visuals)
    print("Build completed")
    for k, v in result.items():
        if k == "qc":
            continue
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
