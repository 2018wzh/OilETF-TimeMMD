from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _missing_summary(df: pd.DataFrame, limit: int = 80) -> Dict[str, float]:
    if df.empty:
        return {}
    rates = (df.isna().sum() / len(df)).sort_values(ascending=False)
    return {k: round(float(v), 4) for k, v in rates.head(limit).items()}


def _time_alignment_check(calendar: pd.DataFrame, panel: pd.DataFrame) -> bool:
    cal_dates = set(calendar.loc[calendar["is_trading_day"] == 1, "date"].astype(str))
    pnl_dates = set(panel["end_date"].astype(str))
    return pnl_dates.issubset(cal_dates)


def _non_missing_numeric(df: pd.DataFrame, col: str, max_missing_rate: float = 0.98) -> bool:
    if col not in df.columns or df.empty:
        return False
    values = pd.to_numeric(df[col], errors="coerce")
    return bool(values.notna().mean() >= (1 - max_missing_rate))


def _text_has_content(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path, keep_default_na=False)
    except Exception:
        return False
    if df.empty or not {"fact", "preds"}.issubset(df.columns):
        return False
    fact_content = df["fact"].astype(str).str.strip().replace({"NA": ""}).str.len().sum()
    preds_content = df["preds"].astype(str).str.strip().replace({"NA": ""}).str.len().sum()
    return bool(fact_content + preds_content > 0)


def build_dataset_card(
    cfg: Dict[str, Any],
    raw_paths: Dict[str, Any],
    outputs: Dict[str, Any],
    split_counts: Dict[str, int],
    paths: Dict[str, Path],
    provenance: List[str],
) -> Dict[str, object]:
    card = {
        "dataset_name": cfg.get("outputs", {}).get("dataset_name", "OilETF-TimeMMD"),
        "dataset_version": cfg.get("outputs", {}).get("dataset_version", "0.0.0"),
        "build_time_utc": datetime.utcnow().isoformat() + "Z",
        "config_path": str(paths["config"]),
        "timezone": cfg.get("timezone", "US/Eastern"),
        "data_sources": {
            "prices": "yfinance",
            "macro": "FRED + yfinance + EIA weekly petroleum XLS",
            "news": "finnhub.company-news",
            "reports": "public RSS/web pages + derived EIA inventory release facts",
        },
        "field_columns": list(outputs["numerical_columns"]),
        "date_range": {
            "start": str(cfg.get("date_range", {}).get("start")),
            "end": str(cfg.get("date_range", {}).get("end")),
        },
        "alignment_rules": {
            "news_effective_date": {
                "timezone": "US/Eastern",
                "cutoff_hour": int(cfg.get("collection", {}).get("news_cutoff_hour_et", 16)),
            },
            "eia_inventory_effective_date": {
                "source_date": "week ending Friday",
                "release_shift_days": 5,
            }
        },
        "file_hashes": {},
        "missing_rates": {},
        "split_counts": split_counts,
        "provenance": provenance,
        "quality_checks": {},
    }

    candidates = {**raw_paths, **outputs}
    for name, item in candidates.items():
        if not isinstance(item, Path):
            continue
        if not item.exists():
            continue
        card["file_hashes"][name] = _md5_file(item)
        try:
            if item.suffix == ".csv":
                df = pd.read_csv(item)
                card["missing_rates"][name] = _missing_summary(df)
            elif item.suffix == ".parquet":
                df = pd.read_parquet(item)
                card["missing_rates"][name] = _missing_summary(df)
        except Exception:
            card["missing_rates"][name] = {}
    return card


def write_quality_report(
    cfg: Dict[str, Any],
    num_df: pd.DataFrame,
    sample_paths: List[Path],
    qc_path: Path,
    calendar_path: Path,
    search_path: Path | None = None,
    report_path: Path | None = None,
    project_root: Path | None = None,
) -> str:
    calendar = pd.read_csv(calendar_path)
    checks = []
    checks.append(("numerical_rows_non_empty", len(num_df) > 0))
    checks.append(("start_date_non_null", bool(num_df["start_date"].notna().all())))
    checks.append(("end_date_non_null", bool(num_df["end_date"].notna().all())))
    checks.append(("dates_chronological", bool(num_df["end_date"].is_monotonic_increasing)))
    checks.append(("time_align", bool(_time_alignment_check(calendar, num_df))))
    checks.append(("market_macro_non_empty", all(_non_missing_numeric(num_df, c) for c in ["spy_ret_1d", "xle_ret_1d", "vix_change"])))
    checks.append(
        (
            "eia_inventory_non_empty",
            all(
                _non_missing_numeric(num_df, c)
                for c in ["eia_crude_inv_change", "eia_gasoline_inv_change", "eia_distillate_inv_change"]
            ),
        )
    )
    if search_path is not None:
        checks.append(("search_text_has_content", _text_has_content(Path(search_path))))
    if report_path is not None:
        checks.append(("report_text_has_content", _text_has_content(Path(report_path))))

    split_ok = True
    split_order_ok = True
    leak_ok = True
    text_ok = True
    paths_ok = True
    root = Path(project_root or ".").resolve()
    for sp in sample_paths:
        try:
            df = pd.read_parquet(sp)
            if not df.empty and not {"train", "val", "test"}.issubset(set(df["split"].unique())):
                split_ok = False
            if not df.empty:
                ordered = df.sort_values("end_date")
                rank = {"train": 0, "val": 1, "test": 2}
                leak_ok = leak_ok and bool((pd.to_datetime(df["y_start"]) > pd.to_datetime(df["x_num_end"])).all())
                text_ok = text_ok and bool((pd.to_datetime(df["text_end"]) <= pd.to_datetime(df["end_date"])).all())
                split_order_ok = split_order_ok and ((ordered["split"].map(rank).diff().fillna(0) >= 0).all())
                for col in ["x_numeric_path", "x_text_embedding_path", "x_image_path"]:
                    if col in df.columns:
                        paths_ok = paths_ok and all((root / str(p)).exists() for p in df[col].dropna().unique())
        except Exception:
            split_ok = False
            split_order_ok = False
            leak_ok = False
            text_ok = False
            paths_ok = False
    checks.append(("sample_split_complete", split_ok))
    checks.append(("sample_split_order", split_order_ok))
    checks.append(("sample_no_future_leak", leak_ok))
    checks.append(("sample_text_alignment", text_ok))
    checks.append(("sample_referenced_paths_exist", paths_ok))
    checks.append(("news_columns_present", "news_count" in num_df.columns and "news_sent_mean" in num_df.columns))

    lines = [
        "# OilETF-TimeMMD Data QC",
        f"build_time_utc: {datetime.utcnow().isoformat()}Z",
        "",
        "## Checks",
    ]
    for name, ok in checks:
        lines.append(f"- {name}: {'PASS' if ok else 'FAIL'}")

    lines.append("")
    lines.append("## Sample split counts")
    for sp in sample_paths:
        try:
            df = pd.read_parquet(sp)
            lines.append(f"- {sp.name}: {df['split'].value_counts().to_dict()}")
        except Exception:
            lines.append(f"- {sp.name}: unavailable")

    qc_text = "\n".join(lines)
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    qc_path.write_text(qc_text, encoding="utf-8")
    return qc_text
