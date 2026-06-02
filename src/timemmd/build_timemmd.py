from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


NUMERIC_FIELDS = [
    "OT",
    "uso_open",
    "uso_high",
    "uso_low",
    "uso_close",
    "uso_adj_close",
    "uso_volume",
    "uso_ret_1d",
    "uso_ret_5d",
    "uso_ret_20d",
    "bno_ret_1d",
    "dbo_ret_1d",
    "vol_5d",
    "vol_20d",
    "ma_5",
    "ma_20",
    "ma_60",
    "ema_12",
    "ema_26",
    "rsi_14",
    "macd",
    "macd_signal",
    "boll_z",
    "atr_14",
    "drawdown_20d",
    "wti_ret_1d",
    "brent_ret_1d",
    "brent_wti_spread",
    "xle_ret_1d",
    "spy_ret_1d",
    "vix_change",
    "dxy_change",
    "eia_crude_inv_change",
    "eia_gasoline_inv_change",
    "eia_distillate_inv_change",
    "news_count",
    "news_sent_mean",
    "news_sent_max",
    "oil_event_count",
]


def _ensure_na_fact(x) -> str:
    if pd.isna(x):
        return "NA"
    x = str(x or "").strip()
    return x if x else "NA"


def _num(row: pd.Series, col: str):
    value = row.get(col)
    return value if pd.notna(value) else None


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.{digits}f}"


def _build_market_text(daily_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in daily_panel.iterrows():
        date = str(row["end_date"])
        facts = []
        preds = []

        uso_close = _num(row, "uso_close")
        ot = _num(row, "OT")
        if uso_close is not None:
            facts.append(f"USO closed at {float(uso_close):.2f} with one-day log return {_fmt(ot)}.")

        wti_ret = _num(row, "wti_ret_1d")
        brent_ret = _num(row, "brent_ret_1d")
        spread = _num(row, "brent_wti_spread")
        macro_bits = []
        if wti_ret is not None:
            macro_bits.append(f"WTI log return {_fmt(wti_ret)}")
        if brent_ret is not None:
            macro_bits.append(f"Brent log return {_fmt(brent_ret)}")
        if spread is not None:
            macro_bits.append(f"Brent-WTI spread {float(spread):.2f}")
        if macro_bits:
            facts.append("; ".join(macro_bits) + ".")

        market_bits = []
        for col, label in [
            ("xle_ret_1d", "XLE"),
            ("spy_ret_1d", "SPY"),
            ("vix_change", "VIX change"),
            ("dxy_change", "DXY change"),
        ]:
            value = _num(row, col)
            if value is not None:
                market_bits.append(f"{label} {_fmt(value)}")
        if market_bits:
            facts.append("; ".join(market_bits) + ".")

        inv_bits = []
        for col, label in [
            ("eia_crude_inv_change", "crude"),
            ("eia_gasoline_inv_change", "gasoline"),
            ("eia_distillate_inv_change", "distillate"),
        ]:
            value = _num(row, col)
            if value is not None and abs(float(value)) > 1e-9:
                inv_bits.append(f"{label} inventory change {float(value):+.0f} thousand barrels")
        if inv_bits:
            facts.append("EIA weekly petroleum update: " + "; ".join(inv_bits) + ".")

        rsi = _num(row, "rsi_14")
        ma_5 = _num(row, "ma_5")
        ma_20 = _num(row, "ma_20")
        vol_20d = _num(row, "vol_20d")
        if rsi is not None and float(rsi) >= 70:
            preds.append("Technical state indicated overbought momentum and elevated pullback risk.")
        elif rsi is not None and float(rsi) <= 30:
            preds.append("Technical state indicated oversold momentum and elevated rebound risk.")
        elif ma_5 is not None and ma_20 is not None:
            if float(ma_5) > float(ma_20):
                preds.append("Short-term trend was above the medium-term average, implying positive momentum risk.")
            elif float(ma_5) < float(ma_20):
                preds.append("Short-term trend was below the medium-term average, implying negative momentum risk.")
        if vol_20d is not None and float(vol_20d) > 0.04:
            preds.append("Realized volatility was high, implying wider near-term price uncertainty.")

        rows.append(
            {
                "start_date": date,
                "end_date": date,
                "fact": " ".join(facts) if facts else "NA",
                "preds": " ".join(preds) if preds else "NA",
            }
        )
    return pd.DataFrame(rows)


def _combine_text(news_value: str, derived_value: str) -> str:
    news_value = _ensure_na_fact(news_value)
    derived_value = _ensure_na_fact(derived_value)
    if news_value == "NA":
        return derived_value
    if derived_value == "NA":
        return news_value
    return f"{news_value} | {derived_value}"


def _build_inventory_report_text(daily_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in daily_panel.iterrows():
        bits = []
        for col, label in [
            ("eia_crude_inv_change", "crude"),
            ("eia_gasoline_inv_change", "gasoline"),
            ("eia_distillate_inv_change", "distillate"),
        ]:
            value = _num(row, col)
            if value is not None and abs(float(value)) > 1e-9:
                bits.append(f"{label} inventories changed {float(value):+.0f} thousand barrels")
        if not bits:
            continue
        date = str(row["end_date"])
        rows.append(
            {
                "start_date": date,
                "end_date": date,
                "fact": "EIA weekly petroleum stock release: " + "; ".join(bits) + ".",
                "preds": "NA",
            }
        )
    return pd.DataFrame(rows, columns=["start_date", "end_date", "fact", "preds"])


def build_numerical_df(daily_panel: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    base = daily_panel.copy()
    cols = ["start_date", "end_date"] + [c for c in NUMERIC_FIELDS if c in base.columns]
    out = base[cols].copy()
    out = out.dropna(subset=["end_date"]).sort_values(["end_date"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8")
    return out


def _map_news_to_effective_date(raw_news: pd.DataFrame, trading_dates: pd.DatetimeIndex, cutoff_hour: int = 16) -> pd.Series:
    if raw_news.empty:
        return pd.Series(dtype="datetime64[ns]")
    trading_index = pd.DatetimeIndex(pd.to_datetime(trading_dates)).normalize()
    trading = set(trading_index.astype(str).tolist())
    max_trading_date = trading_index.max()

    def map_to_day(ts):
        dt = pd.to_datetime(ts)
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        else:
            dt = dt.tz_convert("UTC")
        dt = dt.tz_convert("US/Eastern")
        dt = dt.tz_localize(None)
        d = dt.date()
        t = dt.hour
        if t > cutoff_hour:
            d = (pd.Timestamp(d) + pd.Timedelta(days=1)).date()
        if str(d) not in trading:
            probe = pd.Timestamp(d)
            while str(probe.date()) not in trading:
                probe += pd.Timedelta(days=1)
                if probe > max_trading_date:
                    return pd.NaT
            d = probe.date()
        return pd.Timestamp(d)

    return raw_news["published_at_utc"].map(map_to_day)


def build_search_text(
    daily_panel: pd.DataFrame,
    raw_news: pd.DataFrame,
    out_path: Path,
    trading_dates: pd.DatetimeIndex,
    cutoff_hour: int = 16,
) -> pd.DataFrame:
    derived = _build_market_text(daily_panel)
    news = raw_news.copy()
    if "published_at_utc" not in news.columns:
        news = pd.DataFrame(columns=["published_at_utc", "summary"])
        news["summary"] = []
    if news.empty:
        agg = derived.copy()
    else:
        news["effective_date"] = _map_news_to_effective_date(news, trading_dates, cutoff_hour).dt.strftime("%Y-%m-%d")
        news = news.sort_values("published_at_utc")
        agg = news.groupby("effective_date").agg(
            fact=(
                "summary",
                lambda x: " | ".join([_ensure_na_fact(str(v)) for v in x.fillna("").tolist()[:10]]),
            ),
            preds=(
                "summary",
                lambda x: " | ".join(
                    [
                        _ensure_na_fact(str(v))
                        for v in x.fillna("").tolist()[:5]
                        if any(k in str(v).lower() for k in ["expect", "forecast", "likely", "risk", "may", "could"])
                    ]
                ),
            ),
        )
        agg = agg.reset_index().rename(columns={"effective_date": "start_date"})
        agg["end_date"] = agg["start_date"]
        agg = agg[["start_date", "end_date", "fact", "preds"]]
        agg["fact"] = agg["fact"].replace({"": "NA"})
        agg["preds"] = agg["preds"].replace({"": "NA"})
        agg = derived.merge(agg, on=["start_date", "end_date"], how="left", suffixes=("_derived", "_news"))
        agg["fact"] = agg.apply(lambda r: _combine_text(r["fact_news"], r["fact_derived"]), axis=1)
        agg["preds"] = agg.apply(lambda r: _combine_text(r["preds_news"], r["preds_derived"]), axis=1)
        agg = agg[["start_date", "end_date", "fact", "preds"]]

    agg = agg.sort_values("start_date").reset_index(drop=True)
    for col in ["fact", "preds"]:
        agg[col] = agg[col].fillna("NA").astype(str).str.strip().replace({"": "NA"})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_path, index=False, encoding="utf-8")
    return agg


def build_report_text(raw_reports: pd.DataFrame, out_path: Path, provenance: List[str], daily_panel: pd.DataFrame | None = None) -> pd.DataFrame:
    inventory_reports = _build_inventory_report_text(daily_panel) if daily_panel is not None else pd.DataFrame()
    if raw_reports.empty:
        out = inventory_reports if not inventory_reports.empty else pd.DataFrame(columns=["start_date", "end_date", "fact", "preds"])
        out.to_csv(out_path, index=False, encoding="utf-8")
        return out
    rep = raw_reports.copy()
    rep["start_date"] = pd.to_datetime(rep["start_date"]).dt.strftime("%Y-%m-%d")
    rep["end_date"] = pd.to_datetime(rep["end_date"]).dt.strftime("%Y-%m-%d")
    rep = rep[["start_date", "end_date", "fact", "preds"]].copy()
    rep["fact"] = rep["fact"].fillna("NA").astype(str).str.strip().replace({"": "NA"})
    rep["preds"] = rep["preds"].fillna("NA").astype(str).str.strip().replace({"": "NA"})
    if not inventory_reports.empty:
        rep = pd.concat([rep, inventory_reports], ignore_index=True)
    rep = rep.sort_values("start_date").reset_index(drop=True)
    rep.to_csv(out_path, index=False, encoding="utf-8")
    return rep
