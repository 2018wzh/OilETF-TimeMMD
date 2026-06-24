from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


KEYWORDS = ["crude oil", "wti", "brent", "opec", "eia", "inventory", "supply", "demand"]


def _log_return(series: pd.Series, periods: int = 1) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    prev = values.shift(periods)
    return np.log(values / prev).where((values > 0) & (prev > 0))


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _event_count(text: str) -> int:
    lower = str(text or "").lower()
    return sum(1 for kw in KEYWORDS if kw in lower)


def _window_events(events: pd.DataFrame, end_time: pd.Timestamp, hours: int) -> pd.DataFrame:
    if events.empty:
        return events
    start_time = end_time - pd.Timedelta(hours=hours)
    available = pd.to_datetime(events["available_at_utc"], utc=True, errors="coerce")
    return events[(available <= end_time) & (available > start_time)]


def build_intraday_panel(bars_path: Path, events_path: Path, out_path: Path | None = None) -> pd.DataFrame:
    bars = pd.read_parquet(bars_path).copy()
    events = pd.read_parquet(events_path).copy() if events_path.exists() else pd.DataFrame()
    if bars.empty:
        panel = bars
    else:
        bars["bar_end_utc_dt"] = pd.to_datetime(bars["bar_end_utc"], utc=True)
        bars = bars.sort_values(["symbol", "bar_end_utc_dt"]).reset_index(drop=True)
        panel = bars.copy()
        panel["OT"] = panel.groupby("symbol")["close"].transform(_log_return)
        panel["uso_open"] = panel["open"]
        panel["uso_high"] = panel["high"]
        panel["uso_low"] = panel["low"]
        panel["uso_close"] = panel["close"]
        panel["uso_volume"] = panel["volume"]
        panel["uso_ret_1h"] = panel.groupby("symbol")["close"].transform(_log_return)
        panel["uso_ret_7h"] = panel.groupby("symbol")["close"].transform(lambda x: _log_return(x, 7))
        panel["vol_7h"] = panel.groupby("symbol")["OT"].transform(lambda x: x.rolling(7).std())
        panel["vol_20h"] = panel.groupby("symbol")["OT"].transform(lambda x: x.rolling(20).std())
        panel["ma_7h"] = panel.groupby("symbol")["close"].transform(lambda x: x.rolling(7).mean())
        panel["ma_20h"] = panel.groupby("symbol")["close"].transform(lambda x: x.rolling(20).mean())
        panel["rsi_14h"] = panel.groupby("symbol")["close"].transform(lambda x: _rsi(x, 14))

        if not events.empty:
            events["sentiment"] = pd.to_numeric(events.get("sentiment", 0.0), errors="coerce").fillna(0.0)
            text = events.get("title", "").fillna("").astype(str) + " " + events.get("summary", "").fillna("").astype(str)
            events["oil_event_count"] = text.map(_event_count)

        rows = []
        for end_time in panel["bar_end_utc_dt"]:
            ev1 = _window_events(events, end_time, 1)
            ev6 = _window_events(events, end_time, 6)
            rows.append(
                {
                    "news_count_1h": len(ev1),
                    "news_count_6h": len(ev6),
                    "news_sent_mean_6h": float(ev6["sentiment"].mean()) if len(ev6) else 0.0,
                    "oil_event_count_6h": float(ev6["oil_event_count"].sum()) if len(ev6) else 0.0,
                    "news_agg_6h": " | ".join((ev6.get("summary", pd.Series(dtype=str)).fillna("").astype(str).head(5)).tolist()),
                }
            )
        panel = pd.concat([panel, pd.DataFrame(rows)], axis=1)
        panel["start_date"] = pd.to_datetime(panel["bar_start_utc"], utc=True).map(lambda x: x.isoformat())
        panel["end_date"] = panel["bar_end_utc_dt"].map(lambda x: x.isoformat())
        panel = panel.drop(columns=["bar_end_utc_dt"])

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(out_path, index=False)
    return panel
