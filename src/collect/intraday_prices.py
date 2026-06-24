from __future__ import annotations

from datetime import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf
from tqdm.auto import tqdm


def _flatten_yfinance(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        if symbol in frame.columns.get_level_values(-1):
            frame = frame.xs(symbol, axis=1, level=-1)
        else:
            frame.columns = frame.columns.get_level_values(0)
    return frame.rename(columns={c: str(c).lower().replace(" ", "_") for c in frame.columns})


def normalize_yfinance_hour_bars(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "bar_start_utc",
                "bar_end_utc",
                "bar_start_et",
                "bar_end_et",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "vwap",
                "trade_count",
                "provider",
                "session",
                "bar_minutes_actual",
            ]
        )

    work = _flatten_yfinance(frame.copy(), symbol)
    work.index = pd.to_datetime(work.index)
    if work.index.tz is None:
        work.index = work.index.tz_localize("US/Eastern")
    work["bar_start_et_ts"] = work.index.tz_convert("US/Eastern")
    tod = work["bar_start_et_ts"].dt.time
    work = work[(tod >= time(9, 30)) & (tod < time(16, 0))].copy()
    if work.empty:
        return normalize_yfinance_hour_bars(pd.DataFrame(), symbol)

    work["bar_end_et_ts"] = work["bar_start_et_ts"] + pd.Timedelta(hours=1)
    is_last_bar = work["bar_start_et_ts"].dt.time == time(15, 30)
    work.loc[is_last_bar, "bar_end_et_ts"] = work.loc[is_last_bar, "bar_start_et_ts"] + pd.Timedelta(minutes=30)

    rows = pd.DataFrame(
        {
            "symbol": symbol,
            "bar_start_utc": work["bar_start_et_ts"].dt.tz_convert("UTC").map(lambda x: x.isoformat()),
            "bar_end_utc": work["bar_end_et_ts"].dt.tz_convert("UTC").map(lambda x: x.isoformat()),
            "bar_start_et": work["bar_start_et_ts"].map(lambda x: x.isoformat()),
            "bar_end_et": work["bar_end_et_ts"].map(lambda x: x.isoformat()),
            "open": pd.to_numeric(work["open"], errors="coerce"),
            "high": pd.to_numeric(work["high"], errors="coerce"),
            "low": pd.to_numeric(work["low"], errors="coerce"),
            "close": pd.to_numeric(work["close"], errors="coerce"),
            "volume": pd.to_numeric(work["volume"], errors="coerce").fillna(0),
            "vwap": pd.NA,
            "trade_count": 0,
            "provider": "yfinance",
            "session": "regular",
            "bar_minutes_actual": (work["bar_end_et_ts"] - work["bar_start_et_ts"]).dt.total_seconds().astype(int) // 60,
        }
    )
    return rows.dropna(subset=["open", "high", "low", "close"]).sort_values(["symbol", "bar_start_utc"]).reset_index(drop=True)


def _filter_range(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    idx = pd.to_datetime(frame.index)
    if idx.tz is None:
        idx = idx.tz_localize("US/Eastern")
    start_ts = pd.Timestamp(start, tz="US/Eastern")
    end_ts = pd.Timestamp(end, tz="US/Eastern") + pd.Timedelta(days=1)
    return frame[(idx >= start_ts) & (idx < end_ts)]


def fetch_intraday_hour_bars(symbols: Iterable[str], start: str, end: str, out_path: Path, period: str = "730d") -> pd.DataFrame:
    frames = []
    for symbol in tqdm(list(symbols), desc="yfinance 1h bars", unit="symbol"):
        raw = yf.download(
            symbol,
            period=period,
            interval="1h",
            auto_adjust=False,
            prepost=False,
            progress=False,
            threads=False,
        )
        frames.append(normalize_yfinance_hour_bars(_filter_range(raw, start, end), symbol))
    out = pd.concat(frames, ignore_index=True) if frames else normalize_yfinance_hour_bars(pd.DataFrame(), "")
    if out.empty:
        raise RuntimeError(f"No yfinance 1h data returned for {list(symbols)} in {start}..{end}. Try a shorter lookback.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return out
