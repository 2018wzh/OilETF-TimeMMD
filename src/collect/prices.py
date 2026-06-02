from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

import yfinance as yf


def _to_price_records(symbol: str, frame: pd.DataFrame) -> List[Dict[str, object]]:
    if frame is None or frame.empty:
        return []

    df = frame.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
    else:
        df = pd.concat({symbol: df}, axis=1)
    if symbol not in df.columns.get_level_values(0):
        return []

    sdf = df[symbol].copy()
    sdf = sdf.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    sdf.index = pd.to_datetime(sdf.index).tz_localize(None)
    sdf = sdf[["open", "high", "low", "close", "adj_close", "volume"]]
    sdf = sdf.dropna(subset=["open", "high", "low", "close", "adj_close"], how="any")
    sdf["symbol"] = symbol
    sdf["timestamp"] = sdf.index.strftime("%Y-%m-%d")
    sdf["provider"] = "yfinance"
    return sdf.reset_index(drop=True).to_dict("records")


def fetch_prices(symbols: Sequence[str], start_date: str, end_date: str) -> pd.DataFrame:
    tickers = list(symbols)
    start = str(start_date)
    end_exclusive = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(
        tickers=" ".join(tickers),
        start=start,
        end=end_exclusive,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
    )

    frames: List[pd.DataFrame] = []
    if isinstance(raw.columns, pd.MultiIndex):
        for symbol in tickers:
            rows = _to_price_records(symbol, raw[[symbol]])
            frames.append(pd.DataFrame.from_records(rows))
    else:
        rows = _to_price_records(tickers[0], raw)
        frames.append(pd.DataFrame.from_records(rows))

    if not frames:
        return pd.DataFrame(
            columns=[
                "symbol",
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
                "provider",
            ]
        )
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"]).dt.strftime("%Y-%m-%d")
    return out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def fetch_fred_series(series_id: str, start_date: str, end_date: str) -> pd.Series:
    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}&cosd={start_date}&coed={end_date}"
    )
    try:
        df = pd.read_csv(url)
    except Exception:
        return pd.Series(dtype="float64")
    if df.empty:
        return pd.Series(dtype="float64")
    s = df.iloc[:, 0:2].copy()
    s.columns = ["date", "value"]
    s["date"] = pd.to_datetime(s["date"])
    s["value"] = pd.to_numeric(s["value"], errors="coerce")
    return s.set_index("date")["value"]


def fetch_eia_weekly_stock_series(series_id: str, start_date: str, end_date: str) -> pd.Series:
    url = f"https://www.eia.gov/dnav/pet/hist_xls/{series_id}w.xls"
    try:
        df = pd.read_excel(url, sheet_name="Data 1", skiprows=2)
    except Exception:
        return pd.Series(dtype="float64")
    if df.empty or "Date" not in df.columns:
        return pd.Series(dtype="float64")

    value_col = next((c for c in df.columns if c != "Date"), None)
    if value_col is None:
        return pd.Series(dtype="float64")

    work = df[["Date", value_col]].copy()
    work.columns = ["period_end", "value"]
    work["period_end"] = pd.to_datetime(work["period_end"], errors="coerce")
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    work = work.dropna(subset=["period_end", "value"])
    if work.empty:
        return pd.Series(dtype="float64")

    # EIA weekly petroleum stock rows are labeled by week-ending Friday.
    # The public release is normally the following Wednesday morning, so
    # shifting by five days avoids making Friday inventory values available
    # before publication.
    work["effective_date"] = work["period_end"] + pd.Timedelta(days=5)
    start = pd.to_datetime(start_date) - pd.Timedelta(days=14)
    end = pd.to_datetime(end_date)
    work = work[(work["effective_date"] >= start) & (work["effective_date"] <= end)]
    return work.sort_values("effective_date").set_index("effective_date")["value"]


def _extract_yfinance_close(frame: pd.DataFrame, symbol: str) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype="float64")

    if isinstance(frame.columns, pd.MultiIndex):
        level0 = frame.columns.get_level_values(0)
        level1 = frame.columns.get_level_values(1)
        if symbol in level0 and "Close" in frame[symbol].columns:
            close = frame[symbol]["Close"]
        elif "Close" in level0 and symbol in level1:
            close = frame["Close"][symbol]
        else:
            return pd.Series(dtype="float64")
    elif "Close" in frame.columns:
        close = frame["Close"]
    else:
        return pd.Series(dtype="float64")

    close = close.copy()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return pd.to_numeric(close, errors="coerce")


def fetch_macro_data(start_date: str, end_date: str, cfg: Dict[str, object]) -> pd.DataFrame:
    fred_cfg: Dict[str, str] = cfg.get("fred", {})
    eia_cfg: Dict[str, str] = cfg.get("eia_inventory", {})
    yf_cfg: List[str] = list(cfg.get("yfinance", []))
    out = pd.DataFrame({"date": pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date), freq="D")})

    wti = fetch_fred_series(fred_cfg["wti"], start_date, end_date)
    brent = fetch_fred_series(fred_cfg["brent"], start_date, end_date)
    dxy = fetch_fred_series(fred_cfg["dxy"], start_date, end_date)
    dxy = dxy.rename("dxy_close")
    out["wti_close"] = out["date"].map(wti)
    out["brent_close"] = out["date"].map(brent)
    out["dxy"] = out["date"].map(dxy)
    out["brent_wti_spread"] = out["brent_close"] - out["wti_close"]

    for name, series_id in eia_cfg.items():
        stock = fetch_eia_weekly_stock_series(str(series_id), start_date, end_date)
        col = f"eia_{name}_inv"
        if stock.empty:
            out[col] = np.nan
            continue
        out[col] = out["date"].map(stock).ffill()

    if yf_cfg:
        frame = yf.download(
            tickers=" ".join(yf_cfg),
            start=start_date,
            end=(pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            group_by="ticker",
            auto_adjust=False,
            progress=False,
        )
        if not frame.empty:
            for sym in yf_cfg:
                close = _extract_yfinance_close(frame, sym)
                if close.empty:
                    continue
                out[sym.lower().replace("^", "") + "_close"] = out["date"].map(close)

    out = out.rename(
        columns={
            "vix_close": "vix",
        }
    )
    out["dxy_close"] = out["dxy"]
    out = out.drop(columns=["dxy"])
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out.sort_values("date").reset_index(drop=True)


def write_raw_csv(df: pd.DataFrame, path: Path, provenance: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    meta_path = path.with_suffix(".meta.json")
    pd.Series(provenance).to_json(meta_path)
