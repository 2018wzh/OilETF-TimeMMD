from __future__ import annotations

import pandas as pd
import yfinance as yf


def build_trading_calendar(start_date: str, end_date: str, symbol: str = "SPY") -> pd.DataFrame:
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    full_range = pd.date_range(start=start, end=end, freq="D")

    traded = yf.download(
        tickers=symbol,
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )
    trading_dates = pd.DatetimeIndex(pd.to_datetime(traded.index).normalize().dropna().unique()).sort_values()
    if len(trading_dates) == 0:
        trading_dates = pd.date_range(start=start, end=end, freq="B").normalize()

    trading_dates = trading_dates.normalize()
    trading_index = pd.DatetimeIndex(trading_dates)
    trading_set = set(trading_dates.date.tolist())
    is_trading = pd.Series(full_range.normalize()).isin(trading_index).astype(int).to_numpy()

    prev_trading = []
    next_trading = []
    for dt in full_range:
        cursor = dt.normalize()
        prev_candidates = trading_index[trading_index <= cursor]
        if len(prev_candidates) == 0:
            prev_candidate = cursor
        else:
            prev_candidate = prev_candidates.max()

        next_candidates = trading_index[trading_index >= cursor]
        if len(next_candidates) == 0:
            next_candidate = cursor
        else:
            next_candidate = next_candidates.min()

        prev_trading.append(prev_candidate.strftime("%Y-%m-%d"))
        next_trading.append(next_candidate.strftime("%Y-%m-%d"))

    return pd.DataFrame(
        {
            "date": full_range.strftime("%Y-%m-%d"),
            "is_trading_day": is_trading,
            "previous_trading_day": prev_trading,
            "next_trading_day": next_trading,
            "market_close_time_et": ["16:00:00"] * len(full_range),
        }
    ).astype({"is_trading_day": "int32"})
