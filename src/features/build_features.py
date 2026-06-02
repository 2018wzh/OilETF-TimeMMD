from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd


KEYWORDS = [
    "crude oil",
    "WTI",
    "Brent",
    "OPEC",
    "EIA",
    "API",
    "inventory",
    "supply",
    "demand",
    "sanction",
    "hurricane",
    "russia",
    "saudi",
    "iran",
    "production",
]


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _log_return(series: pd.Series, periods: int = 1) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    prev = values.shift(periods)
    valid = (values > 0) & (prev > 0)
    ratio = (values / prev).where(valid)
    return np.log(ratio)


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _align_news_to_trading(news: pd.DataFrame, calendar: pd.DataFrame, cutoff_hour: int = 16) -> pd.Series:
    if news.empty:
        return pd.Series(dtype="datetime64[ns]")
    trading_dates = set(calendar.loc[calendar["is_trading_day"].astype(bool), "date"].astype(str).tolist())
    max_trading_date = pd.to_datetime(calendar.loc[calendar["is_trading_day"].astype(bool), "date"]).max()

    def map_to_day(ts):
        dt = pd.to_datetime(ts)
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        else:
            dt = dt.tz_convert("UTC")
        dt = dt.tz_convert("US/Eastern")
        dt = dt.tz_localize(None)
        publish_date = dt.date()
        publish_hour = dt.hour
        if publish_hour > cutoff_hour:
            candidate = (pd.Timestamp(publish_date) + pd.Timedelta(days=1)).date()
        else:
            candidate = publish_date
        if str(candidate) in trading_dates:
            return pd.Timestamp(str(candidate)).to_pydatetime().date()
        probe = pd.Timestamp(candidate)
        while str(probe.date()) not in trading_dates:
            probe = probe + pd.Timedelta(days=1)
            if probe > max_trading_date:
                return pd.NaT
        return probe.date()

    mapped = news["published_at_utc"].map(lambda x: map_to_day(x))
    out = pd.Series(mapped, index=news.index, name="effective_date").astype(str)
    return out


def _keyword_count(text: str) -> int:
    lower = (text or "").lower()
    return sum(1 for kw in KEYWORDS if kw.lower() in lower)


def _build_price_features(raw_prices: pd.DataFrame, symbols: Sequence[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.DataFrame(index=dates)
    for symbol in symbols:
        sub = raw_prices[raw_prices["symbol"] == symbol].copy()
        if sub.empty:
            prefix = symbol.lower()
            for suffix in ["open", "high", "low", "close", "adj_close", "volume"]:
                frame[f"{prefix}_{suffix}"] = np.nan
            continue
        sub["timestamp"] = pd.to_datetime(sub["timestamp"]).dt.tz_localize(None)
        sub = sub.set_index("timestamp").sort_index()
        prefix = symbol.lower()
        for col in ["open", "high", "low", "close", "adj_close", "volume"]:
            frame[f"{prefix}_{col}"] = sub[col].reindex(dates)
    return frame


def _add_technical_features(panel: pd.DataFrame, cfg: Dict[str, object]) -> pd.DataFrame:
    tech = cfg.get("technical", {})
    rsi_window = int(tech.get("rsi_window", 14))
    macd_signal = int(tech.get("macd_signal", 9))

    close = panel["uso_adj_close"]
    panel["OT"] = _log_return(close)

    for d in [1, 5, 20]:
        panel[f"uso_ret_{d}d"] = _log_return(close, d)

    panel["vol_5d"] = panel["OT"].rolling(5).std()
    panel["vol_20d"] = panel["OT"].rolling(20).std()

    for d in [5, 20, 60]:
        panel[f"ma_{d}"] = close.rolling(d).mean()
    panel["ema_12"] = _ema(close, 12)
    panel["ema_26"] = _ema(close, 26)

    panel["rsi_14"] = _rsi(close, rsi_window)
    macd_fast = _ema(close, 12)
    macd_slow = _ema(close, 26)
    panel["macd"] = macd_fast - macd_slow
    panel["macd_signal"] = _ema(panel["macd"], macd_signal)

    m20 = close.rolling(20).mean()
    s20 = close.rolling(20).std()
    panel["boll_z"] = (close - m20) / s20

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            panel["uso_high"] - panel["uso_low"],
            (panel["uso_high"] - prev_close).abs(),
            (panel["uso_low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    panel["atr_14"] = tr.rolling(14).mean()
    panel["drawdown_20d"] = close / close.rolling(20).max() - 1

    return panel


def _add_macro_features(panel: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    if macro.empty:
        for col in ["wti_ret_1d", "brent_ret_1d", "brent_wti_spread", "dxy_change", "vix_change", "xle_ret_1d", "spy_ret_1d"]:
            panel[col] = np.nan
        panel["eia_crude_inv_change"] = np.nan
        panel["eia_gasoline_inv_change"] = np.nan
        panel["eia_distillate_inv_change"] = np.nan
        return panel
    m = macro.copy()
    m["date"] = pd.to_datetime(m["date"]).dt.tz_localize(None)
    m = m.sort_values("date").set_index("date")
    m = m.reindex(panel.index)

    panel["wti_ret_1d"] = _log_return(m["wti_close"]) if "wti_close" in m.columns else np.nan
    panel["brent_ret_1d"] = _log_return(m["brent_close"]) if "brent_close" in m.columns else np.nan
    panel["brent_wti_spread"] = m["brent_wti_spread"] if "brent_wti_spread" in m.columns else np.nan
    panel["dxy_change"] = m["dxy_close"].pct_change() if "dxy_close" in m.columns else np.nan
    if "vix" in m.columns:
        panel["vix_change"] = m["vix"].pct_change()
    else:
        panel["vix_change"] = np.nan
    if "spy_close" in m.columns:
        panel["spy_ret_1d"] = _log_return(m["spy_close"])
    else:
        panel["spy_ret_1d"] = np.nan
    if "xle_close" in m.columns:
        panel["xle_ret_1d"] = _log_return(m["xle_close"])
    else:
        panel["xle_ret_1d"] = np.nan

    for name in ["crude", "gasoline", "distillate"]:
        level_col = f"eia_{name}_inv"
        change_col = f"eia_{name}_inv_change"
        if change_col in m.columns:
            panel[change_col] = m[change_col]
        elif level_col in m.columns:
            panel[change_col] = m[level_col].diff()
        else:
            panel[change_col] = np.nan
    return panel


def _add_news_features(panel: pd.DataFrame, news: pd.DataFrame, calendar: pd.DataFrame, cfg: Dict[str, object]) -> pd.DataFrame:
    if news.empty:
        panel["news_count"] = 0.0
        panel["news_sent_mean"] = 0.0
        panel["news_sent_max"] = 0.0
        panel["oil_event_count"] = 0.0
        return panel

    work = news.copy()
    work["effective_date"] = _align_news_to_trading(work, calendar, cfg.get("collection", {}).get("news_cutoff_hour_et", 16))
    work = work[work["effective_date"].notna() & (work["effective_date"].astype(str) != "NaT")]
    if work.empty:
        panel["news_count"] = 0.0
        panel["news_sent_mean"] = 0.0
        panel["news_sent_max"] = 0.0
        panel["oil_event_count"] = 0.0
        return panel
    work["news_day"] = pd.to_datetime(work["effective_date"])
    work["sentiment"] = pd.to_numeric(work["sentiment"], errors="coerce").fillna(0.0)
    text = (work["title"].fillna("").astype(str) + " " + work["summary"].fillna("").astype(str)).str.lower()
    work["event_count"] = text.map(_keyword_count)
    grp = work.groupby("news_day").agg(
        news_count=("article_id", "count"),
        news_sent_mean=("sentiment", "mean"),
        news_sent_max=("sentiment", "max"),
        oil_event_count=("event_count", "sum"),
    )
    work_agg = grp.reindex(panel.index, fill_value=0)
    panel["news_count"] = work_agg["news_count"].astype(float)
    panel["news_sent_mean"] = work_agg["news_sent_mean"].astype(float).fillna(0.0)
    panel["news_sent_max"] = work_agg["news_sent_max"].astype(float).fillna(0.0)
    panel["oil_event_count"] = work_agg["oil_event_count"].astype(float).fillna(0.0)
    panel["news_agg"] = work.groupby("news_day")["summary"].apply(lambda x: " | ".join([y[:120] for y in x.fillna("").astype(str)])).reindex(
        panel.index, fill_value=""
    )
    return panel


def build_daily_panel(
    raw_prices_path: Path,
    raw_macro_path: Path,
    raw_news_path: Path,
    calendar_path: Path,
    symbols: Sequence[str],
    cfg: Dict[str, object],
) -> pd.DataFrame:
    raw_prices = pd.read_csv(raw_prices_path)
    raw_prices["timestamp"] = pd.to_datetime(raw_prices["timestamp"])
    calendar = pd.read_csv(calendar_path)
    calendar["date"] = pd.to_datetime(calendar["date"])
    trading_dates = pd.to_datetime(calendar.loc[calendar["is_trading_day"].astype(bool), "date"])
    dates = pd.DatetimeIndex(trading_dates)

    panel = _build_price_features(raw_prices, symbols, dates)
    panel = _add_technical_features(panel, cfg)

    macro = pd.read_csv(raw_macro_path)
    panel = _add_macro_features(panel, macro)

    news = pd.read_csv(raw_news_path)
    panel = _add_news_features(panel, news, calendar, cfg)
    if "news_agg" not in panel.columns:
        panel["news_agg"] = ""

    # cross-asset derived returns
    if "bno_adj_close" in panel.columns:
        panel["bno_ret_1d"] = _log_return(panel["bno_adj_close"])
    else:
        panel["bno_ret_1d"] = np.nan

    if "dbo_adj_close" in panel.columns:
        panel["dbo_ret_1d"] = _log_return(panel["dbo_adj_close"])
    else:
        panel["dbo_ret_1d"] = np.nan

    if "uso_ret_20d" not in panel.columns:
        panel["uso_ret_20d"] = _log_return(panel["uso_adj_close"], 20)

    if "news_agg" in panel.columns:
        panel["news_agg"] = panel["news_agg"].fillna("")
    panel["end_date"] = pd.to_datetime(panel.index)
    panel["start_date"] = panel["end_date"].dt.strftime("%Y-%m-%d")
    panel["end_date"] = panel["end_date"].dt.strftime("%Y-%m-%d")
    panel = panel.reset_index(drop=True)
    return panel


def write_panel(panel: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(path, index=False)
