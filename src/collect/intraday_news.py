from __future__ import annotations

from hashlib import md5
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from src.collect.news import fetch_news, fetch_news_with_cache


EVENT_COLUMNS = [
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

EVENT_TYPE_RULES: list[tuple[str, list[str]]] = [
    ("opec", ["opec", "opec+"]),
    ("inventory", ["inventory", "inventories", "stockpile", "eia", "crude oil inventory"]),
    ("geo", ["iran", "russia", "saudi", "middle east", "libya", "iraq", "hurricane", "sanction", "conflict", "war"]),
    ("macro", ["inflation", "cpi", "gdp", "fed", "dxy", "dollar", "rate", "yield", "jobs", "employment", "ppi"]),
    ("supply_demand", ["supply", "demand", "production", "output", "export", "import", "refinery", "rig", "crack spread"]),
    ("price", ["oil price", "opec+ supply", "crude oil", "brent", "wti", "oil futures", "energy futures", "crude futures"]),
]


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=EVENT_COLUMNS)


def _series_or_default(frame: pd.DataFrame, name: str, default: object = "") -> pd.Series:
    if name in frame.columns:
        return frame[name]
    return pd.Series([default] * len(frame), index=frame.index)


def _normalize_affected_symbols(row: pd.Series, symbols: Sequence[str]) -> str:
    raw = str(row.get("tickers") or "").strip()
    if raw:
        return raw
    target = str(row.get("target_symbol") or "").strip()
    if target:
        return target
    return ";".join(symbols)


def _classify_event_type(text: str) -> str:
    lower = str(text or "").lower()
    for event_type, keys in EVENT_TYPE_RULES:
        if any(key in lower for key in keys):
            return event_type
    return "news"


def _to_events(news: pd.DataFrame, symbols: Sequence[str]) -> pd.DataFrame:
    if news.empty:
        return _empty_events()

    event_text = (
        _series_or_default(news, "title").fillna("").astype(str)
        + " "
        + _series_or_default(news, "summary").fillna("").astype(str)
    )

    events = pd.DataFrame(
        {
            "event_id": _series_or_default(news, "article_id").astype(str),
            "event_type": event_text.map(_classify_event_type),
            "source": _series_or_default(news, "source", "unknown").fillna("unknown").astype(str),
            "title": _series_or_default(news, "title").fillna("").astype(str),
            "summary": _series_or_default(news, "summary").fillna("").astype(str),
            "published_at_utc": pd.to_datetime(_series_or_default(news, "published_at_utc"), utc=True, errors="coerce").map(
                lambda x: x.isoformat() if pd.notna(x) else pd.NA
            ),
            "available_at_utc": pd.to_datetime(_series_or_default(news, "published_at_utc"), utc=True, errors="coerce").map(
                lambda x: x.isoformat() if pd.notna(x) else pd.NA
            ),
            "affected_symbols": news.apply(lambda row: _normalize_affected_symbols(row, symbols), axis=1),
            "sentiment": pd.to_numeric(_series_or_default(news, "sentiment", 0.0), errors="coerce").fillna(0.0),
            "relevance": pd.to_numeric(_series_or_default(news, "relevance", 1.0), errors="coerce").fillna(1.0),
            "provider": _series_or_default(news, "provider", "news").fillna("news").astype(str),
            "url_hash": _series_or_default(news, "url_hash").fillna("").astype(str),
        }
    )
    events = events[EVENT_COLUMNS].copy()
    events = events.dropna(subset=["published_at_utc"]).drop_duplicates(subset=["event_id"])
    events["published_at_utc"] = pd.to_datetime(events["published_at_utc"], utc=True)
    events["available_at_utc"] = pd.to_datetime(events["available_at_utc"], utc=True)
    events = events.sort_values(["published_at_utc", "event_id"]).reset_index(drop=True)
    return events


def build_intraday_events(
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
    out_path: Path,
    collection_cfg: Dict[str, object] | None = None,
    per_symbol_limit: int = 200,
    news_cache_path: Path | None = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Build intraday event parquet from news sources.

    Uses incremental caching via *news_cache_path* to avoid re-fetching
    already-collected articles across runs.
    """
    if news_cache_path is not None:
        news, miss = fetch_news_with_cache(
            symbols,
            start_date,
            end_date,
            cache_path=news_cache_path,
            per_symbol_limit=per_symbol_limit,
            collection_cfg=collection_cfg,
        )
    else:
        news, miss = fetch_news(
            symbols,
            start_date,
            end_date,
            per_symbol_limit=per_symbol_limit,
            collection_cfg=collection_cfg,
        )
    events = _to_events(news, symbols)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(out_path, index=False)
    return events, miss
