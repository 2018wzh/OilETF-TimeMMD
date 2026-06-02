from __future__ import annotations

from hashlib import md5
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import requests
import yfinance as yf


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return 0.0


def _as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(x) for x in value)
    return str(value)


def _nested(item: Dict[str, object], *keys, default=None):
    value = item
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return value if value is not None else default


def _extract_news_fields(item: Dict[str, object]) -> Dict[str, object]:
    content = item.get("content") if isinstance(item.get("content"), dict) else {}
    ts = item.get("providerPublishTime") or content.get("pubDate") or content.get("displayTime")
    title = item.get("title") or content.get("title") or ""
    summary = (
        item.get("summary")
        or item.get("summary_text")
        or content.get("summary")
        or content.get("description")
        or item.get("provider")
        or ""
    )
    source = item.get("publisher") or _nested(content, "provider", "displayName", default="yfinance")
    url = item.get("link") or item.get("url") or _nested(content, "canonicalUrl", "url", default="") or _nested(
        content, "clickThroughUrl", "url", default=""
    )
    ticker_list = item.get("relatedTickers") or item.get("tickers") or _nested(content, "finance", "stockTickers", default=[])
    return {"ts": ts, "title": title, "summary": summary, "source": source, "url": url, "ticker_list": ticker_list}


def _fetch_yfinance_news(
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
    per_symbol_limit: int = 200,
) -> Tuple[pd.DataFrame, List[str]]:
    rows: List[Dict[str, object]] = []
    miss: List[str] = []
    s_date = pd.Timestamp(start_date)
    e_date = pd.Timestamp(end_date)

    for symbol in symbols:
        try:
            payload = yf.Ticker(symbol).news
        except Exception:
            payload = []
            miss.append(f"{symbol} yfinance.news exception")
        if not isinstance(payload, list):
            miss.append(f"{symbol} yfinance.news payload empty")
            continue

        for item in payload[:per_symbol_limit]:
            if not isinstance(item, dict):
                continue
            fields = _extract_news_fields(item)
            ts = fields["ts"]
            if ts is None:
                continue
            try:
                if isinstance(ts, (int, float)) or str(ts).isdigit():
                    published = pd.to_datetime(ts, unit="s", utc=True)
                else:
                    published = pd.to_datetime(ts, utc=True)
            except Exception:
                try:
                    published = pd.to_datetime(ts, utc=True)
                except Exception:
                    continue
            local_date = published.tz_convert("US/Eastern").date()
            if local_date < s_date.date() or local_date > e_date.date():
                continue
            title = fields["title"] or ""
            summary = fields["summary"] or ""
            if not title and not summary:
                continue
            source = fields["source"] or "yfinance"
            if isinstance(source, dict):
                source = source.get("name", "yfinance")
            source = str(source)
            url = fields["url"] or ""
            ticker_list = fields["ticker_list"] or []
            if isinstance(ticker_list, str):
                tickers = ticker_list
            else:
                tickers = ";".join(map(str, ticker_list))
            text = f"{title} {summary}".strip()
            url_hash = md5(text.encode("utf-8")).hexdigest()
            sentiment = _safe_float(item.get("tone")) if isinstance(item.get("tone"), (int, float, str)) else _safe_float(
                item.get("sentiment", 0.0)
            )
            relevance = _safe_float(item.get("score", 1.0))
            entities = []
            for ent in item.get("entities", []):
                if isinstance(ent, dict):
                    entities.append(ent.get("name", ""))
                else:
                    entities.append(str(ent))
            if not entities:
                entities = []
            rows.append(
                {
                    "article_id": md5((url + str(ts)).encode("utf-8")).hexdigest(),
                    "published_at_utc": published.isoformat(),
                    "source": source,
                    "title": title,
                    "summary": summary,
                    "url_hash": url_hash,
                    "tickers": tickers,
                    "entities": ";".join(filter(None, entities)),
                    "sentiment": sentiment,
                    "relevance": relevance,
                    "provider": "yfinance.news",
                    "target_symbol": symbol,
                }
            )
    if not rows:
        miss.append("no usable news items collected")
        return (
            pd.DataFrame(
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
            ),
            miss,
        )
    df = pd.DataFrame.from_records(rows).drop_duplicates(subset=["article_id"]).sort_values("published_at_utc")
    return df.reset_index(drop=True), miss


def _date_chunks(start_date: str, end_date: str, months: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
    months = max(int(months), 1)
    chunks: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + pd.DateOffset(months=months) - pd.Timedelta(seconds=1), end)
        chunks.append((cursor, chunk_end))
        cursor = (chunk_end + pd.Timedelta(seconds=1)).normalize()
    return chunks


def _gdelt_dt(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y%m%d%H%M%S")


def _parse_gdelt_article(article: Dict[str, object], symbols: Sequence[str]) -> Dict[str, object] | None:
    title = str(article.get("title") or "").strip()
    url = str(article.get("url") or "").strip()
    seen = str(article.get("seendate") or "").strip()
    if not title or not seen:
        return None
    published = pd.to_datetime(seen, utc=True, errors="coerce")
    if pd.isna(published):
        return None

    domain = str(article.get("domain") or article.get("source") or "gdelt").strip()
    language = str(article.get("language") or "").strip()
    source_country = str(article.get("sourcecountry") or "").strip()
    article_key = md5(("gdelt:" + url + seen + title).encode("utf-8")).hexdigest()
    text_hash = md5((title + url).encode("utf-8")).hexdigest()
    return {
        "article_id": article_key,
        "published_at_utc": published.isoformat(),
        "source": domain,
        "title": title,
        "summary": title,
        "url_hash": text_hash,
        "tickers": ";".join(symbols),
        "entities": ";".join(x for x in [language, source_country] if x),
        "sentiment": 0.0,
        "relevance": 1.0,
        "provider": "gdelt.doc",
        "target_symbol": "OIL",
    }


def _fetch_gdelt_news(
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
    cfg: Dict[str, object],
) -> Tuple[pd.DataFrame, List[str]]:
    rows: List[Dict[str, object]] = []
    miss: List[str] = []
    endpoint = str(cfg.get("endpoint", "http://api.gdeltproject.org/api/v2/doc/doc"))
    query = str(cfg.get("query", '("crude oil" OR WTI OR Brent OR OPEC)'))
    requested_start = pd.Timestamp(start_date)
    if cfg.get("start") is not None:
        requested_start = max(requested_start, pd.Timestamp(cfg["start"]))
    requested_end = pd.Timestamp(end_date)
    if cfg.get("end") is not None:
        requested_end = min(requested_end, pd.Timestamp(cfg["end"]))
    if requested_start > requested_end:
        return (
            pd.DataFrame(
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
            ),
            [f"GDELT skipped because configured range {requested_start.date()}..{requested_end.date()} is empty"],
        )
    interval_months = int(cfg.get("interval_months", 6))
    max_records = min(int(cfg.get("max_records_per_query", 100)), 250)
    sort = str(cfg.get("sort", "HybridRel"))
    language = str(cfg.get("language", "")).strip()
    min_interval = float(cfg.get("min_request_interval_seconds", 5.5))
    retries = int(cfg.get("retry_count", 2))
    rate_limit_backoff = float(cfg.get("rate_limit_backoff_seconds", 30.0))
    timeout = int(cfg.get("timeout_seconds", 30))
    session = requests.Session()
    last_request_at = 0.0

    for chunk_start, chunk_end in _date_chunks(str(requested_start.date()), str(requested_end.date()), interval_months):
        params = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": str(max_records),
            "sort": sort,
            "startdatetime": _gdelt_dt(chunk_start),
            "enddatetime": _gdelt_dt(chunk_end),
        }
        payload = None
        for attempt in range(retries + 1):
            elapsed = time.monotonic() - last_request_at
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            last_request_at = time.monotonic()
            try:
                resp = session.get(
                    endpoint,
                    params=params,
                    timeout=timeout,
                    headers={"User-Agent": "OilETF-TimeMMD/0.1 research dataset builder"},
                )
            except Exception as e:
                if attempt >= retries:
                    miss.append(f"GDELT {chunk_start.date()}..{chunk_end.date()} exception: {type(e).__name__}")
                else:
                    time.sleep(min_interval * (attempt + 1))
                continue
            if resp.status_code == 429 and attempt < retries:
                time.sleep(rate_limit_backoff * (attempt + 1))
                continue
            if resp.status_code != 200:
                miss.append(f"GDELT {chunk_start.date()}..{chunk_end.date()} HTTP {resp.status_code}")
                break
            try:
                payload = resp.json()
                break
            except Exception:
                miss.append(f"GDELT {chunk_start.date()}..{chunk_end.date()} invalid JSON")
                break

        articles = (payload or {}).get("articles", [])
        if not isinstance(articles, list):
            continue
        for article in articles:
            if not isinstance(article, dict):
                continue
            if language and str(article.get("language") or "").strip().lower() != language.lower():
                continue
            row = _parse_gdelt_article(article, symbols)
            if row is not None:
                rows.append(row)

    if not rows:
        miss.append("GDELT collected no usable articles")
        return (
            pd.DataFrame(
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
            ),
            miss,
        )
    out = pd.DataFrame.from_records(rows)
    out = out.drop_duplicates(subset=["article_id"]).sort_values("published_at_utc").reset_index(drop=True)
    return out, miss


def fetch_news(
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
    per_symbol_limit: int = 200,
    collection_cfg: Dict[str, object] | None = None,
) -> Tuple[pd.DataFrame, List[str]]:
    collection_cfg = collection_cfg or {}
    sources = collection_cfg.get("news_sources", ["yfinance"])
    if isinstance(sources, str):
        sources = [sources]
    sources = {str(s).lower() for s in sources}

    frames: List[pd.DataFrame] = []
    miss: List[str] = []

    if "yfinance" in sources:
        y_df, y_miss = _fetch_yfinance_news(symbols, start_date, end_date, per_symbol_limit)
        frames.append(y_df)
        miss.extend(y_miss)

    if "gdelt" in sources:
        g_df, g_miss = _fetch_gdelt_news(symbols, start_date, end_date, collection_cfg.get("gdelt", {}))
        frames.append(g_df)
        miss.extend(g_miss)

    frames = [df for df in frames if df is not None and not df.empty]
    if not frames:
        miss.append("no usable news items collected")
        return (
            pd.DataFrame(
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
            ),
            miss,
        )

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["article_id"]).sort_values("published_at_utc").reset_index(drop=True)
    return df, miss


def write_raw_news(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
