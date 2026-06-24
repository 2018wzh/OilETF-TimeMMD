from __future__ import annotations

from hashlib import md5
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import xml.etree.ElementTree as ET

import pandas as pd
import requests
import yfinance as yf
from tqdm.auto import tqdm


def _empty_news_frame() -> pd.DataFrame:
    return pd.DataFrame(
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


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return 0.0


def _strip_xml(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text or "")).strip()


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
        return _empty_news_frame(), miss
    df = pd.DataFrame.from_records(rows).drop_duplicates(subset=["article_id"]).sort_values("published_at_utc")
    return df.reset_index(drop=True), miss


def _fetch_finnhub_news(
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
    cfg: Dict[str, object] | None = None,
) -> Tuple[pd.DataFrame, List[str]]:
    # NOTE: Finnhub free-tier company-news only returns ~30 days of recent articles
    # regardless of the from/to date parameters. Use GDELT or other sources for
    # historical news coverage beyond one month.
    cfg = cfg or {}
    endpoint = str(cfg.get("endpoint", "https://finnhub.io/api/v1/company-news"))
    api_key = str(cfg.get("api_key", "") or "").strip() or str(os.getenv("FINNHUB_API_KEY", "")).strip()
    min_interval = float(cfg.get("min_request_interval_seconds", 1.0))
    retries = int(cfg.get("retry_count", 2))
    backoff = float(cfg.get("retry_backoff_seconds", 5.0))
    timeout = int(cfg.get("timeout_seconds", 30))

    if not api_key:
        raise RuntimeError("FINNHUB_API_KEY is required for finnhub news source")

    rows: List[Dict[str, object]] = []
    miss: List[str] = []
    s_date = pd.Timestamp(start_date).date()
    e_date = pd.Timestamp(end_date).date()
    session = requests.Session()
    last_request_at = 0.0

    for symbol in tqdm(list(symbols), desc="finnhub news", unit="symbol", leave=False):
        params = {
            "symbol": symbol,
            "from": str(s_date),
            "to": str(e_date),
            "token": api_key,
        }
        payload = None
        for attempt in range(retries + 1):
            elapsed = time.monotonic() - last_request_at
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            last_request_at = time.monotonic()

            try:
                resp = session.get(endpoint, params=params, timeout=timeout, headers={"User-Agent": "OilETF-TimeMMD/0.1"})
            except Exception as e:
                if attempt >= retries:
                    miss.append(f"finnhub {symbol} exception: {type(e).__name__}")
                else:
                    time.sleep(backoff * (attempt + 1))
                continue

            if resp.status_code == 429 and attempt < retries:
                time.sleep(backoff * (attempt + 1))
                continue
            if resp.status_code == 404:
                miss.append(f"finnhub {symbol} not available (HTTP 404)")
                break
            if resp.status_code != 200:
                miss.append(f"finnhub {symbol} HTTP {resp.status_code}")
                break
            try:
                payload = resp.json()
                break
            except Exception:
                if attempt < retries:
                    time.sleep(backoff * (attempt + 1))
                    continue
                miss.append(f"finnhub {symbol} invalid JSON")
                break

        if not isinstance(payload, list):
            continue
        for article in payload:
            if not isinstance(article, dict):
                continue
            title = str(article.get("headline", "") or "").strip()
            summary = str(article.get("summary", "") or "").strip()
            if not title and not summary:
                continue
            published_raw = article.get("datetime")
            if published_raw is None:
                continue
            try:
                published = pd.to_datetime(int(published_raw), unit="s", utc=True)
            except Exception:
                continue
            local_date = published.tz_convert("US/Eastern").date()
            if local_date < s_date or local_date > e_date:
                continue
            link = str(article.get("url") or article.get("source") or "")
            text = f"{title} {summary}".strip()
            rows.append(
                {
                    "article_id": md5((link + str(published.value) + symbol).encode("utf-8")).hexdigest(),
                    "published_at_utc": published.isoformat(),
                    "source": str(article.get("source") or "finnhub"),
                    "title": title,
                    "summary": summary,
                    "url_hash": md5((title + (link or summary)).encode("utf-8")).hexdigest(),
                    "tickers": symbol,
                    "entities": ";".join(
                        filter(None, [str(article.get("category", "") or ""), str(article.get("related", "") or "")])
                    ),
                    "sentiment": _safe_float(article.get("sentiment", 0.0)),
                    "relevance": _safe_float(article.get("relevance", 1.0)),
                    "provider": "finnhub.company-news",
                    "target_symbol": symbol,
                }
            )

    if not rows:
        miss.append("finnhub collected no usable articles")
        return _empty_news_frame(), miss
    out = pd.DataFrame.from_records(rows).drop_duplicates(subset=["article_id"]).sort_values("published_at_utc")
    return out.reset_index(drop=True), miss


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


def _fetch_rss_news(
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
    cfg: Dict[str, object] | None = None,
) -> Tuple[pd.DataFrame, List[str]]:
    cfg = cfg or {}
    feeds = cfg.get("feeds", [])
    if not isinstance(feeds, list) or not feeds:
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
            ["rss feeds missing"],
        )

    rows: List[Dict[str, object]] = []
    miss: List[str] = []
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    timeout = int(cfg.get("timeout_seconds", 20))

    for feed in feeds:
        source = str(feed.get("source", "rss-feed")) if isinstance(feed, dict) else "rss-feed"
        url = str(feed.get("url", "") if isinstance(feed, dict) else "").strip()
        if not url:
            miss.append(f"rss feed missing url for source={source}")
            continue
        try:
            resp = requests.get(url, timeout=timeout, headers={"User-Agent": "OilETF-TimeMMD/0.1"})
            if resp.status_code != 200:
                miss.append(f"rss {source} HTTP {resp.status_code}")
                continue
        except Exception as e:
            miss.append(f"rss {source} exception: {type(e).__name__}")
            continue

        try:
            root = ET.fromstring(resp.text)
        except Exception:
            miss.append(f"rss {source} invalid XML")
            continue

        items = root.findall(".//item")
        if not items and root.tag.endswith("feed"):
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            items = root.findall(".//atom:entry", ns)
        if not items:
            miss.append(f"rss {source} no items")
            continue

        max_items = int(feed.get("max_items", cfg.get("max_items_per_feed", 50)) if isinstance(feed, dict) else cfg.get("max_items_per_feed", 50))
        for node in items[: max_items]:
            if not isinstance(node.tag, str):
                continue
            title = node.findtext("title") or node.findtext(".//{*}title") or ""
            if not title:
                continue
            link = node.findtext("link") or node.findtext(".//{*}link") or ""
            if not link and (node.find(".//{*}link") is not None):
                link = node.find(".//{*}link").attrib.get("href", "")
            pub = node.findtext("pubDate") or node.findtext("published") or node.findtext("updated") or ""
            summary = _strip_xml(node.findtext("description") or node.findtext("summary") or "")
            if not summary:
                summary = _strip_xml(title)
            try:
                published = pd.to_datetime(pub, utc=True, errors="coerce")
                if pd.isna(published):
                    published = pd.Timestamp.utcnow()
            except Exception:
                published = pd.Timestamp.utcnow()
            local_date = published.tz_convert("US/Eastern").date()
            if local_date < start.date() or local_date > end.date():
                continue

            article_id = md5((url + str(published.value) + title).encode("utf-8")).hexdigest()
            rows.append(
                {
                    "article_id": article_id,
                    "published_at_utc": published.isoformat(),
                    "source": source,
                    "title": title.strip(),
                    "summary": summary,
                    "url_hash": md5((title + (link or source)).encode("utf-8")).hexdigest(),
                    "tickers": ";".join(map(str, symbols)),
                    "entities": source,
                    "sentiment": 0.0,
                    "relevance": 1.0,
                    "provider": "rss",
                    "target_symbol": symbols[0] if symbols else "USO",
                }
            )

    if not rows:
        miss.append("rss collected no usable articles")
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
    out = pd.DataFrame.from_records(rows).drop_duplicates(subset=["article_id"]).sort_values("published_at_utc").reset_index(drop=True)
    return out, miss


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
                    verify=False,
                )
            except Exception as e:
                if attempt >= retries:
                    miss.append(f"GDELT {chunk_start.date()}..{chunk_end.date()} exception: {type(e).__name__}")
                else:
                    time.sleep(min_interval * (attempt + 1))
                continue
            body_text = (resp.text or "").strip().lower()
            if ("please limit requests" in body_text or "too many requests" in body_text) and (resp.status_code == 200):
                if attempt < retries:
                    time.sleep(rate_limit_backoff * (attempt + 1))
                    continue
                miss.append(
                    f"GDELT {chunk_start.date()}..{chunk_end.date()} rate-limited without retry budget"
                )
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
                if attempt < retries:
                    time.sleep(min_interval * (attempt + 1))
                    continue
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
    sources_cfg = collection_cfg.get("news_sources", ["finnhub"])
    if isinstance(sources_cfg, str):
        sources_cfg = [sources_cfg]
    sources = [str(s).lower() for s in sources_cfg]

    frames: List[pd.DataFrame] = []
    miss: List[str] = []

    if "finnhub" in sources:
        finnhub_df, finnhub_miss = _fetch_finnhub_news(
            symbols,
            start_date,
            end_date,
            collection_cfg.get("finnhub", {}),
        )
        frames.append(finnhub_df)
        miss.extend(finnhub_miss)
        if not finnhub_df.empty:
            # log per-symbol article counts for debugging news coverage
            finnhub_df["published_at_utc"] = pd.to_datetime(finnhub_df["published_at_utc"], utc=True)
            sym_counts = finnhub_df.groupby("target_symbol").size()
            date_min = finnhub_df["published_at_utc"].min().date()
            date_max = finnhub_df["published_at_utc"].max().date()
            miss.append(
                f"finnhub collected {len(finnhub_df)} articles across {len(sym_counts)} symbols "
                f"({date_min}..{date_max}); per-symbol: {sym_counts.to_dict()}"
            )

    if "gdelt" in sources:
        gdelt_df, gdelt_miss = _fetch_gdelt_news(
            symbols,
            start_date,
            end_date,
            collection_cfg.get("gdelt", {}),
        )
        frames.append(gdelt_df)
        miss.extend(gdelt_miss)
        if not gdelt_df.empty:
            gdelt_df["published_at_utc"] = pd.to_datetime(gdelt_df["published_at_utc"], utc=True)
            date_min = gdelt_df["published_at_utc"].min().date()
            date_max = gdelt_df["published_at_utc"].max().date()
            miss.append(f"gdelt collected {len(gdelt_df)} articles ({date_min}..{date_max})")

    for source in sources:
        if source not in {"finnhub", "gdelt"}:
            miss.append(f"{source} news source is not active (supported: finnhub, gdelt)")

    frames = [df for df in frames if df is not None and not df.empty]
    if not frames:
        miss.append("no usable news items collected")
        return _empty_news_frame(), miss

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["article_id"]).sort_values("published_at_utc").reset_index(drop=True)
    return df, miss


def fetch_news_with_cache(
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
    cache_path: Path,
    per_symbol_limit: int = 200,
    collection_cfg: Dict[str, object] | None = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Fetch news with disk-based incremental caching.

    Reads existing cache from *cache_path*, computes the date gap since last
    cached article, fetches only the new range, deduplicates, and persists the
    merged result back to disk.  This avoids redundant API calls for date
    ranges already covered by previous runs.

    Returns (combined_df, miss_log).
    """
    collection_cfg = collection_cfg or {}
    miss: List[str] = []

    # ---- load existing cache ----
    cached = _empty_news_frame()
    if cache_path.exists():
        try:
            cached = pd.read_csv(cache_path)
            cached["published_at_utc"] = pd.to_datetime(cached["published_at_utc"], utc=True)
            miss.append(f"cache loaded: {len(cached)} articles from {cache_path.name}")
        except Exception:
            miss.append(f"cache read failed for {cache_path}, will re-fetch full range")
            cached = _empty_news_frame()

    # ---- determine fetch window ----
    fetch_start = start_date
    if not cached.empty:
        # strip tz so comparison with naive start_date string works
        last_cached = cached["published_at_utc"].max()
        if hasattr(last_cached, "tz") and last_cached.tz is not None:
            last_cached = last_cached.tz_localize(None)
        # overlap 2 days for timezone safety
        fetch_start_dt = max(
            pd.Timestamp(start_date),
            last_cached - pd.Timedelta(days=2),
        )
        fetch_start = str(fetch_start_dt.date())
        if pd.Timestamp(fetch_start) >= pd.Timestamp(end_date):
            miss.append("all dates already covered in cache")
            return cached.reset_index(drop=True), miss
        miss.append(f"incremental fetch window: {fetch_start} .. {end_date}")

    # ---- fetch new data ----
    new_df, new_miss = fetch_news(
        symbols, fetch_start, end_date, per_symbol_limit, collection_cfg
    )
    miss.extend(new_miss)

    if new_df.empty:
        if not cached.empty:
            return cached.reset_index(drop=True), miss
        return new_df, miss

    # normalise datetimes
    new_df["published_at_utc"] = pd.to_datetime(new_df["published_at_utc"], utc=True)
    if not cached.empty:
        cached["published_at_utc"] = pd.to_datetime(cached["published_at_utc"], utc=True)

    # ---- merge & dedupe ----
    combined = pd.concat([cached, new_df], ignore_index=True)
    combined = (
        combined
        .drop_duplicates(subset=["article_id"], keep="last")
        .sort_values("published_at_utc")
        .reset_index(drop=True)
    )

    # ---- persist ----
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(cache_path, index=False)
    miss.append(
        f"cache updated: {len(combined)} articles ({combined['published_at_utc'].min().date()}"
        f"..{combined['published_at_utc'].max().date()}) -> {cache_path.name}"
    )

    return combined, miss


def write_raw_news(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
