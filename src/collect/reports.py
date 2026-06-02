from __future__ import annotations

from hashlib import md5
import re
from typing import Dict, List, Tuple

import pandas as pd
import requests
import xml.etree.ElementTree as ET


def _parse_rss_text(text: str, source: str, max_items: int) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    try:
        root = ET.fromstring(text)
    except Exception:
        return []

    tags = []
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    if root.tag.endswith("rss"):
        tags = root.findall(".//item")
    elif "feed" in root.tag:
        tags = root.findall(".//atom:entry", ns)

    for node in tags[:max_items]:
        title = node.findtext("title") or node.findtext(".//{*}title") or ""
        if not title:
            continue
        link = node.findtext("link") or node.findtext(".//link") or ""
        if not link:
            link_node = node.find(".//{*}link")
            if link_node is not None:
                link = link_node.attrib.get("href", "")
        pub = node.findtext("pubDate") or node.findtext("published") or node.findtext("updated") or ""
        try:
            pub_ts = pd.to_datetime(pub, utc=True)
            if pd.isna(pub_ts):
                pub_ts = pd.Timestamp.utcnow()
        except Exception:
            pub_ts = pd.Timestamp.utcnow()
        fact = node.findtext("description") or node.findtext("summary") or ""
        fact = re.sub(r"<[^>]+>", " ", fact or "")
        fact = re.sub(r"\s+", " ", fact).strip()[:500]
        preds = "NA"
        entries.append(
            {
                "published_at_utc": pub_ts.isoformat(),
                "source": source,
                "title": title.strip(),
                "fact": fact,
                "preds": preds,
                "url": link.strip(),
                "url_hash": md5((source + title + link).encode("utf-8")).hexdigest(),
                "provider": "rss",
            }
        )
    return entries


def collect_reports(start_date: str, end_date: str, max_items: int = 20) -> Tuple[pd.DataFrame, List[str]]:
    url_sources = [
        ("EIA", "https://www.eia.gov/petroleum/supply/weekly/rss.xml"),
        ("OPEC", "https://www.opec.org/opec_web/en/pressroom/pressreleases.xml"),
        ("IEA", "https://www.iea.org/rss/iea-news"),
    ]
    rows: List[Dict[str, object]] = []
    miss: List[str] = []

    for source, url in url_sources:
        try:
            resp = requests.get(url, timeout=20)
            if resp.status_code != 200:
                miss.append(f"{source} HTTP {resp.status_code}")
                continue
            entries = _parse_rss_text(resp.text, source, max_items)
            if entries:
                for row in entries:
                    row["start_date"] = row["published_at_utc"][:10]
                    row["end_date"] = row["start_date"]
                    rows.append(row)
            else:
                miss.append(f"{source} no parsable report items")
        except Exception as e:
            miss.append(f"{source} exception: {type(e).__name__}")

    if not rows:
        return pd.DataFrame(
            columns=[
                "published_at_utc",
                "source",
                "title",
                "fact",
                "preds",
                "url",
                "url_hash",
                "start_date",
                "end_date",
                "provider",
            ]
        ), miss

    df = pd.DataFrame(rows)
    df["published_at_utc"] = pd.to_datetime(df["published_at_utc"], utc=True).astype(str)
    end_plus = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = df[(df["published_at_utc"] >= str(start_date)) & (df["published_at_utc"] <= end_plus)]
    return df.reset_index(drop=True), miss
