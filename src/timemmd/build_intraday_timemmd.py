from __future__ import annotations

from pathlib import Path

import pandas as pd


NUMERIC_FIELDS = [
    "OT",
    "uso_open",
    "uso_high",
    "uso_low",
    "uso_close",
    "uso_volume",
    "uso_ret_1h",
    "uso_ret_7h",
    "vol_7h",
    "vol_20h",
    "ma_7h",
    "ma_20h",
    "rsi_14h",
    "news_count_1h",
    "news_count_6h",
    "news_sent_mean_6h",
    "oil_event_count_6h",
]


def build_intraday_numerical(panel: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    cols = ["start_date", "end_date"] + [c for c in NUMERIC_FIELDS if c in panel.columns]
    out = panel[cols].sort_values("end_date").reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8")
    return out


def build_intraday_search_text(panel: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "start_date": panel["start_date"],
            "end_date": panel["end_date"],
            "fact": panel.get("news_agg_6h", "").fillna("").replace({"": "NA"}),
            "preds": "NA",
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8")
    return out
