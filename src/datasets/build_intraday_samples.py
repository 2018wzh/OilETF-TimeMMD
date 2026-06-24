from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd


def build_intraday_samples(
    numerical: pd.DataFrame,
    settings: Sequence[Dict[str, int]],
    symbol: str,
    output_dir: Path,
    x_numeric_path: str = "data/numerical/OilETF/OilETF_intraday.csv",
    x_text_embedding_path: str = "data/textual/OilETF/OilETF_intraday_search.csv",
) -> List[Path]:
    outputs: List[Path] = []
    n = len(numerical)
    if n == 0:
        return outputs
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)
    for cfg in settings:
        H = int(cfg["H"])
        F = int(cfg["F"])
        rows = []
        for i in range(H - 1, n - F):
            entry_i = i + 1
            exit_i = i + F
            split = "train" if i < train_end else "val" if i < val_end else "test"
            end_time = str(numerical.iloc[i]["end_date"])
            entry_open = float(numerical.iloc[entry_i]["uso_open"])
            exit_close = float(numerical.iloc[exit_i]["uso_close"])
            rows.append(
                {
                    "sample_id": f"{symbol}_{end_time}_H{H}_F{F}",
                    "symbol": symbol,
                    "end_date": end_time,
                    "H": H,
                    "F": F,
                    "x_num_start": str(numerical.iloc[i - H + 1]["start_date"]),
                    "x_num_end": end_time,
                    "text_start": str(numerical.iloc[i - H + 1]["start_date"]),
                    "text_end": end_time,
                    "entry_time": str(numerical.iloc[entry_i]["start_date"]),
                    "y_start": str(numerical.iloc[entry_i]["start_date"]),
                    "y_end": str(numerical.iloc[exit_i]["end_date"]),
                    "y_return": exit_close / entry_open - 1,
                    "y_direction": exit_close > entry_open,
                    "x_numeric_path": x_numeric_path,
                    "x_text_embedding_path": x_text_embedding_path,
                    "y_path": "",
                    "split": split,
                }
            )
        out_path = output_dir / f"intraday_samples_H{H}_F{F}.parquet"
        pd.DataFrame(rows).to_parquet(out_path, index=False)
        outputs.append(out_path)
    return outputs
