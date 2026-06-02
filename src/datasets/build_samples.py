from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd


def build_samples(
    numerical: pd.DataFrame,
    settings: Sequence[Dict[str, int]],
    symbol: str,
    output_dir: Path,
    x_numeric_path: str = "data/numerical/OilETF/OilETF.csv",
    x_text_embedding_path: str = "data/textual/OilETF/OilETF_search.csv",
    x_image_dir: str = "data/images/OilETF",
) -> List[Path]:
    dates = pd.to_datetime(numerical["end_date"])
    outputs = []
    n = len(numerical)
    if n == 0:
        return outputs
    train_ratio = 0.7
    val_ratio = 0.1
    test_ratio = 0.2
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    for cfg in settings:
        H = int(cfg["H"])
        F = int(cfg["F"])
        rows = []
        for i in range(H - 1, n - F):
            end_date = dates.iloc[i]
            start = dates.iloc[i - H + 1]
            y_start = dates.iloc[i + 1]
            y_end = dates.iloc[i + F]
            if i < train_end:
                split = "train"
            elif i < val_end:
                split = "val"
            else:
                split = "test"
            sample_id = f"{symbol}_{end_date.strftime('%Y-%m-%d')}_H{H}_F{F}"
            rows.append(
                {
                    "sample_id": sample_id,
                    "symbol": symbol,
                    "end_date": end_date.strftime("%Y-%m-%d"),
                    "H": H,
                    "F": F,
                    "x_num_start": start.strftime("%Y-%m-%d"),
                    "x_num_end": end_date.strftime("%Y-%m-%d"),
                    "text_start": start.strftime("%Y-%m-%d"),
                    "text_end": end_date.strftime("%Y-%m-%d"),
                    "y_start": y_start.strftime("%Y-%m-%d"),
                    "y_end": y_end.strftime("%Y-%m-%d"),
                    "x_numeric_path": x_numeric_path,
                    "x_text_embedding_path": x_text_embedding_path,
                    "x_image_path": f"{x_image_dir}/{symbol}_{end_date.strftime('%Y-%m-%d')}_H{H}.png",
                    "y_path": "",
                    "split": split,
                }
            )
        df = pd.DataFrame(rows)
        out_path = output_dir / f"samples_H{H}_F{F}.parquet"
        df.to_parquet(out_path, index=False)
        outputs.append(out_path)
    return outputs
