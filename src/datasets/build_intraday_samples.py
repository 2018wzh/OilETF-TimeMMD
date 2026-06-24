from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd


def build_intraday_samples(
    numerical: pd.DataFrame,
    settings: Sequence[Dict[str, int]],
    output_dir: Path,
    x_numeric_path: str = "data/numerical/OilETF/OilETF_intraday.csv",
    x_text_embedding_path: str = "data/textual/OilETF/OilETF_intraday_search.csv",
) -> List[Path]:
    outputs: List[Path] = []
    if "symbol" not in numerical.columns:
        raise ValueError("numerical must contain symbol column for intraday sampling")
    if numerical.empty:
        return outputs
    for cfg in settings:
        H = int(cfg["H"])
        F = int(cfg["F"])
        rows = []
        for symbol, symbol_df in numerical.groupby("symbol", sort=False):
            symbol_df = symbol_df.sort_values("end_date").reset_index(drop=True)
            s_n = len(symbol_df)
            train_end_sym = int(s_n * 0.7)
            val_end_sym = int(s_n * 0.8)
            for i in range(H - 1, s_n - F):
                entry_i = i + 1
                exit_i = i + F
                split = "train" if i < train_end_sym else "val" if i < val_end_sym else "test"
                end_time = str(symbol_df.iloc[i]["end_date"])
                entry_open = float(symbol_df.iloc[entry_i]["uso_open"])
                exit_close = float(symbol_df.iloc[exit_i]["uso_close"])
                rows.append(
                    {
                        "sample_id": f"{symbol}_{end_time}_H{H}_F{F}",
                        "symbol": str(symbol),
                        "end_date": end_time,
                        "H": H,
                        "F": F,
                        "x_num_start": str(symbol_df.iloc[i - H + 1]["start_date"]),
                        "x_num_end": end_time,
                        "text_start": str(symbol_df.iloc[i - H + 1]["start_date"]),
                        "text_end": end_time,
                        "entry_time": str(symbol_df.iloc[entry_i]["start_date"]),
                        "y_start": str(symbol_df.iloc[entry_i]["start_date"]),
                        "y_end": str(symbol_df.iloc[exit_i]["end_date"]),
                        "y_return": exit_close / entry_open - 1,
                        "y_direction": exit_close > entry_open,
                        "x_numeric_path": x_numeric_path,
                        "x_text_embedding_path": x_text_embedding_path,
                        "y_path": "",
                        "split": split,
                    }
                )
        if not rows:
            continue
        out_path = output_dir / f"intraday_samples_H{H}_F{F}.parquet"
        pd.DataFrame(rows).to_parquet(out_path, index=False)
        outputs.append(out_path)
    return outputs
