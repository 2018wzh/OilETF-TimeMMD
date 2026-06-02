from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _plot_window(window: pd.DataFrame, symbol: str, end_date, H: int, out_path: Path) -> None:
    if window.empty:
        return
    fig, ax1 = plt.subplots(figsize=(7, 4), dpi=80)
    x = np.arange(len(window))
    ax1.set_title(f"{symbol} {end_date.strftime('%Y-%m-%d')} H{H}")
    o = window["open"].to_numpy()
    c = window["close"].to_numpy()
    h = window["high"].to_numpy()
    l = window["low"].to_numpy()
    v = window["volume"].to_numpy()
    color = np.where(c >= o, "green", "red")
    ax1.vlines(x, l, h, color="black", linewidth=0.6, alpha=0.8)
    ax1.bar(x, c - o, bottom=o, color=color, alpha=0.6, width=0.45, linewidth=0)
    ax1.plot(x, c, color="blue", linewidth=1.0, alpha=0.7, label="close")
    ma5 = window["ma5"]
    ma20 = window["ma20"]
    ma60 = window["ma60"]
    ax1.plot(x, ma5, color="orange", linewidth=0.8, label="ma5")
    ax1.plot(x, ma20, color="purple", linewidth=0.8, label="ma20")
    ax1.plot(x, ma60, color="gray", linewidth=0.8, label="ma60")
    ax1.set_ylabel("price")
    ax1.tick_params(axis="x", rotation=0, labelsize=7)
    ax1.legend(fontsize=7)
    # simplified volume trace (single line) for speed
    ax2 = ax1.twinx()
    ax2.plot(x, v, alpha=0.2, color="steelblue", linewidth=0.8)
    ax2.set_ylim(0, v.max() * 3 if len(v) else 1.0)
    ax2.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def generate_candles(raw_prices_path: Path, symbols: Sequence[str], windows: Sequence[int], image_dir: Path) -> None:
    raw = pd.read_csv(raw_prices_path)
    if raw.empty:
        return
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    image_dir.mkdir(parents=True, exist_ok=True)
    raw = raw.sort_values("timestamp")

    for symbol in symbols:
        sdf = raw[raw["symbol"] == symbol].sort_values("timestamp")
        if sdf.empty:
            continue
        sdf = sdf.copy().sort_values("timestamp").reset_index(drop=True)
        sdf["ma5"] = sdf["close"].rolling(5).mean()
        sdf["ma20"] = sdf["close"].rolling(20).mean()
        sdf["ma60"] = sdf["close"].rolling(60).mean()
        for H in windows:
            for i in range(H - 1, len(sdf)):
                window = sdf.iloc[i - H + 1 : i + 1]
                end_date = window["timestamp"].iloc[-1]
                out = image_dir / f"{symbol}_{end_date.strftime('%Y-%m-%d')}_H{H}.png"
                if out.exists():
                    continue
                _plot_window(
                    window,
                    symbol,
                    end_date,
                    H,
                    out,
                )
