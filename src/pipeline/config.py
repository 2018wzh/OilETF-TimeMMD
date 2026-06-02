from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd

try:
    import yaml
except Exception:  # pragma: no cover - dependency fallback message is explicit in loader
    yaml = None


@dataclass(frozen=True)
class PathConfig:
    root: Path
    raw: Path
    processed: Path
    timemmd_numerical: Path
    timemmd_textual: Path
    images: Path
    metadata: Path
    outputs: Path


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        if yaml is None:
            raise RuntimeError("缺少依赖 pyyaml，无法解析 YAML 配置。请安装依赖并重试。")
        return yaml.safe_load(f)


def normalize_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(raw or {})
    cfg["date_range"] = cfg.get("date_range", {})
    cfg["date_range"]["start"] = pd.to_datetime(cfg["date_range"].get("start", "2016-01-01")).date()
    cfg["date_range"]["end"] = cfg["date_range"].get("end")
    if cfg["date_range"]["end"] is None:
        cfg["date_range"]["end"] = pd.Timestamp.now(tz="UTC").tz_convert("US/Eastern").date()
    else:
        cfg["date_range"]["end"] = pd.to_datetime(cfg["date_range"]["end"]).date()
    return cfg


def build_paths(cfg: Dict[str, Any]) -> PathConfig:
    root = Path(cfg.get("root", ".")).resolve()
    paths = cfg.get("paths", {})
    raw = root / paths.get("raw", "data/raw")
    processed = root / paths.get("processed", "data/processed")
    timemmd_numerical = root / paths.get("timemmd_numerical", "data/numerical/OilETF")
    timemmd_textual = root / paths.get("timemmd_textual", "data/textual/OilETF")
    images = root / paths.get("images", "data/images/OilETF")
    metadata = root / paths.get("metadata", "data/metadata")
    outputs = root / paths.get("outputs", "outputs")
    for p in [
        raw,
        raw / "prices",
        raw / "oil_macro",
        raw / "news",
        raw / "reports",
        raw / "calendar",
        processed,
        timemmd_numerical.parent,
        timemmd_numerical,
        timemmd_textual.parent,
        timemmd_textual,
        images,
        metadata,
        outputs,
    ]:
        p.mkdir(parents=True, exist_ok=True)
    return PathConfig(
        root=root,
        raw=raw,
        processed=processed,
        timemmd_numerical=timemmd_numerical,
        timemmd_textual=timemmd_textual,
        images=images,
        metadata=metadata,
        outputs=outputs,
    )
