from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Any, Dict

import pandas as pd


def _load_env_file() -> None:
    # ponytail: root-level load only, avoid repeating dotenv logic in every entrypoint.
    env_file = None
    start = Path(__file__).resolve()
    for parent in (start.parent.parent, start.parent.parent.parent):
        candidate = parent / ".env"
        if candidate.exists():
            env_file = candidate
            break
    if env_file is None:
        return

    try:
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)
    except Exception:
        for raw in env_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"").strip("'")
            os.environ.setdefault(key, value)

try:
    import yaml
except Exception:  # pragma: no cover - dependency fallback message is explicit in loader
    yaml = None


_load_env_file()


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
