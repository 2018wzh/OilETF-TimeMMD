from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence
from tqdm.auto import tqdm

from src.pipeline import build_intraday_dataset, build_oil_dataset
from src.pipeline.config import load_config, normalize_config
from src.pipeline.upload_hf_dataset import upload_dataset


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OilETF-TimeMMD datasets")
    parser.add_argument("--config", type=Path, default=Path("configs/data_config.yaml"))
    parser.add_argument("--mode", choices=["daily", "intraday", "all"], default="daily")
    parser.add_argument("--incremental", action="store_true", help="append new daily raw data only")
    parser.add_argument("--with-visuals", action="store_true", help="enable daily candlestick image rendering (slow, ~25k images)")
    parser.add_argument("--upload", action="store_true", help="upload built dataset files to Hugging Face")
    parser.add_argument("--upload-only", action="store_true", help="upload existing local dataset files without rebuilding")
    parser.add_argument("--hf-repo-id", default="", help="Hugging Face dataset repo id, e.g. user/dataset")
    parser.add_argument("--hf-private", action="store_true", help="create/use a private Hugging Face dataset repo")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    stages = []
    if not args.upload_only and args.mode in {"daily", "all"}:
        stages.append(("daily", "build daily dataset"))
    if not args.upload_only and args.mode in {"intraday", "all"}:
        stages.append(("intraday", "build intraday dataset"))
    if args.upload or args.upload_only:
        stages.append(("huggingface", "upload dataset"))
    pbar = tqdm(total=len(stages), desc="build pipeline", unit="stage")
    results = {}
    try:
        for stage, desc in stages:
            pbar.set_description(f"build: {desc}")
            if stage == "daily":
                results["daily"] = build_oil_dataset.build(
                    args.config, incremental=args.incremental, skip_visuals=not args.with_visuals
                )
            elif stage == "intraday":
                results["intraday"] = build_intraday_dataset.build(args.config)
            else:
                cfg = normalize_config(load_config(args.config))
                results["huggingface"] = {
                    "url": upload_dataset(cfg, args.hf_repo_id, scope=args.mode, private=args.hf_private)
                }
            pbar.update(1)
    finally:
        pbar.close()

    print("Build completed")
    for mode, result in results.items():
        print(f"[{mode}]")
        for key, value in result.items():
            if key == "qc":
                continue
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
