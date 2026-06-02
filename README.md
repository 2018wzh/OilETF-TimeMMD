# OilETF-TimeMMD 数据工程闭环

## 一键构建数据集

```bash
python -m src.pipeline.build_oil_dataset --config configs/data_config.yaml
```

Windows 可复现构建：

```powershell
.\data\scripts\rebuild_dataset.ps1
```

如只重建表格和 parquet、复用已有图片：

```powershell
.\data\scripts\rebuild_dataset.ps1 -SkipVisuals
```

参数：
- `--incremental`：基于已有 `data/raw` 做增量采集（只追加新日期）
- `--skip-visuals`：跳过 K 线图像生成

当前 `configs/data_config.yaml` 已固定 `date_range.end: 2026-05-31`，避免后续运行因日期自动前移导致样本数量变化。
新闻源默认包含 `yfinance.news` 和无需 API key 的 GDELT DOC API；GDELT 受公开接口限流影响，默认从 `2024-01-01` 起回补，并保留已有 `raw_news.csv` 缓存以便多次运行逐步补齐。需要扩大历史范围时，修改 `collection.gdelt.start`。

## 输出
- `data/raw/prices/raw_prices.csv`
- `data/raw/oil_macro/raw_oil_macro.csv`
- `data/raw/news/raw_news.csv`
- `data/raw/reports/raw_reports.csv`
- `data/raw/calendar/calendar.csv`
- `data/processed/daily_panel.parquet`
- `data/numerical/OilETF/OilETF.csv`
- `data/textual/OilETF/OilETF_search.csv`
- `data/textual/OilETF/OilETF_report.csv`
- `data/processed/samples_H60_F1.parquet`
- `data/processed/samples_H120_F5.parquet`
- `data/images/OilETF/<symbol>_<date>_H60.png`
- `data/images/OilETF/<symbol>_<date>_H120.png`
- `data/metadata/OilETF_dataset_card.json`
- `data/README.md`
- `outputs/data_qc_report.md`

`data/` 保存实际数据集内容，已在 `.gitignore` 中排除。发布到 Hugging Face 时，将 `data/` 作为 dataset repository 的上传根目录。

## 依赖安装

```bash
pip install -r requirements.txt
```
