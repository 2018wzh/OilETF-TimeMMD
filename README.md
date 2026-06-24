# OilETF-TimeMMD 数据工程闭环

## 一键构建数据集

```bash
python build.py
```

默认构建日频数据集。可显式指定模式：

```bash
python build.py --mode daily
python build.py --mode intraday
python build.py --mode all
```

如只重建表格和 parquet、复用已有图片：

```bash
python build.py --mode daily --incremental --skip-visuals
```

参数：
- `--mode daily|intraday|all`：构建日频、小时级或两者
- `--config`：指定配置文件，默认 `configs/data_config.yaml`
- `--incremental`：基于已有 `data/raw` 做增量采集（只追加新日期）
- `--skip-visuals`：跳过 K 线图像生成
- `--upload`：构建完成后上传到 Hugging Face Dataset
- `--upload-only`：不重新构建，只上传本地已有数据
- `--hf-repo-id`：Hugging Face 数据集仓库 ID，例如 `username/dataset-name`

构建小时级数据并上传：

```bash
python build.py --mode intraday --upload --hf-repo-id <repo_id>
```

只上传本地已有数据：

```bash
python build.py --mode intraday --upload-only --hf-repo-id <repo_id>
```

也可以在 `configs/data_config.yaml` 的 `huggingface.repo_id` 中填写仓库 ID，然后省略 `--hf-repo-id`。上传脚本只上传 processed / numerical / textual / metadata 等发布文件，不上传 `data/raw` 和 `.env`。

兼容入口仍可使用：

```bash
python -m src.pipeline.build_dataset --mode daily --config configs/data_config.yaml
```

当前 `configs/data_config.yaml` 已固定 `date_range.end: 2026-05-31`，避免后续运行因日期自动前移导致样本数量变化。
小时级价格源使用 `yfinance` 的 `1h` 历史数据，默认通过 `intraday.yfinance_period: 730d` 拉取后再按配置日期过滤；如 Yahoo 返回空数据，可缩短 `intraday.lookback_years` 或 `intraday.yfinance_period`。
新闻源默认包含 `yfinance.news` 和无需 API key 的 GDELT DOC API；GDELT 受公开接口限流影响，默认从 `2024-01-01` 起回补，并保留已有 `raw_news.csv` 缓存以便多次运行逐步补齐。需要扩大历史范围时，修改 `collection.gdelt.start`。

## 输出
- `data/raw/prices/raw_prices.csv`
- `data/raw/oil_macro/raw_oil_macro.csv`
- `data/raw/news/raw_news.csv`
- `data/raw/reports/raw_reports.csv`
- `data/raw/calendar/calendar.csv`
- `data/processed/daily_panel.parquet`
- `data/processed/hourly_panel.parquet`
- `data/numerical/OilETF/OilETF.csv`
- `data/numerical/OilETF/OilETF_intraday.csv`
- `data/textual/OilETF/OilETF_search.csv`
- `data/textual/OilETF/OilETF_intraday_search.csv`
- `data/textual/OilETF/OilETF_report.csv`
- `data/processed/samples_H60_F1.parquet`
- `data/processed/samples_H120_F5.parquet`
- `data/processed/intraday_samples_H60_F1.parquet`
- `data/processed/intraday_samples_H120_F7.parquet`
- `data/images/OilETF/<symbol>_<date>_H60.png`
- `data/images/OilETF/<symbol>_<date>_H120.png`
- `data/metadata/OilETF_dataset_card.json`
- `data/metadata/OilETF_intraday_dataset_card.json`
- `data/README.md`
- `outputs/data_qc_report.md`

`data/` 保存实际数据集内容，已在 `.gitignore` 中排除。发布到 Hugging Face 时，将 `data/` 作为 dataset repository 的上传根目录。

## 依赖安装

```bash
pip install -r requirements.txt
```
