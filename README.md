# OilETF-TimeMMD — 石油 ETF 多模态时序预测数据集

## 数据集概览

| 维度 | OilETF (daily) | OilETF-intraday (hourly) |
|------|---------------|-------------------------|
| ETF 标的 | USO, BNO, DBO, USL, SCO | 同上 (UCO 已移除, yfinance 无数据) |
| 价格频率 | 日线 | 小时线 (9:30-16:00 ET) |
| 价格跨度 | 2016-01 → 2026-06 | 2024-06 → 2026-06 |
| 新闻源 | Finnhub + GDELT | Finnhub + GDELT (共享缓存) |
| 新闻跨度 | 2017-10 → 2026-06 | 2017-10 → 2026-06 |
| 事件类型 | — | geo / price / opec / macro / inventory / supply_demand |
| 样本配置 | H60F1 / H120F5 | H60F1 / H120F7 |
| 多模态 | 价格 + 新闻 + 宏观 + 库存 | 价格 + 事件 + 日内新闻 |

## 缓存策略

- Daily 与 Intraday 共享 `data/raw/news/raw_news.csv` 新闻缓存
- 每次构建自动增量：只抓取上次缓存日期之后的新文章
- GDELT 提供 2017+ 历史新闻覆盖，Finnhub 提供近 30 天 per-symbol 新闻

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

当前 `configs/data_config.yaml` 已固定 `date_range.end: 2026-06-24`，避免后续运行因日期自动前移导致样本数量变化。
小时级价格源使用 `yfinance` 的 `1h` 历史数据，默认通过 `intraday.yfinance_period: 730d` 拉取后再按配置日期过滤；如 Yahoo 返回空数据，可缩短 `intraday.lookback_years` 或 `intraday.yfinance_period`。
小时级新闻源已切换为单一官方源：`finnhub`（`company-news`）。
如需构建新闻特征，请在环境变量或配置中提供 `FINNHUB_API_KEY`（或 `collection.finnhub.api_key`），否则构建在新闻采集阶段会直接失败并报错，不会再通过其他源兜底补齐。
请注意：GDELT、yfinance.news、RSS 在本项目当前版本不再作为新闻 fallback，出现 `no usable news items collected` 则是 Finnhub 数据确实缺失或限流结果。

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
