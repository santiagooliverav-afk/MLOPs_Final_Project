# AI Chip ASP Regression: Predicting Accelerator List Prices from Specs

## 1. Problem Statement

**Problem.** Given an AI accelerator's specifications, predict its estimated average selling
price (ASP, in USD). This supports a planning question relevant to the semiconductor market —
"given a chip's specs, what should it be priced at relative to the market?" — using
`datasets/raw/ai_chip_market.csv` (120 rows, 30 distinct chips, 2020–2026, 11 vendors).

**Problem type.** Supervised regression on tabular data.

**System design decisions:**

- **Data source.** Static CSV bundled in `datasets/raw/`, not a live API. The dataset is a
  curated yearly snapshot of public AI accelerator specs/shipments/pricing; no free API
  exposes this kind of cross-vendor semiconductor market data, so a static file is the
  practical choice. This also means the project is fully reproducible offline — no API keys
  or network access required to retrain.
- **Latency requirements.** Fully offline/batch. There is no user-facing request that needs a
  sub-second prediction: new chip launches happen a few times a year, and the on-demand
  workflow (predicting prices for upcoming chips listed in
  `batch_prediction_dataset/on_demand_dataset.csv`) is run manually/on a schedule, not per
  request. This is why the project ships a CD pipeline that retrains on every push to `main`
  (batch retraining) and a separate on-demand batch-prediction script, rather than a real-time
  inference endpoint — a few seconds to minutes of latency is entirely acceptable here.
- **Data transformations** (implemented in `src/transform.py`, see
  `notebooks/experiments.ipynb` §2–3 for the full investigation):
  - *Deduplication*: each chip's specs and ASP are reported once per launch but repeated for
    every year it stayed on the market (all 30 chips have >1 row). Training on raw rows would
    leak the same chip into both the train and test sets, so rows are collapsed to one per
    chip before splitting.
  - *Leakage columns dropped*: `estimated_revenue_usd_m` and `estimated_shipments_units` are
    priced off ASP (`revenue ≈ ASP × shipments`), so keeping them would leak the target.
    `chip_name`, `launch_date`, and `description` are identifiers/free text, not model features.
  - *Invalid rows dropped*: two chips report `estimated_asp_usd == 0` (undisclosed cloud-only
    pricing, not a free chip), and one (Cerebras WSE-3) reports `memory_gb == 0` with a
    `$2.5M` ASP — a wafer-scale **system** price, not a per-chip ASP, that would dominate the
    loss for a 30-row dataset if left in.
  - *Encoding*: `vendor` is one-hot encoded (11 categories); the encoder is fit once during
    training and reused as-is for on-demand predictions (`handle_unknown="ignore"` zero-fills
    a vendor never seen during training instead of raising).
- **Evaluation strategy.** Only ~27 independent samples remain after cleaning. A single K-fold
  split is highly sensitive to its random seed — same model, same data, R² ranged from **-2.05
  to +0.81** depending only on the CV seed (demonstrated in the notebook). The project
  therefore uses **repeated K-fold CV** (5 splits × 20 repeats) everywhere a metric is reported,
  both in the notebook comparison and in `src/train.py`.

## 2. Model Development

Four regressors of increasing complexity were compared, all evaluated identically with the
same repeated 5-fold × 20-repeat CV, and each logged to MLflow as its own run (params, CV
metrics, and the fitted model artifact with an inferred signature). See
`notebooks/experiments.ipynb` for the full comparison code.

| model | MAE | RMSE | R² (mean) | R² (std) |
|---|---|---|---|---|
| **gradient-boosting** (max_depth=2) | **3,200.7** | **5,135.8** | **0.081** | **2.34** |
| random-forest (max_depth=4) | 3,887.9 | 5,876.1 | -0.367 | 4.61 |
| ridge-regression (alpha=10) | 6,697.2 | 10,628.5 | -24.95 | 151.00 |
| linear-regression | 7,042.7 | 9,843.6 | -4.73 | 18.37 |

**Chosen model: `GradientBoostingRegressor(max_depth=2, n_estimators=100)`** — it has the
lowest MAE/RMSE and the least-negative, least-volatile R² of the four. The linear models
perform poorly (often strongly negative R²) because ASP does not scale additively with specs:
chips at the high end of TDP/throughput command disproportionately higher prices, a
nonlinearity that even shallow (depth-2) trees capture, without the model being "overly
complex" — no extra feature engineering, just 100 small trees. This is the model trained by
`src/train.py` and shipped via `main.py` / the CD pipeline.

**MLflow screenshots:**

> _TODO (manual step): run `mlflow ui` from the project root, open
> http://127.0.0.1:5000, select the `ai-chip-asp-regression` experiment, and paste
> screenshots here — one of the 4-run comparison table, and one of the
> `gradient-boosting` run's params/metrics/artifacts page._

## 3. Conclusions

This project builds an end-to-end training pipeline that predicts an AI accelerator's ASP from
its specs: `src/load.py` → `src/transform.py` → `src/train.py` → `src/store.py`, orchestrated
by `main.py` and retrained automatically by the CD pipeline on every push to `main`. Experiments
across 4 candidate models are tracked in MLflow (`notebooks/experiments.ipynb`), and a separate
on-demand workflow (`src/predict.py`) scores new, unreleased chips listed in
`batch_prediction_dataset/on_demand_dataset.csv` using the exact same fitted encoder as
training, writing predictions back into that folder.

The most important finding wasn't the model choice — it was the data itself: with only ~27
independent samples after fixing leakage and removing two data-quality outliers, evaluation
metrics (especially R²) are inherently noisy, and a single train/test split would have been
actively misleading (it produced a misleadingly good R²=0.71 before repeated CV exposed how
seed-dependent that number was). The final model, gradient boosting with shallow trees,
achieves MAE ≈ $3.2K and RMSE ≈ $5.1K against a target ranging roughly $8K–$65K — a reasonable
but not highly precise fit, which is an honest reflection of how little data is available
rather than a modeling shortcoming. More chips/vendors, or finer-grained specs (e.g. process
node, memory bandwidth), would be the most direct way to improve on this.
