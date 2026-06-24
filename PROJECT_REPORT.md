# AI Chip Price Prediction Project

## Contributors: Santiago Olivera Vaquero and Jason Jonas Smith

## What I tried to build

The goal of this project is to predict the average selling price of an AI accelerator chip from a small set of product specifications. The target column is `estimated_asp_usd`, so this is a supervised regression problem.

The dataset is `datasets/raw/ai_chip_market.csv`. It has 120 rows, covering 30 different chips from 2020 to 2026. Most rows are repeated yearly observations of the same chip, not truly new chips, so the raw row count is a little misleading. After cleaning, the model is trained on 27 independent chip examples.

I used a static CSV instead of an API. For this particular topic, that made more sense: there is not a simple free source that gives clean cross-vendor AI chip specs, launch dates, shipment estimates, and ASP values in one place. Keeping the data in the repo also makes the project easy to rerun without API keys or network access.

This is meant to be a batch ML workflow, not a real-time app. Chip launches do not happen every minute, and the likely use case is periodically retraining the model or scoring a small file of upcoming chips. Because of that, I focused on a reproducible training pipeline, saved model artifacts, MLflow experiment tracking, and an on-demand batch prediction script.

## Data cleaning and feature decisions

The first important issue was duplicate chip rows. The same chip appears once for each year that it is present in the market data, but the chip specifications and ASP are basically the same. If I trained directly on the raw rows, the same chip could appear in both training and validation folds. That would make the model look better than it really is. To avoid this, `src/transform.py` keeps only the most recent row for each `chip_name`.

I also removed columns that would either leak the answer or not generalize well:

| Column | Reason for removal |
|---|---|
| `chip_name` | Identifier, not a reusable feature |
| `launch_date` | Mostly redundant with year and too specific for such a small dataset |
| `description` | Free text, not used in this version of the model |
| `estimated_shipments_units` | Downstream business estimate, not a chip spec |
| `estimated_revenue_usd_m` | Closely tied to ASP and shipments, so it leaks target information |

Two types of invalid training rows were removed. Rows with `estimated_asp_usd == 0` were treated as unknown pricing, not free chips. Rows with `memory_gb == 0` were also removed because they represent system-level products rather than normal chip-level products. One example is the Cerebras wafer-scale system, whose price would dominate the loss in a dataset this small.

The final feature set is intentionally simple:

| Feature group | Columns |
|---|---|
| Numeric specs | `year`, `memory_gb`, `fp16_tflops`, `tdp_watts` |
| Categorical feature | `vendor`, one-hot encoded |

The fitted one-hot encoder is saved together with the model, so `src/predict.py` can transform new chips in the same way as training data. Unknown vendors are handled with `handle_unknown="ignore"`, which is useful for batch scoring because a new vendor should not crash the prediction workflow.

## Modeling approach

I compared four models in `notebooks/experiments.ipynb` and logged each one to MLflow:

| Model | MAE | RMSE | Mean R2 | R2 std |
|---|---:|---:|---:|---:|
| Gradient Boosting | 3,199.0 | 5,133.6 | 0.081 | 2.340 |
| Random Forest | 3,887.9 | 5,876.1 | -0.367 | 4.605 |
| Ridge Regression | 6,697.2 | 10,628.5 | -24.952 | 151.004 |
| Linear Regression | 7,042.7 | 9,843.6 | -4.729 | 18.373 |

I used repeated cross-validation instead of one train/test split: 5 folds repeated 20 times. This was necessary because the cleaned dataset is tiny. With only 27 usable chips, a single split can change the result a lot depending on which products land in validation. Repeating the folds gives a less fragile estimate.

The best model was `GradientBoostingRegressor(n_estimators=100, max_depth=2, random_state=42)`. It had the lowest MAE and RMSE, so it is the model used by `main.py`.

I did not choose the most complex possible model. With this little data, a bigger model would be easy to overfit. Shallow gradient boosting is a reasonable compromise: it can model nonlinear relationships between specs and price, but the trees are still constrained.

The weak R2 is also worth being honest about. The model is useful as a rough pricing estimate, but it is not a high-confidence pricing engine. The dataset is too small and the market is too messy for that. The MAE of about $3.2K is acceptable relative to prices that range from about $15K to $65K after cleaning, but there is definitely room for better data.

## MLflow evidence

The experiment is called `ai-chip-asp-regression`.

Screenshot 1: MLflow comparison table with the four completed runs.

![MLflow comparison table](screenshots/mlflow-comparison-table.png)

Screenshot 2: Detail page for the `gradient-boosting` run, showing parameters, metrics, and model artifacts.

![Gradient boosting run details](screenshots/mlflow-gradient-boosting-run.png)

The model runs were logged with parameters, metrics, tags, and a saved sklearn model artifact. The logged artifact includes the fitted model signature and an input example, which makes the run easier to inspect later.

## Pipeline and MLOps pieces

The training pipeline is split into small modules:

| File | Role |
|---|---|
| `src/load.py` | Reads the raw CSV |
| `src/transform.py` | Cleans rows, removes leakage columns, and one-hot encodes vendor |
| `src/train.py` | Trains the final gradient boosting model and evaluates it |
| `src/store.py` | Saves the trained model bundle as a timestamped `.joblib` file |
| `main.py` | Runs the full training flow |
| `src/predict.py` | Loads the latest model and scores the on-demand prediction CSV |

The saved artifact is a bundle containing both the trained model and the fitted transformer. That matters because prediction data must go through the same encoding as training data. Without saving the transformer, the batch prediction step could easily end up with mismatched columns.

There are also GitHub Actions workflows:

| Workflow | Purpose |
|---|---|
| `.github/workflows/ci.yaml` | Installs dependencies and runs the test suite |
| `.github/workflows/cd.yaml` | Runs `main.py` and commits a newly trained model artifact |

This is a simple setup, but it covers the main MLOps flow for a small project: data in the repo, deterministic preprocessing, repeatable training, experiment tracking, model persistence, tests, and a batch prediction path.

## What I would improve next

The biggest limitation is not the algorithm. It is the amount and quality of data. A better version of this project would add more chips, more vendors, and more technical features such as memory bandwidth, process node, die area, interconnect type, and whether the product is sold as a card, chip, or full system.

I would also separate system-level products from chip-level products more formally. Right now those rows are removed with simple rules, which is fine for this dataset, but a larger dataset should have an explicit product category column.

Finally, I would add a small model registry step or a clearer promotion rule. At the moment, the CD workflow retrains and saves a new model, but it does not decide whether the new model is actually better than the previous one. A practical next step would be to compare the new MLflow run against the current production model before saving or promoting it.

## Conclusion

The final project is a compact batch ML system for estimating AI chip ASP from product specs. Gradient boosting performed best among the tested models, with an MAE around $3.2K. The result is useful as a rough estimate, but the evaluation also shows the main reality of the project: with only 27 cleaned examples, model choice matters less than getting more reliable data.

Even so, the pipeline is structured in a way that can grow. More data could be added to the CSV, the same training command could be rerun, MLflow would track the new experiment results, and the prediction script would continue using the saved transformer/model bundle for on-demand scoring.
