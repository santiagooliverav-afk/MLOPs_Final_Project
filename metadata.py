DATASETS_FOLDER = "datasets/raw"
RAW_DATASET_FILE = "ai_chip_market.csv"

MODELS_FOLDER = "models"
MODEL_NAME = "asp-regressor"

BATCH_PREDICTION_FOLDER = "batch_prediction_dataset"
ON_DEMAND_INPUT_FILE = "on_demand_dataset.csv"
ON_DEMAND_OUTPUT_FILE = "on_demand_predictions.csv"

# A chip's specs and ASP are reported once per launch but repeated for every
# year it stayed on the market, so duplicates are collapsed to one row per
# chip before training (see src/transform.py) to avoid train/test leakage.
ID_COLUMN = "chip_name"
TARGET_COLUMN = "estimated_asp_usd"

# Dropped because they are either identifiers/free text or downstream of the
# target (estimated_revenue_usd_m and estimated_shipments_units are priced
# off estimated_asp_usd, so keeping them would leak the target).
COLUMNS_TO_DROP = [
    "launch_date",
    "description",
    "estimated_shipments_units",
    "estimated_revenue_usd_m",
]

NUMERIC_FEATURES = ["memory_gb", "fp16_tflops", "tdp_watts", "year"]
ONE_HOT_ENCODE_COLUMNS = ["vendor"]

# GradientBoostingRegressor(max_depth=2) had the best and most stable
# repeated-CV score among LinearRegression/Ridge/RandomForest/GradientBoosting
# (see notebooks/experiments.ipynb) -- shallow trees suit the ~27-row dataset
# better than linear models, which assume additive effects that don't hold
# for this nonlinear spec-to-price relationship.
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 2,
    "random_state": 42,
}

CV_FOLDS = 5
CV_REPEATS = 20
RANDOM_STATE = 42
