import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold, cross_val_score

from metadata import CV_FOLDS, CV_REPEATS, MODEL_PARAMS, RANDOM_STATE


def train_model(df: pd.DataFrame, target_column: str):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    model = GradientBoostingRegressor(**MODEL_PARAMS)

    # Only ~27 independent samples (one per chip) after dedup, so a single
    # K-fold split is highly seed-sensitive (R2 swung from -5.7 to +0.81 in
    # exploration just by changing the CV seed). Repeating the K-fold split
    # many times and averaging gives a far more honest generalization
    # estimate than any single split.
    cv = RepeatedKFold(
        n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE
    )
    metrics = evaluate_model(model, X, y, cv)

    model.fit(X, y)
    return model, metrics


def evaluate_model(model, X: pd.DataFrame, y: pd.Series, cv) -> dict:
    neg_mae = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
    neg_rmse = cross_val_score(
        model, X, y, cv=cv, scoring="neg_root_mean_squared_error"
    )
    r2 = cross_val_score(model, X, y, cv=cv, scoring="r2")

    return {
        "mae": float(-neg_mae.mean()),
        "rmse": float(-neg_rmse.mean()),
        "r2": float(r2.mean()),
        "r2_std": float(r2.std()),
    }
