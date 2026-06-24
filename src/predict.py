from pathlib import Path

import joblib
import pandas as pd

from metadata import (
    BATCH_PREDICTION_FOLDER,
    MODELS_FOLDER,
    ON_DEMAND_INPUT_FILE,
    ON_DEMAND_OUTPUT_FILE,
    TARGET_COLUMN,
)


def load_latest_model(models_folder: str = MODELS_FOLDER) -> dict:
    model_files = sorted(Path(models_folder).glob("*.joblib"))
    if not model_files:
        raise FileNotFoundError(
            f"No trained model found in {models_folder}/. Run main.py first."
        )
    return joblib.load(model_files[-1])


def predict(
    input_file: str = ON_DEMAND_INPUT_FILE,
    output_file: str = ON_DEMAND_OUTPUT_FILE,
) -> pd.DataFrame:
    artifact = load_latest_model()
    model = artifact["model"]
    transformer = artifact["transformer"]

    raw_df = pd.read_csv(f"{BATCH_PREDICTION_FOLDER}/{input_file}")

    # Reuses the encoder fitted during training (see src/transform.py) so
    # the on-demand input goes through the exact same encoding.
    features_df = transformer.transform(raw_df)
    features_df = features_df.drop(columns=[TARGET_COLUMN], errors="ignore")

    predictions = model.predict(features_df)

    output_df = raw_df.copy()
    output_df[f"predicted_{TARGET_COLUMN}"] = predictions

    output_path = f"{BATCH_PREDICTION_FOLDER}/{output_file}"
    output_df.to_csv(output_path, index=False)
    print(f"Predictions stored as: {output_path}")
    return output_df


if __name__ == "__main__":
    predict()
