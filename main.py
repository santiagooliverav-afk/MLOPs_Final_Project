from src.load import load_data
from src.store import store_model
from src.train import train_model
from src.transform import Transformer
from metadata import MODEL_NAME, RAW_DATASET_FILE, TARGET_COLUMN


def main():
    df = load_data(file_name=RAW_DATASET_FILE)

    transformer = Transformer()
    df = transformer.fit_transform(df, target_column=TARGET_COLUMN)

    model, metrics = train_model(df=df, target_column=TARGET_COLUMN)
    print(f"Cross-validated metrics: {metrics}")

    # Bundle the fitted transformer with the model so the on-demand
    # workflow applies the exact same encoding at prediction time.
    store_model(
        model={"model": model, "transformer": transformer}, model_name=MODEL_NAME
    )


# This allows running this code only when main.py is executed directly,
# not when it's imported (e.g. from tests).
if __name__ == "__main__":
    main()
