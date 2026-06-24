import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from metadata import ID_COLUMN, COLUMNS_TO_DROP, ONE_HOT_ENCODE_COLUMNS


class Transformer:
    """Cleans raw ai_chip_market rows into model-ready numeric features.

    The one-hot encoder is fit once on training data (fit_transform) and
    reused as-is in the on-demand workflow (transform), so a chip from a
    vendor never seen during training doesn't change the model's input
    schema (handle_unknown="ignore" zero-fills it instead of raising).
    """

    def __init__(self):
        self.id_column = ID_COLUMN
        self.drop_columns = COLUMNS_TO_DROP
        self.one_hot_encoding_columns = ONE_HOT_ENCODE_COLUMNS
        self.encoder = OneHotEncoder(
            drop="first", sparse_output=False, handle_unknown="ignore"
        ).set_output(transform="pandas")

    def fit_transform(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        df = self._prepare(df)
        df = self._drop_invalid_rows(df, target_column)
        encoded_df = self.encoder.fit_transform(df[self.one_hot_encoding_columns])
        return self._merge_encoded(df, encoded_df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._prepare(df)
        encoded_df = self.encoder.transform(df[self.one_hot_encoding_columns])
        return self._merge_encoded(df, encoded_df)

    def _drop_invalid_rows(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        # memory_gb == 0 marks system-level products (e.g. Cerebras'
        # wafer-scale systems) rather than a single chip, and a target of
        # 0 marks undisclosed pricing -- neither is a usable training
        # example, and the $2.5M wafer-scale system would otherwise
        # dominate the loss for a 30-row dataset.
        valid_rows = (df["memory_gb"] > 0) & (df[target_column] > 0)
        return df[valid_rows].reset_index(drop=True)

    def drop_duplicate_chips(self, df: pd.DataFrame) -> pd.DataFrame:
        # Specs/ASP are constant per chip across years, so keep only the
        # most recent year's row per chip to avoid train/test leakage.
        return (
            df.sort_values("year")
            .drop_duplicates(subset=self.id_column, keep="last")
            .reset_index(drop=True)
        )

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.drop_duplicate_chips(df)
        columns_to_drop = [
            column
            for column in [self.id_column, *self.drop_columns]
            if column in df.columns
        ]
        return df.drop(columns=columns_to_drop)

    def _merge_encoded(
        self, df: pd.DataFrame, encoded_df: pd.DataFrame
    ) -> pd.DataFrame:
        df = df.drop(columns=self.one_hot_encoding_columns)
        return pd.concat(
            [df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1
        )
