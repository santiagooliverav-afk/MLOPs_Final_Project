import pandas as pd

from src.transform import Transformer


def make_raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": [2020, 2021, 2022],
            "chip_name": ["ChipA", "ChipA", "ChipB"],
            "vendor": ["NVIDIA", "NVIDIA", "AMD"],
            "launch_date": ["2019-01-01", "2019-01-01", "2021-06-01"],
            "memory_gb": [32, 32, 16],
            "fp16_tflops": [100, 100, 50],
            "tdp_watts": [300, 300, 200],
            "estimated_shipments_units": [100, 200, 50],
            "estimated_asp_usd": [15000, 15000, 8000],
            "estimated_revenue_usd_m": [1.5, 3.0, 0.4],
            "description": ["desc-a", "desc-a", "desc-b"],
        }
    )


def test_drop_duplicate_chips_keeps_latest_year():
    transformer = Transformer()
    df = make_raw_df()

    deduped = transformer.drop_duplicate_chips(df)

    assert len(deduped) == 2
    chip_a_year = deduped.loc[deduped["chip_name"] == "ChipA", "year"].iloc[0]
    assert chip_a_year == 2021


def test_fit_transform_drops_id_and_leakage_columns():
    transformer = Transformer()
    df = make_raw_df()

    transformed = transformer.fit_transform(df, target_column="estimated_asp_usd")

    leakage_columns = [
        "chip_name",
        "launch_date",
        "description",
        "estimated_shipments_units",
        "estimated_revenue_usd_m",
    ]
    for column in leakage_columns:
        assert column not in transformed.columns


def test_fit_transform_one_hot_encodes_vendor():
    transformer = Transformer()
    df = make_raw_df()

    transformed = transformer.fit_transform(df, target_column="estimated_asp_usd")

    assert "vendor" not in transformed.columns
    assert any(column.startswith("vendor_") for column in transformed.columns)


def test_fit_transform_drops_undisclosed_pricing_and_system_level_rows():
    transformer = Transformer()
    df = make_raw_df()
    df.loc[df["chip_name"] == "ChipB", "estimated_asp_usd"] = 0

    transformed = transformer.fit_transform(df, target_column="estimated_asp_usd")

    assert len(transformed) == 1
    assert (transformed["estimated_asp_usd"] > 0).all()


def test_transform_reuses_fitted_encoder_for_unseen_vendor():
    transformer = Transformer()
    transformer.fit_transform(make_raw_df(), target_column="estimated_asp_usd")

    new_chip = pd.DataFrame(
        {
            "year": [2026],
            "chip_name": ["ChipC"],
            "vendor": ["UnseenVendor"],
            "launch_date": ["2026-01-01"],
            "memory_gb": [64],
            "fp16_tflops": [200],
            "tdp_watts": [400],
        }
    )

    transformed = transformer.transform(new_chip)

    assert "chip_name" not in transformed.columns
    assert len(transformed) == 1
