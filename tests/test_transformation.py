"""
Unit tests for the Silver Layer temporal transformations.
"""

import pandas as pd
import pytest
from bess_pipeline.transformations.temporal import standardize_system_prices


@pytest.fixture
def raw_bronze_dataframe():
    """Mocks the exact schema returned by the Elexon System Prices endpoint."""
    return pd.DataFrame({
        "settlementDate": ["2026-08-01", "2026-08-01"],
        "settlementPeriod": [1, 2],
        "startTime": ["2026-08-01T23:00:00Z", "2026-08-01T23:30:00Z"],
        "systemSellPrice": [45.50, 48.20],
        "systemBuyPrice": [45.50, 48.20],
        "netImbalanceVolume": [-120.5, 85.0],
    })


def test_standardize_system_prices(raw_bronze_dataframe):
    df_silver = standardize_system_prices(raw_bronze_dataframe)

    expected_cols = {
        "settlementDate",
        "settlementPeriod",
        "system_price_gbp",
        "interval_start_utc",
        "interval_end_utc",
    }
    assert expected_cols.issubset(df_silver.columns), "Missing standardized columns."

    assert df_silver["system_price_gbp"].iloc[0] == 45.50
    assert df_silver["settlementPeriod"].iloc[1] == 2

    delta = df_silver["interval_end_utc"].iloc[0] - df_silver["interval_start_utc"].iloc[0]
    assert delta == pd.Timedelta(minutes=30), "Interval duration is not exactly 30 minutes."


def test_standardize_empty_dataframe():
    """Ensure the pipeline safely handles empty datasets without crashing."""
    df_empty = pd.DataFrame()
    df_result = standardize_system_prices(df_empty)

    assert df_result.empty, "Function should return an empty DataFrame when provided empty input."