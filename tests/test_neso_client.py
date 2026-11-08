"""NESO N2EX parsing tests (offline, synthetic CSV in the real dataset's format)."""

import pandas as pd
import pytest

from bess_pipeline.ingestion.neso_client import parse_n2ex_csv

CSV = """Date,Delivery Period,Price
2026-09-20,22:00 - 23:00,129.95
2026-09-20,23:00 - 00:00,-5.5
2026-09-21,00:00 - 01:00,100.0
"""


def test_hourly_prices_become_two_half_hours_in_utc():
    df = parse_n2ex_csv(CSV)
    assert len(df) == 6
    assert df["interval_start_utc"].iloc[0] == pd.Timestamp("2026-09-20 22:00", tz="UTC")
    assert list(df["da_price_gbp"]) == [129.95, 129.95, -5.5, -5.5, 100.0, 100.0]
    assert (df["interval_start_utc"].diff().dropna() == pd.Timedelta(minutes=30)).all()


def test_rejects_unexpected_columns():
    with pytest.raises(ValueError):
        parse_n2ex_csv("a,b\n1,2\n")
