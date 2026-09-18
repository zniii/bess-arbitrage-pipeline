"""
Tests for ElexonClient.
"""

import pytest
import pandas as pd
from bess_pipeline.ingestion.elexon_client import ElexonClient


def test_client_init():
    client = ElexonClient(timeout=10, max_retries=2)
    assert client.timeout == 10
    assert client.max_retries == 2
    assert "Accept" in client.session.headers


def test_invalid_date_range_raises_error():
    client = ElexonClient()
    # Match the exact exception string
    with pytest.raises(ValueError, match=r"start_date must be strictly before end_date"):
        client.fetch_system_prices("2026-08-10", "2026-08-01")


def test_live_small_window_fetch():
    """Validates the live endpoint returns expected schema fields."""
    client = ElexonClient()
    # Query a known past day window (2026-08-01 to 2026-08-02)
    df = client.fetch_system_prices("2026-08-01", "2026-08-02")

    assert isinstance(df, pd.DataFrame)
    assert not df.empty

    # Elexon Insights schema assertions
    assert "settlementDate" in df.columns
    assert "settlementPeriod" in df.columns
    # Insights returns 'price' or 'systemSellPrice'/'systemBuyPrice'
    price_cols = [c for c in df.columns if "price" in c.lower()]
    assert len(price_cols) > 0, f"Expected price column in: {df.columns.tolist()}"