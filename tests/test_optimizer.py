"""
Unit tests for the MILP BESS Dispatch Optimizer.
"""

import pytest
import pandas as pd
from bess_pipeline.optimization.dispatch_optimizer import BESSDispatcher

@pytest.fixture
def dummy_gold_prices():
    """Generates a 48-period mock DataFrame with extreme price volatility."""
    prices = [10.0 if i % 2 == 0 else 150.0 for i in range(1, 49)]
    
    df = pd.DataFrame({
        "settlementDate": ["2026-08-01"] * 48,
        "settlementPeriod": list(range(1, 49)),
        "interval_start_utc": pd.date_range("2026-08-01 23:00:00", periods=48, freq="30min"),
        "system_price_gbp": prices
    })
    return df

def test_optimizer_maintains_power_boundaries(dummy_gold_prices):
    dispatcher = BESSDispatcher()
    df_solved = dispatcher.optimize_schedule(dummy_gold_prices)
    
    assert df_solved["optimal_charge_mw"].max() <= 10.0, "Charge exceeded 10 MW limit."
    assert df_solved["optimal_charge_mw"].min() >= 0.0, "Charge cannot be negative."
    assert df_solved["optimal_discharge_mw"].max() <= 10.0, "Discharge exceeded 10 MW limit."
    assert df_solved["optimal_discharge_mw"].min() >= 0.0, "Discharge cannot be negative."

def test_optimizer_maintains_energy_boundaries(dummy_gold_prices):
    dispatcher = BESSDispatcher()
    df_solved = dispatcher.optimize_schedule(dummy_gold_prices)
    
    assert df_solved["state_of_energy_mwh"].max() <= 18.0, "SOC exceeded 90% (18 MWh)."
    assert df_solved["state_of_energy_mwh"].min() >= 2.0, "SOC fell below 10% (2 MWh)."

def test_mutual_exclusivity(dummy_gold_prices):
    dispatcher = BESSDispatcher()
    df_solved = dispatcher.optimize_schedule(dummy_gold_prices)
    
    for _, row in df_solved.iterrows():
        if row["optimal_charge_mw"] > 0:
            assert row["optimal_discharge_mw"] == 0.0, "Simultaneous charge/discharge."
        if row["optimal_discharge_mw"] > 0:
            assert row["optimal_charge_mw"] == 0.0, "Simultaneous charge/discharge."