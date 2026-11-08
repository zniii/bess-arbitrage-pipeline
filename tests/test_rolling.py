"""Cost-aware, SOC-carrying dispatch tests on SYNTHETIC prices (no network)."""

import numpy as np
import pandas as pd

from bess_pipeline.forecasting.rolling import simulate
from bess_pipeline.optimization.dispatch_optimizer import BESSDispatcher


def make_days(n=4, spread=60.0, base=100.0):
    days = []
    for d in range(n):
        idx = pd.date_range(pd.Timestamp("2026-09-01", tz="UTC") + pd.Timedelta(days=d), periods=48, freq="30min")
        tod = np.arange(48)
        price = base + spread * np.sin(2 * np.pi * (tod - 12) / 48)
        days.append(pd.DataFrame({"interval_start_utc": idx, "mid_price_gbp": price, "fc": price}))
    return days


def test_soc_is_carried_between_days():
    d = BESSDispatcher(end_soc_min_mwh=8.0)
    res, scheds = simulate(d, make_days(), "fc")
    assert (res["end_soc_mwh"] >= 8.0 - 1e-6).all()           # reserve respected
    assert np.allclose(res["start_soc_mwh"].iloc[1:].to_numpy(), res["end_soc_mwh"].iloc[:-1].to_numpy())
    assert all((s["state_of_energy_mwh"] <= 18.0 + 1e-6).all() and (s["state_of_energy_mwh"] >= 2.0 - 1e-6).all()
               for s in scheds)


def test_costs_reduce_profit_and_stop_trading_below_breakeven():
    days = make_days(spread=60.0)
    free = simulate(BESSDispatcher(), days, "fc")[0]["net_profit_gbp"].sum()
    costly = simulate(BESSDispatcher(degradation_cost_gbp_per_mwh=15.0, trading_fee_gbp_per_mwh=1.0), days, "fc")[0]
    assert costly["net_profit_gbp"].sum() < free
    # spread far below cost + efficiency loss: the battery should not cycle
    flat = make_days(spread=3.0)
    res = simulate(BESSDispatcher(degradation_cost_gbp_per_mwh=15.0), flat, "fc")[0]
    assert res["export_mwh"].sum() < 1e-6


def test_perfect_foresight_not_worse_than_misled_forecast():
    days = make_days()
    for dd in days:
        dd["bad"] = dd["fc"].to_numpy()[::-1]
    d = BESSDispatcher(degradation_cost_gbp_per_mwh=8.0, end_soc_min_mwh=4.0)
    perfect = simulate(d, days, "mid_price_gbp", lookahead=True)[0]["net_profit_gbp"].sum()
    misled = simulate(d, days, "bad")[0]["net_profit_gbp"].sum()
    assert perfect >= misled - 1e-6


def test_net_profit_accounting_identity():
    d = BESSDispatcher(degradation_cost_gbp_per_mwh=8.0, trading_fee_gbp_per_mwh=0.5)
    r = simulate(d, make_days(2), "fc")[0]
    expected = r["revenue_gbp"] - r["charging_cost_gbp"] - r["trading_fees_gbp"] - r["degradation_gbp"]
    assert np.allclose(r["net_profit_gbp"], expected)
