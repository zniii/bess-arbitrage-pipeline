"""
Dispatch strategies compared in the daily job.

Prices are the N2EX GB day-ahead auction (NESO) unless stated. The auction for D clears on D-1,
so a plan made at D-1 08:00 UTC must be based on a FORECAST of it.

* ``da_perfect``         optimize on the actual day-ahead prices, settle at them (upper bound).
* ``da_forecast``        optimize on OUR forecast made at D-1 08:00 UTC, settle at the actual
                         day-ahead price. Headline strategy.
* ``imbalance_forecast`` (experimental) optimize on a forecast of the imbalance price, settle at it.
* ``imbalance_perfect``  (experimental) actual imbalance prices; unattainable upper bound.

These per-day strategies reset the battery daily and ignore costs. The cost-aware, SOC-carrying
strategy lives in ``rolling.py`` and feeds the Max Revenue page.
"""

import pandas as pd

from bess_pipeline.forecasting.backtest import settle_day
from bess_pipeline.optimization.dispatch_optimizer import BESSDispatcher


def _with_settlement(day: pd.DataFrame, col: str) -> pd.DataFrame:
    d = day.copy()
    d["system_price_gbp"] = d[col]
    return d


def run_strategies(
    dispatcher: BESSDispatcher,
    day: pd.DataFrame,
    usable_capacity_mwh: float,
    da_forecast: pd.DataFrame | None = None,
    imbalance_forecast: pd.DataFrame | None = None,
) -> dict:
    """
    Args:
        day: one UK day with ['interval_start_utc', 'system_price_gbp', 'da_price_gbp'].
    Returns {strategy_name: (BESSFinancialMetrics, schedule_df)} for strategies that solved.
    """
    out: dict = {}
    da_day = _with_settlement(day, "da_price_gbp")
    out["da_perfect"] = settle_day(dispatcher, da_day, None, usable_capacity_mwh)
    if da_forecast is not None:
        out["da_forecast"] = settle_day(dispatcher, da_day, da_forecast, usable_capacity_mwh)
    out["imbalance_perfect"] = settle_day(dispatcher, day, None, usable_capacity_mwh)
    if imbalance_forecast is not None:
        out["imbalance_forecast"] = settle_day(dispatcher, day, imbalance_forecast, usable_capacity_mwh)
    result = {}
    actual = day.set_index("interval_start_utc")["system_price_gbp"]
    for name, (kpi, sched) in out.items():
        if kpi is None:
            continue
        sched = sched.copy()
        sched["settlement_price_gbp"] = sched["system_price_gbp"]  # price this strategy is paid at
        sched["system_price_gbp"] = sched["interval_start_utc"].map(actual).to_numpy()  # actual imbalance price
        sched["strategy"] = name
        result[name] = (kpi, sched)
    return result
