"""
Dispatch strategies compared in the pipeline and dashboard.

Settlement basis matters, so each strategy states it:

* ``mid_baseline``       optimize on the published MID price, settle at MID
                         (day-ahead arbitrage with the auction result known).
* ``mid_forecast``       optimize on OUR forecast of MID made before the auction,
                         settle at the actual MID. Value of the model = how close
                         this gets to ``mid_baseline``.
* ``imbalance_forecast`` (experimental) optimize on a forecast of the imbalance
                         price, settle at the actual imbalance price.
* ``imbalance_perfect``  (experimental) optimize on actual imbalance prices; an
                         unattainable upper bound for the imbalance market.
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
    mid_forecast: pd.DataFrame | None = None,
    imbalance_forecast: pd.DataFrame | None = None,
) -> dict:
    """
    Args:
        day: one UK day with ['interval_start_utc', 'system_price_gbp', 'mid_price_gbp'].
    Returns {strategy_name: (BESSFinancialMetrics, schedule_df)} for strategies that solved.
    """
    out: dict = {}
    mid_day = _with_settlement(day, "mid_price_gbp")
    out["mid_baseline"] = settle_day(dispatcher, mid_day, None, usable_capacity_mwh)
    if mid_forecast is not None:
        out["mid_forecast"] = settle_day(dispatcher, mid_day, mid_forecast, usable_capacity_mwh)
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
