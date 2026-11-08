"""
Walk-forward backtest: forecast -> optimize on forecast -> settle on actual.

For each target day the forecaster only sees history available at D-1 (enforced
inside ``model._available``). Reports forecast error and the commercial value of
each forecaster versus the perfect-foresight upper bound.
"""

import logging

import pandas as pd

from bess_pipeline.analytics.financial_kpis import calculate_dispatch_financial_kpis
from bess_pipeline.forecasting.model import day_timestamps
from bess_pipeline.optimization.dispatch_optimizer import BESSDispatcher

logger = logging.getLogger(__name__)


def settle_day(
    dispatcher: BESSDispatcher,
    actual_day: pd.DataFrame,
    forecast_day: pd.DataFrame | None,
    usable_capacity_mwh: float,
):
    """Optimize on forecast (or on actuals if forecast is None), settle on actuals."""
    day = actual_day.copy()
    day["interval_start_utc"] = pd.to_datetime(day["interval_start_utc"], utc=True)
    if forecast_day is None:
        price_col = "system_price_gbp"
    else:
        fc = forecast_day.copy()
        fc["interval_start_utc"] = pd.to_datetime(fc["interval_start_utc"], utc=True)
        day = day.merge(fc, on="interval_start_utc", how="left")
        price_col = "price_forecast"
    schedule = dispatcher.optimize_schedule(day, price_col=price_col)
    if schedule.empty:
        return None, schedule
    return calculate_dispatch_financial_kpis(schedule, usable_capacity_mwh=usable_capacity_mwh), schedule


def walk_forward_backtest(
    history: pd.DataFrame,
    dates: list[str],
    forecasters: dict,
    dispatcher: BESSDispatcher | None = None,
    retrain_every_days: int = 7,
    exogenous: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Returns one row per (date, forecaster) with mae, profit and perfect-foresight profit.
    ``forecasters`` maps name -> object with fit(history) / predict(date, history).
    """
    dispatcher = dispatcher or BESSDispatcher()
    usable = dispatcher.max_soc_mwh - dispatcher.min_soc_mwh
    hist = history.copy()
    hist["interval_start_utc"] = pd.to_datetime(hist["interval_start_utc"], utc=True)
    rows = []
    last_fit: dict[str, pd.Timestamp] = {}

    for date in dates:
        ts = day_timestamps(date)
        actual = hist[hist["interval_start_utc"].isin(ts)]
        if actual.empty:
            continue
        perfect, _ = settle_day(dispatcher, actual, None, usable)
        for name, fcster in forecasters.items():
            day_start = ts[0]
            if name not in last_fit or (day_start - last_fit[name]).days >= retrain_every_days:
                lag = getattr(fcster, "min_lag_days", 2)
                train = hist[hist["interval_start_utc"] < day_start - pd.Timedelta(days=lag - 1)]
                fcster.fit(train, exogenous) if exogenous is not None else fcster.fit(train)
                last_fit[name] = day_start
            fc = fcster.predict(date, hist, exogenous) if exogenous is not None else fcster.predict(date, hist)
            m = actual.merge(fc, on="interval_start_utc", how="left")
            tcol = getattr(fcster, "target_col", "system_price_gbp")
            mae = float((m[tcol] - m["price_forecast"]).abs().mean())
            kpi, _ = settle_day(dispatcher, actual, fc, usable)
            rows.append({
                "date": date,
                "forecaster": name,
                "mae_gbp_per_mwh": round(mae, 3),
                "profit_gbp": kpi.net_profit_gbp if kpi else 0.0,
                "perfect_foresight_profit_gbp": perfect.net_profit_gbp if perfect else 0.0,
            })
    return pd.DataFrame(rows)
