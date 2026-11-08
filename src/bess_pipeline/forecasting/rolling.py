"""
Multi-day rolling dispatch simulation with state of charge carried between days.

Each day is optimized on a DECISION price (a forecast, or the actual price for the
perfect-foresight bound), starting from the previous day's end state of energy, and
is then SETTLED on the actual price including every modelled cost:

    net = revenue - charging cost - trading fees - degradation

``lookahead=True`` (perfect foresight only) optimizes day D together with day D+1 and
commits day D, so the bound is not artificially cut off at midnight.
"""

import logging

import pandas as pd

from bess_pipeline.optimization.dispatch_optimizer import BESSDispatcher

logger = logging.getLogger(__name__)
HOURS = 0.5


def settle_schedule(sched: pd.DataFrame, price: pd.Series, dispatcher: BESSDispatcher) -> dict:
    """Money and energy for one committed schedule, valued at ``price`` (aligned to sched rows)."""
    dis, ch = sched["optimal_discharge_mw"], sched["optimal_charge_mw"]
    revenue = float((dis * price * HOURS).sum())
    cost = float((ch * price * HOURS).sum())
    export = float((dis * HOURS).sum())
    imp = float((ch * HOURS).sum())
    fees = dispatcher.trading_fee * (export + imp)
    degr = dispatcher.degradation_cost * export
    usable = dispatcher.max_soc_mwh - dispatcher.min_soc_mwh
    return {
        "revenue_gbp": revenue,
        "charging_cost_gbp": cost,
        "trading_fees_gbp": fees,
        "degradation_gbp": degr,
        "net_profit_gbp": revenue - cost - fees - degr,
        "export_mwh": export,
        "import_mwh": imp,
        "efc": export / usable if usable > 0 else 0.0,
    }


def simulate(
    dispatcher: BESSDispatcher,
    days: list[pd.DataFrame],
    decision_col: str,
    settle_col: str = "mid_price_gbp",
    carry_soc: bool = True,
    lookahead: bool = False,
    start_soc_mwh: float | None = None,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """
    Args:
        days: consecutive UK days, each with ['interval_start_utc', decision_col, settle_col].
        carry_soc: start each day at the previous day's end SOC (else at the minimum SOC).
        lookahead: optimize day D together with day D+1 (needs decision_col on D+1).
    Returns (per-day results, committed schedules per day).
    """
    soc = dispatcher.min_soc_mwh if start_soc_mwh is None else start_soc_mwh
    rows, schedules = [], []
    for i, day in enumerate(days):
        day = day.sort_values("interval_start_utc").reset_index(drop=True)
        horizon = day
        if lookahead and i + 1 < len(days):
            horizon = pd.concat([day, days[i + 1].sort_values("interval_start_utc")], ignore_index=True)
        sched = dispatcher.optimize_schedule(horizon, price_col=decision_col, initial_soc_mwh=soc)
        if sched.empty:
            logger.error("Solver failed on day index %d; stopping simulation.", i)
            break
        sched = sched.iloc[: len(day)].reset_index(drop=True)
        sched[settle_col] = day[settle_col].to_numpy()
        row = settle_schedule(sched, sched[settle_col], dispatcher)
        row["start_soc_mwh"] = soc
        soc = float(sched["state_of_energy_mwh"].iloc[-1]) if carry_soc else dispatcher.min_soc_mwh
        row["end_soc_mwh"] = float(sched["state_of_energy_mwh"].iloc[-1])
        rows.append(row)
        schedules.append(sched)
    return pd.DataFrame(rows), schedules
