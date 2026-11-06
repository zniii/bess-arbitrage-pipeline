"""
Leak-free exogenous inputs (wind and demand forecasts) for day-ahead models.

For delivery day D the decision is taken on D-1 at ``DECISION_HOUR_UTC`` (08:00 UTC,
i.e. before the ~09:20 UK day-ahead auction deadline in both GMT and BST). Only
forecast publications with ``publishTime <= decision time`` are used, taking the
latest such publication for each target period. A test asserts this.
"""

import pandas as pd

DECISION_HOUR_UTC = 8


def _uk_date(ts: pd.Series | pd.DatetimeIndex) -> pd.Series | pd.DatetimeIndex:
    """UK (Europe/London) calendar date as naive midnight; settlement days are UK days."""
    if isinstance(ts, pd.DatetimeIndex):
        return ts.tz_convert("Europe/London").normalize().tz_localize(None)
    return ts.dt.tz_convert("Europe/London").dt.normalize().dt.tz_localize(None)


def _decision_time(uk_date: pd.Series) -> pd.Series:
    """D-1 08:00 UTC for UK delivery date D (naive midnight dates in)."""
    return (uk_date - pd.Timedelta(days=1) + pd.Timedelta(hours=DECISION_HOUR_UTC)).dt.tz_localize("UTC")


def build_exogenous(
    timestamps: pd.DatetimeIndex,
    wind: pd.DataFrame,
    demand: pd.DataFrame,
) -> pd.DataFrame:
    """
    Args:
        timestamps: UTC half-hour interval starts to build inputs for.
        wind: ['publishTime', 'startTime', 'generation'] (hourly WINDFOR publications).
        demand: ['publishTime', 'forecastDate', 'demand'] (daily NDFD publications).

    Returns ['interval_start_utc', 'wind_forecast_mw', 'demand_forecast_mw'] (NaN if no
    publication existed before the cutoff).
    """
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    out = pd.DataFrame({"interval_start_utc": ts})

    if not wind.empty:
        w = wind.copy()
        w["publishTime"] = pd.to_datetime(w["publishTime"], utc=True)
        w["startTime"] = pd.to_datetime(w["startTime"], utc=True)
        w = w[w["publishTime"] <= _decision_time(_uk_date(w["startTime"]))]
        wl = w.sort_values("publishTime").groupby("startTime")["generation"].last()
        out["wind_forecast_mw"] = wl.reindex(ts.floor("h")).to_numpy()
    else:
        out["wind_forecast_mw"] = float("nan")

    if not demand.empty:
        d = demand.copy()
        d["publishTime"] = pd.to_datetime(d["publishTime"], utc=True)
        d["forecastDate"] = pd.to_datetime(d["forecastDate"])  # UK settlement date, naive
        d = d[d["publishTime"] <= _decision_time(d["forecastDate"])]
        dl = d.sort_values("publishTime").groupby("forecastDate")["demand"].last()
        out["demand_forecast_mw"] = dl.reindex(_uk_date(ts)).to_numpy()
    else:
        out["demand_forecast_mw"] = float("nan")
    return out
