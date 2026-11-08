"""
Leak-free feature engineering for day-ahead GB price forecasting.

Information rule (strict):
    A forecast for delivery day D is produced on the morning of D-1, before the
    day-ahead auction closes. Imbalance (system) prices are only known after each
    settlement period, so prices are available up to the end of D-2 and every
    price feature lags >= 2 days (``min_lag_days=2``). The same holds for MID: it is a
    market index that Elexon publishes as each period happens (NOT the day-ahead
    auction result), so only part of D-1 exists at 08:00 UTC. Exogenous
    forecasts (wind, demand) must already be filtered to what was published before
    the cutoff (see ``exogenous.py``).

Lags are applied in UTC time (not by row offset), so 46/50-period DST days
cannot misalign features.
"""

import warnings

import numpy as np
import pandas as pd

MIN_LAG_DAYS = 2  # imbalance-price default
PRICE_LAGS_DAYS = (1, 2, 3, 7, 14)
EXOGENOUS_COLUMNS = ("wind_forecast_mw", "demand_forecast_mw", "carbon_intensity_forecast")

CALENDAR_COLUMNS = ["tod_sin", "tod_cos", "dow", "is_weekend", "month"]
ROLLING_COLUMNS = ["same_period_mean_7d", "same_period_std_7d", "trailing_mean_24h", "trailing_std_24h"]


def _price_series(history: pd.DataFrame, target_col: str = "system_price_gbp") -> pd.Series:
    s = history.set_index(pd.to_datetime(history["interval_start_utc"], utc=True))[
        target_col
    ].astype(float)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def build_features(
    history: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    exogenous: pd.DataFrame | None = None,
    target_col: str = "system_price_gbp",
    min_lag_days: int = MIN_LAG_DAYS,
) -> pd.DataFrame:
    """
    Build the feature matrix for the given target half-hour timestamps.

    Args:
        history: DataFrame with ['interval_start_utc', target_col].
        timestamps: UTC interval starts to build features for.
        exogenous: optional frame indexed/keyed by 'interval_start_utc' with any of
            EXOGENOUS_COLUMNS (must contain only forecasts available at D-1 09:20).
    """
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    s = _price_series(history, target_col)
    out = pd.DataFrame(index=ts)

    local = ts.tz_convert("Europe/London")
    tod = local.hour * 2 + local.minute // 30  # 0..47 local clock slot
    out["tod_sin"] = np.sin(2 * np.pi * tod / 48)
    out["tod_cos"] = np.cos(2 * np.pi * tod / 48)
    out["dow"] = local.dayofweek
    out["is_weekend"] = (local.dayofweek >= 5).astype(int)
    out["month"] = local.month

    lags = [d for d in PRICE_LAGS_DAYS if d >= min_lag_days]
    for d in lags:
        out[f"price_lag_{d}d"] = s.reindex(ts - pd.Timedelta(days=d)).to_numpy()

    same_period = np.vstack(
        [s.reindex(ts - pd.Timedelta(days=d)).to_numpy() for d in range(min_lag_days, min_lag_days + 7)]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns early in history
        out["same_period_mean_7d"] = np.nanmean(same_period, axis=0)
        out["same_period_std_7d"] = np.nanstd(same_period, axis=0)

    roll_mean = s.rolling("1D").mean()
    roll_std = s.rolling("1D").std()
    anchor = ts - pd.Timedelta(days=min_lag_days)
    # as-of lookup: latest rolling value at or before the anchor
    out["trailing_mean_24h"] = roll_mean.reindex(roll_mean.index.union(anchor)).ffill().reindex(anchor).to_numpy()
    out["trailing_std_24h"] = roll_std.reindex(roll_std.index.union(anchor)).ffill().reindex(anchor).to_numpy()

    cols = CALENDAR_COLUMNS + [f"price_lag_{d}d" for d in lags] + ROLLING_COLUMNS
    if exogenous is not None and not exogenous.empty:
        ex = exogenous.copy()
        if "interval_start_utc" in ex.columns:
            ex = ex.set_index(pd.to_datetime(ex["interval_start_utc"], utc=True))
        for c in EXOGENOUS_COLUMNS:
            if c in ex.columns:
                out[c] = ex[c].reindex(ts).to_numpy()
                cols.append(c)
    out.attrs["feature_columns"] = cols
    return out[cols]
