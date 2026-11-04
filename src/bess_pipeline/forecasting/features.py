"""
Leak-free feature engineering for day-ahead GB price forecasting.

Information rule (strict):
    A forecast for delivery day D is produced on the morning of D-1, before the
    day-ahead auction closes. System (imbalance) prices are only known after
    each settlement period, so at that moment prices are available up to the
    end of D-2. Every price-derived feature therefore uses a lag of at least
    ``MIN_LAG_DAYS`` = 2 days. Exogenous forecasts (wind, demand, ...) are
    forward-looking by nature and may be passed as extra columns.

Lags are applied in UTC time (not by row offset), so 46/50-period DST days
cannot misalign features.
"""

import warnings

import numpy as np
import pandas as pd

MIN_LAG_DAYS = 2
PRICE_LAGS_DAYS = (2, 3, 7, 14)
EXOGENOUS_COLUMNS = ("wind_forecast_mw", "demand_forecast_mw", "carbon_intensity_forecast")

FEATURE_COLUMNS_BASE = [
    "tod_sin",
    "tod_cos",
    "dow",
    "is_weekend",
    "month",
    *[f"price_lag_{d}d" for d in PRICE_LAGS_DAYS],
    "same_period_mean_7d",
    "same_period_std_7d",
    "trailing_mean_24h",
    "trailing_std_24h",
]


def _price_series(history: pd.DataFrame) -> pd.Series:
    s = history.set_index(pd.to_datetime(history["interval_start_utc"], utc=True))[
        "system_price_gbp"
    ].astype(float)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def build_features(
    history: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    exogenous: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build the feature matrix for the given target half-hour timestamps.

    Args:
        history: DataFrame with ['interval_start_utc', 'system_price_gbp'].
        timestamps: UTC interval starts to build features for.
        exogenous: optional frame indexed/keyed by 'interval_start_utc' with any of
            EXOGENOUS_COLUMNS (must contain only forecasts available at D-1 09:20).
    """
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    s = _price_series(history)
    out = pd.DataFrame(index=ts)

    local = ts.tz_convert("Europe/London")
    tod = local.hour * 2 + local.minute // 30  # 0..47 local clock slot
    out["tod_sin"] = np.sin(2 * np.pi * tod / 48)
    out["tod_cos"] = np.cos(2 * np.pi * tod / 48)
    out["dow"] = local.dayofweek
    out["is_weekend"] = (local.dayofweek >= 5).astype(int)
    out["month"] = local.month

    for d in PRICE_LAGS_DAYS:
        out[f"price_lag_{d}d"] = s.reindex(ts - pd.Timedelta(days=d)).to_numpy()

    same_period = np.vstack(
        [s.reindex(ts - pd.Timedelta(days=d)).to_numpy() for d in range(MIN_LAG_DAYS, MIN_LAG_DAYS + 7)]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns early in history
        out["same_period_mean_7d"] = np.nanmean(same_period, axis=0)
        out["same_period_std_7d"] = np.nanstd(same_period, axis=0)

    roll_mean = s.rolling("1D").mean()
    roll_std = s.rolling("1D").std()
    anchor = ts - pd.Timedelta(days=MIN_LAG_DAYS)
    # as-of lookup: latest rolling value at or before the anchor
    out["trailing_mean_24h"] = roll_mean.reindex(roll_mean.index.union(anchor)).ffill().reindex(anchor).to_numpy()
    out["trailing_std_24h"] = roll_std.reindex(roll_std.index.union(anchor)).ffill().reindex(anchor).to_numpy()

    cols = list(FEATURE_COLUMNS_BASE)
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
