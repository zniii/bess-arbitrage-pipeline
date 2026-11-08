"""
Day-ahead price forecasters.

* ``SeasonalNaiveForecaster``: same half-hour one week earlier (the benchmark any
  real model must beat).
* ``GBMPriceForecaster``: gradient-boosted regressor (scikit-learn) on the
  leak-free features from ``features.py``.

Both expose ``fit(history)`` and ``predict(target_date, history)`` where
``predict`` returns ['interval_start_utc', 'price_forecast'] for the UK local
day ``target_date`` and only reads history that would be available at D-1.
"""

import logging
from datetime import datetime, time, timedelta

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from bess_pipeline.forecasting.features import build_features

logger = logging.getLogger(__name__)

LAG_NAIVE_DAYS = 7


def day_timestamps(target_date: str) -> pd.DatetimeIndex:
    """All UTC half-hour starts of a Europe/London calendar day (46/48/50 periods)."""
    d = datetime.strptime(target_date, "%Y-%m-%d").date()
    tz = pd.Timestamp("2000-01-01", tz="Europe/London").tz
    start = pd.Timestamp(datetime.combine(d, time.min)).tz_localize(tz).tz_convert("UTC")
    end = pd.Timestamp(datetime.combine(d + timedelta(days=1), time.min)).tz_localize(tz).tz_convert("UTC")
    return pd.date_range(start, end, freq="30min", inclusive="left")


def _available(history: pd.DataFrame, target_date: str, min_lag_days: int = 2) -> pd.DataFrame:
    """Drop anything not knowable at D-1 morning (imbalance: D-1 onward; MID: D onward)."""
    cutoff = day_timestamps(target_date)[0] - pd.Timedelta(days=min_lag_days - 1)
    h = history.copy()
    h["interval_start_utc"] = pd.to_datetime(h["interval_start_utc"], utc=True)
    return h[h["interval_start_utc"] < cutoff]


class SeasonalNaiveForecaster:
    name = "seasonal_naive_7d"

    def __init__(self, target_col: str = "system_price_gbp", min_lag_days: int = 2) -> None:
        self.target_col = target_col
        self.min_lag_days = min_lag_days

    def fit(self, history: pd.DataFrame) -> "SeasonalNaiveForecaster":
        return self

    def predict(self, target_date: str, history: pd.DataFrame, exogenous=None) -> pd.DataFrame:
        ts = day_timestamps(target_date)
        h = _available(history, target_date, self.min_lag_days)
        s = h.set_index("interval_start_utc")[self.target_col].astype(float)
        s = s[~s.index.duplicated(keep="last")]
        pred = s.reindex(ts - pd.Timedelta(days=LAG_NAIVE_DAYS)).to_numpy()
        out = pd.DataFrame({"interval_start_utc": ts, "price_forecast": pred})
        out["price_forecast"] = out["price_forecast"].fillna(s.mean() if len(s) else 0.0)
        return out


class GBMPriceForecaster:
    name = "gbm"

    def __init__(self, max_iter: int = 200, learning_rate: float = 0.05, random_state: int = 0,
                 target_col: str = "system_price_gbp", min_lag_days: int = 2, name: str | None = None) -> None:
        self.target_col = target_col
        self.min_lag_days = min_lag_days
        if name:
            self.name = name
        self.model = HistGradientBoostingRegressor(
            max_iter=max_iter, learning_rate=learning_rate, random_state=random_state
        )
        self.feature_columns: list[str] | None = None

    def fit(self, history: pd.DataFrame, exogenous: pd.DataFrame | None = None) -> "GBMPriceForecaster":
        h = history.copy()
        h["interval_start_utc"] = pd.to_datetime(h["interval_start_utc"], utc=True)
        X = build_features(h, pd.DatetimeIndex(h["interval_start_utc"]), exogenous,
                           self.target_col, self.min_lag_days)
        y = h.set_index("interval_start_utc")[self.target_col].astype(float)
        y = y[~y.index.duplicated(keep="last")].reindex(X.index)
        ok = y.notna() & X[f"price_lag_{self.min_lag_days}d"].notna()
        if ok.sum() < 200:
            raise ValueError(f"Need >=200 trainable intervals (~4 days after lags), got {int(ok.sum())}.")
        # An all-NaN column (e.g. wind/demand feed unavailable) breaks sklearn's binning; drop it.
        keep = [c for c in X.columns if X.loc[ok, c].notna().any()]
        self.feature_columns = keep
        self.model.fit(X.loc[ok, keep], y[ok])
        logger.info("Fitted %s on %d intervals.", self.name, int(ok.sum()))
        return self

    def predict(self, target_date: str, history: pd.DataFrame, exogenous: pd.DataFrame | None = None) -> pd.DataFrame:
        if self.feature_columns is None:
            raise RuntimeError("Call fit() before predict().")
        ts = day_timestamps(target_date)
        X = build_features(_available(history, target_date, self.min_lag_days), ts, exogenous,
                           self.target_col, self.min_lag_days)
        for c in self.feature_columns:
            if c not in X.columns:
                X[c] = float("nan")
        pred = self.model.predict(X[self.feature_columns])
        return pd.DataFrame({"interval_start_utc": ts, "price_forecast": pred})
