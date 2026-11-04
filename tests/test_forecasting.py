"""Forecasting tests. Price data here is SYNTHETIC (no network), used only to test
mechanics: no-lookahead, shapes, DST day lengths, settle-on-actual accounting."""

import numpy as np
import pandas as pd

from bess_pipeline.forecasting.backtest import settle_day
from bess_pipeline.forecasting.features import build_features
from bess_pipeline.forecasting.model import (
    GBMPriceForecaster,
    SeasonalNaiveForecaster,
    day_timestamps,
)
from bess_pipeline.optimization.dispatch_optimizer import BESSDispatcher


def synthetic_history(days=40, start="2026-08-01"):
    idx = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=days * 48, freq="30min")
    rng = np.random.default_rng(0)
    tod = np.arange(len(idx)) % 48
    price = 60 + 40 * np.sin(2 * np.pi * (tod - 12) / 48) + rng.normal(0, 3, len(idx))
    return pd.DataFrame({"interval_start_utc": idx, "system_price_gbp": price})


def test_day_timestamps_dst_lengths():
    assert len(day_timestamps("2026-03-29")) == 46
    assert len(day_timestamps("2026-10-25")) == 50
    assert len(day_timestamps("2026-08-10")) == 48


def test_features_have_no_lookahead():
    h = synthetic_history()
    ts = day_timestamps("2026-09-05")
    base = build_features(h, ts)
    # Corrupt everything from D-1 start onward: features must not change.
    cutoff = ts[0] - pd.Timedelta(days=1)
    h2 = h.copy()
    h2.loc[h2["interval_start_utc"] >= cutoff, "system_price_gbp"] = 1e6
    after = build_features(h2, ts)
    pd.testing.assert_frame_equal(base, after)


def test_naive_forecast_is_lag_7d():
    h = synthetic_history()
    fc = SeasonalNaiveForecaster().predict("2026-09-05", h)
    ts = day_timestamps("2026-09-05")
    expected = h.set_index("interval_start_utc")["system_price_gbp"].reindex(ts - pd.Timedelta(days=7)).to_numpy()
    np.testing.assert_allclose(fc["price_forecast"].to_numpy(), expected)


def test_gbm_fits_and_predicts_full_day():
    h = synthetic_history()
    train = h[h["interval_start_utc"] < pd.Timestamp("2026-09-03", tz="UTC")]
    f = GBMPriceForecaster(max_iter=50).fit(train)
    fc = f.predict("2026-09-05", h)
    assert len(fc) == 48 and fc["price_forecast"].notna().all()


def test_settlement_uses_actual_prices_not_forecast():
    h = synthetic_history(days=3)
    day = h.iloc[:48].copy()
    wrong = pd.DataFrame({"interval_start_utc": day["interval_start_utc"],
                          "price_forecast": day["system_price_gbp"].to_numpy()[::-1]})
    d = BESSDispatcher()
    perfect, _ = settle_day(d, day, None, 16.0)
    misled, _ = settle_day(d, day, wrong, 16.0)
    assert misled.net_profit_gbp <= perfect.net_profit_gbp + 1e-6
