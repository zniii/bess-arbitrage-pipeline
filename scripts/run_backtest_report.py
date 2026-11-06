"""
Walk-forward backtest that produces the data behind the Streamlit "Findings" page.

For every test day it mirrors the daily job exactly (same forecasters, same
no-lookahead rules, same strategies) and writes to ``app/data/``:
  backtest_daily.csv         per-day profit per strategy, forecast errors, cycles
  mid_forecast_sample.csv    half-hourly actual MID vs model/naive forecasts + wind/demand
  backtest_meta.json         run settings and totals

Usage:
    python scripts/run_backtest_report.py --start 2026-08-01 --end 2026-11-04 [--history-start 2026-01-01]
                                          [--cache-dir DIR]   # reuse prices/mid/wind/demand parquet files
"""

import argparse
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import sys

import pandas as pd

from bess_pipeline.forecasting.exogenous import build_exogenous
from bess_pipeline.forecasting.model import SeasonalNaiveForecaster, day_timestamps
from bess_pipeline.forecasting.strategies import run_strategies
from bess_pipeline.ingestion.elexon_client import ElexonClient
from bess_pipeline.optimization.dispatch_optimizer import BESSDispatcher
from bess_pipeline.orchestration.daily_update import _fit_forecaster
from bess_pipeline.transformations.temporal import standardize_system_prices
from bess_pipeline.forecasting.backtest import settle_day

OUT = Path(__file__).resolve().parent.parent / "app" / "data"
RETRAIN_EVERY_DAYS = 14
logger = logging.getLogger("backtest_report")


def load_inputs(history_start: str, end: str, cache_dir: Path | None):
    """Return (history[system_price_gbp, mid_price_gbp], wind, demand); use cache files if present."""
    names = ["prices", "mid", "wind", "demand"]
    if cache_dir and all((cache_dir / f"{n}.parquet").exists() for n in names):
        logger.info("Reading cached inputs from %s", cache_dir)
        prices, mid, wind, demand = (pd.read_parquet(cache_dir / f"{n}.parquet") for n in names)
    else:
        c = ElexonClient()
        end_excl = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        warm = (datetime.strptime(history_start, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")
        prices = standardize_system_prices(c.fetch_system_prices(history_start, end_excl))
        mid = c.fetch_mid(history_start, end_excl)
        wind = c.fetch_wind_forecasts(warm, end_excl)
        demand = c.fetch_demand_forecasts(warm, end_excl)
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            for n, d in zip(names, (prices, mid, wind, demand)):
                d.to_parquet(cache_dir / f"{n}.parquet")
    prices["interval_start_utc"] = pd.to_datetime(prices["interval_start_utc"], utc=True)
    mid["interval_start_utc"] = pd.to_datetime(mid["interval_start_utc"], utc=True)
    hist = prices.merge(mid[["interval_start_utc", "mid_price_gbp"]], on="interval_start_utc", how="inner")
    return hist, wind, demand


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--history-start", default="2026-01-01")
    p.add_argument("--cache-dir", type=Path, default=None)
    a = p.parse_args()

    hist, wind, demand = load_inputs(a.history_start, a.end, a.cache_dir)
    exog = build_exogenous(pd.DatetimeIndex(hist["interval_start_utc"]), wind, demand)
    exog_idx = exog.set_index("interval_start_utc")
    disp = BESSDispatcher()
    usable = disp.max_soc_mwh - disp.min_soc_mwh
    mid_kw = dict(target_col="mid_price_gbp", min_lag_days=1, name="gbm_mid")
    imb_kw = dict(target_col="system_price_gbp", min_lag_days=2, name="gbm_imbalance")
    naive_mid = SeasonalNaiveForecaster("mid_price_gbp", 1)

    rows, samples, fitted_at, f_mid, f_imb = [], [], None, None, None
    for date in [d.strftime("%Y-%m-%d") for d in pd.date_range(a.start, a.end)]:
        ts = day_timestamps(date)
        actual = hist[hist["interval_start_utc"].isin(ts)]
        if actual.empty:
            continue
        if f_mid is None or (ts[0] - fitted_at).days >= RETRAIN_EVERY_DAYS:
            f_mid = _fit_forecaster(hist, ts[0], exog, **mid_kw)
            f_imb = _fit_forecaster(hist, ts[0], exog, **imb_kw)
            fitted_at = ts[0]
        fc_mid, fc_imb = f_mid.predict(date, hist, exog), f_imb.predict(date, hist, exog)
        fc_naive = naive_mid.predict(date, hist)
        strat = run_strategies(disp, actual, usable, mid_forecast=fc_mid, imbalance_forecast=fc_imb)
        mid_day = actual.assign(system_price_gbp=actual["mid_price_gbp"])
        k_naive, _ = settle_day(disp, mid_day, fc_naive, usable)

        ai = actual.set_index("interval_start_utc")
        err = lambda col, fc: float((ai[col] - fc.set_index("interval_start_utc")["price_forecast"]).abs().mean())  # noqa: E731
        k = strat["mid_forecast"][0]
        rows.append({
            "date": date,
            "profit_mid_baseline": strat["mid_baseline"][0].net_profit_gbp,
            "profit_mid_forecast": k.net_profit_gbp,
            "profit_mid_naive": k_naive.net_profit_gbp,
            "profit_imbalance_forecast": strat["imbalance_forecast"][0].net_profit_gbp,
            "profit_imbalance_perfect": strat["imbalance_perfect"][0].net_profit_gbp,
            "mae_mid_model": err("mid_price_gbp", fc_mid),
            "mae_mid_naive": err("mid_price_gbp", fc_naive),
            "mae_imbalance_model": err("system_price_gbp", fc_imb),
            "export_mwh": k.export_throughput_mwh,
            "efc": k.equivalent_full_cycles,
            "mean_mid_price": float(actual["mid_price_gbp"].mean()),
            "mean_imbalance_price": float(actual["system_price_gbp"].mean()),
        })
        s = pd.DataFrame({"interval_start_utc": actual["interval_start_utc"].to_numpy()})
        s["date"] = date
        s["mid_actual"] = actual["mid_price_gbp"].to_numpy()
        s["imbalance_actual"] = actual["system_price_gbp"].to_numpy()
        s["mid_model"] = s["interval_start_utc"].map(fc_mid.set_index("interval_start_utc")["price_forecast"])
        s["mid_naive"] = s["interval_start_utc"].map(fc_naive.set_index("interval_start_utc")["price_forecast"])
        s["imbalance_model"] = s["interval_start_utc"].map(fc_imb.set_index("interval_start_utc")["price_forecast"])
        s["wind_forecast_mw"] = s["interval_start_utc"].map(exog_idx["wind_forecast_mw"])
        s["demand_forecast_mw"] = s["interval_start_utc"].map(exog_idx["demand_forecast_mw"])
        samples.append(s)
        logger.info("%s mid_baseline £%.0f mid_model £%.0f naive £%.0f", date, rows[-1]["profit_mid_baseline"],
                    rows[-1]["profit_mid_forecast"], rows[-1]["profit_mid_naive"])

    OUT.mkdir(parents=True, exist_ok=True)
    daily = pd.DataFrame(rows)
    daily.round(3).to_csv(OUT / "backtest_daily.csv", index=False)
    pd.concat(samples).round(2).to_csv(OUT / "mid_forecast_sample.csv", index=False)
    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_start": a.start, "test_end": a.end, "history_start": a.history_start, "test_days": len(daily),
        "retrain_every_days": RETRAIN_EVERY_DAYS,
        "battery": {"power_mw": disp.max_power_mw, "energy_mwh": disp.max_energy_mwh,
                    "roundtrip_efficiency": round(disp.eff ** 2, 3), "soc_window_pct": [10, 90],
                    "degradation_cost_gbp_per_mwh": disp.degradation_cost},
        "decision_time": "D-1 08:00 UTC (before ~09:20 UK day-ahead auction)",
        "model": "HistGradientBoostingRegressor on calendar, price lags, rolling stats, WINDFOR, NDFD",
    }
    (OUT / "backtest_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Wrote %s", OUT)


if __name__ == "__main__":
    main()
