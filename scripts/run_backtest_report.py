"""
Walk-forward backtest behind the Streamlit pages, on the N2EX GB DAY-AHEAD auction price.

Each day D is planned at D-1 08:00 UTC from a forecast of the 24 hourly day-ahead prices (the auction
clears ~09:20 UK on D-1), then settled at the actual auction price. Costs are modelled: degradation,
trading fee, 85% round-trip efficiency, 10-90% SOC window, end-of-day reserve, SOC carried across days,
and a terminal value for stored energy tuned on the first 40% of days only.

Writes to app/data/:
  rolling_daily.csv       per day: revenue/costs/net, expected (valued at forecast), SOC, naive and perfect
  plan_halfhourly.csv     half-hourly actual/forecast price and the charge/discharge/SOE plan, incl. NEXT day
  rolling_summary.json    assumptions, bridge from theoretical maximum to achievable, calibration, next day
  sensitivity.csv         degradation x trading fee grid
  backtest_meta.json      run settings

Usage:
    python scripts/run_backtest_report.py --start 2026-08-01 --end 2026-11-07 [--cache-dir DIR]
        [--degradation 8 --fee 0 --reserve-mwh 4] [--next-day YYYY-MM-DD|none] [--no-sensitivity]
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from bess_pipeline.forecasting.exogenous import build_exogenous
from bess_pipeline.forecasting.model import SeasonalNaiveForecaster, day_timestamps
from bess_pipeline.forecasting.rolling import settle_schedule, simulate
from bess_pipeline.ingestion.elexon_client import ElexonClient
from bess_pipeline.ingestion.neso_client import NesoClient
from bess_pipeline.optimization.dispatch_optimizer import BESSDispatcher
from bess_pipeline.orchestration.daily_update import _fit_forecaster

OUT = Path(__file__).resolve().parent.parent / "app" / "data"
RETRAIN_EVERY_DAYS = 14
TV_GRID = [0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0]   # GBP per MWh left in the battery at midnight
SENS_DEG = [0.0, 4.0, 8.0, 15.0, 25.0]
SENS_FEE = [0.0, 1.0, 2.0]
TARGET = "da_price_gbp"
logger = logging.getLogger("backtest_report")


def _tail_start(df: pd.DataFrame, col: str) -> str:
    return pd.to_datetime(df[col], utc=True).max().strftime("%Y-%m-%d")


def load_inputs(history_start: str, da_history_start: str, far: str, cache_dir: Path | None):
    """(day-ahead half-hourly, wind publications, demand publications). Wind/demand cached + topped up."""
    da = NesoClient().fetch_day_ahead()
    da = da[da["interval_start_utc"] >= pd.Timestamp(da_history_start, tz="UTC")]
    c = ElexonClient()
    warm = (datetime.strptime(history_start, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")
    names = ["wind", "demand"]
    if cache_dir and all((cache_dir / f"{n}.parquet").exists() for n in names):
        wind, demand = (pd.read_parquet(cache_dir / f"{n}.parquet") for n in names)
        t = _tail_start(wind, "publishTime")
        wind = pd.concat([wind, c.fetch_wind_forecasts(t, far)]).drop_duplicates(["publishTime", "startTime"], keep="last")
        t = _tail_start(demand, "publishTime")
        demand = pd.concat([demand, c.fetch_demand_forecasts(t, far)]).drop_duplicates(["publishTime", "forecastDate"], keep="last")
    else:
        wind, demand = c.fetch_wind_forecasts(warm, far), c.fetch_demand_forecasts(warm, far)
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        wind.reset_index(drop=True).to_parquet(cache_dir / "wind.parquet")
        demand.reset_index(drop=True).to_parquet(cache_dir / "demand.parquet")
    return da, wind, demand


def make_dispatcher(deg: float, fee: float, reserve: float, tv: float) -> BESSDispatcher:
    return BESSDispatcher(degradation_cost_gbp_per_mwh=deg, trading_fee_gbp_per_mwh=fee,
                          end_soc_min_mwh=reserve, terminal_value_gbp_per_mwh=tv)


def total(df: pd.DataFrame, col: str = "net_profit_gbp") -> float:
    return float(df[col].sum())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True, help="last day to include as 'actual' if its prices are published")
    p.add_argument("--history-start", default="2026-01-01", help="wind/demand warm-up start (those feeds are cached from here)")
    p.add_argument("--da-history-start", default="2021-09-20", help="first day of day-ahead prices used for training (dataset starts 2021-09-20)")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--next-day", default=None, help="UK date to plan (default: tomorrow). 'none' disables.")
    p.add_argument("--degradation", type=float, default=8.0, help="GBP per MWh discharged (placeholder)")
    p.add_argument("--fee", type=float, default=0.0, help="trading/route-to-market fee, GBP per MWh traded")
    p.add_argument("--reserve-mwh", type=float, default=4.0, help="minimum state of energy at the end of each day")
    p.add_argument("--no-sensitivity", action="store_true")
    a = p.parse_args()
    today = datetime.now(ZoneInfo("Europe/London")).date()
    next_day = None if a.next_day == "none" else (a.next_day or (today + timedelta(days=1)).isoformat())
    last = max(filter(None, [a.end, next_day]))
    far = (datetime.strptime(last, "%Y-%m-%d") + timedelta(days=2)).strftime("%Y-%m-%d")

    da, wind, demand = load_inputs(a.history_start, a.da_history_start, far, a.cache_dir)
    hist = da.assign(system_price_gbp=float("nan"))
    chain_dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(a.start, last)]
    all_ts = pd.DatetimeIndex(sorted(set(da["interval_start_utc"]) | {t for d in chain_dates for t in day_timestamps(d)}))
    exog = build_exogenous(all_ts, wind, demand)
    exog_idx = exog.set_index("interval_start_utc")
    da_s = da.set_index("interval_start_utc")[TARGET]

    # decision time D-1 08:00 UTC: the day-ahead auction result of D-1 is already public, D's is not -> lag >= 1 day
    kw = dict(target_col=TARGET, min_lag_days=1, name="gbm_day_ahead")
    naive = SeasonalNaiveForecaster(TARGET, 1)
    disp0 = BESSDispatcher()

    # ---- 1. Walk-forward forecasts for every chain day
    frames, f_model, fitted_at = [], None, None
    for date in chain_dates:
        ts = day_timestamps(date)
        if f_model is None or (ts[0] - fitted_at).days >= RETRAIN_EVERY_DAYS:
            f_model = _fit_forecaster(hist, ts[0], exog, **kw)
            fitted_at = ts[0]
        fc = f_model.predict(date, hist, exog).set_index("interval_start_utc")["price_forecast"]
        fn = naive.predict(date, hist).set_index("interval_start_utc")["price_forecast"]
        f = pd.DataFrame({"interval_start_utc": ts})
        f["date"] = date
        f["da_actual"] = f["interval_start_utc"].map(da_s)
        f["da_model"] = f["interval_start_utc"].map(fc)
        f["da_naive"] = f["interval_start_utc"].map(fn)
        f["actual_known"] = bool(f["da_actual"].notna().all()) and date <= a.end
        f["da_settle"] = f["da_actual"].where(f["actual_known"], f["da_model"])   # forecast stands in only if unpublished
        frames.append(f)
        logger.info("%s forecast ready (actual known: %s)", date, f["actual_known"].iloc[0])
    known = [bool(f["actual_known"].iloc[0]) for f in frames]
    n_known = next((i for i, k in enumerate(known) if not k), len(known))   # contiguous published prefix
    kframes = frames[:n_known]
    n_tune = max(1, int(0.4 * n_known))

    # ---- 2. Terminal value tuned on the first 40% of days only
    tune = {}
    for tv in TV_GRID:
        r, _ = simulate(make_dispatcher(a.degradation, a.fee, a.reserve_mwh, tv), kframes[:n_tune], "da_model", "da_actual")
        tune[tv] = total(r)
        logger.info("tune terminal value £%s/MWh -> net £%.0f on first %d days", tv, tune[tv], n_tune)
    tv_star = max(tune, key=lambda k: (tune[k], -k))
    logger.info("Chosen terminal value £%s/MWh (tuned on first %d of %d days only)", tv_star, n_tune, n_known)

    d_cost = make_dispatcher(a.degradation, a.fee, a.reserve_mwh, tv_star)
    ach, ach_sched = simulate(d_cost, frames, "da_model", "da_settle")                       # includes future days
    naive_res, _ = simulate(d_cost, kframes, "da_naive", "da_actual")
    ach_known = ach.iloc[:n_known]
    d_free = make_dispatcher(0.0, 0.0, disp0.min_soc_mwh, 0.0)
    ub0, _ = simulate(d_free, kframes, "da_actual", "da_actual", carry_soc=False)            # perfect, no costs, daily reset
    ub1, _ = simulate(d_free, kframes, "da_actual", "da_actual", carry_soc=True, lookahead=True)    # + SOC carried
    ub2, _ = simulate(make_dispatcher(a.degradation, a.fee, a.reserve_mwh, 0.0), kframes, "da_actual", "da_actual",
                      carry_soc=True, lookahead=True)                                        # + costs and reserve
    oos = slice(n_tune, n_known)
    mae = lambda col: [float((f["da_actual"] - f[col]).abs().mean()) for f in kframes]
    mae_m, mae_n = mae("da_model"), mae("da_naive")
    summary = {
        "price": "N2EX GB day-ahead auction (NESO), hourly, UTC",
        "assumptions": {"degradation_gbp_per_mwh_discharged": a.degradation, "trading_fee_gbp_per_mwh": a.fee,
                        "end_of_day_reserve_mwh": a.reserve_mwh, "network_levy_charges": "not modelled (unknown)",
                        "terminal_value_gbp_per_mwh": tv_star,
                        "terminal_value_tuning_grid": {str(k): round(v, 2) for k, v in tune.items()},
                        "tuned_on_days": n_tune, "evaluated_days": n_known, "revenue_share_applied": False},
        "bridge_gbp": {
            "perfect_no_costs_daily_reset": round(total(ub0), 2),
            "perfect_no_costs_soc_carry": round(total(ub1), 2),
            "perfect_after_costs_soc_carry": round(total(ub2), 2),
            "achievable_after_costs_soc_carry": round(total(ach_known), 2),
            "naive_after_costs_soc_carry": round(total(naive_res), 2),
        },
        "out_of_sample_gbp": {"days": n_known - n_tune, "perfect_after_costs": round(total(ub2.iloc[oos]), 2),
                              "achievable_after_costs": round(total(ach_known.iloc[oos]), 2),
                              "naive_after_costs": round(total(naive_res.iloc[oos]), 2)},
        "forecast_error_gbp_per_mwh": {"model": round(sum(mae_m) / len(mae_m), 2), "naive": round(sum(mae_n) / len(mae_n), 2)},
        "achievable_breakdown_gbp": {c: round(total(ach_known, c), 2) for c in
                                     ["revenue_gbp", "charging_cost_gbp", "trading_fees_gbp", "degradation_gbp", "net_profit_gbp"]},
    }

    # ---- 3. Per-day table and half-hourly plan (expected = plan valued at the forecast)
    exp = [settle_schedule(sc, fr["da_model"].reset_index(drop=True), d_cost) for fr, sc in zip(frames, ach_sched)]
    plan_rows = []
    for fr, sc in zip(frames, ach_sched):
        pl = fr[["date", "interval_start_utc", "da_actual", "da_model", "da_naive", "actual_known"]].reset_index(drop=True).copy()
        pl["charge_mw"], pl["discharge_mw"], pl["soe_mwh"] = sc["optimal_charge_mw"], sc["optimal_discharge_mw"], sc["state_of_energy_mwh"]
        pl["wind_forecast_mw"] = pl["interval_start_utc"].map(exog_idx["wind_forecast_mw"])
        pl["demand_forecast_mw"] = pl["interval_start_utc"].map(exog_idx["demand_forecast_mw"])
        plan_rows.append(pl)
    roll = ach.copy()
    roll.insert(0, "date", chain_dates[: len(roll)])
    roll["actual_known"] = known[: len(roll)]
    roll["expected_net_gbp"] = [e["net_profit_gbp"] for e in exp][: len(roll)]
    roll["expected_revenue_gbp"] = [e["revenue_gbp"] for e in exp][: len(roll)]
    roll["expected_cost_gbp"] = [e["charging_cost_gbp"] for e in exp][: len(roll)]
    for name, df in (("perfect_costed_net_gbp", ub2), ("perfect_free_net_gbp", ub1), ("perfect_free_reset_net_gbp", ub0),
                     ("naive_net_gbp", naive_res)):
        roll[name] = pd.Series(df["net_profit_gbp"].to_numpy()).reindex(range(len(roll))).to_numpy()
    roll["mae_model"] = pd.Series(mae_m).reindex(range(len(roll))).to_numpy()
    roll["mae_naive"] = pd.Series(mae_n).reindex(range(len(roll))).to_numpy()
    kn = roll[roll["actual_known"]]
    summary["calibration"] = {
        "expected_net_gbp": round(float(kn["expected_net_gbp"].sum()), 2),
        "realised_net_gbp": round(float(kn["net_profit_gbp"].sum()), 2),
        "realised_over_expected": round(float(kn["net_profit_gbp"].sum() / kn["expected_net_gbp"].sum()), 3),
    }
    if next_day and next_day in chain_dates:
        i = chain_dates.index(next_day)
        e, r_ = exp[i], roll.iloc[i]
        summary["next_day"] = {"date": next_day, "actual_published": bool(known[i]),
                               "expected_revenue_gbp": round(e["revenue_gbp"], 2), "expected_charging_cost_gbp": round(e["charging_cost_gbp"], 2),
                               "expected_fees_gbp": round(e["trading_fees_gbp"], 2), "expected_degradation_gbp": round(e["degradation_gbp"], 2),
                               "expected_net_gbp": round(e["net_profit_gbp"], 2), "start_soc_mwh": round(float(r_["start_soc_mwh"]), 2),
                               "end_soc_mwh": round(float(r_["end_soc_mwh"]), 2),
                               "realised_net_gbp": round(float(r_["net_profit_gbp"]), 2) if known[i] else None}
    summary["last_actual_day"] = chain_dates[n_known - 1] if n_known else None

    # ---- 4. Sensitivity: degradation x trading fee (terminal value held at the tuned value)
    OUT.mkdir(parents=True, exist_ok=True)
    if not a.no_sensitivity:
        rows = []
        for deg in SENS_DEG:
            for fee in SENS_FEE:
                a_, _ = simulate(make_dispatcher(deg, fee, a.reserve_mwh, tv_star), kframes, "da_model", "da_actual")
                u_, _ = simulate(make_dispatcher(deg, fee, a.reserve_mwh, 0.0), kframes, "da_actual", "da_actual", carry_soc=True, lookahead=True)
                rows.append({"degradation_gbp_per_mwh": deg, "fee_gbp_per_mwh": fee, "achievable_net_gbp": total(a_),
                             "perfect_net_gbp": total(u_), "achievable_efc_per_day": float(a_["efc"].mean())})
                logger.info("sensitivity deg=%s fee=%s achievable £%.0f perfect £%.0f", deg, fee, rows[-1]["achievable_net_gbp"], rows[-1]["perfect_net_gbp"])
        pd.DataFrame(rows).round(2).to_csv(OUT / "sensitivity.csv", index=False)

    def write(df: pd.DataFrame, name: str, nd: int = 3) -> None:
        df = df.copy()
        num = df.select_dtypes("number").columns
        df[num] = df[num].round(nd)
        df.to_csv(OUT / name, index=False)

    write(roll, "rolling_daily.csv")
    write(pd.concat(plan_rows), "plan_halfhourly.csv")
    (OUT / "rolling_summary.json").write_text(json.dumps(summary, indent=2))
    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_start": a.start, "test_end": summary["last_actual_day"], "history_start": a.da_history_start, "test_days": n_known,
        "retrain_every_days": RETRAIN_EVERY_DAYS,
        "battery": {"power_mw": disp0.max_power_mw, "energy_mwh": disp0.max_energy_mwh,
                    "roundtrip_efficiency": round(disp0.eff ** 2, 3), "soc_window_pct": [10, 90]},
        "price": summary["price"],
        "decision_time": "D-1 08:00 UTC (before the ~09:20 UK day-ahead auction); yesterday's auction prices are public by then",
        "model": "HistGradientBoostingRegressor on calendar, day-ahead price lags (>=1 day), rolling stats, WINDFOR, NDFD",
    }
    (OUT / "backtest_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Wrote %s\n%s", OUT, json.dumps(summary["bridge_gbp"], indent=2))


if __name__ == "__main__":
    main()
