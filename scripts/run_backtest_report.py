"""
Walk-forward backtest that produces the data behind the Streamlit "Findings" page.

For every test day it mirrors the daily job exactly (same forecasters, same
no-lookahead rules, same strategies) and writes to ``app/data/``:
  backtest_daily.csv         per-day profit/revenue/cost per strategy, EXPECTED profit (valued at the
                             forecast), forecast errors, cycles
  mid_forecast_sample.csv    half-hourly actual MID vs model/naive forecasts, wind/demand, and the
                             model's charge/discharge/state-of-energy plan; also holds the plan for
                             the NEXT day (actual MID blank until the auction result is published)
  backtest_meta.json         run settings and totals

Usage:
    python scripts/run_backtest_report.py --start 2026-08-01 --end 2026-11-06 [--history-start 2026-01-01]
        [--cache-dir DIR]                      # reuse/top-up prices/mid/wind/demand parquet files
        [--degradation 8 --fee 0 --reserve-mwh 4]   # cost assumptions (GBP/MWh discharged, GBP/MWh traded, MWh)

Besides the legacy strategies (daily reset, no costs) it runs the COST-AWARE strategy:
state of charge carried across days, degradation + trading fees in the objective, an end-of-day
reserve, and a terminal value tuned on the first part of the window and judged on the rest.
It also writes rolling_summary.json (bridge from the theoretical maximum to the achievable result)
and sensitivity.csv (degradation x trading fee).
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from bess_pipeline.forecasting.backtest import settle_day
from bess_pipeline.forecasting.exogenous import build_exogenous
from bess_pipeline.forecasting.model import SeasonalNaiveForecaster, day_timestamps
from bess_pipeline.forecasting.rolling import settle_schedule, simulate
from bess_pipeline.forecasting.strategies import run_strategies
from bess_pipeline.ingestion.elexon_client import ElexonClient
from bess_pipeline.optimization.dispatch_optimizer import BESSDispatcher
from bess_pipeline.orchestration.daily_update import _fit_forecaster
from bess_pipeline.transformations.temporal import standardize_system_prices

OUT = Path(__file__).resolve().parent.parent / "app" / "data"
RETRAIN_EVERY_DAYS = 14
logger = logging.getLogger("backtest_report")


def _tail_start(df: pd.DataFrame, col: str) -> str:
    return pd.to_datetime(df[col], utc=True).max().strftime("%Y-%m-%d")


def load_inputs(history_start: str, end: str, next_day: str | None, cache_dir: Path | None):
    """
    Return (prices, mid, wind, demand). With a cache dir, cached parquet files are topped up
    with only the missing tail instead of re-downloading everything.
    ``end`` = last settled day (imbalance prices); MID/forecasts run through ``next_day``.
    """
    c = ElexonClient()
    end_excl = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    far = (datetime.strptime(next_day or end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    warm = (datetime.strptime(history_start, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")
    names = ["prices", "mid", "wind", "demand"]
    have = cache_dir and all((cache_dir / f"{n}.parquet").exists() for n in names)

    if have:
        prices, mid, wind, demand = (pd.read_parquet(cache_dir / f"{n}.parquet") for n in names)
        logger.info("Topping up cached inputs in %s", cache_dir)
        t = _tail_start(prices, "interval_start_utc")
        if t < end_excl:
            new = standardize_system_prices(c.fetch_system_prices(t, end_excl))
            prices = pd.concat([prices, new]).drop_duplicates("interval_start_utc", keep="last")
        t = _tail_start(mid, "interval_start_utc")
        mid = pd.concat([mid, c.fetch_mid(t, far)]).drop_duplicates("interval_start_utc", keep="last")
        t = _tail_start(wind, "publishTime")
        wind = pd.concat([wind, c.fetch_wind_forecasts(t, far)]).drop_duplicates(["publishTime", "startTime"], keep="last")
        t = _tail_start(demand, "publishTime")
        demand = pd.concat([demand, c.fetch_demand_forecasts(t, far)]).drop_duplicates(["publishTime", "forecastDate"], keep="last")
    else:
        prices = standardize_system_prices(c.fetch_system_prices(history_start, end_excl))
        mid = c.fetch_mid(history_start, far)
        wind = c.fetch_wind_forecasts(warm, far)
        demand = c.fetch_demand_forecasts(warm, far)
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        for n, d in zip(names, (prices, mid, wind, demand)):
            d.reset_index(drop=True).to_parquet(cache_dir / f"{n}.parquet")
    prices["interval_start_utc"] = pd.to_datetime(prices["interval_start_utc"], utc=True)
    mid["interval_start_utc"] = pd.to_datetime(mid["interval_start_utc"], utc=True)
    return prices, mid, wind, demand


TV_GRID = [0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0]   # GBP per MWh left in the battery at midnight
SENS_DEG = [0.0, 4.0, 8.0, 15.0, 25.0]
SENS_FEE = [0.0, 1.0, 2.0]


def make_dispatcher(deg: float, fee: float, reserve: float, tv: float) -> BESSDispatcher:
    return BESSDispatcher(degradation_cost_gbp_per_mwh=deg, trading_fee_gbp_per_mwh=fee,
                          end_soc_min_mwh=reserve, terminal_value_gbp_per_mwh=tv)


def total(df: pd.DataFrame, col: str = "net_profit_gbp") -> float:
    return float(df[col].sum())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True, help="last settled day (imbalance prices complete)")
    p.add_argument("--history-start", default="2026-01-01")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--next-day", default=None, help="UK date to plan (default: tomorrow). 'none' disables.")
    p.add_argument("--degradation", type=float, default=8.0, help="GBP per MWh discharged (placeholder)")
    p.add_argument("--fee", type=float, default=0.0, help="trading/route-to-market fee, GBP per MWh traded")
    p.add_argument("--reserve-mwh", type=float, default=4.0, help="minimum state of energy at the end of each day")
    p.add_argument("--no-sensitivity", action="store_true")
    a = p.parse_args()
    next_day = None if a.next_day == "none" else (
        a.next_day or (datetime.now(ZoneInfo("Europe/London")).date() + timedelta(days=1)).isoformat())

    prices, mid_all, wind, demand = load_inputs(a.history_start, a.end, next_day, a.cache_dir)
    hist = prices.merge(mid_all[["interval_start_utc", "mid_price_gbp"]], on="interval_start_utc", how="inner")
    mid_hist = mid_all.assign(system_price_gbp=float("nan"))  # MID history incl. days without imbalance prices
    chain_end = next_day or a.end
    chain_dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(a.start, max(chain_end, a.end))]
    all_ts = pd.DatetimeIndex(sorted(set(mid_all["interval_start_utc"]) | set(hist["interval_start_utc"])
                                     | {t for d in chain_dates for t in day_timestamps(d)}))
    exog = build_exogenous(all_ts, wind, demand)
    exog_idx = exog.set_index("interval_start_utc")
    mid_s = mid_all.set_index("interval_start_utc")["mid_price_gbp"]

    disp0 = BESSDispatcher()
    usable = disp0.max_soc_mwh - disp0.min_soc_mwh
    mid_kw = dict(target_col="mid_price_gbp", min_lag_days=2, name="gbm_mid")  # MID is published ex-post: D-1 is NOT known at decision time
    imb_kw = dict(target_col="system_price_gbp", min_lag_days=2, name="gbm_imbalance")
    naive_mid = SeasonalNaiveForecaster("mid_price_gbp", 2)
    hist_days = set(hist["interval_start_utc"].map(lambda t: t.tz_convert("Europe/London").strftime("%Y-%m-%d")))

    # ---- 1. Walk-forward forecasts for every chain day + legacy strategies (daily reset, no costs)
    legacy_rows, samples, fcs, fitted_at, f_mid, f_imb = [], [], {}, None, None, None
    for date in chain_dates:
        ts = day_timestamps(date)
        if f_mid is None or (ts[0] - fitted_at).days >= RETRAIN_EVERY_DAYS:
            f_mid = _fit_forecaster(mid_hist, ts[0], exog, **mid_kw)
            f_imb = _fit_forecaster(hist, ts[0], exog, **imb_kw)
            fitted_at = ts[0]
        fc_mid = f_mid.predict(date, mid_hist, exog)
        fc_naive = naive_mid.predict(date, mid_hist)
        fcs[date] = (fc_mid, fc_naive)
        actual = hist[hist["interval_start_utc"].isin(ts)]
        if date not in hist_days or len(actual) != len(ts) or date > a.end:
            continue
        fc_imb = f_imb.predict(date, hist, exog)
        strat = run_strategies(disp0, actual, usable, mid_forecast=fc_mid, imbalance_forecast=fc_imb)
        mid_day = actual.assign(system_price_gbp=actual["mid_price_gbp"])
        k_naive, _ = settle_day(disp0, mid_day, fc_naive, usable)
        ai = actual.set_index("interval_start_utc")
        err = lambda col, fc: float((ai[col] - fc.set_index("interval_start_utc")["price_forecast"]).abs().mean())
        k = strat["mid_forecast"][0]
        legacy_rows.append({
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
        logger.info("%s legacy: baseline £%.0f model £%.0f", date, legacy_rows[-1]["profit_mid_baseline"],
                    legacy_rows[-1]["profit_mid_forecast"])

    # ---- 2. Cost-aware strategy: SoC carried across days, degradation + fees + end-of-day reserve
    frames, known = [], []
    for date in chain_dates:
        ts = day_timestamps(date)
        fc_mid, fc_naive = fcs[date]
        f = pd.DataFrame({"interval_start_utc": ts})
        f["date"] = date
        f["mid_actual"] = f["interval_start_utc"].map(mid_s)
        f["mid_model"] = f["interval_start_utc"].map(fc_mid.set_index("interval_start_utc")["price_forecast"])
        f["actual_known"] = bool(f["mid_actual"].notna().all())
        f["mid_settle"] = f["mid_actual"].fillna(f["mid_model"])   # forecast stands in only for unpublished days
        frames.append(f)
        known.append(bool(f["actual_known"].iloc[0]))
    n_known = next((i for i, k in enumerate(known) if not k), len(known))   # contiguous published prefix
    kframes = frames[:n_known]
    n_tune = max(1, int(0.4 * n_known))

    def sim(disp, fr, dec, settle, **kw):
        return simulate(disp, fr, dec, settle, **kw)

    tune = {}
    for tv in TV_GRID:
        r, _ = sim(make_dispatcher(a.degradation, a.fee, a.reserve_mwh, tv), kframes[:n_tune], "mid_model", "mid_actual")
        tune[tv] = total(r)
        logger.info("tune terminal value £%s/MWh -> net £%.0f on first %d days", tv, tune[tv], n_tune)
    tv_star = max(tune, key=lambda k: (tune[k], -k))
    logger.info("Chosen terminal value £%s/MWh (tuned on first %d of %d days only)", tv_star, n_tune, n_known)

    d_cost = make_dispatcher(a.degradation, a.fee, a.reserve_mwh, tv_star)
    ach, ach_sched = sim(d_cost, frames, "mid_model", "mid_settle")                        # includes future days
    ach_known = ach.iloc[:n_known]
    d_free = make_dispatcher(0.0, 0.0, disp0.min_soc_mwh, 0.0)
    ub0, _ = sim(d_free, kframes, "mid_actual", "mid_actual", carry_soc=False)             # perfect, no costs, daily reset
    ub1, _ = sim(d_free, kframes, "mid_actual", "mid_actual", carry_soc=True, lookahead=True)   # + SoC carry
    ub2, _ = sim(make_dispatcher(a.degradation, a.fee, a.reserve_mwh, 0.0), kframes, "mid_actual", "mid_actual",
                 carry_soc=True, lookahead=True)                                           # + costs and reserve
    s0, _ = sim(d_free, kframes, "mid_model", "mid_actual", carry_soc=False)               # legacy achievable
    oos = slice(n_tune, n_known)
    summary = {
        "assumptions": {"degradation_gbp_per_mwh_discharged": a.degradation, "trading_fee_gbp_per_mwh": a.fee,
                        "end_of_day_reserve_mwh": a.reserve_mwh, "network_levy_charges": "not modelled (unknown)",
                        "terminal_value_gbp_per_mwh": tv_star, "terminal_value_tuning_grid": {str(k): round(v, 2) for k, v in tune.items()},
                        "tuned_on_days": n_tune, "evaluated_days": n_known, "revenue_share_applied": False},
        "bridge_gbp": {
            "legacy_achievable_no_costs_daily_reset": round(total(s0), 2),
            "perfect_no_costs_daily_reset": round(total(ub0), 2),
            "perfect_no_costs_soc_carry": round(total(ub1), 2),
            "perfect_after_costs_soc_carry": round(total(ub2), 2),
            "achievable_after_costs_soc_carry": round(total(ach_known), 2),
        },
        "out_of_sample_gbp": {"days": n_known - n_tune, "perfect_after_costs": round(total(ub2.iloc[oos]), 2),
                              "achievable_after_costs": round(total(ach_known.iloc[oos]), 2)},
        "achievable_breakdown_gbp": {c: round(total(ach_known, c), 2) for c in
                                     ["revenue_gbp", "charging_cost_gbp", "trading_fees_gbp", "degradation_gbp", "net_profit_gbp"]},
    }

    # expected (valued at forecast) vs realised, per chain day; half-hourly plan
    exp_net, plan_rows = [], []
    for fr, sc in zip(frames, ach_sched):
        e = settle_schedule(sc, fr["mid_model"].reset_index(drop=True), d_cost)
        exp_net.append(e)
        pl = fr[["date", "interval_start_utc", "mid_actual", "mid_model", "actual_known"]].reset_index(drop=True).copy()
        pl["charge_mw"], pl["discharge_mw"], pl["soe_mwh"] = sc["optimal_charge_mw"], sc["optimal_discharge_mw"], sc["state_of_energy_mwh"]
        pl["wind_forecast_mw"] = pl["interval_start_utc"].map(exog_idx["wind_forecast_mw"])
        pl["demand_forecast_mw"] = pl["interval_start_utc"].map(exog_idx["demand_forecast_mw"])
        plan_rows.append(pl)
    roll = ach.copy()
    roll.insert(0, "date", chain_dates[: len(roll)])
    roll["actual_known"] = known[: len(roll)]
    roll["expected_net_gbp"] = [e["net_profit_gbp"] for e in exp_net][: len(roll)]
    roll["expected_revenue_gbp"] = [e["revenue_gbp"] for e in exp_net][: len(roll)]
    roll["expected_cost_gbp"] = [e["charging_cost_gbp"] for e in exp_net][: len(roll)]
    for name, df in (("perfect_costed_net_gbp", ub2), ("perfect_free_net_gbp", ub1), ("perfect_free_reset_net_gbp", ub0)):
        roll[name] = pd.Series(df["net_profit_gbp"].to_numpy(), index=range(len(df))).reindex(range(len(roll)))
    kn = roll[roll["actual_known"]]
    summary["calibration"] = {
        "expected_net_gbp": round(float(kn["expected_net_gbp"].sum()), 2),
        "realised_net_gbp": round(float(kn["net_profit_gbp"].sum()), 2),
        "realised_over_expected": round(float(kn["net_profit_gbp"].sum() / kn["expected_net_gbp"].sum()), 3),
    }
    if next_day and next_day in chain_dates:
        i = chain_dates.index(next_day)
        e, r_ = exp_net[i], roll.iloc[i]
        summary["next_day"] = {"date": next_day, "actual_mid_published": bool(known[i]),
                               "expected_revenue_gbp": round(e["revenue_gbp"], 2), "expected_charging_cost_gbp": round(e["charging_cost_gbp"], 2),
                               "expected_fees_gbp": round(e["trading_fees_gbp"], 2), "expected_degradation_gbp": round(e["degradation_gbp"], 2),
                               "expected_net_gbp": round(e["net_profit_gbp"], 2), "start_soc_mwh": round(float(r_["start_soc_mwh"]), 2),
                               "end_soc_mwh": round(float(r_["end_soc_mwh"]), 2),
                               "realised_net_gbp": round(float(r_["net_profit_gbp"]), 2) if known[i] else None}
    summary["last_actual_day"] = chain_dates[n_known - 1] if n_known else None

    # ---- 3. Sensitivity: degradation x trading fee (terminal value held at the tuned value)
    if not a.no_sensitivity:
        rows = []
        for deg in SENS_DEG:
            for fee in SENS_FEE:
                a_, _ = sim(make_dispatcher(deg, fee, a.reserve_mwh, tv_star), kframes, "mid_model", "mid_actual")
                u_, _ = sim(make_dispatcher(deg, fee, a.reserve_mwh, 0.0), kframes, "mid_actual", "mid_actual",
                            carry_soc=True, lookahead=True)
                rows.append({"degradation_gbp_per_mwh": deg, "fee_gbp_per_mwh": fee,
                             "achievable_net_gbp": total(a_), "perfect_net_gbp": total(u_), "achievable_efc_per_day": float(a_["efc"].mean())})
                logger.info("sensitivity deg=%s fee=%s achievable £%.0f perfect £%.0f", deg, fee, rows[-1]["achievable_net_gbp"], rows[-1]["perfect_net_gbp"])
        pd.DataFrame(rows).round(2).to_csv(OUT / "sensitivity.csv", index=False)

    OUT.mkdir(parents=True, exist_ok=True)
    daily = pd.DataFrame(legacy_rows)
    daily.round(3).to_csv(OUT / "backtest_daily.csv", index=False)
    sample = pd.concat(samples)
    num = sample.select_dtypes("number").columns
    sample[num] = sample[num].round(2)
    sample.to_csv(OUT / "mid_forecast_sample.csv", index=False)
    plan = pd.concat(plan_rows)
    num = plan.select_dtypes("number").columns
    plan[num] = plan[num].round(3)
    plan.to_csv(OUT / "plan_halfhourly.csv", index=False)
    roll_out = roll.copy()
    num = roll_out.select_dtypes("number").columns
    roll_out[num] = roll_out[num].round(3)
    roll_out.to_csv(OUT / "rolling_daily.csv", index=False)
    (OUT / "rolling_summary.json").write_text(json.dumps(summary, indent=2))
    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_start": a.start, "test_end": a.end, "history_start": a.history_start, "test_days": len(daily),
        "retrain_every_days": RETRAIN_EVERY_DAYS,
        "battery": {"power_mw": disp0.max_power_mw, "energy_mwh": disp0.max_energy_mwh,
                    "roundtrip_efficiency": round(disp0.eff ** 2, 3), "soc_window_pct": [10, 90],
                    "degradation_cost_gbp_per_mwh": 0.0},
        "decision_time": "D-1 08:00 UTC (before ~09:20 UK day-ahead auction)",
        "model": "HistGradientBoostingRegressor on calendar, price lags, rolling stats, WINDFOR, NDFD",
    }
    (OUT / "backtest_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Wrote %s\n%s", OUT, json.dumps(summary["bridge_gbp"], indent=2))


if __name__ == "__main__":
    main()
