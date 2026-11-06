"""
Daily incremental update: bring Gold up to the most recent settled day.

For every UK day in the trailing ``--window-days`` window (ending yesterday, the
latest fully settled day) that has no Gold schedule + KPI row yet:
  1. ingest Bronze/Silver imbalance prices if missing (plus ``--history-days`` of
     warm-up history), and fetch MID day-ahead prices plus wind/demand forecasts,
  2. forecast day-ahead MID (our model) and, experimentally, the imbalance price,
     using only information available at D-1 08:00 UTC,
  3. run the four strategies in ``forecasting/strategies.py`` (MID baseline,
     MID forecast, imbalance forecast, imbalance perfect foresight),
  4. upload all schedules (``strategy`` column) and a one-row daily KPI record to Gold.

Re-running is a no-op once every day in the window is present (idempotent).

Usage:
    python -m bess_pipeline.orchestration.daily_update [--window-days 30]
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from bess_pipeline.common.database import AWSDataLakeManager
from bess_pipeline.forecasting.exogenous import build_exogenous
from bess_pipeline.forecasting.model import (
    GBMPriceForecaster,
    SeasonalNaiveForecaster,
    day_timestamps,
)
from bess_pipeline.forecasting.strategies import run_strategies
from bess_pipeline.ingestion.elexon_client import ElexonClient
from bess_pipeline.optimization.dispatch_optimizer import BESSDispatcher
from bess_pipeline.transformations.storage import save_to_silver_parquet
from bess_pipeline.transformations.temporal import standardize_system_prices

logger = logging.getLogger("bess_pipeline.daily_update")
RETRAIN_EVERY_DAYS = 7


def latest_settled_date(now: datetime | None = None) -> datetime:
    """Yesterday in UK local time: the most recent day with a complete set of prices."""
    now = now or datetime.now(timezone.utc)
    return datetime.combine(
        now.astimezone(ZoneInfo("Europe/London")).date() - timedelta(days=1), datetime.min.time()
    )


def ingest_day(date_str: str, lake: AWSDataLakeManager, client: ElexonClient) -> bool:
    """Bronze + Silver for one date. Returns False if the API had no data."""
    raw = client.fetch_daily_raw_records(date_str)
    if not raw:
        logger.warning("No records for %s; skipping ingest.", date_str)
        return False
    lake.upload_raw_json_to_bronze(records=raw, source_name="elexon", target_date=date_str)
    silver = standardize_system_prices(pd.DataFrame(raw))
    save_to_silver_parquet(silver, "data/silver/system_prices")
    lake.sync_silver_partition_to_s3("data/silver/system_prices", date_str, "silver/system_prices")
    return True


def _fit_forecaster(history: pd.DataFrame, day_start: pd.Timestamp, exogenous: pd.DataFrame, **kw):
    """Fit a GBM on data known before ``day_start``; fall back to seasonal-naive if too little data."""
    lag = kw["min_lag_days"]
    train = history[history["interval_start_utc"] < day_start - timedelta(days=lag - 1)]
    train = train.dropna(subset=[kw["target_col"]])
    try:
        return GBMPriceForecaster(**kw).fit(train, exogenous)
    except ValueError as err:
        logger.warning("GBM '%s' not trainable yet (%s); using seasonal-naive.", kw.get("name"), err)
        return SeasonalNaiveForecaster(kw["target_col"], lag)


def run_daily_update(
    window_days: int = 30,
    history_days: int = 60,
    as_of: datetime | None = None,
    lake: AWSDataLakeManager | None = None,
    client: ElexonClient | None = None,
    dispatcher: BESSDispatcher | None = None,
) -> list[dict]:
    lake = lake or AWSDataLakeManager()
    client = client or ElexonClient()
    dispatcher = dispatcher or BESSDispatcher()
    usable = dispatcher.max_soc_mwh - dispatcher.min_soc_mwh

    end = as_of or latest_settled_date()
    fmt = lambda d: d.strftime("%Y-%m-%d")
    targets = [fmt(end - timedelta(days=i)) for i in range(window_days - 1, -1, -1)]
    history_start = fmt(end - timedelta(days=window_days - 1 + history_days))

    # 1a. Ingest anything missing from Silver (targets + warm-up history)
    have_silver = lake.list_silver_dates()
    wanted = [fmt(end - timedelta(days=i)) for i in range(window_days - 1 + history_days, -1, -1)]
    for d in wanted:
        if d not in have_silver:
            logger.info("Ingesting missing Silver day %s", d)
            ingest_day(d, lake, client)

    gold_have = lake.list_gold_dates()
    missing = [d for d in targets if d not in gold_have]
    if not missing:
        logger.info("Gold already up to date through %s. Nothing to do.", targets[-1])
        return []

    imb = lake.read_silver_history(history_start, fmt(end))
    if imb.empty:
        logger.error("No Silver history available; aborting.")
        return []
    imb["interval_start_utc"] = pd.to_datetime(imb["interval_start_utc"], utc=True)

    # 1b. MID prices and wind/demand forecasts (fetched live; N2EX is dropped inside fetch_mid)
    end_excl = fmt(end + timedelta(days=1))
    mid = client.fetch_mid(history_start, end_excl)
    wind = client.fetch_wind_forecasts(fmt(datetime.strptime(history_start, "%Y-%m-%d") - timedelta(days=7)), end_excl)
    demand = client.fetch_demand_forecasts(fmt(datetime.strptime(history_start, "%Y-%m-%d") - timedelta(days=7)), end_excl)
    history = imb.merge(mid[["interval_start_utc", "mid_price_gbp"]], on="interval_start_utc", how="left")
    exog = build_exogenous(pd.DatetimeIndex(history["interval_start_utc"]), wind, demand)

    mid_kw = dict(target_col="mid_price_gbp", min_lag_days=1, name="gbm_mid")
    imb_kw = dict(target_col="system_price_gbp", min_lag_days=2, name="gbm_imbalance")

    results, f_mid, f_imb, fitted_at = [], None, None, None
    for date in missing:
        ts = day_timestamps(date)
        actual = history[history["interval_start_utc"].isin(ts)]
        if actual.empty or actual["mid_price_gbp"].isna().any():
            logger.warning("Missing imbalance or MID prices for %s; skipping.", date)
            continue

        if f_mid is None or (ts[0] - fitted_at).days >= RETRAIN_EVERY_DAYS:
            f_mid = _fit_forecaster(history, ts[0], exog, **mid_kw)
            f_imb = _fit_forecaster(history, ts[0], exog, **imb_kw)
            fitted_at = ts[0]

        fc_mid = f_mid.predict(date, history, exog)
        fc_imb = f_imb.predict(date, history, exog)
        strat = run_strategies(dispatcher, actual, usable, mid_forecast=fc_mid, imbalance_forecast=fc_imb)
        if "mid_forecast" not in strat:
            logger.error("Solver failed for %s; skipping.", date)
            continue

        kpi, _ = strat["mid_forecast"]  # headline strategy
        mae = lambda col, fc: round(float((actual.set_index("interval_start_utc")[col]
                                           - fc.set_index("interval_start_utc")["price_forecast"]).abs().mean()), 3)
        row = {
            "settlementDate": date,
            "forecaster_mid": f_mid.name,
            "forecaster_imbalance": f_imb.name,
            "mid_forecast_mae_gbp_per_mwh": mae("mid_price_gbp", fc_mid),
            "imbalance_forecast_mae_gbp_per_mwh": mae("system_price_gbp", fc_imb),
            "profit_gbp": kpi.net_profit_gbp,  # headline = mid_forecast
            "gross_revenue_gbp": kpi.gross_revenue_gbp,
            "charging_cost_gbp": kpi.charging_cost_gbp,
            "export_mwh": kpi.export_throughput_mwh,
            "import_mwh": kpi.import_throughput_mwh,
            "equivalent_full_cycles": kpi.equivalent_full_cycles,
            "avg_capture_spread_gbp_per_mwh": kpi.avg_capture_spread_gbp_per_mwh,
        }
        for name, (k, _) in strat.items():
            row[f"profit_{name}_gbp"] = k.net_profit_gbp
        schedule = pd.concat([s for _, s in strat.values()], ignore_index=True)
        schedule["settlementDate"] = date
        lake.upload_gold_schedule_to_s3(schedule, settlement_date=date)
        lake.upload_daily_kpis_to_s3(row, date)  # written last: marks the day complete
        results.append(row)
        logger.info("%s done: mid_forecast £%.2f | mid_baseline £%s | imbalance_forecast £%s | MID MAE %.1f",
                    date, row["profit_gbp"], row.get("profit_mid_baseline_gbp"),
                    row.get("profit_imbalance_forecast_gbp"), row["mid_forecast_mae_gbp_per_mwh"])
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    p = argparse.ArgumentParser(description="Update Gold dispatch results to the latest settled day.")
    p.add_argument("--window-days", type=int, default=30, help="Days of results to keep current")
    p.add_argument("--history-days", type=int, default=60, help="Extra warm-up history for the forecaster")
    a = p.parse_args()
    run_daily_update(window_days=a.window_days, history_days=a.history_days)


if __name__ == "__main__":
    main()
