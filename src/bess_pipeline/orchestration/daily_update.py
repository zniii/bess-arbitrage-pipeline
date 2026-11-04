"""
Daily incremental update: bring Gold up to the most recent settled day.

For every UK day in the trailing ``--window-days`` window (ending yesterday, the
latest fully settled day) that has no Gold schedule + KPI row yet:
  1. ingest Bronze/Silver if missing (plus ``--history-days`` of extra history
     so the forecaster has something to learn from),
  2. forecast day-ahead prices using only information available at D-1,
  3. optimize dispatch on the FORECAST,
  4. settle on ACTUAL prices (and also solve the perfect-foresight benchmark),
  5. upload the schedule and a one-row daily KPI record to Gold.

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
from bess_pipeline.forecasting.backtest import settle_day
from bess_pipeline.forecasting.model import (
    GBMPriceForecaster,
    SeasonalNaiveForecaster,
    day_timestamps,
)
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

    # 1. Ingest anything missing from Silver (targets + warm-up history)
    have_silver = lake.list_silver_dates()
    wanted = [fmt(end - timedelta(days=i)) for i in range(window_days - 1 + history_days, -1, -1)]
    for d in wanted:
        if d not in have_silver:
            logger.info("Ingesting missing Silver day %s", d)
            ingest_day(d, lake, client)

    missing = [d for d in targets if d not in lake.list_gold_dates()]
    if not missing:
        logger.info("Gold already up to date through %s. Nothing to do.", targets[-1])
        return []

    history = lake.read_silver_history(history_start, fmt(end))
    if history.empty:
        logger.error("No Silver history available; aborting.")
        return []
    history["interval_start_utc"] = pd.to_datetime(history["interval_start_utc"], utc=True)

    # 2-5. Forecast -> optimize -> settle -> upload, oldest first (walk-forward)
    results, forecaster, fitted_at = [], None, None
    for date in missing:
        ts = day_timestamps(date)
        actual = history[history["interval_start_utc"].isin(ts)]
        if actual.empty:
            logger.warning("No actual prices for %s; skipping.", date)
            continue

        if forecaster is None or (ts[0] - fitted_at).days >= RETRAIN_EVERY_DAYS:
            train = history[history["interval_start_utc"] < ts[0] - timedelta(days=1)]
            try:
                forecaster = GBMPriceForecaster().fit(train)
            except ValueError as err:
                logger.warning("GBM not trainable yet (%s); using seasonal-naive baseline.", err)
                forecaster = SeasonalNaiveForecaster()
            fitted_at = ts[0]

        fc = forecaster.predict(date, history)
        kpi, schedule = settle_day(dispatcher, actual, fc, usable)
        perfect, _ = settle_day(dispatcher, actual, None, usable)
        if kpi is None:
            logger.error("Solver failed for %s; skipping.", date)
            continue

        merged = actual.merge(fc, on="interval_start_utc")
        row = {
            "settlementDate": date,
            "forecaster": forecaster.name,
            "forecast_mae_gbp_per_mwh": round(float((merged["system_price_gbp"] - merged["price_forecast"]).abs().mean()), 3),
            "profit_gbp": kpi.net_profit_gbp,
            "perfect_foresight_profit_gbp": perfect.net_profit_gbp if perfect else None,
            "gross_revenue_gbp": kpi.gross_revenue_gbp,
            "charging_cost_gbp": kpi.charging_cost_gbp,
            "export_mwh": kpi.export_throughput_mwh,
            "import_mwh": kpi.import_throughput_mwh,
            "equivalent_full_cycles": kpi.equivalent_full_cycles,
            "avg_capture_spread_gbp_per_mwh": kpi.avg_capture_spread_gbp_per_mwh,
        }
        schedule["settlementDate"] = date
        lake.upload_gold_schedule_to_s3(schedule, settlement_date=date)
        lake.upload_daily_kpis_to_s3(row, date)  # written last: marks the day complete
        results.append(row)
        logger.info("%s done: profit £%.2f (perfect £%s, MAE %.2f)", date, row["profit_gbp"],
                    row["perfect_foresight_profit_gbp"], row["forecast_mae_gbp_per_mwh"])
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
