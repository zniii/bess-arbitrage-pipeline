"""
BESS Dispatch Pipeline End-to-End Orchestrator (Multi-Day Batch Support).

Commercial & Engineering Context:
---------------------------------
Orchestrates the complete data lifecycle for Battery Energy Storage System (BESS) dispatch:
  1. Bronze: Ingests raw half-hourly pricing telemetry from Elexon Insights REST API into Amazon S3.
  2. Silver: Conforms timestamps, enforces UTC half-open intervals [start, end), partitions Parquet.
  3. Gold: Computes 24h rolling volatility, intraday spreads, and arbitrage tags.
  4. Optimization: Formulates & solves the MILP dispatch problem (MILP + degradation penalty).
  5. Analytics & Gold Delivery: Computes financial P&L and uploads final asset schedules to Amazon S3.
"""

import argparse
from datetime import datetime, timedelta
import logging
import sys
import pandas as pd

from bess_pipeline.common.database import AWSDataLakeManager
from bess_pipeline.ingestion.elexon_client import ElexonClient
from bess_pipeline.transformations.temporal import standardize_system_prices
from bess_pipeline.transformations.storage import save_to_silver_parquet
from bess_pipeline.transformations.features import generate_bess_market_features
from bess_pipeline.optimization.dispatch_optimizer import BESSDispatcher
from bess_pipeline.analytics.financial_kpis import calculate_dispatch_financial_kpis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("bess_pipeline.orchestrator")


def run_single_day(
    target_date: str,
    lake_mgr: AWSDataLakeManager,
    client: ElexonClient,
    dispatcher: BESSDispatcher,
    max_power_mw: float,
    max_energy_mwh: float,
    roundtrip_efficiency: float,
) -> dict:
    """Executes the 5-stage pipeline for a single settlement date."""
    logger.info("--- Processing Settlement Date: %s ---", target_date)

    # 1. Bronze
    raw_records = client.fetch_daily_raw_records(target_date)
    if not raw_records:
        logger.warning("No records returned for %s. Skipping day.", target_date)
        return {}

    lake_mgr.upload_raw_json_to_bronze(
        records=raw_records,
        source_name="elexon",
        target_date=target_date,
    )

    # 2. Silver
    df_raw = pd.DataFrame(raw_records)
    df_silver = standardize_system_prices(df_raw)
    save_to_silver_parquet(df_silver, "data/silver/system_prices")
    lake_mgr.sync_silver_partition_to_s3(
        local_base_dir="data/silver/system_prices",
        settlement_date=target_date,
        s3_prefix="silver/system_prices",
    )

    # 3. Gold Features
    df_gold = generate_bess_market_features(df_silver)

    # 4. Optimization
    df_schedule = dispatcher.optimize_schedule(df_gold)
    if df_schedule.empty:
        logger.error("Solver did not find an optimal schedule for %s.", target_date)
        return {}

    # 5. Financial KPIs & S3 Gold Upload
    usable_capacity = max_energy_mwh * (0.9 - 0.1)
    kpis = calculate_dispatch_financial_kpis(df_schedule, usable_capacity_mwh=usable_capacity)
    lake_mgr.upload_gold_schedule_to_s3(df_schedule, settlement_date=target_date)

    return {
        "date": target_date,
        "revenue": kpis.gross_revenue_gbp,
        "cost": kpis.charging_cost_gbp,
        "profit": kpis.net_profit_gbp,
        "export_mwh": kpis.export_throughput_mwh,
        "import_mwh": kpis.import_throughput_mwh,
        "efc": kpis.equivalent_full_cycles,
        "spread": kpis.avg_capture_spread_gbp_per_mwh,
    }


def run_batch_pipeline(
    start_date: str,
    end_date: str,
    max_power_mw: float = 10.0,
    max_energy_mwh: float = 20.0,
    roundtrip_efficiency: float = 0.85,
) -> None:
    """Orchestrates multi-day pipeline execution across a continuous date range."""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    if start_dt > end_dt:
        logger.error("start_date (%s) cannot be after end_date (%s).", start_date, end_date)
        sys.exit(1)

    lake_mgr = AWSDataLakeManager()
    client = ElexonClient()
    dispatcher = BESSDispatcher(
        max_power_mw=max_power_mw,
        max_energy_mwh=max_energy_mwh,
        roundtrip_efficiency=roundtrip_efficiency,
    )

    results = []
    current_dt = start_dt

    while current_dt <= end_dt:
        date_str = current_dt.strftime("%Y-%m-%d")
        daily_kpis = run_single_day(
            target_date=date_str,
            lake_mgr=lake_mgr,
            client=client,
            dispatcher=dispatcher,
            max_power_mw=max_power_mw,
            max_energy_mwh=max_energy_mwh,
            roundtrip_efficiency=roundtrip_efficiency,
        )
        if daily_kpis:
            results.append(daily_kpis)
        current_dt += timedelta(days=1)

    if not results:
        logger.warning("No data was processed successfully.")
        return

    # Cumulative Backtest Summary
    df_results = pd.DataFrame(results)
    total_rev = df_results["revenue"].sum()
    total_cost = df_results["cost"].sum()
    total_profit = df_results["profit"].sum()
    total_export = df_results["export_mwh"].sum()
    total_efc = df_results["efc"].sum()
    avg_spread = (total_profit / total_export) if total_export > 0 else 0.0

    print("\n" + "=" * 70)
    print(f"       BESS BATCH BACKTEST SUMMARY: {start_date} to {end_date}")
    print("=" * 70)
    print(f" Total Days Evaluated       : {len(df_results):>10}")
    print(f" Cumulative Gross Revenue   : £ {total_rev:>10,.2f}")
    print(f" Cumulative Charging Cost   : £ {total_cost:>10,.2f}")
    print(f" Cumulative Net Arbitrage   : £ {total_profit:>10,.2f}")
    print("-" * 70)
    print(f" Cumulative Energy Exported : {total_export:>10.2f} MWh")
    print(f" Cumulative Total Cycles    : {total_efc:>10.3f} EFC")
    print(f" Average Cycles Per Day     : {total_efc / len(df_results):>10.2f} EFC/day")
    print(f" Portfolio Average Spread   : £ {avg_spread:>10.2f} / MWh")
    print("=" * 70 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch BESS dispatch pipeline.")
    parser.add_argument("--date", type=str, help="Single target settlement date (YYYY-MM-DD)")
    parser.add_argument("--start-date", type=str, help="Start date for range (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date for range (YYYY-MM-DD)")
    parser.add_argument("--power-mw", type=float, default=10.0, help="Power rating in MW")
    parser.add_argument("--energy-mwh", type=float, default=20.0, help="Capacity in MWh")
    parser.add_argument("--efficiency", type=float, default=0.85, help="Round-trip efficiency")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Determine execution mode (single day vs. range)
    if args.date:
        s_date = args.date
        e_date = args.date
    elif args.start_date and args.end_date:
        s_date = args.start_date
        e_date = args.end_date
    else:
        logger.error("Provide either --date YYYY-MM-DD OR both --start-date and --end-date.")
        sys.exit(1)

    run_batch_pipeline(
        start_date=s_date,
        end_date=e_date,
        max_power_mw=args.power_mw,
        max_energy_mwh=args.energy_mwh,
        roundtrip_efficiency=args.efficiency,
    )