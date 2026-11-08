"""
Data access for the live dashboard: reads Gold Parquet through DuckDB from S3/MinIO, or from a local
folder (set ``BESS_LOCAL_DATA_DIR``), and copes with BOTH Gold schedule formats:

* legacy files written by the original ``pipeline_runner`` (no ``strategy`` column, no day-ahead
  columns) -> shown as strategy ``legacy_imbalance_perfect`` (optimised on the actual imbalance price);
* new files written by ``daily_update`` (``strategy``, ``da_price_gbp``, ``settlement_price_gbp``,
  ``price_forecast``).

Partitions of both kinds can coexist in one bucket. Missing results return an empty frame instead of
raising, so the page can explain what to run.
"""

import os

import duckdb
import pandas as pd

LEGACY_STRATEGY = "legacy_imbalance_perfect"
HOURS = 0.5


def connect(local: bool = False) -> duckdb.DuckDBPyConnection:
    """DuckDB connection to S3 (AWS, or MinIO/LocalStack via AWS_ENDPOINT_URL), or a plain one for local files."""
    con = duckdb.connect()
    if local:
        return con
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_region=?", [os.getenv("AWS_REGION", "eu-west-2")])
    con.execute("SET s3_access_key_id=?", [os.getenv("AWS_ACCESS_KEY_ID", "")])
    con.execute("SET s3_secret_access_key=?", [os.getenv("AWS_SECRET_ACCESS_KEY", "")])
    endpoint = os.getenv("AWS_ENDPOINT_URL")
    if endpoint:
        host = endpoint.split("://", 1)[-1].rstrip("/")
        con.execute("SET s3_endpoint=?", [host])
        con.execute("SET s3_url_style='path'")
        con.execute("SET s3_use_ssl=?", [endpoint.startswith("https")])
    return con


def base_location() -> tuple[str, bool]:
    """(base path, is_local). Local folder if BESS_LOCAL_DATA_DIR is set, else s3://<bucket>."""
    local = os.getenv("BESS_LOCAL_DATA_DIR")
    if local:
        return local.rstrip("/"), True
    return f"s3://{os.getenv('AWS_S3_BUCKET') or os.getenv('S3_BUCKET_NAME', '')}", False


def _columns(con, glob: str) -> set[str] | None:
    """Column names across all matching files, or None if no file matches."""
    try:
        rows = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{glob}', hive_partitioning=true, union_by_name=true)"
        ).fetchall()
    except (duckdb.IOException, duckdb.InvalidInputException) as err:
        if "No files found" in str(err) or "No files match" in str(err) or "HTTP 404" in str(err) or "404" in str(err):
            return None
        raise
    return {r[0] for r in rows}


def load_schedules(con, base: str) -> pd.DataFrame:
    """Half-hourly schedules from every Gold partition, normalised to one schema (see module docstring)."""
    glob = f"{base}/gold/dispatch_schedules/**/*.parquet"
    cols = _columns(con, glob)
    if cols is None:
        return pd.DataFrame()

    def col(name: str, fallback: str = "NULL::DOUBLE") -> str:
        return name if name in cols else fallback

    strategy = f"COALESCE(strategy, '{LEGACY_STRATEGY}')" if "strategy" in cols else f"'{LEGACY_STRATEGY}'"
    settle = ("COALESCE(settlement_price_gbp, system_price_gbp)" if "settlement_price_gbp" in cols
              else "system_price_gbp")
    query = f"""
        SELECT CAST(settlementDate AS VARCHAR) AS settlementDate,
               {strategy} AS strategy,
               interval_start_utc,
               system_price_gbp,
               {col('da_price_gbp')} AS da_price_gbp,
               {settle} AS settlement_price_gbp,
               {col('price_forecast')} AS price_forecast,
               optimal_charge_mw, optimal_discharge_mw, net_dispatch_mw, state_of_energy_mwh
        FROM read_parquet('{glob}', hive_partitioning=true, union_by_name=true)
        ORDER BY interval_start_utc
    """
    return con.execute(query).df()


def load_kpis(con, base: str) -> pd.DataFrame:
    """Daily KPI rows written by ``daily_update`` (empty frame if the job has not run yet)."""
    glob = f"{base}/gold/daily_kpis/**/*.parquet"
    if _columns(con, glob) is None:
        return pd.DataFrame()
    df = con.execute(
        f"SELECT * REPLACE (CAST(settlementDate AS VARCHAR) AS settlementDate) "
        f"FROM read_parquet('{glob}', hive_partitioning=true, union_by_name=true) ORDER BY settlementDate"
    ).df()
    return df


def derive_kpis(schedules: pd.DataFrame, usable_capacity_mwh: float = 16.0) -> pd.DataFrame:
    """
    Per-day KPIs computed from schedules (used when ``daily_kpis`` does not exist yet).
    Columns: settlementDate, profit_<strategy>_gbp per strategy, export_mwh, equivalent_full_cycles
    (the latter two from the first strategy available per day: da_forecast, else the first).
    """
    if schedules.empty:
        return pd.DataFrame()
    s = schedules.copy()
    s["profit"] = (s["optimal_discharge_mw"] - s["optimal_charge_mw"]) * s["settlement_price_gbp"] * HOURS
    s["export"] = s["optimal_discharge_mw"] * HOURS
    wide = s.pivot_table(index="settlementDate", columns="strategy", values="profit", aggfunc="sum")
    wide.columns = [f"profit_{c}_gbp" for c in wide.columns]
    main = s.assign(_k=(s["strategy"] != "da_forecast").astype(int)).sort_values(["settlementDate", "_k"])
    first = main.groupby("settlementDate")["strategy"].first()
    exp = s.set_index(["settlementDate", "strategy"])["export"].groupby(level=[0, 1]).sum()
    wide["export_mwh"] = [exp[(d, first[d])] for d in wide.index]
    wide["equivalent_full_cycles"] = wide["export_mwh"] / usable_capacity_mwh
    wide.insert(0, "settlementDate", wide.index)
    return wide.reset_index(drop=True)
