"""
Silver Layer Storage Writer.
"""

from pathlib import Path
import logging
import shutil
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def save_to_silver_parquet(
    df: pd.DataFrame,
    base_dir: str = "data/silver/system_prices",
) -> Path:
    """
    Saves standardized settlement data partitioned by settlementDate (YYYY-MM-DD).
    Enforces idempotency by clearing existing partition folders prior to writing.
    """
    if df.empty:
        raise ValueError("Cannot persist empty DataFrame to Silver layer.")

    required_cols = {"interval_start_utc", "settlementDate", "system_price_gbp"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required schema columns: {missing}")

    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    # Process each partition date to ensure clean, idempotent overwrites
    for settlement_date, group in df.groupby("settlementDate"):
        # Format explicitly to YYYY-MM-DD string to avoid colons in Windows paths
        if hasattr(settlement_date, "strftime"):
            date_str = settlement_date.strftime("%Y-%m-%d")
        else:
            date_str = str(settlement_date).split(" ")[0].split("T")[0]

        partition_dir = base_path / f"settlementDate={date_str}"

        # Clean out stale files from prior runs
        if partition_dir.exists():
            shutil.rmtree(partition_dir)
        partition_dir.mkdir(parents=True, exist_ok=True)

        # Write single standardized Parquet part
        table = pa.Table.from_pandas(group.drop(columns=["settlementDate"]))
        output_file = partition_dir / "part-0.parquet"
        pq.write_table(table, output_file, compression="snappy")

    logger.info("Persisted %d records to Silver Parquet store at: %s", len(df), base_path)
    return base_path