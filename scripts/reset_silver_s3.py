"""
Explicit reset of Silver Layer:
1. Reprocess local silver data to ensure part-0.parquet exists.
2. Wipe existing remote silver/system_prices/ objects in S3.
3. Upload canonical part-0.parquet files.
"""

from pathlib import Path
import duckdb
import pandas as pd
from bess_pipeline.common.database import AWSDataLakeManager
from bess_pipeline.transformations.storage import save_to_silver_parquet


def reset_silver() -> None:
    mgr = AWSDataLakeManager()
    bucket = mgr.bucket_name
    s3_prefix = "silver/system_prices/"

    print("--- 1. Generating local part-0.parquet files ---")
    # Read existing local data using DuckDB to ensure we have all 3 days
    con = duckdb.connect()
    df = con.execute("""
        SELECT * 
        FROM read_parquet('data/silver/system_prices/**/*.parquet', hive_partitioning = true)
    """).df()

    # Re-save using the new idempotent logic (creates part-0.parquet)
    save_to_silver_parquet(df, "data/silver/system_prices")
    print("Local partition directories refreshed with part-0.parquet.")

    print("\n--- 2. Purging remote S3 silver layer ---")
    res = mgr.s3_client.list_objects_v2(Bucket=bucket, Prefix=s3_prefix)
    contents = res.get("Contents", [])

    if contents:
        delete_keys = [{"Key": obj["Key"]} for obj in contents]
        print(f"Deleting {len(delete_keys)} remote objects...")
        mgr.s3_client.delete_objects(
            Bucket=bucket,
            Delete={"Objects": delete_keys, "Quiet": False},
        )
        print("Remote silver layer wiped.")
    else:
        print("Remote silver layer was already empty.")

    print("\n--- 3. Syncing fresh part-0.parquet files to S3 ---")
    dates = ["2026-08-01", "2026-08-02", "2026-08-03"]
    for d in dates:
        synced = mgr.sync_silver_partition_to_s3(
            local_base_dir="data/silver/system_prices",
            settlement_date=d,
            s3_prefix="silver/system_prices",
        )
        print(f"  Partition {d}: {synced} file(s) uploaded.")

    print("\nSilver layer reset completed successfully!")


if __name__ == "__main__":
    reset_silver()