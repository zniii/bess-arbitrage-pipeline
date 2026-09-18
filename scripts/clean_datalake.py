"""
Clean stale UUID Parquet files from S3 Silver and typo key from S3 Bronze,
then re-sync canonical part-0.parquet files.
"""

from bess_pipeline.common.database import AWSDataLakeManager


def clean_datalake() -> None:
    mgr = AWSDataLakeManager()
    bucket = mgr.bucket_name

    # 1. Stale Bronze typo key
    stale_bronze_key = "bronze/elexon/year=2026/month=08payload_2026-08-01.json"
    try:
        mgr.s3_client.delete_object(Bucket=bucket, Key=stale_bronze_key)
        print(f"Purged malformed Bronze key: {stale_bronze_key}")
    except Exception as e:
        print(f"Could not delete Bronze key: {e}")

    # 2. Collect all legacy UUID files in Silver
    res = mgr.s3_client.list_objects_v2(Bucket=bucket, Prefix="silver/system_prices/")
    silver_keys = [
        {"Key": obj["Key"]}
        for obj in res.get("Contents", [])
        if not obj["Key"].endswith("part-0.parquet")
    ]

    if silver_keys:
        print(f"Purging {len(silver_keys)} legacy UUID Silver file(s)...")
        mgr.s3_client.delete_objects(Bucket=bucket, Delete={"Objects": silver_keys})
        for k in silver_keys:
            print(f"  Deleted: {k['Key']}")

    # 3. Sync clean local part-0.parquet files to S3
    dates = ["2026-08-01", "2026-08-02", "2026-08-03"]
    print("\nSyncing local part-0.parquet files to S3 Silver...")
    for date_str in dates:
        count = mgr.sync_silver_partition_to_s3(
            local_base_dir="data/silver/system_prices",
            settlement_date=date_str,
            s3_prefix="silver/system_prices",
        )
        print(f"  Partition {date_str}: {count} file(s) synchronized.")

    print("\nDatalake cleanup complete!")


if __name__ == "__main__":
    clean_datalake()