"""
Utility script to purge legacy/duplicate Parquet files from S3 Silver layer,
retaining strictly the idempotent 'part-0.parquet' files per partition.
"""

from bess_pipeline.common.database import AWSDataLakeManager

def purge_stale_silver_files() -> None:
    mgr = AWSDataLakeManager()
    bucket = mgr.bucket_name
    prefix = "silver/system_prices/"

    paginator = mgr.s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    to_delete = []
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Match any parquet file that is NOT part-0.parquet
            if key.endswith(".parquet") and not key.endswith("part-0.parquet"):
                to_delete.append({"Key": key})

    if not to_delete:
        print("No stale files detected. S3 Silver layer is already clean.")
        return

    print(f"Found {len(to_delete)} stale Parquet file(s) to delete...")
    
    # Delete in batches of up to 1000 (S3 API limit)
    for i in range(0, len(to_delete), 1000):
        batch = to_delete[i : i + 1000]
        res = mgr.s3_client.delete_objects(
            Bucket=bucket,
            Delete={"Objects": batch, "Quiet": False},
        )
        for deleted in res.get("Deleted", []):
            print(f"Purged: {deleted['Key']}")

    print("\nS3 Silver cleanup complete! Partitions now only contain canonical part-0.parquet files.")


if __name__ == "__main__":
    purge_stale_silver_files()