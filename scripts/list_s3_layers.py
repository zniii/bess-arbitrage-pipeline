"""
Utility script to inspect objects across Bronze, Silver, and Gold S3 layers.
"""

from bess_pipeline.common.database import AWSDataLakeManager


def inspect_s3_layers() -> None:
    mgr = AWSDataLakeManager()
    bucket = mgr.bucket_name

    for prefix in ["bronze/elexon/", "silver/system_prices/", "gold/dispatch_schedules/"]:
        print(f"\n=== S3 LAYER: {prefix} ===")
        res = mgr.s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        contents = res.get("Contents", [])
        
        if not contents:
            print("  (Empty)")
            continue

        for item in contents:
            key = item["Key"]
            size = item["Size"]
            print(f"  {key:<85} | {size:>8} bytes")


if __name__ == "__main__":
    inspect_s3_layers()