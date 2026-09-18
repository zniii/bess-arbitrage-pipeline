"""
AWS Cloud Storage & Data Lake Manager.

Commercial & Engineering Context:
---------------------------------
Encapsulates Amazon Web Services (AWS) S3 operations for the Medallion Architecture:
  1. Bronze Layer: Ingests immutable, raw JSON API responses into S3 prefixes.
  2. Silver Layer: Syncs partition-pruned Apache Parquet tables to S3, making them
     immediately queryable by serverless engines like Amazon Athena or local DuckDB.
"""


from datetime import datetime
import json
import logging
import os
from pathlib import Path
from typing import Optional
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class AWSDataLakeManager:
    """Manage Bronze JSON uploads and Silver Parquet syncing to Amazon S3"""

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        region_name: Optional[str] = None,
    )-> None:
        """Initialize S3 client using explicit credentials"""
        self.bucket_name = bucket_name or os.getenv("AWS_S3_BUCKET")
        self.region_name = region_name or os.getenv("AWS_REGION", "ew-west-2")

        if not self.bucket_name:
            raise ValueError("AWS_S3_BUCKET is not set in environment or config.")

        #Initialize boto3 S3 client
        self.s3_client = boto3.client(
            "s3",
            region_name=self.region_name,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )

    #Raw ingestion to S3
    def upload_raw_json_to_bronze(
            self,
            records: list[dict],
            source_name: str,
            target_date: str,
    ) -> str:
        """
        Streams raw JSON payload directly to the Bronze S3 prefix.

        S3 Key Convention:
            bronze/<source>/year=YYYY/month=MM/payload_YYYY-MM-DD.json

        Args:
            records: Raw API list of dictionaries.
            source_name: Data source identifier (e.g., 'elexon' or 'neso').
            target_date: Date string formatted as 'YYYY-MM-DD'.

        Returns:
            str: S3 URI (s3://bucket/key).
        """

        date_obj = datetime.strptime(target_date, "%Y-%m-%d")
        year_str = date_obj.strftime("%Y")
        month_str = date_obj.strftime("%m")

        s3_key = (
            f"bronze/{source_name}/year={year_str}/month={month_str}/payload_{target_date}.json"
        )

        try:
            json_bytes = json.dumps(records, indent=2).encode("utf-8")
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json_bytes,
                ContentType="application/json",
            )
            s3_uri = f"s3://{self.bucket_name}/{s3_key}"
            logger.info("Bronze payload uploaded successfully to: %s", s3_uri)
            return s3_uri

        except ClientError as err:
            logger.error("Failed to upload Bronze payload to S3: %s", err)
            raise


    #Silver syncing local partitioned parquet to S3
    def sync_silver_partition_to_s3(
        self,
        local_base_dir: str,
        settlement_date: str,
        s3_prefix: str = "silver/system_prices",
    ) -> int:
        """
        Synchronizes ONLY the specific partition folder for the processed date to S3.
        
        Args:
            local_base_dir: Root local silver directory.
            settlement_date: Partition key value (YYYY-MM-DD).
            s3_prefix: Target S3 prefix root.
        """
        import os
        from pathlib import Path

        partition_folder = f"settlementDate={settlement_date}"
        local_partition_dir = Path(local_base_dir) / partition_folder

        if not local_partition_dir.exists():
            logger.warning("Partition folder %s does not exist locally.", local_partition_dir)
            return 0

        synced_count = 0
        for file_path in local_partition_dir.glob("*.parquet"):
            s3_key = f"{s3_prefix}/{partition_folder}/{file_path.name}"
            logger.info("Syncing partition file %s -> s3://%s/%s", file_path.name, self.bucket_name, s3_key)
            self.s3_client.upload_file(str(file_path), self.bucket_name, s3_key)
            synced_count += 1

        return synced_count


    def upload_gold_schedule_to_s3(
            self,
            df_schedule: pd.DataFrame,
            settlement_date:str,
    ) -> str:
        """
        Serializes and uploads the solved BESS dispatch schedule directly to the S3 Gold layer.

        Key Convention:
            gold/dispatch_schedules/settlementDate=YYYY-MM-DD/schedule.parquet
        """

        import io
        import pyarrow as pa
        import pyarrow.parquet as pq

        s3_key = f"gold/dispatch_schedules/settlementDate={settlement_date}/schedule.parquet"
        
        # Serialize to in-memory buffer to avoid local file pollution
        table = pa.Table.from_pandas(df_schedule)
        buffer = io.BytesIO()
        pq.write_table(table, buffer, compression="snappy")
        buffer.seek(0)

        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=s3_key,
            Body=buffer.getvalue(),
            ContentType="application/octet-stream",
        )

        s3_uri = f"s3://{self.bucket_name}/{s3_key}"
        logger.info("Gold dispatch schedule uploaded to %s", s3_uri)
        return s3_uri