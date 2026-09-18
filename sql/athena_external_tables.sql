-- =============================================================================
-- AWS Athena / AWS Glue External Table Definitions
-- S3 Data Lake: bess-dispatch-datalake-jimee
-- =============================================================================

-- 1. Create Analytics Database
CREATE DATABASE IF NOT EXISTS bess_analytics;

-- 2. Silver Conformed System Prices Table (Partitioned by settlementDate)
CREATE EXTERNAL TABLE IF NOT EXISTS bess_analytics.silver_system_prices (
    interval_start_utc TIMESTAMP,
    interval_end_utc TIMESTAMP,
    settlementPeriod INT,
    system_price_gbp DOUBLE
)
PARTITIONED BY (
    settlementDate STRING
)
STORED AS PARQUET
LOCATION 's3://bess-dispatch-datalake-jimee/silver/system_prices/'
TBLPROPERTIES (
    'parquet.compression' = 'SNAPPY'
);

-- Discover newly written partitions in S3
MSCK REPAIR TABLE bess_analytics.silver_system_prices;

-- 3. Gold BESS Optimal Dispatch Schedules Table
CREATE EXTERNAL TABLE IF NOT EXISTS bess_analytics.gold_dispatch_schedules (
    interval_start_utc TIMESTAMP,
    system_price_gbp DOUBLE,
    optimal_charge_mw DOUBLE,
    optimal_discharge_mw DOUBLE,
    net_dispatch_mw DOUBLE,
    state_of_energy_mwh DOUBLE,
    price_rolling_mean_24h DOUBLE,
    daily_gross_spread DOUBLE,
    is_charge_opportunity BOOLEAN,
    is_discharge_opportunity BOOLEAN
)
PARTITIONED BY (
    settlementDate STRING
)
STORED AS PARQUET
LOCATION 's3://bess-dispatch-datalake-jimee/gold/dispatch_schedules/'
TBLPROPERTIES (
    'parquet.compression' = 'SNAPPY'
);

-- Discover newly written gold partitions in S3
MSCK REPAIR TABLE bess_analytics.gold_dispatch_schedules;