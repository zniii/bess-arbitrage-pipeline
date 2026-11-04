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
    price_forecast DOUBLE,
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

-- 4. Gold daily KPIs (one row per settled day)
CREATE EXTERNAL TABLE IF NOT EXISTS bess_analytics.gold_daily_kpis (
    forecaster STRING,
    forecast_mae_gbp_per_mwh DOUBLE,
    profit_gbp DOUBLE,
    perfect_foresight_profit_gbp DOUBLE,
    gross_revenue_gbp DOUBLE,
    charging_cost_gbp DOUBLE,
    export_mwh DOUBLE,
    import_mwh DOUBLE,
    equivalent_full_cycles DOUBLE,
    avg_capture_spread_gbp_per_mwh DOUBLE
)
PARTITIONED BY (settlementDate STRING)
STORED AS PARQUET
LOCATION 's3://bess-dispatch-datalake-jimee/gold/daily_kpis/';

MSCK REPAIR TABLE bess_analytics.gold_daily_kpis;
