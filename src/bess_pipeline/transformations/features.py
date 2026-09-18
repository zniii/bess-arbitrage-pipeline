"""
Gold Layer Feature Engineering Module for BESS Dispatch Optimization.

Commercial & Mathematical Context:
----------------------------------
Battery Energy Storage Systems (BESS) generate arbitrage revenue by:
  1. Charging during low/negative price periods (buying power).
  2. Discharging during high evening peak price periods (selling power).

To inform linear programming (LP) solvers and price forecasting models, this
module transforms Silver-level half-hourly price intervals into analytical
features:
  - 24-hour Rolling Mean & Standard Deviation (volatility & baseline trend).
  - Daily Price Spreads (gross available revenue envelope per settlement day).
  - Arbitrage Classification Flags (top/bottom price percentiles).
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def generate_bess_market_features(df_silver: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Gold-layer dispatch features from standardized Silver settlement prices.

    Args:
        df_silver: Standardized DataFrame from the Silver layer containing:
                   ['interval_start_utc', 'settlementDate', 'settlementPeriod', 'system_price_gbp']

    Returns:
        pd.DataFrame: Analytical feature mart with rolling statistics and arbitrage signals.

    Raises:
        ValueError: If input DataFrame lacks required Silver columns.
    """
    required = {"interval_start_utc", "settlementDate", "settlementPeriod", "system_price_gbp"}
    missing = required - set(df_silver.columns)
    if missing:
        raise ValueError(f"Input Silver DataFrame missing required columns: {missing}")

    if df_silver.empty:
        logger.warning("Empty Silver DataFrame provided for feature generation.")
        return pd.DataFrame()

    gold = df_silver.copy().sort_values("interval_start_utc").reset_index(drop=True)

    # -------------------------------------------------------------------------
    # 1. Rolling 24-Hour Metrics (48 half-hour settlement periods)
    # -------------------------------------------------------------------------
    gold["price_rolling_mean_24h"] = (
        gold["system_price_gbp"]
        .rolling(window=48, min_periods=1)
        .mean()
        .round(4)
    )
    gold["price_rolling_std_24h"] = (
        gold["system_price_gbp"]
        .rolling(window=48, min_periods=1)
        .std()
        .fillna(0.0)
        .round(4)
    )

    # -------------------------------------------------------------------------
    # 2. Daily Intraday Spreads (Gross Arbitrage Envelope)
    # -------------------------------------------------------------------------
    daily_stats = (
        gold.groupby("settlementDate")["system_price_gbp"]
        .agg(
            daily_min_price="min",
            daily_max_price="max",
        )
        .reset_index()
    )
    daily_stats["daily_gross_spread"] = (
        daily_stats["daily_max_price"] - daily_stats["daily_min_price"]
    ).round(4)

    # Left join to preserve original row count and columns
    gold = gold.merge(daily_stats, on="settlementDate", how="left")

    # -------------------------------------------------------------------------
    # 3. Vectorized Arbitrage Signals (Index-Safe)
    # -------------------------------------------------------------------------
    # Using transform avoids Pandas index-mangling and keeps settlementDate as a column
    q25 = gold.groupby("settlementDate")["system_price_gbp"].transform(lambda s: s.quantile(0.25))
    q75 = gold.groupby("settlementDate")["system_price_gbp"].transform(lambda s: s.quantile(0.75))

    gold["is_charge_opportunity"] = gold["system_price_gbp"] <= q25
    gold["is_discharge_opportunity"] = gold["system_price_gbp"] >= q75

    # Explicitly guarantee settlementDate remains a plain column
    gold = gold.reset_index(drop=True)

    logger.info("Successfully engineered Gold features for %d intervals.", len(gold))
    return gold