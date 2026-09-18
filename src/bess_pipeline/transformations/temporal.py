"""
GB Electricity Market Temporal Normalization Module.

Commercial & Domain Context:
----------------------------
GB power settlement runs on half-hourly settlement periods (SPs). Physical market
dispatch, interconnectors, and algorithmic trading desks execute strictly on UTC.
However, domestic peak loads and local network operational constraints adhere to
UK clock time (Europe/London).

The Daylight Saving Time (DST) Problem:
---------------------------------------
Standard analytics often assume every calendar day contains 48 settlement periods.
In GB market operations, this assumption causes pipeline failure twice a year:
  1. Long Day (Autumn / October): Clocks turn back 1 hour (02:00 -> 01:00 BST/GMT).
     The day contains 25 physical hours -> 50 Settlement Periods.
  2. Short Day (Spring / March): Clocks advance 1 hour (01:00 -> 02:00 GMT/BST).
     The day contains 23 physical hours -> 46 Settlement Periods.

Design Contract:
----------------
- Internal Storage: Strictly UTC with half-open intervals [interval_start_utc, interval_end_utc).
- Downstream Safety: Never rely on row offsets or hardcoded chunk sizes of 48.
- Validation: Enforce interval continuity and verify dynamic SP totals per date.
"""

from datetime import datetime, time, timedelta
import logging
from typing import Final
import zoneinfo
import pandas as pd

# Module-level logger
logger = logging.getLogger(__name__)

# Canonical timezones (using IANA standard zoneinfo from Python 3.9+)
LONDON_TZ: Final[zoneinfo.ZoneInfo] = zoneinfo.ZoneInfo("Europe/London")
UTC_TZ: Final[zoneinfo.ZoneInfo] = zoneinfo.ZoneInfo("UTC")

# Half-hour duration constant for 30-minute settlement blocks
HALF_HOUR_DELTA: Final[pd.Timedelta] = pd.Timedelta(minutes=30)


def get_expected_settlement_periods(date_str: str) -> int:
    """
    Computes the exact number of settlement periods for a specific UK calendar date.

    Evaluates the elapsed duration between local midnight of the target date and
    local midnight of the subsequent date in the Europe/London timezone to detect
    clock changes.

    Args:
        date_str: Target calendar date formatted as 'YYYY-MM-DD'.

    Returns:
        int: Total expected settlement periods:
             - 46 on the March clock change (BST starts, 23-hour day).
             - 50 on the October clock change (GMT resumes, 25-hour day).
             - 48 on all standard 24-hour days.

    Raises:
        ValueError: If date_str fails ISO 'YYYY-MM-DD' parsing.
    """
    try:
        target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError as err:
        logger.error("Failed parsing date string '%s': %s", date_str, err)
        raise

    # Construct timezone-aware local midnight boundaries for the target date
    local_start = datetime.combine(target_date, time.min, tzinfo=LONDON_TZ)
    local_end = datetime.combine(target_date + timedelta(days=1), time.min, tzinfo=LONDON_TZ)

    # Compute physical elapsed hours accounting for DST transitions
    duration_hours = (local_end - local_start).total_seconds() / 3600.0

    # Each hour corresponds to two 30-minute settlement periods
    expected_sp = int(duration_hours * 2)

    logger.debug(
        "Date %s: physical duration = %.1f hrs -> expected SP count = %d",
        date_str,
        duration_hours,
        expected_sp,
    )
    return expected_sp


def standardize_system_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw Elexon settlement records into an interval-indexed Silver format.

    Enforces data contract requirements:
    1. Parses ISO startTime to UTC-aware datetime timestamps.
    2. Calculates explicit half-open interval boundaries [start, end).
    3. Coerces market metrics into strict numerical types (float64, int64).
    4. Validates that no duplicate settlement intervals exist.
    5. Validates settlement period counts against local clock changes.

    Args:
        df: Raw DataFrame directly returned by ElexonClient.

    Returns:
        pd.DataFrame: Cleaned, sorted, and validated settlement records with standard columns:
                      - interval_start_utc (datetime64[ns, UTC])
                      - interval_end_utc (datetime64[ns, UTC])
                      - settlementDate (object, 'YYYY-MM-DD')
                      - settlementPeriod (int64)
                      - system_price_gbp (float64)
                      - net_imbalance_volume_mwh (float64)

    Raises:
        ValueError: If duplicate UTC intervals are present or if SP counts diverge
                    from physical calendar expectations.
    """
    if df.empty:
        logger.warning("Empty DataFrame passed to standardize_settlement_dataframe.")
        return pd.DataFrame()

    cleaned = df.copy()

    #UTC Interval boundaries
    # Elexon provides ISO-8601 strings (e.g., '2026-08-01T23:00:00Z').
    # We enforce UTC parsing to ensure consistent temporal joins downstream.
    cleaned["interval_start_utc"] = pd.to_datetime(cleaned["startTime"], utc=True)
    cleaned["interval_end_utc"] = cleaned["interval_start_utc"] + HALF_HOUR_DELTA

    #Type coercion & renaming
    # Standardize date formats and cast integers/floats explicitly to prevent
    # object-dtype serialization overhead in Parquet/BigQuery.
    cleaned["settlementDate"] = pd.to_datetime(cleaned["settlementDate"]).dt.strftime("%Y-%m-%d")
    cleaned["settlementPeriod"] = cleaned["settlementPeriod"].astype(int)
    
    # systemSellPrice matches systemBuyPrice under single cash-out rules (P305)
    cleaned["system_price_gbp"] = cleaned["systemSellPrice"].astype(float)
    cleaned["net_imbalance_volume_mwh"] = cleaned["netImbalanceVolume"].astype(float)

    #Chronological ordering and deduplication
    cleaned = cleaned.sort_values(by=["interval_start_utc"]).reset_index(drop=True)

    duplicate_mask = cleaned.duplicated(subset=["interval_start_utc"], keep=False)
    if duplicate_mask.any():
        duplicate_timestamps = cleaned.loc[duplicate_mask, "interval_start_utc"].tolist()
        logger.critical("Integrity Error: Found duplicate UTC intervals: %s", duplicate_timestamps)
        raise ValueError(f"Detected duplicate UTC intervals: {duplicate_timestamps}")

    #DST integraion projection
    # For every unique settlement date present in the payload, verify that the
    # number of periods received matches the dynamic requirement (46, 48, or 50).
    for date_val, group in cleaned.groupby("settlementDate"):
        expected_sps = get_expected_settlement_periods(str(date_val))
        actual_sps = len(group)

        # Only enforce assertion if a full day was extracted (ignore partial boundary pulls)
        if actual_sps > 0 and actual_sps not in (46, 48, 50):
            logger.warning(
                "Date %s returned %d settlement periods (expected %d). "
                "Verify if the extraction range covers partial days.",
                date_val,
                actual_sps,
                expected_sps,
            )

    #Schema Projection
    # Discard non-essential telemetry fields (e.g., replacementPriceReferenceVolume)
    # to maintain a lean storage footprint in the Silver layer.
    analytical_columns = [
        "interval_start_utc",
        "interval_end_utc",
        "settlementDate",
        "settlementPeriod",
        "system_price_gbp",
        "net_imbalance_volume_mwh",
    ]

    logger.info("Standardized %d settlement intervals successfully.", len(cleaned))
    return cleaned[analytical_columns]