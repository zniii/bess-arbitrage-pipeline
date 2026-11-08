"""
Elexon Insights Solution REST API Ingestion Client.

Commercial & Domain Context:
----------------------------
In the GB electricity market, the Balancing Mechanism (BM) and settlement cash-out
prices are administered by Elexon under the Balancing and Settlement Code (BSC).
Battery Energy Storage Systems (BESS) trade around these prices either directly
(via imbalance exposure) or by using system prices as a benchmark for day-ahead
and intraday arbitrage spreads.

Data Contract & Endpoint Notes:
-------------------------------
- Provider: Elexon Insights Solution (successor to legacy BMRS)
- Target Endpoint: /balancing/settlement/system-prices/{settlementDate}
- Grain: Half-hourly Settlement Periods (typically 48 periods per calendar day,
  varying to 46 or 50 during daylight saving shifts).
- Pricing Model: Post-P305 reform utilizes a Single Cash-Out Price where
  systemSellPrice == systemBuyPrice in almost all operational scenarios.
"""

import logging
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

# Configure module-level logger inheriting the root application settings
logger = logging.getLogger(__name__)


class ElexonClient:
    """
    HTTP client responsible for extracting wholesale settlement price telemetry
    from the public Elexon Insights API.
    
    Attributes:
        BASE_URL (str): The root path parameter endpoint for daily system prices.
        timeout (int): Socket read/connect timeout in seconds to avoid hanging workers.
        max_retries (int): Maximum consecutive retry attempts per endpoint request.
        backoff_factor (float): Multiplier used to compute exponential backoff intervals.
        session (requests.Session): Persistent HTTP session enabling connection pooling.
    """

    BASE_URL: str = "https://data.elexon.co.uk/bmrs/api/v1/balancing/settlement/system-prices"

    def __init__(
        self,
        timeout: int = 15,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
    ) -> None:
        """
        Initializes the client with connection pooling and resilient network defaults.

        Args:
            timeout: Maximum seconds to wait for server response before raising Timeout.
            max_retries: Total attempts to resolve temporary network dropouts or 429s.
            backoff_factor: Multiplier for exponential delay: wait = backoff_factor ** attempt.
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

        # Persistent connection pool: avoids renegotiating TCP/TLS handshakes
        # on every daily request when iterating through historical months.
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "BESS-Dispatch-Pipeline/0.1.0",
        })

    def _get_daily_prices(self, date_str: str) -> list[dict] | None:
        """
        Fetches all half-hourly settlement periods for a single target calendar day.

        The Elexon Insights API structures this resource using path parameters
        rather than query parameters (e.g., .../system-prices/2026-08-01).

        Fault Tolerance Strategy:
        -------------------------
        1. HTTP 429 (Rate Limit): Back off exponentially and retry without terminating the batch.
        2. HTTP 5xx / Network Drops: Catch transient socket errors and retry up to max_retries.
        3. Hard Failures: Raise HTTPError on client errors (e.g., 404, 401) or exhausted retries.

        Args:
            date_str: Target settlement date formatted as 'YYYY-MM-DD'.

        Returns:
            list[dict]: Unprocessed JSON list of settlement records, or None if unavailable.
        """
        url = f"{self.BASE_URL}/{date_str}"
        params = {"format": "json"}

        for attempt in range(1, self.max_retries + 1):
            try:
                # Issue the HTTP GET with an enforced timeout to protect worker threads
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.timeout,
                )

                # Defensive handling for Elexon rate throttling
                if response.status_code == 429:
                    wait_time = self.backoff_factor ** attempt
                    logger.warning(
                        "Rate limit hit (HTTP 429) on %s. Throttling for %.1f seconds "
                        "(Attempt %d/%d)...",
                        date_str,
                        wait_time,
                        attempt,
                        self.max_retries,
                    )
                    time.sleep(wait_time)
                    continue

                # Raise an HTTPError if the response status is 4xx or 5xx
                response.raise_for_status()
                payload = response.json()
                
                # Elexon schema wraps the payload rows inside the top-level 'data' key
                return payload.get("data", [])

            except (requests.exceptions.RequestException, requests.exceptions.Timeout) as err:
                wait_time = self.backoff_factor ** attempt
                logger.error(
                    "Network error fetching date %s: %s. Retrying in %.1f seconds "
                    "(Attempt %d/%d)...",
                    date_str,
                    err,
                    wait_time,
                    attempt,
                    self.max_retries,
                )
                
                # If we have reached the final attempt, fail loud so orchestration detects it
                if attempt == self.max_retries:
                    logger.critical(
                        "Exhausted all %d retries for date %s. Aborting extraction.",
                        self.max_retries,
                        date_str,
                    )
                    raise
                
                time.sleep(wait_time)

        return None

    def fetch_system_prices(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Extracts settlement prices across an inclusive-exclusive date range.

        Iterates sequentially day-by-day to comply with Elexon's single-day
        path parameter structure while appending records into an in-memory batch.

        Args:
            start_date: Beginning of extraction window 'YYYY-MM-DD' (inclusive).
            end_date: Termination boundary 'YYYY-MM-DD' (exclusive).

        Returns:
            pd.DataFrame: Tabular raw settlement period data across the requested period.

        Raises:
            ValueError: If start_date is equal to or occurs after end_date.
        """
        # Validate temporal logic before initiating network traffic
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        final_end = datetime.strptime(end_date, "%Y-%m-%d")

        if current_date >= final_end:
            raise ValueError("start_date must be strictly before end_date")

        records: list[dict] = []

        # Sequential day-level chunking loop
        while current_date < final_end:
            date_str = current_date.strftime("%Y-%m-%d")
            logger.info("Extracting Elexon settlement prices for: %s", date_str)

            day_data = self._get_daily_prices(date_str)
            if day_data:
                records.extend(day_data)

            # Advance by exactly 1 calendar day
            current_date += timedelta(days=1)
            
            # Politeness interval (200ms) to avoid tripping Elexon's per-minute rate thresholds
            time.sleep(0.2)

        if not records:
            logger.warning("No records returned for date window: %s to %s", start_date, end_date)
            return pd.DataFrame()

        # Compile flat list of record dictionaries into a structured DataFrame
        df = pd.DataFrame(records)
        logger.info(
            "Successfully extracted %d settlement records across %s to %s.",
            len(df),
            start_date,
            end_date,
        )
        return df

    def fetch_daily_raw_records(self, date_str: str) -> list[dict]:
        """
        Public gateway to fetch raw unparsed JSON records for a single settlement day.
        Ideal for landing raw payloads directly into the Bronze data lake.
        """
        records = self._get_daily_prices(date_str)
        return records or []
    # ------------------------------------------------------------------
    # Dataset endpoints (verified live): MID, WINDFOR, NDFD
    # ------------------------------------------------------------------
    DATASETS_URL: str = "https://data.elexon.co.uk/bmrs/api/v1/datasets"

    def _get_dataset(self, dataset: str, params: dict) -> list[dict]:
        """GET /datasets/{dataset} with the same 429/5xx retry policy as system prices."""
        url = f"{self.DATASETS_URL}/{dataset}"
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(url, params={**params, "format": "json"}, timeout=self.timeout)
                if resp.status_code == 429:
                    time.sleep(self.backoff_factor ** attempt)
                    continue
                resp.raise_for_status()
                return resp.json().get("data", [])
            except requests.exceptions.RequestException as err:
                if attempt == self.max_retries:
                    logger.critical("Exhausted retries for dataset %s: %s", dataset, err)
                    raise
                time.sleep(self.backoff_factor ** attempt)
        return []

    def _chunked(self, dataset: str, start: str, end: str, from_key: str, to_key: str,
                 chunk_days: int = 7) -> pd.DataFrame:
        """Pull [start, end) in <=7-day windows (API range limit) and concatenate."""
        cur = datetime.strptime(start, "%Y-%m-%d")
        stop = datetime.strptime(end, "%Y-%m-%d")
        rows: list[dict] = []
        while cur < stop:
            nxt = min(cur + timedelta(days=chunk_days), stop)
            rows += self._get_dataset(dataset, {
                from_key: cur.strftime("%Y-%m-%dT%H:%MZ"),
                to_key: nxt.strftime("%Y-%m-%dT%H:%MZ"),
            })
            cur = nxt
            time.sleep(0.2)
        return pd.DataFrame(rows)

    def fetch_mid(self, start: str, end: str) -> pd.DataFrame:
        """
        Market Index Data, half-hourly, [start, end) in UTC days.

        Only the APX provider is kept: N2EX reports price 0 / volume 0 in live data.
        Returns ['interval_start_utc', 'mid_price_gbp', 'mid_volume_mwh'].
        """
        df = self._chunked("MID", start, end, "from", "to")
        if df.empty:
            return pd.DataFrame(columns=["interval_start_utc", "mid_price_gbp", "mid_volume_mwh"])
        df = df[df["dataProvider"] == "APXMIDP"]
        out = pd.DataFrame({
            "interval_start_utc": pd.to_datetime(df["startTime"], utc=True),
            "mid_price_gbp": df["price"].astype(float),
            "mid_volume_mwh": df["volume"].astype(float),
        })
        return out.drop_duplicates("interval_start_utc", keep="last").sort_values("interval_start_utc").reset_index(drop=True)

    def fetch_wind_forecasts(self, start: str, end: str) -> pd.DataFrame:
        """All WINDFOR publications with publishTime in [start, end): publishTime, startTime, generation."""
        df = self._chunked("WINDFOR", start, end, "publishDateTimeFrom", "publishDateTimeTo")
        if df.empty:
            return pd.DataFrame(columns=["publishTime", "startTime", "generation"])
        df["publishTime"] = pd.to_datetime(df["publishTime"], utc=True)
        df["startTime"] = pd.to_datetime(df["startTime"], utc=True)
        return df[["publishTime", "startTime", "generation"]]

    def fetch_demand_forecasts(self, start: str, end: str) -> pd.DataFrame:
        """All NDFD (national day-ahead demand forecast) publications: publishTime, forecastDate, demand."""
        df = self._chunked("NDFD", start, end, "publishDateTimeFrom", "publishDateTimeTo")
        if df.empty:
            return pd.DataFrame(columns=["publishTime", "forecastDate", "demand"])
        df["publishTime"] = pd.to_datetime(df["publishTime"], utc=True)
        df["forecastDate"] = pd.to_datetime(df["forecastDate"])
        return df[["publishTime", "forecastDate", "demand"]]
