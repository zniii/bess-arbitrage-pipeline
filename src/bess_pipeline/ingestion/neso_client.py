"""
NESO Data Portal client: N2EX GB day-ahead auction prices.

Dataset: "Day-Ahead Power Exchange prices (Nord Pool)", resource "N2EX GB Day-Ahead Price"
(https://www.neso.energy/data-portal/day-ahead-power-exchange-prices-nordpool).
Columns: Date, Delivery Period ('HH:MM - HH:MM'), Price (GBP/MWh). Verified live:
* hourly resolution, 24 rows per day, 2021-09-20 onwards, no gaps;
* timestamps are UTC (best alignment with Elexon MID; correlation 0.91);
* the auction for delivery day D clears on D-1 (~09:20 UK), so D-1 prices are public
  before our D-1 08:00 UTC decision time, and D's own prices are not.
"""

import io
import logging

import pandas as pd
import requests

logger = logging.getLogger(__name__)

RESOURCE_ID = "4f27eea5-7038-4f73-9740-e3e4ad47c26a"
DUMP_URL = f"https://api.neso.energy/datastore/dump/{RESOURCE_ID}"


def parse_n2ex_csv(text: str) -> pd.DataFrame:
    """CSV text -> ['interval_start_utc', 'da_price_gbp'] at HALF-HOUR resolution.

    The auction is hourly, so each hourly price applies to both half-hour settlement periods.
    """
    raw = pd.read_csv(io.StringIO(text))
    need = {"Date", "Delivery Period", "Price"}
    if not need <= set(raw.columns):
        raise ValueError(f"Unexpected NESO columns: {list(raw.columns)}")
    start = raw["Delivery Period"].str.slice(0, 5)
    ts = pd.to_datetime(raw["Date"].astype(str) + " " + start, utc=True)
    hourly = pd.Series(raw["Price"].astype(float).to_numpy(), index=ts).sort_index()
    hourly = hourly[~hourly.index.duplicated(keep="last")]
    half = hourly.resample("30min").ffill()
    # resample ffill extends only up to the last hourly stamp; add its second half-hour
    half.loc[hourly.index[-1] + pd.Timedelta(minutes=30)] = hourly.iloc[-1]
    return pd.DataFrame({"interval_start_utc": half.index, "da_price_gbp": half.to_numpy()}).reset_index(drop=True)


class NesoClient:
    def __init__(self, timeout: int = 60, session: requests.Session | None = None) -> None:
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "BESS-Dispatch-Pipeline/0.1.0"})

    def fetch_day_ahead(self) -> pd.DataFrame:
        """Full N2EX GB day-ahead history (~1.6 MB), half-hourly."""
        resp = self.session.get(DUMP_URL, timeout=self.timeout)
        resp.raise_for_status()
        df = parse_n2ex_csv(resp.text)
        logger.info("Fetched %d N2EX day-ahead half-hours, %s to %s", len(df),
                    df["interval_start_utc"].min(), df["interval_start_utc"].max())
        return df
