"""Daily-update mechanics with an in-memory fake lake and SYNTHETIC prices (no network)."""

from datetime import datetime

import numpy as np
import pandas as pd

from bess_pipeline.orchestration.daily_update import run_daily_update


class FakeLake:
    def __init__(self, silver: pd.DataFrame):
        self.silver = silver
        self.schedules: dict[str, pd.DataFrame] = {}
        self.kpis: dict[str, dict] = {}

    def list_silver_dates(self):
        return set(self.silver["settlementDate"].unique())

    def list_gold_dates(self):
        return set(self.schedules) & set(self.kpis)

    def read_silver_history(self, start, end):
        m = (self.silver["settlementDate"] >= start) & (self.silver["settlementDate"] <= end)
        return self.silver[m].copy()

    def upload_gold_schedule_to_s3(self, df, settlement_date):
        self.schedules[settlement_date] = df

    def upload_daily_kpis_to_s3(self, row, settlement_date):
        self.kpis[settlement_date] = row


def make_silver(end: datetime, days: int) -> pd.DataFrame:
    start = pd.Timestamp(end.date(), tz="UTC") - pd.Timedelta(days=days - 1)
    idx = pd.date_range(start, periods=days * 48, freq="30min")
    rng = np.random.default_rng(1)
    tod = np.arange(len(idx)) % 48
    price = 60 + 40 * np.sin(2 * np.pi * (tod - 12) / 48) + rng.normal(0, 3, len(idx))
    df = pd.DataFrame({"interval_start_utc": idx, "system_price_gbp": price})
    df["settlementDate"] = df["interval_start_utc"].dt.strftime("%Y-%m-%d")
    return df


def test_backfills_window_then_is_noop():
    end = datetime(2026, 9, 30)
    lake = FakeLake(make_silver(end, days=40))
    first = run_daily_update(window_days=5, history_days=20, as_of=end, lake=lake, client=object())
    assert len(first) == 5
    assert set(lake.kpis) == {f"2026-09-{d}" for d in range(26, 31)}
    for row in lake.kpis.values():
        assert row["profit_gbp"] <= row["perfect_foresight_profit_gbp"] + 1e-6
        assert "price_forecast" in lake.schedules[row["settlementDate"]].columns
    second = run_daily_update(window_days=5, history_days=20, as_of=end, lake=lake, client=object())
    assert second == []
