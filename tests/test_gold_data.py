"""Dashboard data access against real Parquet files in legacy, new and mixed Gold layouts (local, no AWS)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import gold_data


def _day(date: str, new: bool, strategy: str = "da_forecast") -> pd.DataFrame:
    ts = pd.date_range(pd.Timestamp(date, tz="UTC"), periods=48, freq="30min")
    price = 100 + 50 * np.sin(np.arange(48) / 48 * 2 * np.pi)
    df = pd.DataFrame({
        "interval_start_utc": ts, "settlementDate": date, "settlementPeriod": np.arange(1, 49),
        "system_price_gbp": price,
        "optimal_charge_mw": np.where(np.arange(48) < 4, 10.0, 0.0),
        "optimal_discharge_mw": np.where((np.arange(48) >= 30) & (np.arange(48) < 34), 10.0, 0.0),
        "state_of_energy_mwh": 5.0,
    })
    df["net_dispatch_mw"] = df["optimal_discharge_mw"] - df["optimal_charge_mw"]
    if new:
        df["strategy"] = strategy
        df["da_price_gbp"] = price
        df["settlement_price_gbp"] = price
        df["price_forecast"] = price * 1.05
    else:
        df["price_rolling_mean_24h"] = 100.0  # extra legacy feature columns
    return df


def _write(root: Path, kind: str, date: str, df: pd.DataFrame, fname: str) -> None:
    d = root / "gold" / kind / f"settlementDate={date}"
    d.mkdir(parents=True)
    df.to_parquet(d / fname, index=False)


def _kpi(date: str) -> pd.DataFrame:
    return pd.DataFrame([{"settlementDate": date, "forecaster_da": "gbm", "profit_gbp": 10.0, "profit_da_forecast_gbp": 10.0,
                          "profit_da_perfect_gbp": 12.0, "equivalent_full_cycles": 0.5, "da_forecast_mae_gbp_per_mwh": 20.0}])


@pytest.fixture
def con():
    return gold_data.connect(local=True)


def test_empty_folder_returns_empty_frames(tmp_path, con):
    assert gold_data.load_schedules(con, str(tmp_path)).empty
    assert gold_data.load_kpis(con, str(tmp_path)).empty


def test_legacy_only_schedules_are_normalised(tmp_path, con):
    for d in ("2026-08-01", "2026-08-02"):
        _write(tmp_path, "dispatch_schedules", d, _day(d, new=False), "schedule.parquet")
    s = gold_data.load_schedules(con, str(tmp_path))
    assert set(s["strategy"]) == {gold_data.LEGACY_STRATEGY}
    assert s["price_forecast"].isna().all() and (s["settlement_price_gbp"] == s["system_price_gbp"]).all()
    assert sorted(s["settlementDate"].unique()) == ["2026-08-01", "2026-08-02"]
    k = gold_data.derive_kpis(s)
    assert len(k) == 2 and "profit_legacy_imbalance_perfect_gbp" in k.columns
    assert k["equivalent_full_cycles"].iloc[0] == pytest.approx(20.0 / 16.0)  # 4 periods x 10 MW x 0.5 h exported = 20 MWh


def test_new_and_mixed_partitions(tmp_path, con):
    _write(tmp_path, "dispatch_schedules", "2026-08-01", _day("2026-08-01", new=False), "schedule.parquet")  # legacy
    new = pd.concat([_day("2026-08-02", True, "da_forecast"), _day("2026-08-02", True, "da_perfect")])
    _write(tmp_path, "dispatch_schedules", "2026-08-02", new, "schedule.parquet")
    _write(tmp_path, "daily_kpis", "2026-08-02", _kpi("2026-08-02"), "kpis.parquet")
    s = gold_data.load_schedules(con, str(tmp_path))
    assert set(s["strategy"]) == {gold_data.LEGACY_STRATEGY, "da_forecast", "da_perfect"}
    assert s.loc[s["settlementDate"] == "2026-08-01", "strategy"].eq(gold_data.LEGACY_STRATEGY).all()
    k = gold_data.load_kpis(con, str(tmp_path))
    assert k["settlementDate"].tolist() == ["2026-08-02"] and k["profit_da_forecast_gbp"].iloc[0] == 10.0
    assert gold_data.derive_kpis(s)["profit_da_forecast_gbp"].notna().sum() == 1


# ---- full dashboard under Streamlit's test harness, against local folders -------------------------
APP = str(Path(__file__).resolve().parent.parent / "app" / "dashboard.py")


def _run_app(monkeypatch, root):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("BESS_LOCAL_DATA_DIR", str(root))
    return AppTest.from_file(APP, default_timeout=60).run()


def _legacy_days(root, dates):
    for d in dates:
        _write(root, "dispatch_schedules", d, _day(d, new=False), "schedule.parquet")


def _new_days(root, dates, with_kpis=True):
    for d in dates:
        both = pd.concat([_day(d, True, "da_forecast"), _day(d, True, "da_perfect")])
        _write(root, "dispatch_schedules", d, both, "schedule.parquet")
        if with_kpis:
            _write(root, "daily_kpis", d, _kpi(d), "kpis.parquet")


def test_dashboard_empty_bucket_explains_what_to_run(tmp_path, monkeypatch):
    at = _run_app(monkeypatch, tmp_path)
    assert not at.exception and any("daily_update" in w.value for w in at.warning)


def test_dashboard_legacy_only(tmp_path, monkeypatch):
    _legacy_days(tmp_path, ["2026-08-01", "2026-08-02", "2026-08-03"])
    at = _run_app(monkeypatch, tmp_path)
    assert not at.exception, [e.value for e in at.exception]
    assert any("derived from the stored schedules" in i.value for i in at.info)
    assert any(m.label == "Net Profit (legacy schedules)" for m in at.metric)


def test_dashboard_new_schema_with_kpis(tmp_path, monkeypatch):
    _new_days(tmp_path, ["2026-08-01", "2026-08-02", "2026-08-03"])
    at = _run_app(monkeypatch, tmp_path)
    assert not at.exception, [e.value for e in at.exception]
    labels = [m.label for m in at.metric]
    assert "Net Profit (day-ahead forecast model)" in labels and "Forecast MAE" in labels


def test_dashboard_mixed_partitions_and_missing_kpis(tmp_path, monkeypatch):
    _legacy_days(tmp_path, ["2026-08-01"])
    _new_days(tmp_path, ["2026-08-02", "2026-08-03"], with_kpis=False)
    at = _run_app(monkeypatch, tmp_path)
    assert not at.exception, [e.value for e in at.exception]
    assert any("derived from the stored schedules" in i.value for i in at.info)
    at.sidebar.selectbox[0].set_value("2026-08-01").run()   # pick the legacy day: strategy list must adapt
    assert not at.exception, [e.value for e in at.exception]
