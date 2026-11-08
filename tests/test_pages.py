"""Smoke tests: both Streamlit pages load on the committed results in app/data (offline, no AWS)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app"


def run(path: Path) -> AppTest:
    return AppTest.from_file(str(path), default_timeout=90).run()


def test_findings_page_loads_with_headline_metrics():
    at = run(APP / "Findings.py")
    assert not at.exception, [e.value for e in at.exception]
    labels = [m.label for m in at.metric]
    assert "Our forecast model" in labels and "Forecast error (MAE)" in labels
    at.select_slider[0].set_value(at.select_slider[0].options[10]).run()   # pick another day
    assert not at.exception, [e.value for e in at.exception]


def test_max_revenue_page_loads_and_period_switching_works():
    at = run(APP / "pages" / "Max_Revenue.py")
    assert not at.exception, [e.value for e in at.exception]
    assert "Expected net profit" in [m.label for m in at.metric]
    for period in ("7 days", "30 days", "1 day"):
        at.radio[0].set_value(period).run()
        assert not at.exception, (period, [e.value for e in at.exception])
    at.sidebar.slider[0].set_value(15).run()
    assert not at.exception


@pytest.mark.parametrize("page", ["Findings.py", "pages/Max_Revenue.py"])
def test_pages_do_not_reference_removed_dashboard(page):
    assert "gold_data" not in (APP / page).read_text()
