"""
Findings: walk-forward backtest of day-ahead price forecasting for BESS arbitrage (N2EX GB day-ahead).

Reads the pre-computed results in ``app/data/`` (written by ``scripts/run_backtest_report.py``), so it
works offline and on hosted Streamlit without AWS credentials.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).resolve().parent.parent / "data"
GREEN, GREY, AMBER, BLACK, RED = "#10b981", "#94a3b8", "#f59e0b", "#0f172a", "#ef4444"

st.set_page_config(page_title="BESS Forecasting Findings", page_icon="🔮", layout="wide")


@st.cache_data
def load():
    daily = pd.read_csv(DATA / "rolling_daily.csv", parse_dates=["date"])
    plan = pd.read_csv(DATA / "plan_halfhourly.csv", parse_dates=["interval_start_utc"])
    summ = json.loads((DATA / "rolling_summary.json").read_text())
    meta = json.loads((DATA / "backtest_meta.json").read_text())
    return daily, plan, summ, meta


try:
    daily_all, plan, summ, meta = load()
except FileNotFoundError:
    st.error("Backtest results not found. Run `python scripts/run_backtest_report.py --start ... --end ...`.")
    st.stop()

daily = daily_all[daily_all["actual_known"]].copy()
B, A, OOS, ERR = summ["bridge_gbp"], summ["assumptions"], summ["out_of_sample_gbp"], summ["forecast_error_gbp_per_mwh"]
perfect, model, naive = B["perfect_after_costs_soc_carry"], B["achievable_after_costs_soc_carry"], B["naive_after_costs_soc_carry"]
b = meta["battery"]
wins = int((daily["net_profit_gbp"] > daily["naive_net_gbp"]).sum())

st.title("🔮 Can a forecast capture day-ahead battery arbitrage?")
st.markdown(
    f"A **{b['power_mw']:.0f} MW / {b['energy_mwh']:.0f} MWh** battery in Great Britain on the **N2EX day-ahead auction price**, "
    f"back-tested over **{meta['test_days']} days ({meta['test_start']} to {meta['test_end']})**. Each day is planned at "
    f"**{meta['decision_time']}**. Profits are **after costs** (degradation £{A['degradation_gbp_per_mwh_discharged']:.0f}/MWh discharged "
    f"[placeholder], trading fee £{A['trading_fee_gbp_per_mwh']:.2f}/MWh, {A['end_of_day_reserve_mwh']:.0f} MWh end-of-day reserve, state of charge carried across days)."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Perfect foresight (max)", f"£{perfect:,.0f}", help="Profit if every auction price were known in advance.")
c2.metric("Our forecast model", f"£{model:,.0f}", f"{model / perfect * 100:.0f}% of upper bound")
c3.metric("Naive (same hour last week)", f"£{naive:,.0f}", f"{naive / perfect * 100:.0f}% of upper bound", delta_color="off")
c4.metric("Forecast error (MAE)", f"£{ERR['model']:.1f}/MWh", f"{(ERR['model'] / ERR['naive'] - 1) * 100:.0f}% vs naive £{ERR['naive']:.1f}", delta_color="inverse")

st.success(
    f"**Finding:** the gradient-boosted model (price history, wind and demand forecasts) captures **{model / perfect * 100:.0f}%** of the "
    f"perfect-foresight profit and halves the forecast error, but the profit advantage over simply repeating last week's prices is small: "
    f"**{(model - naive) / perfect * 100:.0f} percentage points** over all {meta['test_days']} days, and "
    f"**{(OOS['achievable_after_costs'] - OOS['naive_after_costs']) / OOS['perfect_after_costs'] * 100:.0f} points** on the last {OOS['days']} days "
    f"(not used for tuning). It beat naive on **{wins} of {len(daily)} days**."
)

with st.expander("How it works", expanded=False):
    st.markdown(
        f"""
- **Price:** N2EX GB day-ahead auction (NESO open data), hourly, UTC. The auction for day D clears around 09:20 UK on D-1, so
  yesterday's prices are public at decision time and tomorrow's are not.
- **Plan:** a mixed-integer linear program (PuLP/CBC) with a {b['power_mw']:.0f} MW limit, {b['roundtrip_efficiency']:.0%} round-trip efficiency,
  10-90% state-of-charge window and an end-of-day reserve. Degradation and trading fees are in the objective.
- **Forecast model:** gradient-boosted regressor on calendar, lagged day-ahead prices (>= 1 day), rolling statistics, Elexon wind (WINDFOR)
  and national demand (NDFD) forecasts; retrained every {meta['retrain_every_days']} days. A seasonal-naive benchmark repeats the same hour a week earlier.
- **Value of leftover energy overnight** (£{A['terminal_value_gbp_per_mwh']:.0f}/MWh) was tuned on the first {A['tuned_on_days']} days only.
- **Settlement:** every plan is valued at the actual auction price afterwards.
"""
    )

st.divider()
left, right = st.columns([3, 2])

# --- Cumulative profit -------------------------------------------------------
with left:
    cum = daily.set_index("date")[["perfect_costed_net_gbp", "net_profit_gbp", "naive_net_gbp"]].cumsum()
    fig = go.Figure()
    for col, label, colour, dash in [("perfect_costed_net_gbp", "Perfect foresight", GREY, "solid"),
                                     ("net_profit_gbp", "Our forecast model", GREEN, "solid"),
                                     ("naive_net_gbp", "Naive (last week)", AMBER, "dash")]:
        fig.add_scatter(x=cum.index, y=cum[col], name=label, mode="lines", line=dict(color=colour, width=2.5, dash=dash))
    fig.update_layout(title="Cumulative net profit after costs (£)", height=380, hovermode="x unified",
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)

# --- Forecast error by month ------------------------------------------------
with right:
    m = daily.assign(month=daily["date"].dt.strftime("%Y-%m")).groupby("month").agg(
        model=("mae_model", "mean"), naive=("mae_naive", "mean"), days=("date", "count"))
    fig = go.Figure()
    fig.add_bar(x=m.index, y=m["naive"], name="Naive (last week)", marker_color=AMBER)
    fig.add_bar(x=m.index, y=m["model"], name="Our model", marker_color=GREEN)
    fig.update_layout(barmode="group", title="Day-ahead forecast error by month (£/MWh MAE)", height=380,
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)
    if m["days"].min() < 15:
        st.caption(f"Months with fewer than 15 test days ({', '.join(m.index[m['days'] < 15])}) are partial.")

st.divider()

# --- Forecast vs actual for a chosen day --------------------------------------
st.subheader("Forecast vs actual, day by day")
kd = plan[plan["actual_known"]]
dates = sorted(kd["date"].astype(str).unique())
pick = st.select_slider("Delivery day", options=dates, value=dates[-1])
day = kd[kd["date"].astype(str) == pick].copy()
day["local"] = day["interval_start_utc"].dt.tz_convert("Europe/London").dt.tz_localize(None)
fig = go.Figure()
fig.add_scatter(x=day["local"], y=day["da_actual"], name="Actual auction price", line=dict(color=BLACK, width=3, shape="hv"))
fig.add_scatter(x=day["local"], y=day["da_model"], name="Our forecast", line=dict(color=GREEN, width=2.5, shape="hv"))
fig.add_scatter(x=day["local"], y=day["da_naive"], name="Naive (last week)", line=dict(color=AMBER, width=2, dash="dash", shape="hv"))
fig.update_layout(height=360, hovermode="x unified", yaxis_title="£/MWh", margin=dict(l=10, r=10, t=10, b=10),
                  legend=dict(orientation="h", y=-0.15))
st.plotly_chart(fig, use_container_width=True)
row = daily[daily["date"].dt.strftime("%Y-%m-%d") == pick].iloc[0]
d1, d2, d3, d4 = st.columns(4)
d1.metric("Perfect-foresight profit", f"£{row.perfect_costed_net_gbp:,.0f}")
d2.metric("Model profit", f"£{row.net_profit_gbp:,.0f}")
d3.metric("Naive profit", f"£{row.naive_net_gbp:,.0f}")
d4.metric("Model MAE", f"£{row.mae_model:.1f}/MWh")

adv = daily.set_index("date")["net_profit_gbp"] - daily.set_index("date")["naive_net_gbp"]
fig = go.Figure(go.Bar(x=adv.index, y=adv.values, marker_color=[GREEN if v >= 0 else RED for v in adv.values]))
fig.update_layout(title="Daily profit: model minus naive (£)", height=300, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Summary")
summary = pd.DataFrame({
    "Strategy": ["Perfect foresight (upper bound)", "Our forecast model", "Naive (same hour last week)"],
    "Total net profit (£)": [perfect, model, naive],
    "£ per day": [perfect / len(daily), model / len(daily), naive / len(daily)],
    "% of upper bound": [100, model / perfect * 100, naive / perfect * 100],
    f"Last {OOS['days']} days (£)": [OOS["perfect_after_costs"], OOS["achievable_after_costs"], OOS["naive_after_costs"]],
})
st.dataframe(summary.style.format({"Total net profit (£)": "£{:,.0f}", "£ per day": "£{:,.0f}", "% of upper bound": "{:.0f}%",
                                   f"Last {OOS['days']} days (£)": "£{:,.0f}"}), hide_index=True, use_container_width=True)

with st.expander("Caveats and what this does not show", expanded=True):
    st.markdown(
        f"""
- **Idealised trading:** every plan is assumed to trade at the hourly auction price, with no volume limits or price impact. Real orders must be
  submitted before the price is known, and linking hours through the battery's state of charge is hard in practice.
- **Costs are placeholders:** degradation £{A['degradation_gbp_per_mwh_discharged']:.0f}/MWh is an assumption, the trading fee is £{A['trading_fee_gbp_per_mwh']:.2f}/MWh, and network/levy charges, taxes and revenue share are not included.
- **Limited sample:** {meta['test_days']} days in one season; model settings are untuned and this is a single run. Results will differ in winter and other years.
- **Other revenue streams** (frequency response, capacity market, Balancing Mechanism) are not modelled.
- Wind and demand forecast publication times are filtered to the decision time, but historical timestamps are taken as published by Elexon.
- Earlier versions of this project used Elexon MID (an index published as each half-hour happens) as the price. It is not tradable in advance and was replaced by the day-ahead auction price.
"""
    )

st.download_button("Download daily results (CSV)", (DATA / "rolling_daily.csv").read_bytes(), "bess_daily_results.csv", "text/csv")
st.caption(f"Generated {meta['generated_at_utc']} by scripts/run_backtest_report.py. Model: {meta['model']}.")
