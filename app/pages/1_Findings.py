"""
Findings: walk-forward backtest of the BESS dispatch strategies on real Elexon data.

Reads the pre-computed results in ``app/data/`` (produced by
``scripts/run_backtest_report.py``), so it works offline and on hosted Streamlit
without AWS credentials.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).resolve().parent.parent / "data"
GREEN, GREY, AMBER, PURPLE = "#10b981", "#94a3b8", "#f59e0b", "#8b5cf6"

st.set_page_config(page_title="BESS Forecasting Findings", page_icon="🔮", layout="wide")


@st.cache_data
def load():
    daily = pd.read_csv(DATA / "backtest_daily.csv", parse_dates=["date"])
    sample = pd.read_csv(DATA / "mid_forecast_sample.csv", parse_dates=["interval_start_utc"])
    meta = json.loads((DATA / "backtest_meta.json").read_text())
    return daily, sample, meta


try:
    daily, sample, meta = load()
except FileNotFoundError:
    st.error("Backtest results not found. Run `python scripts/run_backtest_report.py --start ... --end ...`.")
    st.stop()

tot = daily.filter(like="profit_").sum()
base, model, naive = tot["profit_mid_baseline"], tot["profit_mid_forecast"], tot["profit_mid_naive"]
mae_model, mae_naive = daily["mae_mid_model"].mean(), daily["mae_mid_naive"].mean()
b = meta["battery"]

st.title("🔮 Can a forecast capture day-ahead battery arbitrage?")
st.markdown(
    f"A **{b['power_mw']:.0f} MW / {b['energy_mwh']:.0f} MWh** battery in Great Britain, back-tested over "
    f"**{meta['test_days']} days ({meta['test_start']} to {meta['test_end']})** on real Elexon data. "
    f"Each day's schedule is decided using only what was known at **{meta['decision_time']}**, "
    f"with models trained on data from {meta['history_start']} up to the previous retrain."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("MID baseline profit", f"£{base:,.0f}", help="Day-ahead arbitrage if the auction prices were known in advance.")
c2.metric("Our forecast model", f"£{model:,.0f}", f"{model / base * 100:.0f}% of baseline")
c3.metric("Naive forecast (last week)", f"£{naive:,.0f}", f"{naive / base * 100:.0f}% of baseline", delta_color="off")
c4.metric("Forecast error (MAE)", f"£{mae_model:.1f}/MWh", f"{(mae_model / mae_naive - 1) * 100:.0f}% vs naive £{mae_naive:.1f}", delta_color="inverse")

st.success(
    f"**Finding:** the gradient-boosted model, using wind and demand forecasts plus price history, captures "
    f"**{model / base * 100:.0f}%** of the MID baseline profit versus **{naive / base * 100:.0f}%** for repeating "
    f"last week's prices, and cuts forecast error by **{(1 - mae_model / mae_naive) * 100:.0f}%**. "
    f"It beat the naive forecast on **{int((daily.profit_mid_forecast > daily.profit_mid_naive).sum())} of "
    f"{len(daily)} days**."
)

with st.expander("How the strategies work", expanded=False):
    st.markdown(
        """
| Strategy | Optimizes on | Settled at | Role |
|---|---|---|---|
| **MID baseline** | published day-ahead (MID) price | MID | Benchmark: auction result known |
| **MID forecast model** | our forecast of MID, made before the auction | actual MID | Headline model |
| **MID naive** | same half-hour one week earlier | actual MID | Simple benchmark |
| **Imbalance forecast** *(experimental)* | forecast imbalance price | imbalance price | Different market |
| **Imbalance perfect foresight** *(experimental)* | actual imbalance price | imbalance price | Unattainable upper bound |

The optimizer is a mixed-integer linear program (PuLP/CBC) with a 10 MW power limit, 85% round-trip
efficiency and a 10-90% state-of-charge window; each day starts at the minimum state of charge.
The forecast model is a gradient-boosted regressor on calendar features, lagged prices, Elexon wind (WINDFOR)
and national demand (NDFD) forecasts, retrained every """
        + f"{meta['retrain_every_days']} days."
    )

st.divider()
left, right = st.columns([3, 2])

# --- Cumulative profit -------------------------------------------------------
with left:
    show_imb = st.checkbox("Show imbalance-price strategies (experimental, different settlement price)", value=False)
    cum = daily.set_index("date").filter(like="profit_").cumsum()
    fig = go.Figure()
    lines = [("profit_mid_baseline", "MID baseline", GREY, "solid"),
             ("profit_mid_forecast", "MID forecast model", GREEN, "solid"),
             ("profit_mid_naive", "MID naive (last week)", AMBER, "dash")]
    if show_imb:
        lines += [("profit_imbalance_forecast", "Imbalance forecast model", PURPLE, "solid"),
                  ("profit_imbalance_perfect", "Imbalance perfect foresight", PURPLE, "dot")]
    for col, label, colour, dash in lines:
        fig.add_scatter(x=cum.index, y=cum[col], name=label, mode="lines", line=dict(color=colour, width=2.5, dash=dash))
    fig.update_layout(title="Cumulative net profit (£)", height=380, hovermode="x unified",
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)

# --- Forecast error by month ------------------------------------------------
with right:
    m = daily.assign(month=daily["date"].dt.strftime("%Y-%m")).groupby("month").agg(
        model=("mae_mid_model", "mean"), naive=("mae_mid_naive", "mean"), days=("date", "count"))
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
dates = sorted(sample["date"].dt.strftime("%Y-%m-%d").unique() if hasattr(sample["date"], "dt") else sample["date"].unique())
pick = st.select_slider("Delivery day", options=dates, value=dates[-1])
day = sample[sample["date"].astype(str).str[:10] == pick]
fig = go.Figure()
fig.add_scatter(x=day["interval_start_utc"], y=day["mid_actual"], name="Actual MID", line=dict(color="#0f172a", width=3))
fig.add_scatter(x=day["interval_start_utc"], y=day["mid_model"], name="Our forecast", line=dict(color=GREEN, width=2.5))
fig.add_scatter(x=day["interval_start_utc"], y=day["mid_naive"], name="Naive (last week)", line=dict(color=AMBER, width=2, dash="dash"))
fig.update_layout(height=360, hovermode="x unified", yaxis_title="£/MWh", margin=dict(l=10, r=10, t=10, b=10),
                  legend=dict(orientation="h", y=-0.15))
st.plotly_chart(fig, use_container_width=True)
row = daily[daily["date"].dt.strftime("%Y-%m-%d") == pick].iloc[0]
d1, d2, d3, d4 = st.columns(4)
d1.metric("Baseline profit", f"£{row.profit_mid_baseline:,.0f}")
d2.metric("Model profit", f"£{row.profit_mid_forecast:,.0f}")
d3.metric("Naive profit", f"£{row.profit_mid_naive:,.0f}")
d4.metric("Model MAE", f"£{row.mae_mid_model:.1f}/MWh")

# --- Daily advantage over naive ------------------------------------------------
adv = daily.set_index("date")["profit_mid_forecast"] - daily.set_index("date")["profit_mid_naive"]
fig = go.Figure(go.Bar(x=adv.index, y=adv.values, marker_color=[GREEN if v >= 0 else "#ef4444" for v in adv.values]))
fig.update_layout(title="Daily profit: model minus naive (£)", height=300, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Summary by strategy")
summary = pd.DataFrame({
    "Strategy": ["MID baseline", "MID forecast model", "MID naive", "Imbalance forecast model*", "Imbalance perfect foresight*"],
    "Settled at": ["MID", "MID", "MID", "Imbalance", "Imbalance"],
    "Total profit (£)": [tot[c] for c in ["profit_mid_baseline", "profit_mid_forecast", "profit_mid_naive",
                                           "profit_imbalance_forecast", "profit_imbalance_perfect"]],
    "£ per day": [tot[c] / len(daily) for c in ["profit_mid_baseline", "profit_mid_forecast", "profit_mid_naive",
                                                 "profit_imbalance_forecast", "profit_imbalance_perfect"]],
})
summary["% of own-market upper bound"] = [100, model / base * 100, naive / base * 100,
                                          tot["profit_imbalance_forecast"] / tot["profit_imbalance_perfect"] * 100, 100]
st.dataframe(summary.style.format({"Total profit (£)": "£{:,.0f}", "£ per day": "£{:,.0f}", "% of own-market upper bound": "{:.0f}%"}),
             hide_index=True, use_container_width=True)
st.caption("*Experimental. Imbalance and MID profits use different settlement prices and are not directly comparable. "
           "'Upper bound' = perfect foresight within the same market.")

with st.expander("Caveats and what this does not show", expanded=True):
    st.markdown(
        f"""
- **MID trading is idealised:** the MID strategies assume the battery can trade its full schedule in the day-ahead auction at the MID price, with no fees, no volume limits and no imbalance exposure.
- **Imbalance profits are theoretical:** the imbalance price is only known after the fact, and a battery cannot freely trade at it. The perfect-foresight figure is an upper bound, not an achievable result.
- **Limited sample:** {meta['test_days']} test days in one season; model settings were not tuned and this is a single run. Results will differ in winter and in other years.
- **Other revenue streams** (frequency response, capacity market, Balancing Mechanism) are not modelled, and degradation cost is £{b['degradation_cost_gbp_per_mwh']:.0f}/MWh.
- **Each day starts at the minimum state of charge**, so energy is not carried across midnight; this likely understates real earnings.
- Wind and demand forecast publication times are filtered to the decision time, but historical publication timestamps are taken as published by Elexon.
"""
    )

st.download_button("Download daily results (CSV)", (DATA / "backtest_daily.csv").read_bytes(), "bess_backtest_daily.csv", "text/csv")
st.caption(f"Generated {meta['generated_at_utc']} by scripts/run_backtest_report.py. Model: {meta['model']}.")
