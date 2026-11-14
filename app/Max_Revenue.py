"""
Maximum revenue from the day-ahead price forecast, net of costs: tomorrow's plan, revenue history, and
how the achievable result compares with the theoretical maximum.

Reads the pre-computed files in ``app/data/`` written by ``scripts/run_backtest_report.py``.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

DATA = Path(__file__).resolve().parent / "data"
BLUE, BLACK, GREEN, RED, PURPLE, GREY, AMBER = "#3b82f6", "#0f172a", "#10b981", "#ef4444", "#8b5cf6", "#94a3b8", "#f59e0b"
ETA = 0.85  # round-trip efficiency

st.set_page_config(page_title="BESS Max Revenue", page_icon="💷", layout="wide")


def _stamp() -> tuple:
    """Changes whenever a results file changes, so the cache below can't serve stale data after a refresh."""
    return tuple((p.name, p.stat().st_mtime_ns) for p in sorted(DATA.glob("*")) if p.suffix in {".csv", ".json"})


@st.cache_data(ttl=3600)
def load(stamp):
    plan = pd.read_csv(DATA / "plan_halfhourly.csv", parse_dates=["interval_start_utc"])
    roll = pd.read_csv(DATA / "rolling_daily.csv", parse_dates=["date"])
    sens = pd.read_csv(DATA / "sensitivity.csv")
    summ = json.loads((DATA / "rolling_summary.json").read_text())
    return plan, roll, sens, summ


try:
    plan, roll, sens, summ = load(_stamp())
except FileNotFoundError:
    st.error("Results not found. Run `python scripts/run_backtest_report.py --start ... --end ...`.")
    st.stop()

A, B = summ["assumptions"], summ["bridge_gbp"]
plan["local"] = plan["interval_start_utc"].dt.tz_convert("Europe/London").dt.tz_localize(None)
known = roll[roll["actual_known"]].copy()
last_day = known["date"].max()

st.title("BESS Optimization")
st.subheader("Highest revenue from day-ahead price forecasting")
st.caption("Price = N2EX GB day-ahead auction (NESO data, hourly). The auction for tomorrow clears at ~09:20 UK today, so plans are "
           "made at 08:00 UTC from a forecast and valued afterwards at the actual auction price.")

share = st.sidebar.slider("Optimiser / route-to-market revenue share (%)", 0, 30, 0,
                          help="Applied to positive net profit when reporting. Does not change the dispatch.") / 100
st.sidebar.markdown(
    f"**Cost assumptions in this run**\n- Degradation: £{A['degradation_gbp_per_mwh_discharged']:.0f}/MWh discharged (placeholder)\n"
    f"- Trading fee: £{A['trading_fee_gbp_per_mwh']:.2f}/MWh traded\n- End-of-day reserve: {A['end_of_day_reserve_mwh']:.0f} MWh\n"
    f"- Network/levy charges: not modelled")


def after_share(x: float) -> float:
    return x * (1 - share) if x > 0 else x


tab_plan, tab_hist, tab_max = st.tabs(["Forecast & plan", "Revenue history (1 / 7 / 30 days)", "Maximum vs achievable"])

# =============================================================================
# Tab 1: day-ahead price forecast and the arbitrage plan built on it
# =============================================================================
with tab_plan:
    nd = summ.get("next_day")
    dates = list(plan["date"].unique())
    labels = {d: (f"{d}  (forecast, not yet settled)" if d == (nd or {}).get("date") else d) for d in dates}
    pick = st.selectbox("Delivery day", dates, index=len(dates) - 1, format_func=labels.get)
    day = plan[plan["date"] == pick].reset_index(drop=True)
    r = roll[roll["date"].dt.strftime("%Y-%m-%d") == pick].iloc[0]
    is_known = bool(day["actual_known"].iloc[0])

    # KPIs first, then the chart
    other = r["trading_fees_gbp"] + r["degradation_gbp"]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Expected revenue", f"£{r['expected_revenue_gbp']:,.0f}", help="Discharge valued at the forecast price.")
    m2.metric("Expected charging cost", f"£{r['expected_cost_gbp']:,.0f}")
    m3.metric("Degradation + fees", f"£{other:,.0f}")
    exp_net = r["expected_net_gbp"]
    m4.metric("Expected net profit", f"£{after_share(exp_net):,.0f}")
    ratio = summ["calibration"]["realised_over_expected"]
    if is_known:
        m5.metric("Realised net profit", f"£{after_share(r['net_profit_gbp']):,.0f}", f"{r['net_profit_gbp'] / exp_net * 100 - 100:+.0f}% vs expected" if exp_net else None)
    else:
        m5.metric("Calibrated estimate", f"£{after_share(exp_net * ratio):,.0f}",
                  help=f"Expected × {ratio:.2f}: over the backtest, realised profit was {ratio:.0%} of expected.")

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.4, 0.3, 0.3],
                        subplot_titles=("Day-ahead price: forecast vs actual (£/MWh)",
                                        "Battery plan: discharge (+) / charge (-) MW", "State of energy (MWh)"))
    fig.add_scatter(x=day["local"], y=day["da_model"], name="Day-ahead forecast", line=dict(color=BLUE, width=3), row=1, col=1)
    if is_known:
        fig.add_scatter(x=day["local"], y=day["da_actual"], name="Actual auction price", line=dict(color=BLACK, width=2), row=1, col=1)
    fig.add_bar(x=day["local"], y=day["discharge_mw"], name="Discharge (sell)", marker_color=GREEN, row=2, col=1)
    fig.add_bar(x=day["local"], y=-day["charge_mw"], name="Charge (buy)", marker_color=RED, row=2, col=1)
    fig.add_scatter(x=day["local"], y=day["soe_mwh"], name="State of energy", line=dict(color=PURPLE, width=2), row=3, col=1)
    fig.add_hline(y=18, line_dash="dash", line_color=GREY, annotation_text="max 90%", row=3, col=1)
    fig.add_hline(y=2, line_dash="dash", line_color=GREY, annotation_text="min 10%", row=3, col=1)
    fig.update_layout(height=720, barmode="relative", hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.08))
    st.plotly_chart(fig, use_container_width=True)

    # plain-language plan
    def windows(col: str) -> list[str]:
        on = day[col] > 0.05
        grp = (on != on.shift()).cumsum()
        out = []
        for _, g in day[on].groupby(grp[on]):
            end = g["local"].iloc[-1] + pd.Timedelta(minutes=30)
            out.append(f"{g['local'].iloc[0]:%H:%M}–{end:%H:%M} (avg forecast £{g['da_model'].mean():.0f}/MWh, "
                       f"{(g[col] * 0.5).sum():.1f} MWh)")
        return out or ["none"]
    c1, c2 = st.columns(2)
    c1.markdown("**Charge (buy)**  \n" + "  \n".join(windows("charge_mw")))
    c2.markdown("**Discharge (sell)**  \n" + "  \n".join(windows("discharge_mw")))

    st.info(
        f"**How reliable is 'expected profit'?** Over {len(known)} settled days the plans' expected net profit "
        f"(£{summ['calibration']['expected_net_gbp']:,.0f}) was **{1 / ratio - 1:.0%} higher** than what was realised at the actual auction price "
        f"(£{summ['calibration']['realised_net_gbp']:,.0f}). The forecast is smoother than reality and the plan is picked to look best on it, so treat "
        "expected profit as an optimistic estimate and the calibrated figure as the more realistic one."
    )
    if pick == (nd or {}).get("date"):
        st.caption("Tomorrow's plan uses the same information cutoff as the backtest (publications up to D-1 08:00 UTC) and starts "
                   f"from {r['start_soc_mwh']:.1f} MWh, the end of the previous planned day.")

# =============================================================================
# Tab 2: revenue history
# =============================================================================
with tab_hist:
    period = st.radio("Period ending " + f"{last_day:%Y-%m-%d}", ["1 day", "7 days", "30 days"], horizontal=True)
    n = {"1 day": 1, "7 days": 7, "30 days": 30}[period]
    win = known[known["date"] > last_day - pd.Timedelta(days=n)]
    rev, chg = win["revenue_gbp"].sum(), win["charging_cost_gbp"].sum()
    oth = win["trading_fees_gbp"].sum() + win["degradation_gbp"].sum()
    net = win["net_profit_gbp"].sum()
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Revenue generated (gross)", f"£{rev:,.0f}", help="Energy sold at the actual day-ahead auction price.")
    h2.metric("Charging cost", f"£{chg:,.0f}")
    h3.metric("Degradation + fees", f"£{oth:,.0f}")
    h4.metric("Net profit", f"£{after_share(net):,.0f}", f"{(1 - share) * 100:.0f}% kept after revenue share" if share else None, delta_color="off")

    if n == 1:
        d1 = plan[(plan["date"] == last_day.strftime("%Y-%m-%d"))].copy()
        d1["revenue"] = d1["discharge_mw"] * d1["da_actual"] * 0.5
        d1["cost"] = d1["charge_mw"] * d1["da_actual"] * 0.5
        d1["other"] = (A["degradation_gbp_per_mwh_discharged"] * d1["discharge_mw"] * 0.5
                       + A["trading_fee_gbp_per_mwh"] * (d1["discharge_mw"] + d1["charge_mw"]) * 0.5)
        d1["net"] = d1["revenue"] - d1["cost"] - d1["other"]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.6, 0.4],
                            subplot_titles=(f"{last_day:%Y-%m-%d}: half-hourly revenue and cost (£)", "Cumulative net profit (£)"))
        fig.add_bar(x=d1["local"], y=d1["revenue"], name="Revenue", marker_color=GREEN, row=1, col=1)
        fig.add_bar(x=d1["local"], y=-(d1["cost"] + d1["other"]), name="Charging cost + other costs", marker_color=RED, row=1, col=1)
        fig.add_scatter(x=d1["local"], y=d1["net"].cumsum(), name="Cumulative net profit", line=dict(color=BLUE, width=3), row=2, col=1)
        fig.update_layout(barmode="relative", height=520, hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10),
                          legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.6, 0.4],
                            subplot_titles=(f"Daily revenue and costs, last {n} days (£)", "Cumulative net profit (£)"))
        fig.add_bar(x=win["date"], y=win["revenue_gbp"], name="Revenue", marker_color=GREEN, row=1, col=1)
        fig.add_bar(x=win["date"], y=-(win["charging_cost_gbp"] + win["trading_fees_gbp"] + win["degradation_gbp"]),
                    name="Charging cost + other costs", marker_color=RED, row=1, col=1)
        fig.add_scatter(x=win["date"], y=win["net_profit_gbp"].cumsum(), name="Cumulative net profit",
                        line=dict(color=BLACK, width=3), row=2, col=1)
        fig.update_layout(barmode="relative", height=560, hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10),
                          legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig, use_container_width=True)
        fig = go.Figure(go.Bar(x=win["date"], y=win["net_profit_gbp"], marker_color=[GREEN if v >= 0 else RED for v in win["net_profit_gbp"]]))
        fig.update_layout(height=260, title="Daily net profit (£)", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Strategy: day-ahead forecast model, degradation £{A['degradation_gbp_per_mwh_discharged']:.0f}/MWh, fee "
               f"£{A['trading_fee_gbp_per_mwh']:.2f}/MWh, state of charge carried across days. Network/levy charges not modelled.")

# =============================================================================
# Tab 3: maximum vs achievable, sensitivity
# =============================================================================
with tab_max:
    ach = B["achievable_after_costs_soc_carry"]
    ub2, ub1, ub0 = B["perfect_after_costs_soc_carry"], B["perfect_no_costs_soc_carry"], B["perfect_no_costs_daily_reset"]
    oos = summ["out_of_sample_gbp"]
    days = A["evaluated_days"]
    k1, k2, k3 = st.columns(3)
    k1.metric("Theoretical max (after costs)", f"£{after_share(ub2):,.0f}", f"£{ub2 / days:,.0f}/day", delta_color="off")
    k2.metric("Achievable with our forecast (after costs)", f"£{after_share(ach):,.0f}", f"{ach / ub2:.0%} of maximum")
    k3.metric("Out-of-sample achievable", f"£{after_share(oos['achievable_after_costs']):,.0f}",
              f"{oos['achievable_after_costs'] / oos['perfect_after_costs']:.0%} of max, last {oos['days']} days", delta_color="off")

    wf = go.Figure(go.Waterfall(
        x=["Perfect foresight, no\ncosts, daily reset", "SOC carried\nacross days", "Degradation, fees,\nend-of-day reserve",
           "Forecast error", "Achievable\n(after costs)"],
        measure=["absolute", "relative", "relative", "relative", "total"],
        y=[ub0, ub1 - ub0, ub2 - ub1, ach - ub2, ach],
        text=[f"£{ub0:,.0f}", f"{ub1 - ub0:+,.0f}", f"{ub2 - ub1:+,.0f}", f"{ach - ub2:+,.0f}", f"£{ach:,.0f}"],
        textposition="outside", increasing=dict(marker_color=GREEN), decreasing=dict(marker_color=RED),
        totals=dict(marker_color=BLUE)))
    wf.update_yaxes(range=[0, ub0 * 1.15])
    wf.update_layout(height=420, title=f"From theoretical maximum to achievable profit ({days} days, £)", margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(wf, use_container_width=True)
    st.markdown(
        f"- **Biggest loss is forecast error** (£{ub2 - ach:,.0f}), not costs (£{ub1 - ub2:,.0f}).\n"
        f"- **Carrying state of charge across days adds only £{ub1 - ub0:,.0f}** ({(ub1 - ub0) / ub0:.1%}) even with perfect knowledge; "
        f"under forecast uncertainty the value placed on energy left in the battery overnight was tuned to **£{A['terminal_value_gbp_per_mwh']:.0f}/MWh**.\n"
        f"- The terminal value was tuned on the first {A['tuned_on_days']} days only; the last {oos['days']} days are out of sample."
    )

    st.subheader("Sensitivity to degradation and trading fees")
    piv = sens.pivot(index="degradation_gbp_per_mwh", columns="fee_gbp_per_mwh", values="achievable_net_gbp")
    hm = go.Figure(go.Heatmap(z=piv.values, x=[f"£{c:g}" for c in piv.columns], y=[f"£{i:g}" for i in piv.index],
                              text=[[f"£{v:,.0f}" for v in row] for row in piv.values], texttemplate="%{text}",
                              colorscale="Greens", showscale=False))
    hm.update_layout(height=340, title="Achievable net profit over the period", xaxis_title="Trading fee (£/MWh traded)",
                     yaxis_title="Degradation (£/MWh discharged)", margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(hm, use_container_width=True)
    st.caption("Terminal value and reserve held at this run's settings. Battery cost per MWh discharged ≈ capex (£/MWh) ÷ (warranted full cycles × depth of discharge): "
               "plug in your own capex and cycle life to place yourself on this grid.")

    st.subheader("Break-even: how high must the sell price be?")
    deg = st.select_slider("Degradation (£/MWh discharged)", sorted(sens["degradation_gbp_per_mwh"].unique()), value=A["degradation_gbp_per_mwh_discharged"])
    fee = st.select_slider("Trading fee (£/MWh traded)", sorted(sens["fee_gbp_per_mwh"].unique()), value=A["trading_fee_gbp_per_mwh"])
    lows = np.array([40, 80, 120, 160, 200])
    need = lows / ETA + deg + fee * (1 + 1 / ETA)
    st.dataframe(pd.DataFrame({"Buy price (£/MWh)": lows, "Break-even sell price (£/MWh)": need.round(1),
                               "Spread needed (£/MWh)": (need - lows).round(1)}), hide_index=True, use_container_width=True)
    cell = sens[(sens["degradation_gbp_per_mwh"] == deg) & (sens["fee_gbp_per_mwh"] == fee)].iloc[0]
    st.write(f"At these costs: achievable £{after_share(cell['achievable_net_gbp']):,.0f}, theoretical maximum "
             f"£{after_share(cell['perfect_net_gbp']):,.0f}, {cell['achievable_efc_per_day']:.2f} full cycles/day.")

    with st.expander("What is and isn't included", expanded=True):
        st.markdown(
            f"""
- **Included:** degradation (£{A['degradation_gbp_per_mwh_discharged']:.0f}/MWh discharged, a *placeholder*), trading fee (£{A['trading_fee_gbp_per_mwh']:.2f}/MWh, default zero because no public figure exists: route-to-market fees are contractual), 85% round-trip efficiency, 10–90% SOC window, {A['end_of_day_reserve_mwh']:.0f} MWh end-of-day reserve, SOC carried across days.
- **Not included:** network and levy charges (unknown for your site, likely the biggest missing cost), optimiser revenue share unless you set the sidebar slider, taxes, other revenue streams (frequency response, capacity market, Balancing Mechanism).
- **Day-ahead price, idealised trading:** the model assumes the battery can buy and sell its whole schedule at the hourly auction price, with no fees (unless set), volume limits or price impact, and that hours can be linked through the battery's state of charge. Real orders are submitted before the price is known; this is an upper-level estimate, not a trading result.
- **One season (autumn 2026), untuned model, single run.** The £8/MWh degradation figure is a placeholder and may be low or high for your battery.
"""
        )
