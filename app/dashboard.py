"""
BESS Asset Commercial Dispatch Dashboard.
Interactive analytics interface querying Gold Layer Parquet schedules.
"""

import os

import duckdb
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from plotly.subplots import make_subplots

load_dotenv()

st.set_page_config(
    page_title="BESS Arbitrage & Dispatch Dashboard",
    page_icon="⚡",
    layout="wide",
)


def _connect():
    """DuckDB connection to S3 (AWS, or MinIO/LocalStack via AWS_ENDPOINT_URL)."""
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_region=?", [os.getenv("AWS_REGION", "eu-west-2")])
    con.execute("SET s3_access_key_id=?", [os.getenv("AWS_ACCESS_KEY_ID", "")])
    con.execute("SET s3_secret_access_key=?", [os.getenv("AWS_SECRET_ACCESS_KEY", "")])
    endpoint = os.getenv("AWS_ENDPOINT_URL")
    if endpoint:
        host = endpoint.split("://", 1)[-1].rstrip("/")
        con.execute("SET s3_endpoint=?", [host])
        con.execute("SET s3_url_style='path'")
        con.execute("SET s3_use_ssl=?", [endpoint.startswith("https")])
    return con


def _bucket() -> str:
    return os.getenv("AWS_S3_BUCKET") or os.getenv("S3_BUCKET_NAME", "")


@st.cache_data(ttl=300)
def load_daily_kpis() -> pd.DataFrame:
    """One row per settled day: forecast-based vs perfect-foresight profit, EFC, MAE."""
    return _connect().execute(f"""
        SELECT * FROM read_parquet('s3://{_bucket()}/gold/daily_kpis/**/*.parquet', hive_partitioning = true)
        ORDER BY settlementDate
    """).df()


@st.cache_data(ttl=300)
def load_gold_schedules() -> pd.DataFrame:
    """Half-hourly dispatch schedules (with the forecast the optimizer used)."""
    return _connect().execute(f"""
        SELECT settlementDate, strategy, interval_start_utc, system_price_gbp, da_price_gbp,
               settlement_price_gbp, price_forecast, optimal_charge_mw, optimal_discharge_mw, net_dispatch_mw, state_of_energy_mwh
        FROM read_parquet('s3://{_bucket()}/gold/dispatch_schedules/**/*.parquet', hive_partitioning = true)
        ORDER BY interval_start_utc ASC
    """).df()


# -----------------------------------------------------------------------------
# Header & Data Loading
# -----------------------------------------------------------------------------
st.title("⚡ BESS Arbitrage & Dispatch Dashboard")
st.caption("Headline: dispatch optimized on our N2EX day-ahead price forecast and settled at the actual day-ahead auction price (per-day view: no costs, battery reset daily; see the Max Revenue page for the cost-aware view)")

try:
    df_all = load_gold_schedules()
    df_kpi = load_daily_kpis()
except Exception as err:
    st.error(f"Failed to load Gold data from S3: {err}")
    st.stop()

if df_all.empty or df_kpi.empty:
    st.warning("No dispatch results found in the Gold data lake. Run the daily update first.")
    st.stop()

# -----------------------------------------------------------------------------
# Period summary (1 day / 7 days / 1 month, ending at the latest settled day)
# -----------------------------------------------------------------------------
PERIODS = {"1 day": 1, "7 days": 7, "1 month (30 days)": 30}
period_label = st.sidebar.radio("Period", list(PERIODS), index=0)
show_imb = st.sidebar.checkbox("Show imbalance-price models (experimental)", value=False,
                               help="Optimizes on forecast imbalance prices and settles at the volatile imbalance price.")
latest = pd.to_datetime(df_kpi["settlementDate"]).max()
window = df_kpi[pd.to_datetime(df_kpi["settlementDate"]) > latest - pd.Timedelta(days=PERIODS[period_label])]

STRATEGIES = {  # column -> (label, colour, experimental?)
    "profit_da_perfect_gbp": ("Day-ahead perfect foresight (upper bound)", "#94a3b8", False),
    "profit_da_forecast_gbp": ("Day-ahead forecast model (ours)", "#10b981", False),
    "profit_imbalance_forecast_gbp": ("Imbalance forecast model", "#8b5cf6", True),
    "profit_imbalance_perfect_gbp": ("Imbalance perfect foresight (upper bound)", "#f59e0b", True),
}
active = {c: v for c, v in STRATEGIES.items() if (show_imb or not v[2]) and c in window.columns}

st.subheader(f"{period_label} ending {latest:%Y-%m-%d}  ({len(window)} day(s) of data)")
p_profit = window["profit_da_forecast_gbp"].sum()
p_base = window["profit_da_perfect_gbp"].sum()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Net Profit (day-ahead forecast model)", f"£{p_profit:,.2f}")
c2.metric("Day-ahead Perfect Foresight", f"£{p_base:,.2f}")
c3.metric("Baseline Captured", f"{(p_profit / p_base * 100) if p_base else 0:.0f}%")
c4.metric("Cycles (EFC)", f"{window['equivalent_full_cycles'].sum():.2f}")
c5.metric("Forecast MAE", f"£{window['da_forecast_mae_gbp_per_mwh'].mean():.2f}/MWh")
if show_imb:
    st.caption(
        f"Imbalance forecast model: £{window['profit_imbalance_forecast_gbp'].sum():,.2f} "
        f"(MAE £{window['imbalance_forecast_mae_gbp_per_mwh'].mean():.2f}/MWh) vs perfect-foresight "
        f"upper bound £{window['profit_imbalance_perfect_gbp'].sum():,.2f}. "
        "Different settlement price from the day-ahead strategies: not directly comparable."
    )

if len(window) > 1:
    bar = go.Figure()
    for col, (label, colour, _) in active.items():
        bar.add_bar(x=window["settlementDate"], y=window[col], name=label, marker_color=colour)
    bar.update_layout(barmode="group", height=340, margin=dict(l=20, r=20, t=30, b=20), title="Daily net profit (£)")
    st.plotly_chart(bar, use_container_width=True)

st.divider()

# -----------------------------------------------------------------------------
# Date Filtering
# -----------------------------------------------------------------------------
st.subheader("Intraday detail")
available_dates = sorted(df_all["settlementDate"].unique().tolist())
selected_date = st.sidebar.selectbox("Select Settlement Date", available_dates, index=len(available_dates) - 1)

strategy_options = ["da_forecast", "da_perfect"] + (["imbalance_forecast", "imbalance_perfect"] if show_imb else [])
strategy = st.sidebar.selectbox("Strategy (intraday detail)", strategy_options)
df_day = df_all[(df_all["settlementDate"] == selected_date) & (df_all["strategy"] == strategy)].copy()
df_day["system_price_gbp"] = df_day["settlement_price_gbp"]  # price this strategy is settled at
df_day["interval_start_utc"] = pd.to_datetime(df_day["interval_start_utc"])

# -----------------------------------------------------------------------------
# Commercial KPIs
# -----------------------------------------------------------------------------
export_mwh = (df_day["optimal_discharge_mw"] * 0.5).sum()
import_mwh = (df_day["optimal_charge_mw"] * 0.5).sum()
gross_rev = (df_day["optimal_discharge_mw"] * df_day["system_price_gbp"] * 0.5).sum()
charge_cost = (df_day["optimal_charge_mw"] * df_day["system_price_gbp"] * 0.5).sum()
net_margin = gross_rev - charge_cost
ENERGY_MWH = float(os.getenv("BESS_ENERGY_MWH", "20"))
MIN_SOE, MAX_SOE = 0.1 * ENERGY_MWH, 0.9 * ENERGY_MWH
efc = export_mwh / (MAX_SOE - MIN_SOE)  # usable depth of discharge
avg_spread = (net_margin / export_mwh) if export_mwh > 0 else 0.0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Net Arbitrage Profit", f"£{net_margin:,.2f}")
col2.metric("Gross Revenue", f"£{gross_rev:,.2f}")
col3.metric("Charging Cost", f"£{charge_cost:,.2f}")
col4.metric("Cycles (EFC)", f"{efc:.2f} cycles")
col5.metric("Captured Spread", f"£{avg_spread:.2f}/MWh")

st.divider()

# -----------------------------------------------------------------------------
# Dispatch & Asset Telemetry Charts
# -----------------------------------------------------------------------------
fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=(
        "Elexon System Price (£/MWh)",
        "Optimal Dispatch: Charge (-) vs. Discharge (+) [MW]",
        "Battery State of Energy (MWh)",
    ),
)

# 1. System Price
fig.add_trace(
    go.Scatter(
        x=df_day["interval_start_utc"],
        y=df_day["system_price_gbp"],
        mode="lines",
        line=dict(color="#f59e0b", width=2),
        name="Settlement Price (£/MWh)",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=df_day["interval_start_utc"],
        y=df_day["price_forecast"],
        mode="lines",
        line=dict(color="#8b5cf6", width=2, dash="dot"),
        name="Day-ahead Forecast (£/MWh)",
    ),
    row=1,
    col=1,
)

# 2. MW Dispatch
fig.add_trace(
    go.Bar(
        x=df_day["interval_start_utc"],
        y=df_day["optimal_discharge_mw"],
        marker_color="#10b981",
        name="Export / Discharge (MW)",
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Bar(
        x=df_day["interval_start_utc"],
        y=-df_day["optimal_charge_mw"],
        marker_color="#ef4444",
        name="Import / Charge (MW)",
    ),
    row=2,
    col=1,
)

# 3. State of Energy Trajectory
fig.add_trace(
    go.Scatter(
        x=df_day["interval_start_utc"],
        y=df_day["state_of_energy_mwh"],
        mode="lines+markers",
        line=dict(color="#3b82f6", width=2),
        name="State of Energy (MWh)",
    ),
    row=3,
    col=1,
)

# Add SOC Constraint Boundaries
fig.add_hline(y=MAX_SOE, line_dash="dash", line_color="gray", annotation_text="Max SOC (90%)", row=3, col=1)
fig.add_hline(y=MIN_SOE, line_dash="dash", line_color="gray", annotation_text="Min SOC (10%)", row=3, col=1)

fig.update_layout(
    height=800,
    hovermode="x unified",
    showlegend=True,
    margin=dict(l=20, r=20, t=40, b=20),
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Detailed Dispatch Table
# -----------------------------------------------------------------------------
with st.expander("View Raw Schedule Data"):
    st.dataframe(
        df_day[[
            "interval_start_utc",
            "system_price_gbp",
            "optimal_charge_mw",
            "optimal_discharge_mw",
            "net_dispatch_mw",
            "state_of_energy_mwh",
        ]].style.format({
            "system_price_gbp": "£{:.2f}",
            "optimal_charge_mw": "{:.2f} MW",
            "optimal_discharge_mw": "{:.2f} MW",
            "net_dispatch_mw": "{:.2f} MW",
            "state_of_energy_mwh": "{:.2f} MWh",
        }),
        use_container_width=True,
    )


