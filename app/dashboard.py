"""
BESS Asset Commercial Dispatch Dashboard.
Interactive analytics interface querying Gold Layer Parquet schedules.
"""

import os
from dotenv import load_dotenv
import duckdb
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

load_dotenv()

st.set_page_config(
    page_title="BESS Arbitrage & Dispatch Dashboard",
    page_icon="⚡",
    layout="wide",
)


@st.cache_data(ttl=300)
def load_gold_schedules() -> pd.DataFrame:
    """Queries Gold dispatch schedules directly from S3 using DuckDB."""
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    
    aws_region = os.getenv("AWS_REGION", "eu-west-2")
    aws_key = os.getenv("AWS_ACCESS_KEY_ID", "")
    aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    bucket = os.getenv("S3_BUCKET_NAME", "bess-dispatch-datalake-jimee")

    con.execute(f"""
        SET s3_region='{aws_region}';
        SET s3_access_key_id='{aws_key}';
        SET s3_secret_access_key='{aws_secret}';
    """)

    query = f"""
        SELECT 
            settlementDate,
            interval_start_utc,
            system_price_gbp,
            optimal_charge_mw,
            optimal_discharge_mw,
            net_dispatch_mw,
            state_of_energy_mwh
        FROM read_parquet('s3://{bucket}/gold/dispatch_schedules/**/*.parquet', hive_partitioning = true)
        ORDER BY interval_start_utc ASC;
    """
    return con.execute(query).df()


# -----------------------------------------------------------------------------
# Header & Data Loading
# -----------------------------------------------------------------------------
st.title("⚡ BESS Arbitrage & Dispatch Dashboard")
st.caption("Wholesale Power Arbitrage & State-of-Charge Monitoring (GB Balancing Mechanism)")

try:
    df_all = load_gold_schedules()
except Exception as err:
    st.error(f"Failed to load Gold schedules from S3: {err}")
    st.stop()

if df_all.empty:
    st.warning("No dispatch schedules found in the Gold data lake.")
    st.stop()

# -----------------------------------------------------------------------------
# Date Filtering
# -----------------------------------------------------------------------------
available_dates = sorted(df_all["settlementDate"].unique().tolist())
selected_date = st.sidebar.selectbox("Select Settlement Date", available_dates, index=len(available_dates) - 1)

df_day = df_all[df_all["settlementDate"] == selected_date].copy()
df_day["interval_start_utc"] = pd.to_datetime(df_day["interval_start_utc"])

# -----------------------------------------------------------------------------
# Commercial KPIs
# -----------------------------------------------------------------------------
export_mwh = (df_day["optimal_discharge_mw"] * 0.5).sum()
import_mwh = (df_day["optimal_charge_mw"] * 0.5).sum()
gross_rev = (df_day["optimal_discharge_mw"] * df_day["system_price_gbp"] * 0.5).sum()
charge_cost = (df_day["optimal_charge_mw"] * df_day["system_price_gbp"] * 0.5).sum()
net_margin = gross_rev - charge_cost
efc = export_mwh / 16.0  # 16 MWh usable DoD (80%)
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
        name="System Price (£/MWh)",
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
fig.add_hline(y=18.0, line_dash="dash", line_color="gray", annotation_text="Max SOC (90%)", row=3, col=1)
fig.add_hline(y=2.0, line_dash="dash", line_color="gray", annotation_text="Min SOC (10%)", row=3, col=1)

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


