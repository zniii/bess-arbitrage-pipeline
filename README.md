# ⚡ Utility-Scale BESS Arbitrage & Dispatch Engine

An automated, cloud-native data and optimization pipeline designed for a 10 MW / 20 MWh Battery Energy Storage System (BESS) operating within the Great Britain (GB) wholesale power market.

The system ingests half-hourly settlement prices from the Elexon Insights API, conforms telemetry into a partitioned S3 Medallion Lakehouse, formulates an optimal physical dispatch schedule using Mixed-Integer Linear Programming (MILP), and delivers commercial analytics via an interactive Streamlit dashboard.

---

## 🏗 System Architecture

The pipeline implements an end-to-end Medallion Data Architecture on Amazon S3:

1. **Bronze (Raw Zone):** Ingests raw JSON payloads from Elexon Insights API partitioned by `year=YYYY/month=MM/`.
2. **Silver (Conformed Zone):** Standardizes intervals into UTC half-open boundaries `[interval_start_utc, interval_end_utc)`. Persists single-part Snappy-compressed Apache Parquet files partitioned by `settlementDate=YYYY-MM-DD` with strict idempotency.
3. **Gold (Curated Features & Schedules):** Computes 24h rolling price volatility envelopes and holds solved optimal dispatch schedules.
4. **Optimization Engine:** Formulates a Mixed-Integer Linear Program (MILP) using `PuLP` and the CBC solver to maximize gross arbitrage while penalizing cell degradation.
5. **Serving & Observability:** Direct in-memory SQL querying over S3 via DuckDB (`httpfs`), Amazon Athena DDL definitions, and a Plotly/Streamlit operations cockpit.

---

## ⚙️ Mathematical Model & Physical Constraints

The dispatch engine solves the following objective over a 48 settlement period horizon ($t \in \{1, \dots, 48\}$):

$$\max \sum_{t=1}^{T} \left[ \left( P_t^{\text{dis}} \cdot \lambda_t - P_t^{\text{ch}} \cdot \lambda_t - C_{\text{deg}} \cdot (P_t^{\text{ch}} + P_t^{\text{dis}}) \right) \cdot \Delta t \right]$$

Subject to:
* **Power Rating:** $0 \le P_t^{\text{ch}} \le 10 \cdot u_t$, $0 \le P_t^{\text{dis}} \le 10 \cdot v_t$ (where $u_t + v_t \le 1, u_t, v_t \in \{0, 1\}$).
* **Usable Energy Envelope:** $2.0 \text{ MWh} \le \text{SOE}_t \le 18.0 \text{ MWh}$ (enforces 10%–90% SOC operating limits).
* **Round-Trip Efficiency ($\eta = 0.85$):**
  $$\text{SOE}_t = \text{SOE}_{t-1} + \left( P_t^{\text{ch}} \cdot \sqrt{\eta} - \frac{P_t^{\text{dis}}}{\sqrt{\eta}} \right) \cdot \Delta t$$
* **Asset Degradation Penalty ($C_{\text{deg}}$):** optional throughput cost (`degradation_cost_gbp_per_mwh` on `BESSDispatcher`, default 0) that prevents cycling when spreads are too narrow to justify battery wear.

## 🔮 Forecast-driven dispatch (daily update)

A real operator cannot optimize against prices that have not happened yet, so the daily job does this for every settled day:

1. **Forecast** the 48 half-hourly prices using only information available the morning before (`forecasting/`; every price feature is lagged ≥ 2 days, enforced by a no-lookahead test). Models: a seasonal-naive benchmark (same half-hour last week) and a gradient-boosted regressor (scikit-learn), retrained weekly.
2. **Optimize** the MILP against the *forecast*.
3. **Settle** the resulting schedule at the *actual* system price.
4. Also solve the **perfect-foresight** benchmark, so the dashboard shows how much of the theoretical profit the forecast captures.

Each run writes `gold/dispatch_schedules/` and a one-row-per-day `gold/daily_kpis/` record. The dashboard shows 1-day, 7-day and 30-day views ending at the latest settled day.

> No free hosted API for GB price forecasts was found, so the model is built in-repo. It currently uses lagged prices and calendar features only; Elexon wind/demand forecasts and the NESO Carbon Intensity forecast are the planned exogenous inputs (the feature builder already accepts `wind_forecast_mw`, `demand_forecast_mw`, `carbon_intensity_forecast` columns, but no ingestion for them exists yet).

---

## 🚀 Getting Started

```bash
git clone https://github.com/zniii/bess-arbitrage-pipeline.git
cd bess-arbitrage-pipeline
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env    # then fill in credentials
```

### Storage: AWS S3 or local MinIO
Set `AWS_ENDPOINT_URL=http://localhost:9000` (see `.env.example`) and run `docker compose up -d` followed by `docker compose run --rm create-bucket`; the pipeline and dashboard then use MinIO instead of AWS. Leave `AWS_ENDPOINT_URL` unset to use real S3.

### Run
```bash
# Update Gold to the latest settled day (idempotent; backfills any gaps in the window)
python -m bess_pipeline.orchestration.daily_update --window-days 30

# Original perfect-foresight backtest for specific dates
python -m bess_pipeline.orchestration.pipeline_runner --start-date 2026-08-01 --end-date 2026-08-03

# Dashboard
streamlit run app/dashboard.py

# Tests (test_live_small_window_fetch needs internet access to Elexon)
pytest
```
`.github/workflows/daily-update.yml` runs the daily update on a schedule (needs the AWS secrets configured).

> PuLP is pinned `<4`: version 4 removed the `LpVariable(lowBound=...)` API used by the optimizer.
