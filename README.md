# ⚡ Utility-Scale BESS Arbitrage & Dispatch Engine

An automated, cloud-native data and optimization pipeline designed for a 10 MW / 20 MWh Battery Energy Storage System (BESS) operating within the Great Britain (GB) wholesale power market.

The system ingests half-hourly settlement prices from the Elexon Insights API, conforms telemetry into a partitioned S3 Medallion Lakehouse, formulates an optimal physical dispatch schedule using Mixed-Integer Linear Programming (MILP), and presents the results in a Streamlit app (**Findings** and **Max Revenue** pages, `app/Findings.py`).

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

## 🔮 Price, forecasting and strategies

**Main price: the N2EX GB day-ahead auction** (hourly, from NESO open data, `ingestion/neso_client.py`). The auction for day D clears around 09:20 UK on D-1, so a plan made at **D-1 08:00 UTC** must rely on a *forecast*; yesterday's auction prices are already public. (Elexon MID, an index published as each half-hour happens, was used in earlier versions; it is not tradable in advance and has been replaced.)

Forecasting (`forecasting/`): gradient-boosted regressor (scikit-learn) on calendar features, day-ahead price lags (>= 1 day), rolling statistics, and Elexon **wind (WINDFOR)** and **national demand (NDFD)** forecasts. Only forecast publications made before the cutoff are used (no-lookahead tests). A seasonal-naive forecaster (same hour last week) is the benchmark. Retrained every 14 days.

Strategies run by the daily job (`strategy` column in `gold/dispatch_schedules/`, one row per day in `gold/daily_kpis/`):

| Strategy | Optimizes on | Settled at | Role |
|---|---|---|---|
| `da_forecast` | **our forecast** of the day-ahead price | actual day-ahead price | **Headline** |
| `da_perfect` | actual day-ahead prices | actual day-ahead price | Upper bound |
| `imbalance_forecast` | forecast imbalance (system) price | imbalance price | Experimental (daily job only) |
| `imbalance_perfect` | actual imbalance prices | imbalance price | Experimental upper bound |

These daily-job strategies reset the battery to 10% SOC each day and ignore costs. The **Max Revenue** page uses the cost-aware strategy from `forecasting/rolling.py`: state of charge carried across days, degradation (default £8/MWh discharged, a placeholder), trading fee (default £0), an end-of-day reserve (default 4 MWh) and a terminal value for stored energy (tuned on the first 40% of days, judged on the rest). Network and levy charges are not modelled. `scripts/run_backtest_report.py` produces everything those pages read (`app/data/`): the bridge from the theoretical maximum to the achievable result, a degradation x fee sensitivity grid, tomorrow's plan with expected profit, and 1/7/30-day revenue history.

> No free hosted API for GB price *forecasts* was found, so the model is built in-repo. The NESO Carbon Intensity forecast was considered as a further feature but is not wired in. Trading is idealised: no volume limits or price impact, and real orders are submitted before the price is known.

The daily job's Gold results (`gold/dispatch_schedules/`, `gold/daily_kpis/`) are no longer displayed by the app; the Streamlit pages read the pre-computed files in `app/data/` written by `scripts/run_backtest_report.py`. Schedules written by the original `pipeline_runner` use an older format (no `strategy` column).

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
Set `AWS_ENDPOINT_URL=http://localhost:9000` (see `.env.example`) and run `docker compose up -d` followed by `docker compose run --rm create-bucket`; the pipeline then uses MinIO instead of AWS. Leave `AWS_ENDPOINT_URL` unset to use real S3.

### Run
```bash
# Update Gold to the latest settled day (idempotent; backfills any gaps in the window)
python -m bess_pipeline.orchestration.daily_update --window-days 30

# Original perfect-foresight backtest for specific dates
python -m bess_pipeline.orchestration.pipeline_runner --start-date 2026-08-01 --end-date 2026-08-03

# Streamlit app: Findings (landing page) and Max Revenue (sidebar). Works offline from app/data
streamlit run app/Findings.py

# Regenerate the results behind both pages (needs internet access to Elexon and NESO)
python scripts/run_backtest_report.py --start 2026-08-01 --end 2026-11-08

# Tests (test_live_small_window_fetch needs internet access to Elexon)
pytest
```
`.github/workflows/daily-update.yml` runs the daily update on a schedule (needs the AWS secrets configured).

> PuLP is pinned `<4`: version 4 removed the `LpVariable(lowBound=...)` API used by the optimizer.
