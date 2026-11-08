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

## 🔮 Strategies and forecasting (daily update)

A real operator cannot optimize against prices that have not happened yet. The daily job runs four strategies per settled day and stores all of them (`strategy` column in `gold/dispatch_schedules/`, one row per day in `gold/daily_kpis/`):

| Strategy | Optimizes on | Settled at | Role |
|---|---|---|---|
| `mid_baseline` | actual MID prices, known in advance | MID | **Upper bound** (name kept in the data): profit if MID were perfectly known |
| `mid_forecast` | **our own forecast of MID**, made at D-1 08:00 UTC | actual MID | **Headline model**: how much of the upper bound a forecast can capture |
| `imbalance_forecast` | forecast of the imbalance (system) price | imbalance price | Experimental, toggled in the dashboard |
| `imbalance_perfect` | actual imbalance prices | imbalance price | Experimental unattainable upper bound |

Forecasting (`forecasting/`): gradient-boosted regressor (scikit-learn) on calendar features, lagged prices, and Elexon **wind (WINDFOR)** and **national demand (NDFD)** day-ahead forecasts. Everything respects the decision time D-1 08:00 UTC: price lags are >= 2 days for both MID and the imbalance price (MID is published as each half-hour happens, so most of D-1 is not yet known), and only forecast publications made before the cutoff are used (covered by no-lookahead tests). A seasonal-naive forecaster (same half-hour last week) is the benchmark. The model is retrained weekly.

**MID** (Elexon Market Index Data, APX provider) is a wholesale price *index* published as each period happens. It is not the day-ahead auction result, and it is not directly tradable: every MID strategy assumes trading at the index price.

> No free hosted API for GB price forecasts was found, so the model is built in-repo. The NESO Carbon Intensity forecast was considered as a further feature but is not wired in.
> MID is not the same price as the imbalance price. The legacy strategies above ignore costs and reset the battery to 10% SOC every day; the **Max Revenue** page adds the cost-aware strategy below.

### Cost-aware strategy (Max Revenue page)
`scripts/run_backtest_report.py` also runs a strategy that carries state of charge across days and includes degradation (default £8/MWh discharged, a placeholder), trading fees (default £0), an end-of-day reserve (default 4 MWh) and a terminal value for stored energy (tuned on the first 40% of days, judged on the rest). It reports the theoretical maximum (perfect MID, after costs), the achievable forecast-based result, a bridge between them, a degradation x fee sensitivity grid, tomorrow's planned schedule with expected profit, and 1/7/30-day revenue history. Network and levy charges are not modelled.

The dashboard shows 1-day, 7-day and 30-day views ending at the latest settled day. The imbalance models are hidden unless enabled with the sidebar checkbox.

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

# Dashboard (live Gold results from S3/MinIO) and the offline "Findings" page (sidebar)
streamlit run app/dashboard.py

# Regenerate the backtest behind the Findings page (needs internet access to Elexon)
python scripts/run_backtest_report.py --start 2026-08-01 --end 2026-11-04 --history-start 2026-01-01

# Tests (test_live_small_window_fetch needs internet access to Elexon)
pytest
```
`.github/workflows/daily-update.yml` runs the daily update on a schedule (needs the AWS secrets configured).

> PuLP is pinned `<4`: version 4 removed the `LpVariable(lowBound=...)` API used by the optimizer.
