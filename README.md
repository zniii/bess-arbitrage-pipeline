# ⚡ BESS Arbitrage & Dispatch Engine

**Forecast tomorrow's Great Britain day-ahead power price, plan when a 10 MW / 20 MWh battery should charge and discharge, and measure what that plan actually earns after costs.**

The project is an end-to-end pipeline: market-data ingestion into an S3 Medallion lakehouse, a gradient-boosted price forecaster, a mixed-integer linear program (MILP) that schedules the battery, settlement of every plan at the real auction price, and a Streamlit app that presents the results.

## Headline result

Walk-forward backtest on the **N2EX GB day-ahead auction price**, 99 days (1 Aug – 7 Nov 2026). Each day is planned at 08:00 UTC the day before, using only data public at that time, then settled at the actual auction price. Profits are **after costs** (£8/MWh degradation placeholder, 4 MWh end-of-day reserve, state of charge carried across days).

| Strategy | Net profit | Share of perfect foresight |
|---|---|---|
| Perfect foresight (unattainable upper bound) | £117,527 | 100% |
| **Forecast model (this project)** | **£96,405** | **82%** |
| Naive: same hour last week | £85,540 | 73% |

* Forecast error (MAE): **£22.2/MWh** for the model vs £43.1/MWh for the naive benchmark.
* On the last 60 days, which were not used for tuning: £64,303 (model) vs £58,363 (naive) vs £78,096 (perfect foresight).
* The model beat the naive benchmark on 57 of 99 days. Forecast error, not costs, is the biggest source of lost profit.

> Trading is idealised: the whole schedule is assumed to trade at the hourly auction price with no volume limits or price impact, and network/levy charges are not modelled. Read these as an upper-level estimate, not a trading result. See [Limitations](#limitations).

## The app

Two offline Streamlit pages read pre-computed results from `app/data/`, so they run without AWS credentials.

<!-- TODO: add screenshots to docs/images/ and keep the links below -->
| Max Revenue (main dashboard) | Findings |
|---|---|
| ![Max Revenue page](docs/images/max_revenue.png) | ![Findings page](docs/images/findings.png) |

* **Max Revenue**: the day-ahead forecast and the resulting charge/discharge plan for any day (expected revenue, charging cost, degradation and fees, net profit), 1/7/30-day revenue history, and a waterfall from the theoretical maximum to the achievable result with a degradation × fee sensitivity grid.
* **Findings**: does a forecast capture day-ahead arbitrage? Cumulative profit versus perfect foresight and the naive benchmark, forecast error by month, and forecast vs actual for any day.

## How it works

**Decision timeline.** The auction for day D clears around 09:20 UK on D-1, so a plan made earlier must rely on a forecast.

| When | What happens |
|---|---|
| D-1, 08:00 UTC | The model forecasts the 48 half-hourly prices for day D from data public by now: auction prices up to D-1, plus wind and demand forecasts published by 08:00 UTC. The MILP turns the forecast into a schedule. |
| D-1, ~09:20 UK | The day-ahead auction for D clears. |
| Day D | The battery follows the plan. |
| After D | The plan is settled at the **actual** auction price, net of costs. |

**Pipeline.**

```
Elexon Insights + NESO open data
        │  ingestion/
        ▼
Bronze (raw JSON)  →  Silver (UTC-conformed Parquet)  →  Gold (schedules, daily KPIs)
                                   │
                                   ▼
              forecasting/ (gradient-boosted price model)
                                   │
                                   ▼
              optimization/ (MILP dispatch, PuLP + CBC)
                                   │
                                   ▼
              settlement at actual price, net of costs  →  Streamlit app
```

* **Bronze:** raw API payloads, partitioned by `year=YYYY/month=MM/`.
* **Silver:** typed data on UTC half-open intervals `[interval_start_utc, interval_end_utc)`, with duplicate and DST period-count checks; idempotent, partitioned Snappy Parquet.
* **Gold:** solved dispatch schedules and daily KPIs per strategy.
* **Serving:** DuckDB queries Parquet directly from S3/MinIO; Athena table definitions are in `sql/`.

### Forecast model

A scikit-learn `HistGradientBoostingRegressor` predicts the hourly N2EX day-ahead price for tomorrow. Features: calendar (time of day, weekday, month), day-ahead price lags of ≥ 1 day, rolling statistics, and Elexon wind (`WINDFOR`) and national demand (`NDFD`) forecasts. Only information published before the 08:00 UTC cutoff is used, and unit tests enforce the no-lookahead rule. The benchmark is a seasonal-naive forecast (same hour one week earlier). The model is retrained every 14 days. No free hosted API for GB price forecasts exists, so the model is built in-repo.

### Optimiser

For each half-hour *t* (Δt = 0.5 h) the MILP chooses charge power, discharge power and state of energy, to maximise

$$\sum_{t} \Big[ \lambda_t (P^{\text{dis}}_t - P^{\text{ch}}_t) - c_{\text{deg}} P^{\text{dis}}_t - c_{\text{fee}} (P^{\text{ch}}_t + P^{\text{dis}}_t) \Big] \Delta t \;+\; v \cdot \text{SOE}_T$$

where λ is the price, *c*<sub>deg</sub> the degradation cost per MWh discharged, *c*<sub>fee</sub> a trading fee per MWh traded, and *v* the value placed on energy left in the battery at the end of the day. Constraints:

* **Power:** 0 ≤ P<sup>ch</sup> ≤ 10·u<sub>t</sub> and 0 ≤ P<sup>dis</sup> ≤ 10·(1 − u<sub>t</sub>) MW, with u<sub>t</sub> binary (no simultaneous charge and discharge).
* **Energy window:** 2 MWh ≤ SOE ≤ 18 MWh (10%–90% of 20 MWh).
* **Efficiency (round trip η = 0.85):** SOE<sub>t</sub> = SOE<sub>t−1</sub> + (P<sup>ch</sup>·√η − P<sup>dis</sup>/√η)·Δt.
* **End-of-day reserve:** SOE at the end of the day ≥ a configurable minimum (default 4 MWh).

## Getting started

Requires Python 3.10+.

```bash
git clone https://github.com/zniii/bess-arbitrage-pipeline.git
cd bess-arbitrage-pipeline
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### Explore the app (no credentials needed)

```bash
streamlit run app/Max_Revenue.py     # Max Revenue is the landing page; Findings is in the sidebar
```

### Storage (only for the pipeline jobs): AWS S3 or local MinIO

```bash
cp .env.example .env     # fill in AWS credentials, or set AWS_ENDPOINT_URL=http://localhost:9000 for MinIO
docker compose up -d && docker compose run --rm create-bucket    # MinIO only
```

Leave `AWS_ENDPOINT_URL` unset to use real S3.

### Run the pipeline

```bash
# Update Gold to the latest settled day (idempotent; backfills gaps in the trailing window)
python -m bess_pipeline.orchestration.daily_update --window-days 30

# Perfect-foresight backtest for a date range
python -m bess_pipeline.orchestration.pipeline_runner --start-date YYYY-MM-DD --end-date YYYY-MM-DD

# Regenerate the results behind the app (needs internet access to Elexon and NESO)
python scripts/run_backtest_report.py --start YYYY-MM-DD --end YYYY-MM-DD \
    [--degradation 8 --fee 0 --reserve-mwh 4 --cache-dir DIR]

# Tests (test_live_small_window_fetch needs internet access to Elexon)
pytest
```

`.github/workflows/daily-update.yml` runs the daily update on a schedule (needs the AWS secrets configured).

> PuLP is pinned `<4`: version 4 removed the `LpVariable(lowBound=...)` API used by the optimiser.

## Project layout

```
app/                  Streamlit app: Max_Revenue.py (landing page), pages/Findings.py, data/ (pre-computed results)
scripts/              run_backtest_report.py (produces app/data/) and S3 maintenance helpers
src/bess_pipeline/
  ingestion/          Elexon and NESO API clients
  transformations/    UTC/DST handling, Silver Parquet storage
  forecasting/        features, model, backtest, strategies, cost-aware rolling backtest
  optimization/       BESSDispatcher (MILP)
  analytics/          revenue, cost and cycle KPIs
  orchestration/      pipeline_runner (CLI), daily_update (incremental job)
sql/                  Athena external-table DDL
tests/                optimiser, transformations, API clients, forecasting, rolling backtest, daily job, app pages
docs/                 Technical overview
```

## Limitations

* **Idealised trading:** no volume limits or price impact; real orders are submitted before the price is known.
* **Incomplete costs:** degradation is a placeholder (£8/MWh), the trading fee defaults to £0, and network/levy charges and taxes are not modelled. Other revenue streams (frequency response, capacity market, Balancing Mechanism) are excluded.
* **Small sample:** one autumn (99 days), a single run with untuned model settings; results will differ in other seasons and years.
* **Daily job vs app:** the scheduled daily job still uses simpler per-day strategies (battery reset each day, no costs); only the backtest report behind the app uses the cost-aware strategy.

## Documentation

See [`docs/PROJECT_SUMMARY.md`](docs/PROJECT_SUMMARY.md) for the technical overview: assumptions, data sources, model and optimiser details, full results and sensitivity tables, design notes and roadmap.

## Author

Golshan Yakkha · golshanjimee@gmail.com / https://www.linkedin.com/in/golshan-yakkha/.
