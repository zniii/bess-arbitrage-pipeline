# BESS Arbitrage Pipeline: Technical Overview

A technical description of the pipeline, models, assumptions and results. For a short introduction and setup instructions, see the [README](../README.md).

*Results come from a walk-forward backtest on public Elexon and NESO data, 1 Aug – 7 Nov 2026 (99 days). Limits and unverified items are stated explicitly in [section 12](#12-limitations).*

## Contents

1. [Scope](#1-scope)
2. [Battery specification and model assumptions](#2-battery-specification-and-model-assumptions)
3. [Data sources](#3-data-sources)
4. [Workflow](#4-workflow)
5. [Repository structure](#5-repository-structure)
6. [Forecasting model](#6-forecasting-model)
7. [Optimiser (MILP)](#7-optimiser-milp)
8. [Strategies compared](#8-strategies-compared)
9. [Results](#9-results)
10. [Streamlit app](#10-streamlit-app)
11. [Infrastructure](#11-infrastructure)
12. [Limitations](#12-limitations)
13. [Design notes and lessons](#13-design-notes-and-lessons)
14. [Roadmap](#14-roadmap)

---

## 1. Scope

An automated data, forecasting and optimisation pipeline for a **10 MW / 20 MWh battery energy storage system (BESS)** trading in the **Great Britain wholesale electricity market**. It:

1. ingests market data (Elexon Insights, NESO open data),
2. stores it in a Bronze / Silver / Gold lakehouse (Amazon S3, or local MinIO),
3. **forecasts the next day's day-ahead price** with a gradient-boosted model,
4. **optimises** the battery's half-hourly charge/discharge schedule with a mixed-integer linear program (MILP),
5. settles the plan at the actual price, net of costs, and
6. presents the results in a Streamlit app.

The headline strategy trades on the **N2EX day-ahead auction price**. An experimental imbalance-price strategy is also available in the daily job.

---

## 2. Battery specification and model assumptions

| Parameter | Value | Notes |
|---|---|---|
| Power rating | **10 MW** | Charge and discharge limit; set by `--power-mw` / `BESSDispatcher(max_power_mw)` |
| Energy capacity | **20 MWh** | 2-hour system |
| Operating SOC window | **10% to 90%** | State of energy kept between **2 MWh and 18 MWh** |
| Usable energy | **16 MWh** | 18 minus 2; equivalent full cycles are measured against this |
| Round-trip efficiency | **85%** | Split as √0.85 ≈ 92.2% on each of charge and discharge |
| Settlement period | 30 minutes (0.5 h) | Max 5 MWh at the grid per period (10 MW × 0.5 h) |
| Simultaneous charge/discharge | Forbidden | Binary variable per period |
| Starting state of energy | 2 MWh (first day) | Afterwards each day starts from the previous day's end state (cost-aware strategy) |
| End-of-day reserve | **4 MWh** (default) | Minimum state of energy at the end of each planned day |
| Value of energy left overnight | **£25/MWh** | Tuned on the first 39 days only; the last 60 days are out of sample |
| Degradation cost | **£8/MWh discharged** | **Placeholder**; derive a site-specific value as capex (£/MWh) ÷ (warranted cycles × depth of discharge) |
| Trading / route-to-market fee | **£0/MWh** | No public figure exists (contractual); fee and revenue-share sensitivity provided |
| Network / levy charges | **Not modelled** | Site-specific; likely the largest missing real-world cost |

---

## 3. Data sources

All sources are free and need no API key.

| Data | Source | Endpoint / notes |
|---|---|---|
| **Day-ahead price (main)** | NESO Data Portal, "N2EX GB Day-Ahead Price" | `https://api.neso.energy/datastore/dump/4f27eea5-7038-4f73-9740-e3e4ad47c26a`. Hourly, **UTC**, 20 Sep 2021 onwards, no gaps, 591 negative-price hours. Each hourly price is applied to both half-hours. Alignment checked against Elexon MID: correlation 0.91 (best at zero hour shift) |
| Imbalance (system) price | Elexon Insights | `/balancing/settlement/system-prices/{date}` (half-hourly; 46/48/50 periods on DST days) |
| MID (index price) | Elexon Insights | `/datasets/MID`; use the `APXMIDP` provider (`N2EXMIDP` reports zero). **Published as each half-hour happens; it is not the day-ahead auction result.** |
| Wind generation forecast | Elexon Insights | `/datasets/WINDFOR` (hourly MW, with `publishTime`) |
| Demand forecast | Elexon Insights | `/datasets/NDFD` (national day-ahead demand, daily MW, with `publishTime`) |

No free hosted API that forecasts GB prices was found (Modo Energy is paid; energyforecast.de and Rabot cover other markets), so the forecast model is built in this repository. The NESO Carbon Intensity API was considered as an extra feature but is not used.

---

## 4. Workflow

### 4.1 The daily decision timeline

| When | What happens |
|---|---|
| D-2, ~09:20 UK | Day-ahead auction for D-1 clears; results become public |
| **D-1, 08:00 UTC** | **Decision time.** The model forecasts the 48 half-hourly prices for day D using only data public by now: auction prices up to D-1, wind and demand forecasts published by 08:00 UTC |
| D-1, ~09:20 UK | Day-ahead auction for D clears (the forecast is compared with this later) |
| Day D | Battery follows the plan |
| After D | Plan is **settled at the actual auction price**, net of costs |

### 4.2 Pipeline stages

1. **Bronze (raw):** raw JSON from Elexon, `bronze/elexon/year=YYYY/month=MM/payload_DATE.json`.
2. **Silver (conformed):** UTC half-open intervals `[interval_start_utc, interval_end_utc)`, typed columns, duplicate and DST period-count checks, partitioned Snappy Parquet `silver/system_prices/settlementDate=YYYY-MM-DD/part-0.parquet` (idempotent overwrite).
3. **Forecast:** gradient-boosted regressor on leak-free features ([section 6](#6-forecasting-model)).
4. **Optimise:** MILP with PuLP and the CBC solver ([section 7](#7-optimiser-milp)).
5. **Settle and report:** revenue, charging cost, degradation, fees, net profit and equivalent full cycles, written to Gold (`gold/dispatch_schedules/`, `gold/daily_kpis/`) and to the results files in `app/data/`.
6. **Serve:** Streamlit pages; optional Athena DDL; DuckDB reads Parquet straight from S3/MinIO.

### 4.3 Two code paths

| Path | Used by | Strategy details |
|---|---|---|
| **Daily job** (`orchestration/daily_update.py`) | Gold tables in S3/MinIO (not displayed by the app) | Per-day strategies; **battery reset to 10% SOC each day; no costs** |
| **Report script** (`scripts/run_backtest_report.py`) | Max Revenue and Findings pages (offline) | **Cost-aware, SOC carried across days, end-of-day reserve, tuned terminal value** |

The daily job is idempotent: it finds missing days in a trailing window (default 30), backfills them, and a second run does nothing. It is scheduled with a GitHub Actions cron (`17 6 * * *`).

---

## 5. Repository structure

```
bess-arbitrage-pipeline/
├── README.md                        Overview and run instructions
├── pyproject.toml / requirements.txt   Dependencies (pulp>=2.8,<4; scikit-learn; boto3; duckdb; streamlit; plotly; pandas; pyarrow)
├── .env.example                     AWS or MinIO settings (AWS_ENDPOINT_URL for MinIO)
├── docker-compose.yml               Local MinIO (S3-compatible) + bucket-creation helper
├── .github/workflows/daily-update.yml   Scheduled daily update
├── docs/PROJECT_SUMMARY.md          This document
├── sql/athena_external_tables.sql   Athena/Glue table DDL (silver, gold schedules, daily KPIs)
├── app/
│   ├── Max_Revenue.py               Landing page (offline): plan, 1/7/30-day revenue, max vs achievable, sensitivity
│   ├── pages/Findings.py            Offline: does a forecast capture day-ahead arbitrage?
│   └── data/                        Pre-computed results the pages read
│       ├── rolling_daily.csv        Per-day revenue, costs, net, expected vs realised, SOC, naive, perfect, forecast error
│       ├── plan_halfhourly.csv      Half-hourly actual/forecast price and the charge/discharge/SOE plan (incl. the next day)
│       ├── rolling_summary.json     Assumptions, bridge to the theoretical maximum, calibration, next-day plan
│       ├── sensitivity.csv          Degradation × trading-fee grid
│       └── backtest_meta.json       Run settings
├── scripts/
│   ├── run_backtest_report.py       Produces everything in app/data/ (walk-forward backtest)
│   └── clean_datalake.py, list_s3_layers.py, purge_stale_s3_parquet.py, reset_silver_s3.py   S3 maintenance
├── src/bess_pipeline/
│   ├── ingestion/    elexon_client.py (system prices, MID, WINDFOR, NDFD), neso_client.py (N2EX day-ahead)
│   ├── transformations/   temporal.py (UTC/DST), storage.py (Silver Parquet), features.py (legacy gold features)
│   ├── forecasting/  features.py, exogenous.py, model.py, backtest.py, strategies.py, rolling.py
│   ├── optimization/ dispatch_optimizer.py (BESSDispatcher MILP)
│   ├── analytics/    financial_kpis.py (revenue, cost, EFC, spread)
│   ├── common/       database.py (AWSDataLakeManager: S3/MinIO I/O)
│   └── orchestration/ pipeline_runner.py (CLI), daily_update.py (incremental job)
└── tests/            optimiser, transformations, Elexon and NESO clients, forecasting, rolling backtest, daily update, app pages
```

Requires Python 3.10 or newer.

---

## 6. Forecasting model

* **Target:** the hourly N2EX day-ahead price for tomorrow, expanded to 48 half-hours.
* **Algorithm:** scikit-learn `HistGradientBoostingRegressor` (200 iterations, learning rate 0.05). All-NaN feature columns are dropped (they break its binning).
* **Features** (all built in UTC, so 46/50-period DST days cannot misalign):
  * Calendar (UK local): time-of-day sine/cosine, day of week, weekend flag, month.
  * Day-ahead price lags of 1, 2, 3, 7 and 14 days (only those at or above the minimum lag).
  * Same-period mean and standard deviation over the previous 7 days; trailing 24-hour mean and standard deviation.
  * Wind forecast (`WINDFOR`, MW) and national demand forecast (`NDFD`, MW).
* **No-lookahead rules (enforced by tests):**
  * Day-ahead target: minimum lag 1 day, because yesterday's auction result is public at 08:00 UTC.
  * Imbalance and MID targets: minimum lag 2 days (their D-1 values are not fully known).
  * Wind/demand: only the latest publication with `publishTime` at or before D-1 08:00 UTC, matched by UK delivery date.
* **Retraining:** every 14 days in the report script; every 7 days in the daily job. Training data starts 20 Sep 2021 (wind/demand features are blank before Dec 2025).
* **Benchmark:** seasonal-naive, the same hour one week earlier.
* **Fallback:** if there is too little data to train, the job uses the naive forecaster.

---

## 7. Optimiser (MILP)

Per half-hour period *t* (Δt = 0.5 h), the decision variables are charge MW, discharge MW, state of energy (MWh) and a binary "is charging".

**Objective (maximise):** the sum over *t* of
`(discharge − charge) × price × Δt − degradation × discharge × Δt − fee × (charge + discharge) × Δt`
plus `terminal value × final state of energy`.

**Constraints:**
* `charge ≤ 10 × u_t`, `discharge ≤ 10 × (1 − u_t)` (no simultaneous charge and discharge).
* `SOE_t = SOE_(t−1) + charge × √η × Δt − discharge / √η × Δt`, with η = 0.85.
* `2 ≤ SOE_t ≤ 18` MWh; `SOE_final ≥ end-of-day reserve`; the initial state of energy is a parameter (clipped into the window).

The **perfect-foresight bound** optimises day D together with day D+1 (the lookahead) and commits day D. **Expected profit** is a plan valued at the forecast price; **realised profit** is the same plan valued at the actual price.

---

## 8. Strategies compared

| Strategy | Decides on | Settled at | Where used |
|---|---|---|---|
| `da_forecast` (headline) | Day-ahead price forecast | Actual day-ahead price | Daily job, report |
| `da_perfect` | Actual day-ahead prices | Actual day-ahead price | Upper bound |
| Naive | Same hour last week | Actual day-ahead price | Benchmark (report) |
| `imbalance_forecast` *(experimental)* | Imbalance-price forecast | Imbalance price | Daily job only |
| `imbalance_perfect` *(experimental)* | Actual imbalance prices | Imbalance price | Daily job only |

Imbalance and day-ahead profits use different settlement prices and are not directly comparable.

---

## 9. Results

Walk-forward backtest on the day-ahead price, 99 days (1 Aug – 7 Nov 2026). Costs: £8/MWh degradation (placeholder), £0 fee, 4 MWh end-of-day reserve, charge carried across days.

| Strategy | Net profit | Share of perfect foresight |
|---|---|---|
| Perfect foresight (upper bound) | **£117,527** | 100% |
| **Forecast model** | **£96,405** | **82%** |
| Naive (same hour last week) | £85,540 | 73% |

| Check | Value |
|---|---|
| Forecast error (MAE) | **£22.2/MWh** model vs £43.1 naive |
| Last 60 days (not used for tuning) | £64,303 model vs £58,363 naive vs £78,096 perfect |
| Days the model beat naive | 57 of 99 |
| Achievable breakdown | Revenue £421,339; charging cost £306,542; degradation £18,392; fees £0; net £96,405 |
| Bridge | Perfect, no costs, daily reset £136,299 → +£281 carrying charge across days → −£19,053 costs and reserve → −£21,122 forecast error → £96,405 |
| Expected vs realised | Expected profit was about **25% higher** than realised (realised = 0.80 × expected) |

**Sensitivity of the achievable result (net profit over the period):**

| Degradation (£/MWh discharged) | Fee £0 | Fee £1 | Fee £2 |
|---|---|---|---|
| £25 | £62,590 | £58,787 | £55,449 |
| £15 | £80,743 | £76,315 | £72,551 |
| £8 | £96,405 | £91,508 | £86,476 |
| £4 | £105,675 | £100,455 | £95,586 |
| £0 | £115,900 | £110,001 | £104,887 |

**Sensitivity of the perfect-foresight maximum:**

| Degradation (£/MWh discharged) | Fee £0 | Fee £1 | Fee £2 |
|---|---|---|---|
| £25 | £85,348 | £81,905 | £78,636 |
| £15 | £103,195 | £99,087 | £95,081 |
| £8 | £117,527 | £112,902 | £108,415 |
| £4 | £126,613 | £121,609 | £116,760 |
| £0 | £136,178 | £130,888 | £125,792 |

**Achievable full cycles per day:**

| Degradation (£/MWh discharged) | Fee £0 | Fee £1 | Fee £2 |
|---|---|---|---|
| £25 | 1.18 | 1.17 | 1.13 |
| £15 | 1.33 | 1.30 | 1.25 |
| £8 | 1.45 | 1.41 | 1.37 |
| £4 | 1.53 | 1.49 | 1.45 |
| £0 | 1.59 | 1.56 | 1.52 |

---

## 10. Streamlit app

Both pages read the pre-computed files in `app/data/` (written by `scripts/run_backtest_report.py`), so they work offline and on hosted Streamlit without AWS credentials. The Gold tables written by the daily job are not displayed.

* **Max Revenue** (`app/Max_Revenue.py`, landing page), three tabs:
  1. *Forecast & plan*: key figures (expected revenue, charging cost, degradation and fees, expected net profit, realised or calibrated profit), the day-ahead forecast, the charge/discharge/state-of-energy plan, and charge and discharge windows in UK time (any day, default the next planned day).
  2. *Revenue history (1 / 7 / 30 days)*: revenue, charging cost, degradation and fees, net profit, cumulative line.
  3. *Maximum vs achievable*: waterfall from the theoretical maximum to the achievable result, the sensitivity heatmap, break-even sell prices, and a revenue-share slider.
* **Findings** (`app/pages/Findings.py`, secondary page): headline result, cumulative profit (perfect, model, naive), forecast error by month, forecast vs actual for any day, daily advantage over naive, summary table, caveats and a CSV download.

Run with `streamlit run app/Max_Revenue.py`.

---

## 11. Infrastructure

* **Storage:** S3 (`AWS_S3_BUCKET`, region `eu-west-2`) or local **MinIO** by setting `AWS_ENDPOINT_URL` (see `docker-compose.yml`; bucket `bess-datalake`). One configurable endpoint in `AWSDataLakeManager`.
* **Query layer:** DuckDB (`httpfs`) and optional Amazon Athena DDL.
* **Scheduling:** `.github/workflows/daily-update.yml` (needs the AWS secrets).
* **Solver:** CBC via PuLP. PuLP is pinned `<4` because version 4 removed the `LpVariable(lowBound=...)` API the optimiser uses.

---

## 12. Limitations

* **Storage layer not validated end-to-end:** the S3 and MinIO integration has not been exercised against a live bucket. The Streamlit pages are covered by Streamlit's test harness.
* **Idealised trading:** assumes the whole schedule trades at the hourly auction price with no volume limits or price impact. Real orders are submitted before the price is known, and linking hours through the state of charge is hard in practice. Results are an estimate, not a trading result.
* **Incomplete costs:** degradation is a placeholder, fees default to £0, and network/levy charges, taxes and revenue share (except via the slider) are not included.
* **Small sample:** one autumn (99 days), untuned model settings, a single run. Results will differ in winter and in other years.
* **Daily job vs report:** the daily job uses the simpler per-day strategies (daily reset, no costs) and its output is not displayed; only the report behind the app uses the cost-aware, charge-carrying strategy.
* **Not modelled:** frequency response, capacity market, Balancing Mechanism revenue, forecast uncertainty ranges, intra-day re-forecasting, the Carbon Intensity forecast.
* **Wind/demand publication times** are taken as published by Elexon; the 08:00 UTC cutoff is conservative against the ~09:20 UK auction deadline.

---

## 13. Design notes and lessons

1. **MID is not the day-ahead auction result.** Elexon MID is an index published as each half-hour happens (only 19 of 48 half-hours existed at 09:25 UTC on the delivery day), so it is not a price that can be traded in advance. Earlier versions of this project used it and have been replaced with the real auction price. A 1-day price lag also leaked partly-unknown data for MID; removing it lowered that model's profit share from 62% to 57% before costs.
2. **Actual auction history is free; forecasts are not.** NESO publishes the realised N2EX price history, which made a realistic backtest possible. The forecasting model itself has to be built.
3. **Imbalance prices are very hard to forecast** and are only known afterwards. Over 96 days the imbalance forecast earned about £73k against a £378k perfect-foresight bound (about 19%), so that strategy remains experimental.
4. **Forecast error is the main loss**, much larger than costs: £21,122 of forecast error against £19,053 of costs and reserve in the bridge above, with carrying charge across days adding only £281 on the day-ahead price.
5. **Leakage control:** all features respect the 08:00 UTC cutoff, with unit tests that fail if a feature uses data published after it.
6. **Training on all history** (since 2021) rather than only 2026 raised profit from £90.4k to £96.4k with a similar forecast error. This choice was made after comparing both, so treat it as modest evidence.
7. **Expected profit is optimistic.** The forecast is smoother than reality and the plan is picked to look best on it, so expected profit overstates what is realised (about 25% over the backtest); the app also shows a calibrated estimate.

---

## 14. Roadmap

1. Correct forecast bias and tune the model (more features, quantile outputs for risk-aware dispatch).
2. Replace the placeholder costs with real degradation (capex and warranted cycles), trading fee and network charges.
3. Move the cost-aware, charge-carrying strategy into the daily job, and decide whether to display its results.
4. Validate against a live S3 or MinIO bucket and deploy the daily schedule.
5. Add intra-day re-forecasting and other revenue streams.
