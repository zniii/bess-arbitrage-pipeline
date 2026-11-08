# BESS Arbitrage Pipeline: Project Summary

*Branch `claude/repo-summary-pbwblf`. Results generated 8 Nov 2026 on real Elexon and NESO data. Everything below is what was built and measured in this project; limits and unverified items are stated explicitly.*

---

## 1. What the project is

An automated data, forecasting and optimisation pipeline for a **10 MW / 20 MWh battery energy storage system (BESS)** trading in the **Great Britain wholesale electricity market**. It:

1. ingests market data (Elexon Insights, NESO open data),
2. stores it in a Bronze / Silver / Gold lakehouse (Amazon S3, or local MinIO),
3. **forecasts the next day's day-ahead price** with a gradient-boosted model,
4. **optimises** the battery's half-hourly charge/discharge schedule with a mixed-integer linear programme (MILP),
5. settles the plan at the actual price, net of costs, and
6. presents results in Streamlit.

**Where it started:** a manual backtest that optimised on the *actual* imbalance price (perfect foresight, so an unattainable upper bound), with one-day static results.
**Where it ended:** a forecast-driven, cost-aware plan on the **N2EX day-ahead auction price**, with a daily-update job, a live dashboard, and two offline results pages.

---

## 2. Battery specification and model assumptions

| Parameter | Value | Notes |
|---|---|---|
| Power rating | **10 MW** | Charge and discharge limit; set by `--power-mw` / `BESSDispatcher(max_power_mw)` |
| Energy capacity | **20 MWh** | 2-hour system |
| Operating SOC window | **10% to 90%** | State of energy kept between **2 MWh and 18 MWh** |
| Usable energy | **16 MWh** | 18 minus 2; equivalent full cycles are measured against this |
| Round-trip efficiency | **85%** | Split as sqrt(0.85) = about 92.2% on each of charge and discharge |
| Settlement period | 30 minutes (0.5 h) | Max 5 MWh at the grid per period (10 MW x 0.5 h) |
| Simultaneous charge/discharge | Forbidden | Binary variable per period |
| Starting state of energy | 2 MWh (first day) | After that, each day starts from the previous day's end state of energy (cost-aware strategy) |
| End-of-day reserve | **4 MWh** (default) | Minimum state of energy at the end of each planned day |
| Value of energy left overnight | **£25/MWh** | Tuned on the first 39 days only; the last 60 days are out of sample |
| Degradation cost | **£8/MWh discharged** | **Placeholder**; derive your own as capex (£/MWh) / (warranted cycles x depth of discharge) |
| Trading / route-to-market fee | **£0/MWh** | No public figure exists (contractual); fee and revenue-share sensitivity provided |
| Network / levy charges | **Not modelled** | Unknown for the site; likely the largest missing real-world cost |

---

## 3. Data sources (all free, no API key)

| Data | Source | Endpoint / notes |
|---|---|---|
| **Day-ahead price (main)** | NESO Data Portal, "N2EX GB Day-Ahead Price" | `https://api.neso.energy/datastore/dump/4f27eea5-7038-4f73-9740-e3e4ad47c26a`. Hourly, **UTC**, 20 Sep 2021 onwards, no gaps, 591 negative-price hours. Each hourly price is applied to both half-hours. Alignment checked against Elexon MID: correlation 0.91 (best at zero hour shift) |
| Imbalance (system) price | Elexon Insights | `/balancing/settlement/system-prices/{date}` (half-hourly; 46/48/50 periods on DST days) |
| MID (index price) | Elexon Insights | `/datasets/MID`; use the `APXMIDP` provider (`N2EXMIDP` reports zero). **Published as each half-hour happens; it is not the day-ahead auction result.** Used in earlier versions only |
| Wind generation forecast | Elexon Insights | `/datasets/WINDFOR` (hourly MW, with `publishTime`) |
| Demand forecast | Elexon Insights | `/datasets/NDFD` (national day-ahead demand, daily MW, with `publishTime`) |

**Not found:** no free hosted API that forecasts GB prices (Modo Energy is paid; energyforecast.de / Rabot cover other markets). The forecast model is therefore built in this repo. The NESO Carbon Intensity API was considered as an extra feature but is not used.

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
3. **Forecast:** gradient-boosted regressor on leak-free features (section 6).
4. **Optimise:** MILP with PuLP + the CBC solver (section 7).
5. **Settle and report:** revenue, charging cost, degradation, fees, net profit, equivalent full cycles, written to Gold (`gold/dispatch_schedules/`, `gold/daily_kpis/`) and to the offline results files in `app/data/`.
6. **Serve:** Streamlit pages; optional Athena DDL; DuckDB reads Parquet straight from S3/MinIO.

### 4.3 Two code paths (important)

| Path | Used by | Strategy details |
|---|---|---|
| **Daily job** (`orchestration/daily_update.py`) | Live dashboard (reads S3/MinIO) | Per-day strategies; **battery reset to 10% SOC each day; no costs** |
| **Report script** (`scripts/run_backtest_report.py`) | Findings and Max Revenue pages (offline) | **Cost-aware, SOC carried across days, end-of-day reserve, tuned terminal value** |

The daily job is idempotent: it finds missing days in a trailing window (default 30), backfills them, and a second run does nothing. Its schedule is a GitHub Actions cron (`17 6 * * *`).

---

## 5. Folder structure

```
bess-arbitrage-pipeline/
├── README.md                        Project overview and run instructions
├── pyproject.toml / requirements.txt   Dependencies (pulp>=2.8,<4; scikit-learn; boto3; duckdb; streamlit; plotly; pandas; pyarrow)
├── .env.example                     AWS or MinIO settings (AWS_ENDPOINT_URL for MinIO)
├── docker-compose.yml               Local MinIO (S3-compatible) + bucket-creation helper
├── .github/workflows/daily-update.yml   Scheduled daily update
├── docs/PROJECT_SUMMARY.md          This document
├── sql/athena_external_tables.sql   Athena/Glue table DDL (silver, gold schedules, daily KPIs)
├── app/
│   ├── dashboard.py                 Live dashboard: 1/7/30-day results from S3/MinIO, imbalance-model toggle
│   ├── pages/1_Findings.py          Offline: does a forecast capture day-ahead arbitrage?
│   ├── pages/2_Max_Revenue.py       Offline: plan for tomorrow, 1/7/30-day revenue, max vs achievable, sensitivity
│   └── data/                        Pre-computed results the offline pages read
│       ├── rolling_daily.csv        Per-day revenue, costs, net, expected vs realised, SOC, naive, perfect, forecast error
│       ├── plan_halfhourly.csv      Half-hourly actual/forecast price and the charge/discharge/SOE plan (incl. tomorrow)
│       ├── rolling_summary.json     Assumptions, bridge to the theoretical maximum, calibration, next-day plan
│       ├── sensitivity.csv          Degradation x trading-fee grid
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
│   └── orchestration/ pipeline_runner.py (original CLI), daily_update.py (incremental job)
└── tests/            22 tests (optimiser, transformations, Elexon client, NESO parser, forecasting, rolling, daily update)
```

About 3,600 lines of Python and SQL including tests and scripts. Python 3.13 was used in the sandbox (`requires-python >= 3.10`).

---

## 6. The forecasting model

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

## 7. The optimiser (MILP)

Per half-hour period *t* (Δt = 0.5 h), decision variables: charge MW, discharge MW, state of energy (MWh), and a binary "is charging".

**Objective (maximise):** the sum over *t* of
`(discharge - charge) x price x Δt  -  degradation x discharge x Δt  -  fee x (charge + discharge) x Δt`
plus `terminal value x final state of energy`.

**Constraints:**
* `charge <= 10 x u_t`, `discharge <= 10 x (1 - u_t)` (no simultaneous charge and discharge).
* `SOE_t = SOE_(t-1) + charge x sqrt(eta) x Δt - discharge / sqrt(eta) x Δt`, with eta = 0.85.
* `2 <= SOE_t <= 18` MWh; `SOE_final >= end-of-day reserve`; initial state of energy is a parameter (clipped into the window).

**Perfect-foresight bound** optimises day D together with day D+1 (the lookahead) and commits day D. **Expected profit** is a plan valued at the forecast price; **realised profit** is the same plan valued at the actual price.

---

## 8. Strategies compared

| Strategy | Decides on | Settled at | Where used |
|---|---|---|---|
| `da_forecast` (headline) | Our day-ahead forecast | Actual day-ahead price | Daily job, dashboard, report |
| `da_perfect` | Actual day-ahead prices | Actual day-ahead price | Upper bound |
| Naive | Same hour last week | Actual day-ahead price | Benchmark (report) |
| `imbalance_forecast` *(experimental)* | Imbalance-price forecast | Imbalance price | Dashboard toggle |
| `imbalance_perfect` *(experimental)* | Actual imbalance prices | Imbalance price | Dashboard toggle |

Imbalance and day-ahead profits use different settlement prices and are not directly comparable.

---

## 9. Results (final: day-ahead price, 99 days, 1 Aug to 7 Nov 2026)

Costs: £8/MWh degradation (placeholder), £0 fee, 4 MWh end-of-day reserve, charge carried across days.

| Strategy | Net profit | Share of perfect foresight |
|---|---|---|
| Perfect foresight (upper bound) | **£117,527** | 100% |
| **Our forecast model** | **£96,405** | **82%** |
| Naive (same hour last week) | £85,540 | 73% |

| Check | Value |
|---|---|
| Forecast error (MAE) | **£22.2/MWh** model vs £43.1 naive |
| Last 60 days (not used for tuning) | £64,303 model vs £58,363 naive vs £78,096 perfect |
| Days the model beat naive | 57 of 99 |
| Achievable breakdown | Revenue £421,339; charging cost £306,542; degradation £18,392; fees £0; net £96,405 |
| Bridge | Perfect, no costs, daily reset £136,299 → +£281 carrying charge across days → -£19,053 costs and reserve → -£21,122 forecast error → £96,405 |
| Expected vs realised | Expected profit was about **25% higher** than realised (realised = 0.80 x expected) |
| Tomorrow (9 Nov 2026) plan | Expected revenue £10,284, charging cost £7,870, net about £2,178 (about £1,740 calibrated) |

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

## 10. The pages

* **Live dashboard (`app/dashboard.py`)**: reads Gold results from S3/MinIO via DuckDB. 1-day / 7-day / 30-day selector; daily profit bars; forecast vs actual price, dispatch and state of energy for a chosen day; sidebar toggle for the experimental imbalance models.
* **Findings (`1_Findings.py`)**: headline result, cumulative profit (perfect, model, naive), forecast error by month, forecast vs actual for any day, daily advantage over naive, summary table, caveats, CSV download.
* **Max Revenue (`2_Max_Revenue.py`)**, three tabs:
  1. *Forecast & plan*: the day-ahead forecast, the charge/discharge/state-of-energy plan, charge and discharge windows in UK time, expected profit and a calibrated estimate (any day, default tomorrow).
  2. *Revenue history (1 / 7 / 30 days)*: revenue, charging cost, degradation and fees, net profit, cumulative line.
  3. *Maximum vs achievable*: waterfall from the theoretical maximum to the achievable result, the sensitivity heatmap, break-even sell prices, and a revenue-share slider.

---

## 11. Infrastructure

* **Storage:** S3 (`AWS_S3_BUCKET`, region `eu-west-2`) or local **MinIO** by setting `AWS_ENDPOINT_URL` (see `docker-compose.yml`; bucket `bess-datalake`). One configurable endpoint in `AWSDataLakeManager` and in the dashboard's DuckDB connection.
* **Query layer:** DuckDB (`httpfs`) and optional Amazon Athena DDL.
* **Scheduling:** `.github/workflows/daily-update.yml` (needs the AWS secrets).
* **Session environment (cloud sandbox):** a dedicated environment `bess-arbitrage-pipeline` allows `data.elexon.co.uk`, `api.carbonintensity.org.uk`, `www.neso.energy` and `api.neso.energy`.
* **Solver:** CBC (via PuLP). PuLP is pinned `<4` because version 4 removed the `LpVariable(lowBound=...)` API the optimiser uses.

---

## 12. Key decisions, corrections and findings along the way

1. **MID is not the day-ahead auction result.** Early versions treated Elexon MID as a known-in-advance price. It is an index published as each half-hour happens (only 19 of 48 half-hours existed at 09:25 UTC on the day). The first MID model also used a 1-day price lag that leaked partly-unknown data; fixing that lowered its profit share from 62% to 57% before costs. MID was then replaced by the real auction price.
2. **Free GB price forecasts do not exist;** the model is built in-repo. A free *actual* auction-price history does (NESO), which made the project realistic.
3. **Imbalance prices are very hard to forecast** and are only known afterwards. On 96 days the imbalance forecast earned about £73k against a £378k perfect-foresight bound (about 19%). They remain an experimental toggle only.
4. **Forecast error is the main loss**, much larger than costs. On the earlier MID analysis, carrying charge across days added only about 1.4%; on the day-ahead price it added about £281.
5. **Bugs found and fixed:** missing `pandas` import in `database.py` (would crash on import); `ew-west-2` region typo; mismatched bucket variable names; UTC-vs-UK day mismatch when looking up demand forecasts around BST; all-NaN feature columns breaking model training; PuLP 4 API break.
6. **Training on all history** (since 2021) rather than only 2026 raised profit from £90.4k to £96.4k with a similar forecast error. This choice was made after seeing both, so treat it as modest evidence.
7. **Recent bias:** over the last seven known days the forecasts ran about £20/MWh above the actual prices; tomorrow's high forecast (peak near £390/MWh) is probably overstated.

---

## 13. How to run

```bash
pip install -e ".[dev]"
cp .env.example .env                  # fill in AWS credentials, or set AWS_ENDPOINT_URL for MinIO

# Local S3 (optional)
docker compose up -d && docker compose run --rm create-bucket

# Update Gold to the latest settled day (idempotent)
python -m bess_pipeline.orchestration.daily_update --window-days 30

# Original single-day / range backtest (perfect foresight on imbalance prices)
python -m bess_pipeline.orchestration.pipeline_runner --start-date 2026-08-01 --end-date 2026-08-03

# Regenerate the data behind the Findings and Max Revenue pages (needs internet access to Elexon and NESO)
python scripts/run_backtest_report.py --start 2026-08-01 --end 2026-11-08 \
    [--degradation 8 --fee 0 --reserve-mwh 4 --cache-dir DIR]

# Dashboard and pages
streamlit run app/dashboard.py

# Tests (the live Elexon test needs internet)
pytest
```

---

## 14. Limitations and what has not been verified

* **Never run against real S3 or MinIO:** the cloud session had no AWS credentials or Docker. The dashboard was tested only with Streamlit's test harness on synthetic Gold data. The offline pages were checked in a browser.
* **Idealised trading:** assumes the whole schedule trades at the hourly auction price with no volume limits or price impact. Real orders are submitted before the price is known, and linking hours through the state of charge is hard in practice. Results are an estimate, not a trading result.
* **Costs are incomplete:** degradation is a placeholder, fees default to £0, and network/levy charges, taxes, and revenue share (except the slider) are not included.
* **Small sample:** one autumn (99 days), untuned model settings, a single run. Results will differ in winter and other years.
* **Daily job vs report:** the live daily job still uses the simpler per-day strategies (daily reset, no costs); only the offline pages use the cost-aware, charge-carrying strategy.
* **Not modelled:** frequency response, capacity market, Balancing Mechanism revenue, forecast uncertainty ranges, intra-day re-forecasting, the Carbon Intensity forecast.
* **Wind/demand publication times** are taken as published by Elexon; the 08:00 UTC cutoff is conservative against the ~09:20 UK auction deadline.

## 15. Suggested next steps

1. Fix the forecast bias and tune the model (more features, quantile outputs for risk-aware dispatch).
2. Put the real degradation cost (capex and warranted cycles), trading fee and network charges into the model.
3. Move the cost-aware, charge-carrying strategy into the daily job and live dashboard.
4. Test against real S3/MinIO and deploy the daily schedule.
5. Add intra-day re-forecasting and other revenue streams.
