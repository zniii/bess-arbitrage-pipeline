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
* **Asset Degradation Penalty ($C_{\text{deg}}$):** Prevents cycling when market spreads are too narrow to justify battery wear.
---
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)
---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Clone the repository and install the project in editable mode:

```bash
git clone [https://github.com/your-username/bess-arbitrage-pipeline.git](https://github.com/your-username/bess-arbitrage-pipeline.git)
cd bess-arbitrage-pipeline
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
