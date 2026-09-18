"""
Financial & Operational KPI Analytics for BESS Asset Dispatch.

Commercial Context:
-------------------
Quantifies the economic viability and cell wear of the dispatch schedule:
  - Wholesale Arbitrage Revenue (£): Value captured from power export.
  - Charging Cost (£): Cost incurred importing power from the grid.
  - Net Arbitrage Profit (£): Net commercial margin.
  - Equivalent Full Cycles (EFC): Normalized cycle wear tracking degradation warranties.
"""

from dataclasses import asdict, dataclass
import logging
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BESSFinancialMetrics:
    """Standardized commercial performance indicators for a dispatch run."""

    gross_revenue_gbp: float
    charging_cost_gbp: float
    net_profit_gbp: float
    export_throughput_mwh: float
    import_throughput_mwh: float
    equivalent_full_cycles: float
    avg_capture_spread_gbp_per_mwh: float

    def to_dict(self) -> dict:
        return asdict(self)


def calculate_dispatch_financial_kpis(
    df_schedule: pd.DataFrame,
    usable_capacity_mwh: float = 16.0,  # 80% usable DoD (20 MWh * (0.9 - 0.1))
    hours_per_period: float = 0.5,
) -> BESSFinancialMetrics:
    """
    Computes financial P&L and cycle wear metrics from a solved dispatch schedule.

    Args:
        df_schedule: DataFrame containing optimal dispatch decisions:
                     ['system_price_gbp', 'optimal_charge_mw', 'optimal_discharge_mw']
        usable_capacity_mwh: Usable energy capacity (MWh) between minimum and maximum SOC.
        hours_per_period: Duration of each settlement period in hours (0.5 for GB).

    Returns:
        BESSFinancialMetrics dataclass holding all commercial KPIs.
    """
    required_cols = {"system_price_gbp", "optimal_charge_mw", "optimal_discharge_mw"}
    missing = required_cols - set(df_schedule.columns)
    if missing:
        raise ValueError(f"Missing required columns in dispatch schedule: {missing}")

    if df_schedule.empty:
        logger.warning("Empty schedule provided. Returning zeroed metrics.")
        return BESSFinancialMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # 1. Energy Volumes (MWh)
    export_mwh = (df_schedule["optimal_discharge_mw"] * hours_per_period).sum()
    import_mwh = (df_schedule["optimal_charge_mw"] * hours_per_period).sum()

    # 2. Cash Flows (£)
    gross_revenue = (
        df_schedule["optimal_discharge_mw"]
        * df_schedule["system_price_gbp"]
        * hours_per_period
    ).sum()

    charging_cost = (
        df_schedule["optimal_charge_mw"]
        * df_schedule["system_price_gbp"]
        * hours_per_period
    ).sum()

    net_profit = gross_revenue - charging_cost

    # 3. Degradation & Cycling Metrics
    # EFC = Total discharged energy / Usable battery capacity
    efc = (export_mwh / usable_capacity_mwh) if usable_capacity_mwh > 0 else 0.0

    # 4. Average Captured Margin (£/MWh discharged)
    avg_capture_spread = (net_profit / export_mwh) if export_mwh > 0 else 0.0

    return BESSFinancialMetrics(
        gross_revenue_gbp=round(float(gross_revenue), 2),
        charging_cost_gbp=round(float(charging_cost), 2),
        net_profit_gbp=round(float(net_profit), 2),
        export_throughput_mwh=round(float(export_mwh), 2),
        import_throughput_mwh=round(float(import_mwh), 2),
        equivalent_full_cycles=round(float(efc), 3),
        avg_capture_spread_gbp_per_mwh=round(float(avg_capture_spread), 2),
    )