"""
BESS Linear Programming Dispatch Optimizer.

Commercial & Mathematical Context:
----------------------------------
Solves a Mixed-Integer Linear Program (MILP) to determine the optimal half-hourly
charge and discharge schedule for a Battery Energy Storage System (BESS) over a 
given pricing horizon, respecting capacity, power ratings, and round-trip efficiency.
"""

import logging
import pandas as pd
import pulp

logger = logging.getLogger(__name__)


class BESSDispatcher:
    """Optimizes BESS asset dispatch using Mixed-Integer Linear Programming."""

    def __init__(
        self,
        max_power_mw: float = 10.0,
        max_energy_mwh: float = 20.0,  # 2-hour duration system
        roundtrip_efficiency: float = 0.85,
        min_soc_pct: float = 0.1,
        max_soc_pct: float = 0.9,
    ) -> None:
        self.max_power_mw = max_power_mw
        self.max_energy_mwh = max_energy_mwh
        self.eff = roundtrip_efficiency ** 0.5  # Split efficiency across charge/discharge
        self.min_soc_mwh = max_energy_mwh * min_soc_pct
        self.max_soc_mwh = max_energy_mwh * max_soc_pct

    def optimize_schedule(self, df_gold: pd.DataFrame) -> pd.DataFrame:
        """
        Computes optimal half-hourly charge/discharge schedule maximizing arbitrage revenue.

        Args:
            df_gold: Gold feature DataFrame containing ['interval_start_utc', 'system_price_gbp']

        Returns:
            pd.DataFrame: Augmented DataFrame with decision variables [charge_mw, discharge_mw, soc_mwh].
        """
        if df_gold.empty:
            logger.warning("Empty DataFrame provided for optimization.")
            return pd.DataFrame()

        # Sort chronologically to maintain temporal continuity
        data = df_gold.sort_values("interval_start_utc").reset_index(drop=True)
        T = len(data)
        hours_per_period = 0.5  # 30-minute settlement periods

        # Initialize PuLP Minimization Problem (Maximize Revenue = Minimize -Revenue)
        prob = pulp.LpProblem("BESS_Arbitrage_Optimization", pulp.LpMaximize)

        # Create Decision Variables
        charge = [pulp.LpVariable(f"charge_{t}", lowBound=0, upBound=self.max_power_mw) for t in range(T)]
        discharge = [pulp.LpVariable(f"discharge_{t}", lowBound=0, upBound=self.max_power_mw) for t in range(T)]
        soc = [pulp.LpVariable(f"soc_{t}", lowBound=self.min_soc_mwh, upBound=self.max_soc_mwh) for t in range(T)]
        
        # Binary variables to prevent simultaneous charging and discharging (u_t = 1 if charging)
        is_charging = [pulp.LpVariable(f"is_charging_{t}", cat="Binary") for t in range(T)]

        # Objective Function: Maximize total revenue from discharging minus charging costs
        revenue_expr = []
        for t in range(T):
            price = data.loc[t, "system_price_gbp"]
            # Revenue from export minus cost of import (accounting for price and efficiency)
            revenue_expr.append(discharge[t] * price * hours_per_period - charge[t] * price * hours_per_period)
        
        prob += pulp.lpSum(revenue_expr)

        # Constraints
        initial_soc = self.min_soc_mwh  # Start day at minimum safe SOC

        for t in range(T):
            # 1. Mutually exclusive charge/discharge limits using binary indicators
            prob += charge[t] <= self.max_power_mw * is_charging[t]
            prob += discharge[t] <= self.max_power_mw * (1 - is_charging[t])

            # 2. State of Charge (SOC) transition equation
            prev_soc = initial_soc if t == 0 else soc[t - 1]
            energy_in = charge[t] * self.eff * hours_per_period
            energy_out = (discharge[t] / self.eff) * hours_per_period
            
            prob += soc[t] == prev_soc + energy_in - energy_out

        # Solve the MILP model using CBC (default open-source solver included with PuLP)
        logger.info("Solving MILP optimization model across %d settlement periods...", T)
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        status = pulp.LpStatus[prob.status]
        logger.info("Optimization status: %s", status)

        if status != "Optimal":
            logger.error("Optimization failed to find an optimal solution. Status: %s", status)
            return pd.DataFrame()

        # Extract solved decision variables back into DataFrame
        result_df = data.copy()
        result_df["optimal_charge_mw"] = [v.varValue for v in charge]
        result_df["optimal_discharge_mw"] = [v.varValue for v in discharge]
        result_df["state_of_energy_mwh"] = [v.varValue for v in soc]
        result_df["net_dispatch_mw"] = result_df["optimal_discharge_mw"] - result_df["optimal_charge_mw"]

        logger.info("BESS dispatch optimization successfully solved.")
        return result_df