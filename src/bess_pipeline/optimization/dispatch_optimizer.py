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
        degradation_cost_gbp_per_mwh: float = 0.0,
        trading_fee_gbp_per_mwh: float = 0.0,
        end_soc_min_mwh: float | None = None,
        terminal_value_gbp_per_mwh: float = 0.0,
    ) -> None:
        """
        Args:
            degradation_cost_gbp_per_mwh: wear cost per MWh DISCHARGED (grid side). Derive it as
                battery capex (GBP/MWh) / (warranted full cycles x depth of discharge).
            trading_fee_gbp_per_mwh: exchange / route-to-market fee on every MWh bought or sold.
            end_soc_min_mwh: state of energy that must remain at the END of the optimized horizon
                (reserve so the next day can start charged). Defaults to the minimum SOC.
            terminal_value_gbp_per_mwh: value placed on each MWh left in the battery at the end of
                the horizon (stored energy can be sold tomorrow). 0 = no value.
        """
        self.max_power_mw = max_power_mw
        self.max_energy_mwh = max_energy_mwh
        self.eff = roundtrip_efficiency ** 0.5  # Split efficiency across charge/discharge
        self.min_soc_mwh = max_energy_mwh * min_soc_pct
        self.max_soc_mwh = max_energy_mwh * max_soc_pct
        self.degradation_cost = degradation_cost_gbp_per_mwh
        self.trading_fee = trading_fee_gbp_per_mwh
        self.end_soc_min_mwh = self.min_soc_mwh if end_soc_min_mwh is None else end_soc_min_mwh
        self.terminal_value = terminal_value_gbp_per_mwh

    def optimize_schedule(
        self,
        df_gold: pd.DataFrame,
        price_col: str = "system_price_gbp",
        initial_soc_mwh: float | None = None,
    ) -> pd.DataFrame:
        """
        Computes optimal half-hourly charge/discharge schedule maximizing arbitrage revenue.

        Args:
            df_gold: Gold feature DataFrame containing ['interval_start_utc', price_col].
            price_col: Column the optimizer DECIDES on (e.g. a forecast). Settlement
                against actual prices happens downstream in the KPI step.
            initial_soc_mwh: starting state of energy (e.g. the previous day's end SOC).
                Defaults to the minimum SOC; clipped into the allowed window.

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
            price = data.loc[t, price_col]
            # Revenue from export minus cost of import, minus throughput wear
            revenue_expr.append(
                (discharge[t] - charge[t]) * price * hours_per_period
                - self.degradation_cost * discharge[t] * hours_per_period
                - self.trading_fee * (charge[t] + discharge[t]) * hours_per_period
            )
        
        prob += pulp.lpSum(revenue_expr) + self.terminal_value * soc[T - 1]

        # Constraints
        initial_soc = self.min_soc_mwh if initial_soc_mwh is None else initial_soc_mwh
        initial_soc = min(max(initial_soc, self.min_soc_mwh), self.max_soc_mwh)
        # Reserve at the end of the horizon (reachable: it is within the SOC window)
        prob += soc[T - 1] >= min(self.end_soc_min_mwh, self.max_soc_mwh)

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