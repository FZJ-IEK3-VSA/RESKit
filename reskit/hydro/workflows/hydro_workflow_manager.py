import numpy as np

from ...workflow_manager import WorkflowManager


class HydroWorkflowManager(WorkflowManager):
    """Workflow manager for hydropower simulations.

    This manager is intentionally lightweight so run-of-river, reservoir,
    and pumped-storage logic can be added incrementally while keeping the same
    in/out behavior as other RESKit technologies.
    """

    WATER_DENSITY_KG_M3 = 1000.0
    GRAVITY_M_S2 = 9.81

    def __init__(self, placements):
        super().__init__(placements)
        assert "capacity" in self.placements.columns, "Placement dataframe needs 'capacity' column"

    def simulate_run_of_river(self, inflow_m3s, net_head_m, efficiency=0.9):
        # Inline implementation from hydro/core/run_of_river.py
        inflow_m3s = np.asarray(inflow_m3s, dtype=float)
        capacity_kw = np.asarray(self.placements["capacity"].to_numpy(dtype=float)).reshape(1, -1)

        if inflow_m3s.ndim != 2:
            raise ValueError("inflow_m3s must have shape (time, locations)")
        if capacity_kw.shape[1] != inflow_m3s.shape[1]:
            raise ValueError("capacity_kw length must match inflow_m3s second dimension")

        if np.isscalar(net_head_m):
            head = np.full((1, inflow_m3s.shape[1]), float(net_head_m))
        else:
            head = np.asarray(net_head_m, dtype=float).reshape(1, inflow_m3s.shape[1])

        if not (0 <= float(efficiency) <= 1):
            raise ValueError("efficiency must be in the interval [0, 1]")

        power_output_w = (
            float(self.WATER_DENSITY_KG_M3)
            * float(self.GRAVITY_M_S2)
            * np.maximum(inflow_m3s, 0.0)
            * head
            * float(efficiency)
        )
        capacity_factor = np.clip(power_output_w / (capacity_kw * 1000.0), 0.0, 1.0)
        total_system_generation = power_output_w / 1000.0

        outputs = {
            "inflow_m3s": inflow_m3s,
            "net_head_m": np.broadcast_to(head, inflow_m3s.shape),
            "power_output_w": power_output_w,
            "capacity_factor": capacity_factor,
            "total_system_generation": total_system_generation,
        }

        for key, value in outputs.items():
            self.sim_data[key] = value

        return self

    def simulate_run_of_river_from_daily_discharge(
        self,
        discharge_m3_per_day,
        net_head_m,
        efficiency=0.88,
        cap_production_by_capacity=True,
    ):
        # Inline implementation from hydro/core/run_of_river.py
        if np.ma.isMaskedArray(discharge_m3_per_day):
            discharge_m3_per_day = np.ma.filled(discharge_m3_per_day, fill_value=0.0)
        discharge_m3_per_day = np.asarray(discharge_m3_per_day, dtype=float)
        capacity_kw = np.asarray(self.placements["capacity"].to_numpy(dtype=float), dtype=float).reshape(-1)

        if discharge_m3_per_day.ndim != 2:
            raise ValueError("discharge_m3_per_day must have shape (locations, time)")

        n_locations = discharge_m3_per_day.shape[0]
        if capacity_kw.shape[0] != n_locations:
            raise ValueError("capacity_kw length must match discharge_m3_per_day first dimension")

        if np.isscalar(net_head_m):
            head = np.full(n_locations, float(net_head_m))
        else:
            head = np.asarray(net_head_m, dtype=float).reshape(-1)
            if head.shape[0] != n_locations:
                raise ValueError("net_head_m length must match discharge_m3_per_day first dimension")

        if not (0 <= float(efficiency) <= 1):
            raise ValueError("efficiency must be in the interval [0, 1]")

        positive_discharge_m3_per_day = np.maximum(discharge_m3_per_day, 0.0)
        potential_system_generation = (
            positive_discharge_m3_per_day
            * head[:, None]
            * float(self.GRAVITY_M_S2)
            * float(self.WATER_DENSITY_KG_M3)
            * float(efficiency)
            / 3.6e6
        )

        if cap_production_by_capacity:
            capacity_limit_system_generation = capacity_kw[:, None] * 24.0
            total_system_generation = np.minimum(potential_system_generation, capacity_limit_system_generation)
        else:
            total_system_generation = potential_system_generation

        usable_discharge_m3_per_day = (
            total_system_generation
            * 3.6e6
            / (head[:, None] * float(self.GRAVITY_M_S2) * float(self.WATER_DENSITY_KG_M3) * float(efficiency))
        )

        capacity_factor = total_system_generation / (capacity_kw[:, None] * 24.0)

        outputs = {
            "discharge_m3_per_day": discharge_m3_per_day,
            "usable_discharge_m3_per_day": usable_discharge_m3_per_day,
            "capacity_factor": np.clip(capacity_factor, 0.0, 1.0)
            if cap_production_by_capacity
            else capacity_factor,
            "total_system_generation": total_system_generation,
        }

        # Convert to (time, location) for WorkflowManager/xarray conventions.
        for key, value in outputs.items():
            self.sim_data[key] = np.asarray(value).T

        return self
