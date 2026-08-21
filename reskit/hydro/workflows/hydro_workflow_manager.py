import warnings

import numpy as np
import pandas as pd

from ...workflow_manager import WorkflowManager
from ..core import extract_selected_discharge_alluvium


class HydroWorkflowManager(WorkflowManager):
    """Workflow manager for discharge extraction and hydropower calculations."""

    WATER_DENSITY_KG_M3 = 1000.0
    GRAVITY_M_S2 = 9.81
    PARFLOW_1DAY_URL = (
        "https://service.tereno.net/thredds/dodsC/forecastnrw/products/ParFlow-DE06-HC_v03/"
        "sfd_DE05_ECMWF-HRES_hindcast_r1i1p2_FZJ-IBG3-ParFlowCLM380_"
        "hgfadapter-h00-v03bJuwelsGpuProdClimatologyTl_1day_{year}0101-{year}1231.nc"
    )
    PARFLOW_3HOUR_URL = (
        "https://service.tereno.net/thredds/dodsC/forecastnrw/products/tmp_ice2/"
        "sfd_DE06_ECMWF-HRES_hindcast_r1i1p1_FZJ-IBG3-ParFlowCLM380_"
        "hgfadapter-h00-v03bJuwelsGpuProdClimatologyTl_3hours_"
        "{year}0101-{year}1231.ICE2.nc"
    )
    DISCHARGE_PRODUCTS = {
        "parflow-1day": {
            "native_interval": pd.Timedelta(days=1),
            "native_unit": "m3_per_timestep",
            "url_template": PARFLOW_1DAY_URL,
        },
        "parflow-3hour": {
            "native_interval": pd.Timedelta(hours=3),
            "native_unit": "m3_per_timestep",
            "url_template": PARFLOW_3HOUR_URL,
        },
    }
    DISCHARGE_PRODUCT_ALIASES = {"parflow": "parflow-1day"}

    def __init__(self, placements):
        super().__init__(placements)

    @staticmethod
    def _validate_time_index(time_index):
        time_index = pd.DatetimeIndex(time_index)
        if len(time_index) == 0:
            raise ValueError("time_index must contain at least one timestamp")
        if time_index.has_duplicates or not time_index.is_monotonic_increasing:
            raise ValueError("time_index must be unique and monotonically increasing")
        return time_index

    @staticmethod
    def _time_step(time_index):
        if len(time_index) < 2:
            raise ValueError("time_index must contain at least two timestamps to infer its interval")
        steps = np.diff(time_index.asi8)
        if not np.all(steps == steps[0]):
            raise ValueError("time_index must have a regular interval")
        return pd.Timedelta(int(steps[0]), unit="ns")

    @staticmethod
    def _location_values(value, n_locations, name):
        if np.isscalar(value):
            result = np.full(n_locations, float(value))
        else:
            result = np.asarray(value, dtype=float).reshape(-1)
        if result.size != n_locations:
            raise ValueError(f"{name} must contain one value per location")
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must contain only finite values")
        return result

    @staticmethod
    def _validate_efficiency(efficiency):
        efficiency = float(efficiency)
        if not 0 < efficiency <= 1:
            raise ValueError("efficiency must be in the interval (0, 1]")
        return efficiency

    def _hydraulic_power_w(self, discharge_m3s, head_m, efficiency):
        return (
            self.WATER_DENSITY_KG_M3
            * self.GRAVITY_M_S2
            * np.maximum(discharge_m3s, 0.0)
            * head_m[None, :]
            * efficiency
        )

    @staticmethod
    def _align_discharge(frame, requested_index, native_interval, requested_interval):
        """Align mean discharge rates without interpolating interval volumes."""
        if requested_interval == native_interval:
            return frame.reindex(requested_index), False, None
        if requested_interval < native_interval:
            # The final native value represents the final native interval. A
            # repeated right edge lets interpolation cover that whole interval.
            right_edge = frame.index[-1] + native_interval
            extended = pd.concat(
                [frame, pd.DataFrame([frame.iloc[-1].to_numpy()], index=[right_edge])]
            )
            aligned = extended.reindex(extended.index.union(requested_index)).interpolate(
                method="time", limit_area="inside"
            )
            return aligned.reindex(requested_index), True, "linearly interpolated"

        # Downsampling mean rates conserves volume for regular native steps.
        aligned = frame.resample(requested_interval, origin=requested_index[0]).mean()
        return aligned.reindex(requested_index), True, "time-averaged"

    def extract_discharge(self, product, time_index, product_options=None):
        """Extract and align discharge, storing it as m3/s in (time, location)."""
        product = str(product).lower()
        product = self.DISCHARGE_PRODUCT_ALIASES.get(product, product)
        if product not in self.DISCHARGE_PRODUCTS:
            available = ", ".join(sorted(self.DISCHARGE_PRODUCTS))
            raise ValueError(f"Unknown discharge product '{product}'. Available products: {available}")

        requested_index = self._validate_time_index(time_index)
        requested_interval = self._time_step(requested_index)
        options = {} if product_options is None else dict(product_options)

        if product.startswith("parflow-"):
            required = ("root_dir", "alluvium_mask_file", "indicator_file")
            missing = [name for name in required if name not in options]
            if missing:
                raise ValueError("Missing ParFlow product options: " + ", ".join(missing))
            years = requested_index.year.unique()
            year = int(options.pop("year", years[0]))
            if len(years) != 1 or year != int(years[0]):
                raise ValueError("ParFlow extraction currently supports one calendar year per call")
            extraction = extract_selected_discharge_alluvium(
                year=year,
                placements=self.placements,
                root_dir=options.pop("root_dir"),
                alluvium_mask_file=options.pop("alluvium_mask_file"),
                indicator_file=options.pop("indicator_file"),
                fallback_mode=options.pop("fallback_mode", "max_annual"),
                data_url=self.DISCHARGE_PRODUCTS[product]["url_template"].format(year=year),
            )
            if options:
                raise ValueError("Unknown ParFlow product options: " + ", ".join(sorted(options)))
            native_interval = self.DISCHARGE_PRODUCTS[product]["native_interval"]
            if "selected_discharge_m3_per_timestep" in extraction:
                native_volume = extraction["selected_discharge_m3_per_timestep"]
            else:
                native_volume = extraction["selected_discharge_m3_per_day"]
            native = np.ma.filled(native_volume, np.nan)
            native = np.asarray(native, dtype=float).T / native_interval.total_seconds()
            native_index = pd.date_range(
                f"{year}-01-01", periods=native.shape[0], freq=native_interval
            )
            self.placements["selected_candidate_idx"] = extraction["selected_candidate_idx"]
            self.placements["selected_from_alluvium"] = extraction["selected_from_alluvium"].astype(int)
            self.selected_cell_overview = extraction["selected_cell_overview"]
        else:  # pragma: no cover
            raise NotImplementedError(product)

        native_interval = self.DISCHARGE_PRODUCTS[product]["native_interval"]
        product_end = native_index[-1] + native_interval
        if (
            requested_index[0] < native_index[0]
            or requested_index[-1] + requested_interval > product_end
        ):
            raise ValueError("Requested time_index extends beyond the discharge product coverage")
        frame = pd.DataFrame(native, index=native_index)
        frame, resampled, resampling_method = self._align_discharge(
            frame, requested_index, native_interval, requested_interval
        )
        if resampled:
            warnings.warn(
                f"Requested interval {requested_interval} differs from the native {product} "
                f"interval {native_interval}; discharge rates are {resampling_method}.",
                UserWarning,
                stacklevel=2,
            )
        self.set_time_index(requested_index)
        self.sim_data["discharge_m3s"] = frame.to_numpy(dtype=float)
        self.workflow_parameters["discharge_product"] = product
        self.workflow_parameters["native_time_interval"] = str(native_interval)
        self.workflow_parameters["requested_time_interval"] = str(requested_interval)
        self.workflow_parameters["temporal_resampling_applied"] = str(resampled)
        self.workflow_parameters["temporal_resampling_method"] = resampling_method or "none"
        return self

    def calculate_hydropower(
        self,
        discharge_m3s=None,
        time_index=None,
        efficiency=0.88,
        cap_production_by_capacity=True,
    ):
        """Convert discharge or prescribed release into power and interval energy."""
        if "head" not in self.placements.columns or "capacity" not in self.placements.columns:
            raise ValueError("placements must contain 'head' and 'capacity' columns")
        if time_index is not None:
            given_index = self._validate_time_index(time_index)
            if self.time_index is not None and not given_index.equals(self.time_index):
                raise ValueError("time_index does not match the manager's existing time index")
            if self.time_index is None:
                self.set_time_index(given_index)
        if self.time_index is None:
            raise ValueError("time_index is required when no discharge has been extracted")

        interval_hours = self._time_step(self.time_index) / pd.Timedelta(hours=1)
        n_locations = len(self.placements)
        if discharge_m3s is None:
            if "discharge_m3s" not in self.sim_data:
                raise ValueError("discharge_m3s is required when no discharge has been extracted")
            discharge = np.asarray(self.sim_data["discharge_m3s"], dtype=float)
        else:
            discharge = np.asarray(discharge_m3s, dtype=float)
            self.sim_data["discharge_m3s"] = discharge
        if discharge.shape != (len(self.time_index), n_locations):
            raise ValueError("discharge_m3s must have shape (time, locations)")

        head = self._location_values(self.placements["head"], n_locations, "head")
        capacity_kw = self._location_values(self.placements["capacity"], n_locations, "capacity")
        if np.any(head <= 0) or np.any(capacity_kw <= 0):
            raise ValueError("head and capacity must be greater than zero")
        efficiency = self._validate_efficiency(efficiency)
        potential_power_kw = self._hydraulic_power_w(discharge, head, efficiency) / 1000.0
        power_kw = (
            np.minimum(potential_power_kw, capacity_kw[None, :])
            if cap_production_by_capacity
            else potential_power_kw
        )
        usable = discharge * np.divide(
            power_kw, potential_power_kw, out=np.zeros_like(power_kw), where=potential_power_kw > 0
        )
        self.sim_data.update(
            potential_power_kw=potential_power_kw,
            power_kw=power_kw,
            generation_kwh=power_kw * float(interval_hours),
            capacity_factor=power_kw / capacity_kw[None, :],
            usable_discharge_m3s=usable,
            spilled_discharge_m3s=np.maximum(discharge, 0.0) - usable,
        )
        self.workflow_parameters["hydropower_efficiency"] = efficiency
        self.workflow_parameters["capacity_capped"] = str(bool(cap_production_by_capacity))
        return self

    # Legacy interfaces retained for compatibility.
    def simulate_run_of_river(self, inflow_m3s, net_head_m, efficiency=0.9):
        inflow = np.asarray(inflow_m3s, dtype=float)
        if inflow.ndim != 2:
            raise ValueError("inflow_m3s must have shape (time, locations)")
        n_locations = inflow.shape[1]
        if n_locations != len(self.placements):
            raise ValueError("capacity_kw length must match inflow_m3s second dimension")
        head = self._location_values(net_head_m, n_locations, "net_head_m")
        efficiency = self._validate_efficiency(efficiency)
        capacity_kw = self._location_values(self.placements["capacity"], n_locations, "capacity")
        power_w = self._hydraulic_power_w(inflow, head, efficiency)
        self.sim_data.update(
            inflow_m3s=inflow,
            net_head_m=np.broadcast_to(head, inflow.shape),
            power_output_w=power_w,
            capacity_factor=np.clip(power_w / (capacity_kw[None, :] * 1000), 0, 1),
            total_system_generation=power_w / 1000,
        )
        return self

    def simulate_run_of_river_from_daily_discharge(
        self, discharge_m3_per_day, net_head_m, efficiency=0.88, cap_production_by_capacity=True
    ):
        discharge = np.asarray(np.ma.filled(discharge_m3_per_day, 0.0), dtype=float)
        if discharge.ndim != 2:
            raise ValueError("discharge_m3_per_day must have shape (locations, time)")
        n_locations = discharge.shape[0]
        if n_locations != len(self.placements):
            raise ValueError("capacity_kw length must match discharge_m3_per_day first dimension")
        head = self._location_values(net_head_m, n_locations, "net_head_m")
        efficiency = self._validate_efficiency(efficiency)
        capacity_kw = self._location_values(self.placements["capacity"], n_locations, "capacity")
        potential = (
            np.maximum(discharge, 0) * head[:, None] * self.GRAVITY_M_S2
            * self.WATER_DENSITY_KG_M3 * efficiency / 3.6e6
        )
        generation = np.minimum(potential, capacity_kw[:, None] * 24) if cap_production_by_capacity else potential
        usable = generation * 3.6e6 / (
            head[:, None] * self.GRAVITY_M_S2 * self.WATER_DENSITY_KG_M3 * efficiency
        )
        capacity_factor = generation / (capacity_kw[:, None] * 24)
        outputs = {
            "discharge_m3_per_day": discharge,
            "usable_discharge_m3_per_day": usable,
            "capacity_factor": np.clip(capacity_factor, 0, 1) if cap_production_by_capacity else capacity_factor,
            "total_system_generation_kWh_per_day": generation,
            "total_system_generation": generation,
        }
        for key, value in outputs.items():
            self.sim_data[key] = value.T
        return self
