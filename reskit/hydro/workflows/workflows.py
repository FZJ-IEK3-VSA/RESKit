import os
import numpy as np
import pandas as pd

from .hydro_workflow_manager import HydroWorkflowManager
from ..core import extract_selected_discharge_alluvium


def extract_discharge(
    placements,
    product,
    time_index,
    product_options=None,
    output_netcdf_path=None,
    output_variables=None,
):
    """Extract discharge for placements and return it on ``time_index``."""
    wf = HydroWorkflowManager(placements)
    wf.extract_discharge(
        product=product,
        time_index=time_index,
        product_options=product_options,
    )
    return wf.to_xarray(
        output_netcdf_path=output_netcdf_path,
        output_variables=output_variables,
    )


def run_of_river_hydropower(
    placements,
    product,
    time_index,
    efficiency=0.88,
    cap_production_by_capacity=True,
    product_options=None,
    output_netcdf_path=None,
    output_variables=None,
):
    """Extract discharge and calculate run-of-river generation."""
    wf = HydroWorkflowManager(placements)
    wf.extract_discharge(
        product=product,
        time_index=time_index,
        product_options=product_options,
    )
    wf.calculate_hydropower(
        efficiency=efficiency,
        cap_production_by_capacity=cap_production_by_capacity,
    )
    return wf.to_xarray(
        output_netcdf_path=output_netcdf_path,
        output_variables=output_variables,
    )


def release_generation(
    placements,
    discharge_m3s,
    time_index,
    efficiency=0.88,
    cap_production_by_capacity=True,
    output_netcdf_path=None,
    output_variables=None,
):
    """Calculate generation from a user-supplied turbine release series."""
    wf = HydroWorkflowManager(placements)
    wf.calculate_hydropower(
        discharge_m3s=discharge_m3s,
        time_index=time_index,
        efficiency=efficiency,
        cap_production_by_capacity=cap_production_by_capacity,
    )
    return wf.to_xarray(
        output_netcdf_path=output_netcdf_path,
        output_variables=output_variables,
    )


def run_of_river_workflow(
    placements,
    inflow_m3s,
    net_head_m,
    time_index,
    efficiency=0.9,
    output_netcdf_path=None,
    output_variables=None,
):
    """Run a run-of-river workflow and return xarray output."""
    wf = HydroWorkflowManager(placements)

    if not isinstance(inflow_m3s, np.ndarray):
        inflow_m3s = np.asarray(inflow_m3s)

    if not isinstance(time_index, pd.DatetimeIndex):
        time_index = pd.DatetimeIndex(time_index)

    wf.set_time_index(time_index)

    if inflow_m3s.shape[0] != len(wf.time_index):
        raise ValueError("inflow_m3s first dimension must match workflow time_index length")

    wf.simulate_run_of_river(
        inflow_m3s=inflow_m3s,
        net_head_m=net_head_m,
        efficiency=efficiency,
    )

    return wf.to_xarray(output_netcdf_path=output_netcdf_path, output_variables=output_variables)


def run_of_river_daily_discharge_workflow(
    placements,
    discharge_m3_per_day,
    net_head_m,
    time_index,
    efficiency=0.88,
    cap_production_by_capacity=True,
    output_netcdf_path=None,
    output_variables=None,
):
    """Run run-of-river workflow from daily discharge volumes.

    Parameters
    ----------
    discharge_m3_per_day : np.ndarray
        Shape must be (locations, time).
    """
    wf = HydroWorkflowManager(placements)

    if not isinstance(discharge_m3_per_day, np.ndarray):
        discharge_m3_per_day = np.asarray(discharge_m3_per_day)

    if not isinstance(time_index, pd.DatetimeIndex):
        time_index = pd.DatetimeIndex(time_index)
    wf.set_time_index(time_index)

    if discharge_m3_per_day.ndim != 2:
        raise ValueError("discharge_m3_per_day must have shape (locations, time)")
    if discharge_m3_per_day.shape[1] != len(wf.time_index):
        raise ValueError("discharge_m3_per_day second dimension must match workflow time_index length")

    wf.simulate_run_of_river_from_daily_discharge(
        discharge_m3_per_day=discharge_m3_per_day,
        net_head_m=net_head_m,
        efficiency=efficiency,
        cap_production_by_capacity=cap_production_by_capacity,
    )

    return wf.to_xarray(output_netcdf_path=output_netcdf_path, output_variables=output_variables)


def run_of_river_parflow_alluvium_workflow(
    placements,
    year,
    time_index,
    net_head_m,
    extraction_root_dir,
    alluvium_mask_file,
    indicator_file,
    efficiency=0.88,
    cap_production_by_capacity=True,
    fallback_mode="max_annual",
    output_selected_alluvium_candidate_path=None,
    output_netcdf_path=None,
    output_variables=None,
):
    """Run full chain: ParFlow discharge extraction -> alluvium-aware selection -> hydropower calculation."""
    required_cols = ["lat", "lon"]
    for col in required_cols:
        if col not in placements.columns:
            raise ValueError(f"placements must contain '{col}' column")

    extraction_result = extract_selected_discharge_alluvium(
        year=year,
        placements=placements,
        root_dir=extraction_root_dir,
        alluvium_mask_file=alluvium_mask_file,
        indicator_file=indicator_file,
        fallback_mode=fallback_mode,
    )
    # output selected alluvium candidate overview if requested
    if output_selected_alluvium_candidate_path is not None:
        os.makedirs(os.path.dirname(output_selected_alluvium_candidate_path), exist_ok=True)
        pd.DataFrame(extraction_result["selected_cell_overview"]).to_csv(
            output_selected_alluvium_candidate_path, index=False
        )

    ds = run_of_river_daily_discharge_workflow(
        placements=placements,
        discharge_m3_per_day=extraction_result["selected_discharge_m3_per_day"],
        net_head_m=net_head_m,
        time_index=time_index,
        efficiency=efficiency,
        cap_production_by_capacity=cap_production_by_capacity,
        output_netcdf_path=output_netcdf_path,
        output_variables=output_variables,
    )

    ds["selected_candidate_idx"] = ("location", extraction_result["selected_candidate_idx"])  # metadata only
    ds["selected_from_alluvium"] = ("location", extraction_result["selected_from_alluvium"])  # metadata only

    return ds
