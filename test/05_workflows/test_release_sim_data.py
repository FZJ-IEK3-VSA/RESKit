from collections.abc import Callable

import geokit as gk
import numpy as np
import pandas as pd
import pytest
import xarray

from reskit import WorkflowManager, data
from reskit.csp.workflows.workflows import csp_ptr_era5_specific_dataset
from reskit.solar.workflows.workflows import openfield_pv_era5
from reskit.wind.workflows.workflows import wind_era5_PenaSanchezDunkelWinklerEtAl2025

FIXTURES = data.paths("test_suite")


def _manager(time_selection: np.ndarray | None = None) -> WorkflowManager:
    placements = pd.DataFrame({"lon": [6.1, 6.2], "lat": [50.5, 50.6]})
    manager = WorkflowManager(placements)

    time_index = pd.date_range("2020-01-01", periods=4, freq="h")
    manager.set_time_index(time_index)

    if time_selection is None:
        time_step_count = 4
    else:
        time_step_count = int(np.sum(time_selection))
    manager._time_sel_ = time_selection

    placement_count = 2
    values_a = np.arange(time_step_count * placement_count, dtype=float)
    manager.sim_data["a"] = values_a.reshape(time_step_count, placement_count)
    manager.sim_data["b"] = np.ones((time_step_count, placement_count), dtype=np.float32)
    return manager


def test_release_sim_data_keeps_needed_and_requested_variables() -> None:
    manager = _manager()
    manager.sim_data["c"] = np.zeros((4, 2))

    # without output_variables, everything is part of the output
    manager.release_sim_data([], None)
    assert list(manager.sim_data) == ["a", "b", "c"]

    manager.release_sim_data(["a"], output_variables="c")
    assert list(manager.sim_data) == ["a", "c"]


@pytest.mark.parametrize("time_selection", [None, np.array([False, True, True, False])])
@pytest.mark.parametrize("output_variables", [None, ["a"]])
def test_to_xarray_release_gives_the_same_result(
    time_selection: np.ndarray | None,
    output_variables: list[str] | None,
) -> None:
    copying_manager = _manager(time_selection)
    expected = copying_manager.to_xarray(output_variables=output_variables)

    releasing_manager = _manager(time_selection)
    result = releasing_manager.to_xarray(output_variables=output_variables, release=True)

    assert result.identical(expected)
    assert len(releasing_manager.sim_data) == 0


def _run_pv(output_variables: list[str] | None) -> xarray.Dataset:
    all_placements = gk.vector.extractFeatures(FIXTURES["turbine_placements_shp"])
    placements = all_placements.iloc[:20]
    placements["capacity"] = 2000
    return openfield_pv_era5(
        placements=placements,
        era5_path=FIXTURES["era5"],
        global_solar_atlas_ghi_path=FIXTURES["gsa_ghi"],
        global_solar_atlas_dni_path=FIXTURES["gsa_dni"],
        output_variables=output_variables,
    )


def _run_wind(output_variables: list[str] | None) -> xarray.Dataset:
    all_placements = gk.vector.extractFeatures(FIXTURES["turbine_placements_shp"])
    placements = all_placements.iloc[:20]
    placements["hub_height"] = 120
    placements["capacity"] = 3000
    placements["rotor_diam"] = 150

    inputs = data.paths("wind_era5_PenaSanchezDunkelWinklerEtAl2025", test=True)
    height_scaling_data = {50: inputs["gwa_50m"], 200: inputs["gwa_200m"]}
    return wind_era5_PenaSanchezDunkelWinklerEtAl2025(
        placements=placements,
        era5_path=inputs["era5"],
        gwa_100m_path=inputs["gwa_100m"],
        height_scaling_data=height_scaling_data,
        output_variables=output_variables,
    )


def _run_csp(output_variables: list[str] | None) -> xarray.Dataset:
    placements = pd.DataFrame({"lon": [-6.8, -6.8], "lat": [31.0, 31.4], "land_area_m2": [1e6, 5e6]})
    return csp_ptr_era5_specific_dataset(
        placements=placements,
        era5_path=FIXTURES["era5_csp"],
        global_solar_atlas_dni_path=FIXTURES["csp_gsa_dni"],
        datasetname="Dataset_SolarSalt_2030",
        JITaccelerate=False,
        return_self=False,
        output_variables=output_variables,
    )


@pytest.mark.parametrize(
    "run, outputs",
    [
        (_run_pv, ["capacity_factor", "total_system_generation"]),
        (_run_wind, ["capacity_factor"]),
        (_run_csp, ["capacity_factor_sf", "capacity_factor_plant", "lcoe_EURct_per_kWh_el"]),
    ],
    ids=["pv", "wind", "csp"],
)
def test_workflows_give_the_same_outputs_when_releasing_interim_data(
    run: Callable[[list[str] | None], xarray.Dataset],
    outputs: list[str],
) -> None:
    # the workflows drop interim variables as soon as later steps no longer need them,
    # dropping one too early would raise or change the requested outputs
    full_result = run(None)
    limited_result = run(outputs)

    limited_variables = sorted(limited_result.data_vars)
    expected_variables = sorted(outputs)
    assert limited_variables == expected_variables
    for output_name in outputs:
        limited_values = limited_result[output_name].values
        full_values = full_result[output_name].values
        np.testing.assert_array_equal(limited_values, full_values)
