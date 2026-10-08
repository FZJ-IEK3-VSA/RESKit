import geokit as gk
import numpy as np
import pandas as pd
import pytest

from reskit import WorkflowManager, data
from reskit.csp.workflows.workflows import csp_ptr_era5_specific_dataset
from reskit.solar.workflows.workflows import openfield_pv_era5
from reskit.wind.workflows.workflows import wind_era5_PenaSanchezDunkelWinklerEtAl2025

FIXTURES = data.paths("test_suite")


def _manager(time_sel=None) -> WorkflowManager:
    man = WorkflowManager(pd.DataFrame({"lon": [6.1, 6.2], "lat": [50.5, 50.6]}))
    man.set_time_index(pd.date_range("2020-01-01", periods=4, freq="h"))
    rows = 4 if time_sel is None else int(np.sum(time_sel))
    man._time_sel_ = time_sel
    man.sim_data["a"] = np.arange(rows * 2, dtype=float).reshape(rows, 2)
    man.sim_data["b"] = np.ones((rows, 2), dtype=np.float32)
    return man


def test_release_sim_data_keeps_needed_and_requested_variables():
    man = _manager()
    man.sim_data["c"] = np.zeros((4, 2))

    # without output_variables, everything is part of the output
    man.release_sim_data([], None)
    assert list(man.sim_data) == ["a", "b", "c"]

    man.release_sim_data(["a"], output_variables="c")
    assert list(man.sim_data) == ["a", "c"]


@pytest.mark.parametrize("time_sel", [None, np.array([False, True, True, False])])
@pytest.mark.parametrize("output_variables", [None, ["a"]])
def test_to_xarray_release_gives_the_same_result(time_sel, output_variables):
    expected = _manager(time_sel).to_xarray(output_variables=output_variables)

    man = _manager(time_sel)
    result = man.to_xarray(output_variables=output_variables, release=True)

    assert result.identical(expected)
    assert len(man.sim_data) == 0


def _run_pv(output_variables):
    placements = gk.vector.extractFeatures(FIXTURES["turbine_placements_shp"]).iloc[:20]
    placements["capacity"] = 2000
    return openfield_pv_era5(
        placements=placements,
        era5_path=FIXTURES["era5"],
        global_solar_atlas_ghi_path=FIXTURES["gsa_ghi"],
        global_solar_atlas_dni_path=FIXTURES["gsa_dni"],
        output_variables=output_variables,
    )


def _run_wind(output_variables):
    placements = gk.vector.extractFeatures(FIXTURES["turbine_placements_shp"]).iloc[:20]
    placements["hub_height"] = 120
    placements["capacity"] = 3000
    placements["rotor_diam"] = 150
    inputs = data.paths("wind_era5_PenaSanchezDunkelWinklerEtAl2025", test=True)
    return wind_era5_PenaSanchezDunkelWinklerEtAl2025(
        placements=placements,
        era5_path=inputs["era5"],
        gwa_100m_path=inputs["gwa_100m"],
        height_scaling_data={50: inputs["gwa_50m"], 200: inputs["gwa_200m"]},
        output_variables=output_variables,
    )


def _run_csp(output_variables):
    return csp_ptr_era5_specific_dataset(
        placements=pd.DataFrame({"lon": [-6.8, -6.8], "lat": [31.0, 31.4], "land_area_m2": [1e6, 5e6]}),
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
def test_workflows_give_the_same_outputs_when_releasing_interim_data(run, outputs):
    # the workflows drop interim variables as soon as later steps no longer need them,
    # dropping one too early would raise or change the requested outputs
    full = run(None)
    limited = run(outputs)

    assert sorted(limited.data_vars) == sorted(outputs)
    for name in outputs:
        np.testing.assert_array_equal(limited[name].values, full[name].values)
