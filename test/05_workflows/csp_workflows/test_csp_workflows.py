import numpy as np
import pandas as pd
import pytest
import xarray as xr

import reskit as rk
from reskit import data

FIXTURES = data.paths("test_suite")


@pytest.fixture
def pt_pv_placements() -> pd.DataFrame:
    placements = pd.DataFrame()
    # noor 2 ptr plant, morocco
    # Longitude  # [ -6.8, -6.8, -6.8]
    placements["lon"] = [-6.8, -6.8, -6.8]
    # Latitude  # [ 31.0, 31.4, 31.0,]
    placements["lat"] = [
        31.0,
        31.4,
        31.0,
    ]
    placements["land_area_m2"] = [1e6, 5e6, 6e6]
    repeats = int(3 / 3)
    placements = placements.loc[placements.index.repeat(repeats)].reset_index(drop=True)
    return placements


# Make a placements dataframe

# %%


# with output_variables, the workflow drops interim variables as soon as later steps no
# longer need them; dropping one too early would raise or change the requested outputs.
@pytest.mark.parametrize(
    "output_variables",
    [
        None,
        [
            "solar_multiple",
            "storage_hours",
            "daily_net_electricity_output",
            "daily_backup_heating_electricity",
            "lcoe",
        ],
    ],
    ids=["all_variables", "release_interim_variables"],
)
def test_CSP_PTR_ERA5(pt_pv_placements: pd.DataFrame, output_variables: list[str] | None) -> None:
    out = rk.csp.csp_ptr_era5(
        placements=pt_pv_placements,
        era5_path=FIXTURES["era5_csp"],
        global_solar_atlas_dni_path=FIXTURES["csp_gsa_dni"],
        global_solar_atlas_tamb_path=FIXTURES["csp_gsa_temp"],
        verbose=True,
        cost_year=2030,
        JITaccelerate=False,
        debug_vars=True,
        onlynightuse=True,
        output_variables=output_variables,
    )

    print("Simulation done")

    # datasets
    a = np.array(["Dataset_SolarSalt_2030", "Dataset_Therminol_2030", "Dataset_SolarSalt_2030"])
    assert (out["dataset"].values == a).all()

    if output_variables is None:
        # the workflow gives both areas in the output
        assert "land_area_m2" in out.variables
        assert "aperture_area_m2" in out.variables

        assert np.allclose(
            out["LRA_factor_direct_normal_irradiance"].values,
            [0.92240995, 0.85188336, 0.92240995],
        )

        # direct_horizontal_irradiance:
        assert out["direct_horizontal_irradiance"].values.shape == (8760, 3)
        assert np.isclose(out["direct_horizontal_irradiance"].values.mean(), 189.95024229234605)
        assert np.isclose(out["direct_horizontal_irradiance"].values.std(), 268.22838885782073)
        assert np.isclose(out["direct_horizontal_irradiance"].values.min(), 0.0)
        assert np.isclose(out["direct_horizontal_irradiance"].values.max(), 966.579790643025)

        # direct_horizontal_irradiance:
        assert out["direct_normal_irradiance"].values.shape == (8760, 3)
        assert np.isclose(out["direct_normal_irradiance"].values.mean(), 278.73085651103776)
        assert np.isclose(out["direct_normal_irradiance"].values.std(), 332.3526428406074)
        assert np.isclose(out["direct_normal_irradiance"].values.min(), 0.0)
        assert np.isclose(out["direct_normal_irradiance"].values.max(), 982.451222209881)

        # heat_to_htf
        assert out["heat_to_htf"].values.shape == (8760, 3)
        assert np.isclose(out["heat_to_htf"].values.mean(), 209780548.79719985)
        assert np.isclose(out["heat_to_htf"].values.std(), 324059983.1510693)
        assert np.isclose(out["heat_to_htf"].values.min(), 0.0)
        assert np.isclose(out["heat_to_htf"].values.max(), 1245394974.4195464)

        # heat_to_plant
        assert out["heat_to_plant"].values.shape == (8760, 3)
        assert np.isclose(out["heat_to_plant"].values.mean(), 159338864.09594935)
        assert np.isclose(out["heat_to_plant"].values.std(), 276057742.9582942)
        assert np.isclose(out["heat_to_plant"].values.min(), 0.0)
        assert np.isclose(out["heat_to_plant"].values.max(), 1102811547.598732)

        # solar_field_heating
        assert out["solar_field_heating"].values.shape == (8760, 3)
        assert np.isclose(out["solar_field_heating"].values.mean(), 8567187.817746798)
        assert np.isclose(out["solar_field_heating"].values.std(), 20261718.074392248)
        assert np.isclose(out["solar_field_heating"].values.min(), 0.0)
        assert np.isclose(out["solar_field_heating"].values.max(), 70676641.74470554)

    # solar_multiple
    a = np.array([2.0, 2.0, 2.0])
    assert np.isclose(out["solar_multiple"].values, a).all()
    # storage_hours
    a = np.array([12, 12, 12])
    assert np.isclose(out["storage_hours"].values, a).all()

    # daily_net_electricity_output
    assert out["daily_net_electricity_output"].values.shape == (365, 3)
    assert np.isclose(out["daily_net_electricity_output"].values.mean(), 1082442335.686746)
    assert np.isclose(out["daily_net_electricity_output"].values.std(), 797437011.9432147)
    assert np.isclose(out["daily_net_electricity_output"].values.min(), 0.0)
    assert np.isclose(out["daily_net_electricity_output"].values.max(), 2560268088.640056)

    # daily_backup_heating_electricity
    assert out["daily_backup_heating_electricity"].values.shape == (365, 3)
    assert np.isclose(out["daily_backup_heating_electricity"].values.mean(), 6517471.755673256)
    assert np.isclose(out["daily_backup_heating_electricity"].values.std(), 62620895.67712642)
    assert np.isclose(out["daily_backup_heating_electricity"].values.min(), 0.0)
    assert np.isclose(out["daily_backup_heating_electricity"].values.max(), 1054026635.4118232)

    # lcoe, in EUR/kWh
    a = np.array([0.14875259498468504, 0.17115158016447037, 0.14875259498468509])
    assert np.isclose(out["lcoe"].values, a).all()


def _run(placements, **kwargs):
    """Runs csp_ptr_era5 on the test data."""
    return rk.csp.csp_ptr_era5(
        placements=placements,
        era5_path=FIXTURES["era5_csp"],
        global_solar_atlas_dni_path=FIXTURES["csp_gsa_dni"],
        global_solar_atlas_tamb_path=FIXTURES["csp_gsa_temp"],
        cost_year=2030,
        **kwargs,
    )


def test_single_dataset_returns_the_results(pt_pv_placements: pd.DataFrame) -> None:
    """With one dataset, the results are returned, not the workflow manager, and carry their units."""
    out = _run(pt_pv_placements, HTF_sel=["SolarSalt"])
    assert out["lcoe"].attrs["units"] == "EUR/kWh_el"
    assert "annuity" not in out


def test_several_datasets_write_netcdf(pt_pv_placements: pd.DataFrame, tmp_path) -> None:
    """The merged results of several datasets are written once; location_id stays the coordinate."""
    placements = pt_pv_placements.assign(location_id=["a", "b", "c"])
    path = str(tmp_path / "csp.nc")
    assert _run(placements, output_netcdf_path=path, output_variables=["lcoe"]) == path
    written = xr.load_dataset(path)
    assert list(written["location"].values) == ["a", "b", "c"]
    assert list(written["dataset"].values) == [
        "Dataset_SolarSalt_2030",
        "Dataset_Therminol_2030",
        "Dataset_SolarSalt_2030",
    ]
    # the caller's placements stay unchanged
    assert list(placements.columns) == ["lon", "lat", "land_area_m2", "location_id"]


def test_return_self_is_deprecated(pt_pv_placements: pd.DataFrame) -> None:
    with pytest.warns(DeprecationWarning, match="return_self"):
        man = _run(pt_pv_placements, HTF_sel=["SolarSalt"], return_self=True)
    assert isinstance(man, rk.csp.workflows.csp_workflow_manager.PTRWorkflowManager)


if __name__ == "__main__":
    placements = pt_pv_placements()
    test_CSP_PTR_ERA5(placements)

# %%
