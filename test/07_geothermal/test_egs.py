import geokit as gk
import pandas as pd
import pytest
import xarray as xr
from numpy import allclose

import reskit as rk

# Default variables
sourceTemperature = rk.geothermal.data.path_temperatures
sourceSustainableHeatflow = rk.geothermal.data.path_heat_flow_sustainable_W_per_m2

LCOE_GRINGARTEN = [0.32707465, 0.12043728, 0.03419219, 0.10778247]


@pytest.fixture
def placements():
    """Creates a DataFrame of geothermal placements with geometries."""
    df = pd.DataFrame()
    df["lat"] = [51.00, 37.0, 64.922, 0.0]
    df["lon"] = [9.00, -114.0, -18.854, 114.0]

    geoms = [gk.geom.point(lon, lat, srs=gk.srs.loadSRS(4326)) for lon, lat in zip(df["lon"], df["lat"])]
    df["geom"] = geoms

    return df


@pytest.fixture
def egs_output(placements):
    """Runs the egs_workflow and returns the output."""
    return rk.geothermal.egs_workflow(
        placements=placements,
        sourceTemperature=sourceTemperature,
        sourceSustainableHeatflow=sourceSustainableHeatflow,
        manual_values={"x_ED_1": 8},
    )


def test_output_dimensions(placements, egs_output):
    """The results lie along 'location', without the geometries, and the caller's placements stay unchanged."""
    assert dict(egs_output.sizes) == {"location": len(placements)}
    assert "geom" not in egs_output
    assert list(placements.columns) == ["lat", "lon", "geom"]


def test_units(egs_output):
    """Every result variable has its unit in the dataset attributes."""
    assert egs_output.attrs["lcoe_gringarten"] == "EUR/kWh"
    for variable in egs_output.data_vars:
        assert variable in ("lon", "lat") or variable in egs_output.attrs, variable


def test_lcoe_volume(egs_output):
    """Validate lcoe_volume values."""
    expected = [0.83377206, 0.52975674, 0.26825743, 0.50673514]
    assert allclose(egs_output.lcoe_volume, expected), "lcoe_volume values do not match!"


def test_lcoe_gringarten(egs_output):
    """Validate lcoe_gringarten values."""
    assert allclose(egs_output.lcoe_gringarten, LCOE_GRINGARTEN), "lcoe_gringarten values do not match!"


def test_lcoe_sustainable(egs_output):
    """Validate lcoe_sustainable values."""
    expected = [51.79376029, 28.61874858, 7.62909725, 20.23294058]
    assert allclose(egs_output.lcoe_sustainable, expected), "lcoe_sustainable values do not match!"


def test_provenance(egs_output):
    """The results name the workflow and list the files it read, including the Gringarten curves."""
    provenance = rk.util.provenance.read_provenance(egs_output)
    assert provenance["workflow"] == "reskit.geothermal.workflows.workflows.egs_workflow"
    roles = {role for record in provenance["input_files"] for role in record["roles"]}
    assert roles == {"argument:sourceTemperature", "argument:sourceSustainableHeatflow", "gringarten_curve"}


def test_output_netcdf_path(placements, tmp_path):
    """The requested variables are written to NetCDF and the path is returned."""
    path = str(tmp_path / "egs.nc")
    returned = rk.geothermal.egs_workflow(
        placements=placements, output_netcdf_path=path, output_variables=["lon", "lat", "lcoe_gringarten"]
    )
    assert returned == path
    assert set(xr.load_dataset(path).data_vars) == {"lon", "lat", "lcoe_gringarten"}


def test_geometry_only_placements(placements):
    """Placements with only point geometries give the same results; they used to fail."""
    output = rk.geothermal.egs_workflow(placements=placements[["geom"]], manual_values={"x_ED_1": 8})
    assert allclose(output.lcoe_gringarten, LCOE_GRINGARTEN)


@pytest.mark.parametrize("suffix", [".nc4", ".csv"])
def test_deprecated_savepath(placements, tmp_path, suffix):
    """The deprecated savepath still writes NetCDF or CSV, with a warning, and returns the path."""
    path = str(tmp_path / f"egs{suffix}")
    with pytest.warns(DeprecationWarning, match="savepath"):
        returned = rk.geothermal.egs_workflow(placements=placements, savepath=path)
    assert returned == path
    if suffix == ".csv":
        assert len(pd.read_csv(path)) == len(placements)
    else:
        assert xr.load_dataset(path).sizes["location"] == len(placements)
