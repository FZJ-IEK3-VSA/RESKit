import geokit as gk
import pandas as pd
import pytest
import xarray as xr
from numpy import allclose

import reskit as rk
from reskit.util.provenance import read_provenance

# Default variables
sourceTemperature = rk.geothermal.data.path_temperatures
sourceSustainableHeatflow = rk.geothermal.data.path_heat_flow_sustainable_W_per_m2


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
    """The results have one value per placement along 'location', like the other workflows, and no geometries."""
    assert dict(egs_output.sizes) == {"location": len(placements)}
    assert "geom" not in egs_output
    # the caller's placements stay unchanged
    assert list(placements.columns) == ["lat", "lon", "geom"]


def test_lcoe_vm(egs_output):
    """Validate LCOE_VM_EUR_per_kWh values."""
    expected = [0.83377206, 0.52975674, 0.26825743, 0.50673514]
    assert allclose(egs_output.LCOE_VM_EUR_per_kWh, expected), "LCOE_VM values do not match!"


def test_lcoe_gr(egs_output):
    """Validate LCOE_GR_EUR_per_kWh values."""
    expected = [0.32707465, 0.12043728, 0.03419219, 0.10778247]
    assert allclose(egs_output.LCOE_GR_EUR_per_kWh, expected), "LCOE_GR values do not match!"


def test_lcoe_su(egs_output):
    """Validate LCOE_SU_EUR_per_kWh values."""
    expected = [51.79376029, 28.61874858, 7.62909725, 20.23294058]
    assert allclose(egs_output.LCOE_SU_EUR_per_kWh, expected), "LCOE_SU values do not match!"


def test_provenance(egs_output):
    """The results name the workflow, its arguments and the files it read."""
    provenance = read_provenance(egs_output)
    assert provenance["workflow"] == "reskit.geothermal.workflows.workflows.egs_workflow"
    assert provenance["parameters"]["manual_values"] == {"x_ED_1": 8}
    roles = {role for record in provenance["input_files"] for role in record["roles"]}
    assert roles == {"argument:sourceTemperature", "argument:sourceSustainableHeatflow", "gringarten_curve"}


def test_output_netcdf_path(placements, egs_output, tmp_path):
    """The results are written and the path returned; an earlier file at the path is no input."""
    path = tmp_path / "egs.nc"
    path.write_text("an earlier result, which is no input")
    returned = rk.geothermal.egs_workflow(
        placements=placements, manual_values={"x_ED_1": 8}, output_netcdf_path=str(path)
    )
    assert returned == str(path)
    written = xr.load_dataset(path)
    assert allclose(written.LCOE_GR_EUR_per_kWh, egs_output.LCOE_GR_EUR_per_kWh)
    assert str(path) not in [record["path"] for record in read_provenance(written)["input_files"]]


def test_output_variables(placements):
    """Only the requested variables are returned."""
    output = rk.geothermal.egs_workflow(placements=placements, output_variables=["lon", "lat", "LCOE_GR_EUR_per_kWh"])
    assert set(output.data_vars) == {"lon", "lat", "LCOE_GR_EUR_per_kWh"}


def test_location_id(placements):
    """A column 'location_id' becomes the coordinate of 'location', as in the other workflows."""
    output = rk.geothermal.egs_workflow(placements=placements.assign(location_id=["a", "b", "c", "d"]))
    assert list(output.location.values) == ["a", "b", "c", "d"]


def test_geometry_only_placements(placements):
    """Placements with only point geometries are accepted, as in the other workflows."""
    output = rk.geothermal.egs_workflow(placements=placements[["geom"]], manual_values={"x_ED_1": 8})
    assert allclose(output.lat, placements.lat)
    assert allclose(output.LCOE_GR_EUR_per_kWh, [0.32707465, 0.12043728, 0.03419219, 0.10778247])


@pytest.mark.parametrize("suffix", [".nc4", ".csv"])
def test_deprecated_savepath(placements, tmp_path, suffix):
    """The deprecated savepath still writes NetCDF or CSV, with a warning, and returns the path."""
    path = tmp_path / f"egs{suffix}"
    with pytest.warns(DeprecationWarning, match="savepath"):
        returned = rk.geothermal.egs_workflow(placements=placements, manual_values={"x_ED_1": 8}, savepath=str(path))
    assert returned == str(path)
    if suffix == ".csv":
        assert len(pd.read_csv(path)) == len(placements)
    else:
        assert xr.load_dataset(path).sizes["location"] == len(placements)


def test_deprecated_savepath_rejects_other_formats(placements, tmp_path):
    """Shapefiles were never written successfully and are refused."""
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="savepath must end in"):
        rk.geothermal.egs_workflow(placements=placements, savepath=str(tmp_path / "egs.shp"))
