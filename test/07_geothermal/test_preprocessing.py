"""Check the scientific transformations with independent synthetic inputs."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from osgeo import gdal

from reskit.geothermal import create_geothermal_resource


@pytest.fixture
def sources(tmp_path):
    """Create geographically varying fields in the original delivery formats."""
    lat = np.arange(-89.5, 90, 1)
    lon = np.arange(-179.5, 180, 1)
    index = pd.MultiIndex.from_product([lon, lat], names=[("lon", "degree"), ("lat", "degree")])
    table = pd.DataFrame(
        {
            ("Similarity method: mean HF", "mW/m2"): 70 + index.get_level_values(1) * 0.1,
            ("Heat prod (provinces)", "uW/m3"): np.full(len(index), 2.0),
        },
        index=index,
    )
    table.loc[(-178.5, -88.5), ("Heat prod (provinces)", "uW/m3")] = np.nan
    table_path = tmp_path / "Supplementary material.txt"
    # Deliberately reverse table order; geography must come from coordinates.
    table.iloc[::-1].reset_index().to_csv(table_path, index=False)

    classes = np.ones((360, 720), dtype=np.int16)
    # Southwest cell: unconsolidated, siliciclastic, water, raster nodata.
    classes[-2:, :2] = [[1, 3], [11, -9999]]
    raster_path = tmp_path / "lithology.tif"
    raster = gdal.GetDriverByName("GTiff").Create(str(raster_path), 720, 360, 1, gdal.GDT_Int16)
    raster.SetGeoTransform((-180, 0.5, 0, 90, 0, -0.5))
    raster.GetRasterBand(1).SetNoDataValue(-9999)
    raster.GetRasterBand(1).WriteArray(classes)
    raster = None

    coefficients = [1.9, 1.8, 3.4, 2.9, 3.3, 3.7, 1.4, 3.5, 2.5, 1.8, 2.5, 1.4, 2.1, 5.4, 2.5, 2.2]
    workbook_path = tmp_path / "conductivity.xlsx"
    pd.DataFrame(
        {"Thermal Conductivity [W/mK] [2]": coefficients}, index=pd.Index(range(1, 17), name="OBJECTID")
    ).to_excel(workbook_path)

    half_lat = np.arange(-89.75, 90, 0.5)
    half_lon = np.arange(-179.75, 180, 0.5)
    temperature = 20 + half_lat[:, None] + half_lon[None, :]
    temperature[:, -1] = np.nan
    nasa_path = tmp_path / "power.nc"
    xr.Dataset(
        {"TS": (("time", "lat", "lon"), temperature[None], {"units": "C"})},
        coords={"time": [13], "lat": half_lat, "lon": half_lon},
    ).to_netcdf(nasa_path)
    return {
        "goutorbe_table": table_path,
        "lithology_raster": raster_path,
        "surface_temperature": nasa_path,
        "conductivity_table": workbook_path,
    }


def test_reconstructs_physical_fields_and_writes_provenance(sources, tmp_path):
    """Check units, averaging, orientation, overrides, masks and output layout."""
    output = tmp_path / "output"
    result = create_geothermal_resource(**sources, output_dir=output)
    assert result.temperature.dims == ("lat", "lon", "depth")
    np.testing.assert_array_equal(result.depth, np.arange(1000, 10001, 1000))

    southwest = result.sel(lat=-89.5, lon=-179.5)
    k = (1.9 + 3.4 + 2.5 + 2.5) / 4
    assert southwest.surface_temperature == pytest.approx(-249.0)
    # q=61.05 mW/m2, A=2 uW/m3: the 1 km terms are 61.05/k and 1/k.
    assert southwest.temperature.sel(depth=1000) == pytest.approx(-249 + 60.05 / k)
    assert southwest.heat_flow_sustainable_W_per_m2 == pytest.approx(0.04305)
    assert southwest.temperature.sel(depth=10000) == pytest.approx(-249 + (610.5 - 100) / k)

    northeast = result.sel(lat=89.5, lon=179.5)
    # Filling the last column from 179.25 shifts its block's mean by -0.25 C.
    assert northeast.surface_temperature == pytest.approx(288.75)
    assert northeast.temperature.sel(depth=1000) == pytest.approx(288.75 + 77.95 / 1.9)
    assert northeast.heat_flow_sustainable_W_per_m2 == pytest.approx(0.06095)
    missing = result.sel(lat=-88.5, lon=-178.5)
    assert missing.isnull().to_array().all()

    provenance = json.loads(result.attrs["source_files"])
    assert provenance["goutorbe_table"]["sha256"] == hashlib.sha256(sources["goutorbe_table"].read_bytes()).hexdigest()
    assert "10.1016/j.renene.2025.123199" in result.attrs["references"]
    assert "not specified in the paper" in result.attrs["sustainable_heat_flow_method"]
    for name, variables in [
        ("Temperatures.nc4", ["temperature", "surface_temperature"]),
        ("heat_flow_sustainable_W_per_m2.nc4", ["heat_flow_sustainable_W_per_m2"]),
    ]:
        with xr.open_dataset(output / name) as reopened:
            xr.testing.assert_identical(reopened, result[variables])
    assert result.temperature.attrs["units"] == "degree_Celsius"
    assert result.heat_flow_sustainable_W_per_m2.attrs["units"] == "W m-2"


@pytest.mark.parametrize(
    "invalid", ["duplicate_cell", "heat_flow_units", "raster_alignment", "temperature_units", "conductivity"]
)
def test_rejects_inputs_that_would_silently_change_the_calculation(sources, invalid):
    """Fail on misaligned grids, ambiguous units and invalid physical values."""
    if invalid in {"duplicate_cell", "heat_flow_units"}:
        path = sources["goutorbe_table"]
        lines = path.read_text().splitlines()
        if invalid == "duplicate_cell":
            lines[-1] = lines[-2]
            message = "every global 1-degree cell exactly once"
        else:
            lines[1] = lines[1].replace("mW/m2", "W/m2")
            message = "incorrect units"
        path.write_text("\n".join(lines) + "\n")
    elif invalid == "raster_alignment":
        raster = gdal.Open(str(sources["lithology_raster"]), gdal.GA_Update)
        raster.SetGeoTransform((-179.75, 0.5, 0, 90, 0, -0.5))
        raster = None
        message = "aligned global 0.5-degree raster"
    elif invalid == "temperature_units":
        with xr.open_dataset(sources["surface_temperature"]) as source:
            modified = source.load()
        modified.TS.attrs["units"] = "K"
        modified.to_netcdf(sources["surface_temperature"])
        message = "Celsius"
    else:
        path = sources["conductivity_table"]
        mapping = pd.read_excel(path, index_col=0)
        mapping.iloc[0, 0] = 0
        mapping.to_excel(path)
        message = "finite and positive"
    with pytest.raises(ValueError, match=message):
        create_geothermal_resource(**sources)


def test_checks_both_output_paths_before_writing(sources, tmp_path):
    """An existing second output must not lead to replacing the first one."""
    existing = tmp_path / "heat_flow_sustainable_W_per_m2.nc4"
    existing.write_bytes(b"previous release")
    with pytest.raises(FileExistsError):
        create_geothermal_resource(**sources, output_dir=tmp_path)
    assert existing.read_bytes() == b"previous release"
    assert not (tmp_path / "Temperatures.nc4").exists()
