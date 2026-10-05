"""Era5ZarrSource on real Earth Data Hub (EDH) data.

era5-edh/era5-edh.zarr is cut from the EDH ERA5 single-levels store by
scripts/make_era5_edh_test_data.py and keeps its layout: a 'valid_time' axis,
descending latitudes, float32 and only the raw variables, so the radiation RESKit
uses is derived from the raw accumulations. It covers the box and the 140 hours of
the 'era5' netCDF4 fixtures, plus the hour before them.

Its values are not those of the netCDF4 fixtures -- those were packed to 16 bit
integers, the EDH ones are bit-rounded -- so it is compared with them loosely: to
1 % of each variable's range. That is about 15 times the largest difference
measured, and still catches a wrong unit, offset or time step.
"""

from os import sep
from os.path import join

import numpy as np
import pytest
import xarray as xr

from reskit import TEST_DATA
from reskit.weather import Era5Source, Era5ZarrSource

EDH = TEST_DATA["era5-edh.zarr"]
# The 140 hours of the 'era5' netCDF4 fixtures, on RESKit's time index
ERA5_HOURS = slice("2014-12-31 23:30", "2015-01-06 18:30")
TOLERANCE = 0.01  # of the range of the variable


@pytest.fixture(scope="module")
def sources():
    """Both sources over the same 140 hours: EDH via Era5ZarrSource, 'era5' via Era5Source."""
    return (
        Era5ZarrSource(EDH, time_slice=ERA5_HOURS, verbose=False),
        Era5Source(TEST_DATA["era5"], verbose=False),
    )


def test_the_store_has_the_layout_of_the_edh_store():
    with xr.open_zarr(EDH) as ds:
        assert "valid_time" in ds.dims and "time" not in ds.variables
        assert ds["latitude"].values[0] > ds["latitude"].values[-1]
        assert set(ds.data_vars) == set(Era5Source.CDS_TO_NC_NAME.values())
        assert all(variable.dtype == np.float32 for variable in ds.data_vars.values())
        assert ds.attrs["reskit_source_url"].startswith("https://data.earthdatahub.destine.eu/")


def test_Era5ZarrSource_reads_the_edh_store(sources):
    edh, netcdf = sources

    assert edh.time_name == "valid_time"
    assert edh.time_index.equals(netcdf.time_index)
    assert (edh.lats == netcdf.lats).all()
    assert (edh.lons == netcdf.lons).all()
    assert edh.variables["derived_from"].dropna().to_dict() == {"ssrd_t_adj": "ssrd", "fdir_t_adj": "fdir"}

    edh.sload_global_horizontal_irradiance()
    edh.sload_direct_horizontal_irradiance()
    assert not np.isnan(edh.data["global_horizontal_irradiance"]).any()
    assert not np.isnan(edh.data["direct_horizontal_irradiance"]).any()


def _assert_close(sources, loader, variable):
    edh, netcdf = sources
    getattr(edh, loader)()
    getattr(netcdf, loader)()
    expected = netcdf.data[variable]
    actual = edh.data[variable]

    assert actual.shape == expected.shape
    tolerance = TOLERANCE * (expected.max() - expected.min())
    assert np.abs(actual - expected).max() <= tolerance, variable


@pytest.mark.parametrize(
    "loader, variable",
    [
        ("sload_elevated_wind_speed", "elevated_wind_speed"),
        ("sload_surface_wind_speed", "surface_wind_speed"),
        ("sload_surface_pressure", "surface_pressure"),
        ("sload_surface_air_temperature", "surface_air_temperature"),
        ("sload_surface_dew_temperature", "surface_dew_temperature"),
        ("sload_boundary_layer_height", "boundary_layer_height"),
        # derived from the raw accumulations, see Era5ZarrSource._derive_solar_variables
        ("sload_global_horizontal_irradiance", "global_horizontal_irradiance"),
        ("sload_direct_horizontal_irradiance", "direct_horizontal_irradiance"),
    ],
)
def test_edh_data_matches_the_netcdf_fixtures(sources, loader, variable):
    _assert_close(sources, loader, variable)


def test_the_store_is_registered_as_a_single_fixture():
    """Under its relative path and its bare name, while the files inside it are not."""
    assert (
        TEST_DATA["era5-edh.zarr"]
        == TEST_DATA[join("era5-edh", "era5-edh.zarr")]
        == join(TEST_DATA["era5-edh"], "era5-edh.zarr")
    )
    assert not [key for key in TEST_DATA if ".zarr" + sep in key]
