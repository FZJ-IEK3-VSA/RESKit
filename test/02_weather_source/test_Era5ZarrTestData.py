"""Era5ZarrSource on real ERA5 data from an online Zarr store.

era5-zarr/era5.zarr is cut from an online ERA5 Zarr store by
scripts/make_era5_zarr_test_data.py and keeps its layout: a 'valid_time' axis,
descending latitudes, float32 and only the raw variables, so the radiation RESKit
uses is derived from the raw accumulations. It covers the box and the 140 hours of
the 'era5' netCDF4 fixtures, plus the hour before them.

Its values are not those of the netCDF4 fixtures -- those were packed to 16 bit
integers, the Zarr ones are bit-rounded -- so each element may differ from them by
what both encodings round away, see test/era5_encoding.py. A wrong unit or time
step exceeds that, which the negative controls below make sure of.
"""

from os import sep
from os.path import join

import numpy as np
import pytest
import xarray as xr
from era5_encoding import ZARR_MANTISSA_BITS, assert_matches_encoding

from reskit import TEST_DATA
from reskit.weather import Era5Source, Era5ZarrSource

STORE = TEST_DATA["era5.zarr"]
# The 140 hours of the 'era5' netCDF4 fixtures, on RESKit's time index
ERA5_HOURS = slice("2014-12-31 23:30", "2015-01-06 18:30")
LOADERS = {
    "elevated_wind_speed": "sload_elevated_wind_speed",
    "surface_wind_speed": "sload_surface_wind_speed",
    "surface_pressure": "sload_surface_pressure",
    "surface_air_temperature": "sload_surface_air_temperature",
    "surface_dew_temperature": "sload_surface_dew_temperature",
    "boundary_layer_height": "sload_boundary_layer_height",
    # derived from the raw accumulations, see Era5ZarrSource._derive_solar_variables
    "global_horizontal_irradiance": "sload_global_horizontal_irradiance",
    "direct_horizontal_irradiance": "sload_direct_horizontal_irradiance",
}


@pytest.fixture(scope="module")
def sources():
    """Both sources over the same 140 hours: the store via Era5ZarrSource, 'era5' via Era5Source."""
    return (
        Era5ZarrSource(STORE, time_slice=ERA5_HOURS, verbose=False),
        Era5Source(TEST_DATA["era5"], verbose=False),
    )


def test_the_store_has_the_layout_of_its_online_source():
    with xr.open_zarr(STORE) as ds:
        assert "valid_time" in ds.dims and "time" not in ds.variables
        assert ds["latitude"].values[0] > ds["latitude"].values[-1]
        assert set(ds.data_vars) == set(Era5Source.CDS_TO_NC_NAME.values())
        assert all(variable.dtype == np.float32 for variable in ds.data_vars.values())
        assert ds.attrs["reskit_source_url"].startswith("https://")


def test_the_store_keeps_the_mantissa_bits_its_tolerance_assumes():
    dropped_bits = (1 << (23 - ZARR_MANTISSA_BITS)) - 1
    with xr.open_zarr(STORE) as ds:
        for name, variable in ds.data_vars.items():
            assert not (variable.values.view(np.uint32) & dropped_bits).any(), name


def test_Era5ZarrSource_reads_the_store(sources):
    zarr, netcdf = sources

    assert zarr.time_name == "valid_time"
    assert zarr.time_index.equals(netcdf.time_index)
    assert (zarr.lats == netcdf.lats).all()
    assert (zarr.lons == netcdf.lons).all()
    assert zarr.variables["derived_from"].dropna().to_dict() == {"ssrd_t_adj": "ssrd", "fdir_t_adj": "fdir"}

    zarr.sload_global_horizontal_irradiance()
    zarr.sload_direct_horizontal_irradiance()
    assert not np.isnan(zarr.data["global_horizontal_irradiance"]).any()
    assert not np.isnan(zarr.data["direct_horizontal_irradiance"]).any()


def _load(sources, variable):
    zarr, netcdf = sources
    for source in sources:
        getattr(source, LOADERS[variable])()
    return zarr.data[variable], netcdf.data[variable]


@pytest.mark.parametrize("variable", LOADERS)
def test_zarr_data_matches_the_netcdf_fixtures(sources, variable):
    assert_matches_encoding(*_load(sources, variable), variable)


@pytest.mark.parametrize("variable", LOADERS)
def test_the_tolerance_catches_a_time_step_off_by_one_hour(sources, variable):
    actual, expected = _load(sources, variable)
    with pytest.raises(AssertionError):
        assert_matches_encoding(actual[1:], expected[:-1], variable)


@pytest.mark.parametrize(
    "variable, wrong_unit",
    [
        ("surface_air_temperature", lambda celsius: celsius + 273.15),
        ("surface_dew_temperature", lambda celsius: celsius + 273.15),
        ("surface_pressure", lambda pascal: pascal / 100),
        ("global_horizontal_irradiance", lambda w_per_m2: w_per_m2 * 3600),
    ],
)
def test_the_tolerance_catches_a_wrong_unit(sources, variable, wrong_unit):
    actual, expected = _load(sources, variable)
    with pytest.raises(AssertionError):
        assert_matches_encoding(wrong_unit(actual), expected, variable)


def test_the_store_is_registered_as_a_single_fixture():
    """Under its relative path and its bare name, while the files inside it are not."""
    assert (
        TEST_DATA["era5.zarr"] == TEST_DATA[join("era5-zarr", "era5.zarr")] == join(TEST_DATA["era5-zarr"], "era5.zarr")
    )
    assert not [key for key in TEST_DATA if ".zarr" + sep in key]
