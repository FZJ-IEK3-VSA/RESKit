from reskit import TEST_DATA
from reskit.weather_source import MerraSource
import pytest
import numpy as np
import pandas as pd
import geokit as gk
import netCDF4 as nc


@pytest.fixture
def pt_MerraSource():
    return MerraSource(TEST_DATA['merra-like'], verbose=False)


def test_MerraSource___init__():
    raw = nc.Dataset(TEST_DATA['merra-like.nc4'])
    rawLats = raw["lat"][:]
    rawLons = raw["lon"][:]
    rawTimes = pd.DatetimeIndex(
        nc.num2date(raw["time"][:], raw["time"].units, only_use_cftime_datetimes=False, only_use_python_datetimes=True),
        tz="GMT")

    # Unbounded source
    ms = MerraSource(TEST_DATA['merra-like.nc4'], verbose=False)

    # ensure lats, lons and times are okay
    assert (ms.lats == rawLats).all()
    assert (ms.lons == rawLons).all()
    assert (ms.time_index == rawTimes).all()

    # Initialize a MerraSource with Aachen boundaries
    aachenExt = gk.Extent.fromVector(gk._test_data_['aachenShapefile.shp']).pad(0.5).fit(0.01)
    aachenLats = np.array([50.0, 50.5, 51.0, 51.5])
    aachenLons = np.array([5.625, 6.250])

    ms = MerraSource(TEST_DATA['merra-like.nc4'], bounds=aachenExt, index_pad=1, verbose=False)

    # ensure lats, lons and times are okay
    assert np.isclose(ms.lats, aachenLats).all()

    assert np.isclose(ms.lons, aachenLons).all()

    assert (ms.time_index == rawTimes).all()


def test_MerraSource_context_area_at_index(pt_MerraSource): return
def test_MerraSource_sload_elevated_wind_speed(pt_MerraSource): return
def test_MerraSource_sload_surface_wind_speed(pt_MerraSource): return
def test_MerraSource_sload_wind_speed_at_2m(pt_MerraSource): return
def test_MerraSource_sload_wind_speed_at_10m(pt_MerraSource): return
def test_MerraSource_sload_wind_speed_at_50m(pt_MerraSource): return
def test_MerraSource__load_wind_dir(pt_MerraSource): return
def test_MerraSource_sload_elevated_wind_direction(pt_MerraSource): return
def test_MerraSource_sload_surface_wind_direction(pt_MerraSource): return
def test_MerraSource_sload_wind_direction_at_2m(pt_MerraSource): return
def test_MerraSource_sload_wind_direction_at_10m(pt_MerraSource): return
def test_MerraSource_sload_wind_direction_at_50m(pt_MerraSource): return
def test_MerraSource_sload_surface_pressure(pt_MerraSource): return
def test_MerraSource_sload_surface_air_temperature(pt_MerraSource): return
def test_MerraSource_sload_surface_dew_temperature(pt_MerraSource): return
def test_MerraSource_sload_global_horizontal_irradiance(pt_MerraSource): return
