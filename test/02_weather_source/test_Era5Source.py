from os.path import join

import geokit as gk
import netCDF4 as nc
import numpy as np
import pandas as pd
import pytest

from reskit import TEST_DATA
from reskit.weather import Era5Source


@pytest.fixture
def pt_Era5Source():
    return Era5Source(TEST_DATA["era5-like"], verbose=False)


@pytest.fixture
def pt_BoundedEra5Source():
    aachenExt = gk.Extent.fromVector(gk._test_data_["aachenShapefile.shp"])
    return Era5Source(TEST_DATA["era5-like"], bounds=aachenExt, index_pad=1, verbose=False)


def test_Era5Source___init__():
    raw = nc.Dataset(join(TEST_DATA["era5-like"], "surface_pressure.nc"), mode="r")
    rawLats = raw["latitude"][::-1]
    rawLons = raw["longitude"][:]
    rawTimes = pd.DatetimeIndex(
        nc.num2date(
            raw["time"][:],
            raw["time"].units,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        )
    ) - pd.Timedelta(minutes=30)

    # Unbounded source
    ms = Era5Source(TEST_DATA["era5-like"], verbose=False)

    # ensure lats, lons and times are okay
    assert (ms.lats == rawLats).all()
    assert (ms.lons == rawLons).all()
    assert (ms.time_index == rawTimes).all()

    # Initialize a Era5Source with Aachen boundaries
    aachenExt = gk.Extent.fromVector(gk._test_data_["aachenShapefile.shp"]).pad(0.5).fit(0.01)

    ms = Era5Source(TEST_DATA["era5-like"], bounds=aachenExt, index_pad=1, verbose=False)

    # ensure lats, lons and times are okay
    assert np.isclose(ms.lats[0], 49.5)
    assert np.isclose(ms.lats[-1], 51.75)
    assert ms.lats.size == 10

    assert np.isclose(ms.lons[0], 5.0)
    assert np.isclose(ms.lons[-1], 7.25)
    assert ms.lons.size == 10

    assert (ms.time_index == rawTimes).all()


def test_Era5Source_loc_to_index(pt_Era5Source, pt_BoundedEra5Source):
    idx = pt_Era5Source.loc_to_index((6.03, 50.81))
    assert idx.yi == 7
    assert idx.xi == 4

    idx = pt_Era5Source.loc_to_index(
        [
            (6.03, 50.81),
            (6.44, 50.47),
        ]
    )
    assert idx[0].yi == 7
    assert idx[1].yi == 6
    assert idx[0].xi == 4
    assert idx[1].xi == 6

    idx = pt_Era5Source.loc_to_index(
        [
            (6.03, 50.81),
            (6.44, 50.47),
        ],
        as_int=False,
    )
    assert np.isclose(idx[0].yi, 7.240000000000009)
    assert np.isclose(idx[1].yi, 5.8799999999999955)
    assert np.isclose(idx[0].xi, 4.120000000000001)
    assert np.isclose(idx[1].xi, 5.760000000000002)

    idx = pt_BoundedEra5Source.loc_to_index((6.03, 50.81))
    assert idx.yi == 3
    assert idx.xi == 2

    idx = pt_BoundedEra5Source.loc_to_index(
        [
            (6.03, 50.81),
            (6.44, 50.47),
        ]
    )
    assert idx[0].yi == 3
    assert idx[1].yi == 2
    assert idx[0].xi == 2
    assert idx[1].xi == 4

    idx = pt_BoundedEra5Source.loc_to_index(
        [
            (6.03, 50.81),
            (6.44, 50.47),
        ],
        as_int=False,
    )
    assert np.isclose(idx[0].yi, 3.240000000000009)
    assert np.isclose(idx[1].yi, 1.8799999999999955)
    assert np.isclose(idx[0].xi, 2.120000000000001)
    assert np.isclose(idx[1].xi, 3.7600000000000016)


def test_Era5Source__sload_wind_speed(pt_Era5Source, pt_BoundedEra5Source):
    """
    This test is to check the internal function _sload_wind_speed, which is used by the standard loader functions for wind speed variables.
    Since the following 4 functions are testing for pre-calculated wind speed, here this function tests the wind speed calculated from u and v components.
    The numbers from the following 4 functions are also listed out here in a comment for comparison. The difference is at 4th digit after the decimal point, which is acceptable given the precision of the data and the calculations.
    """
    for var in ["elevated_wind_speed", "wind_speed_at_100m"]:
        pt_Era5Source._sload_wind_speed(height=100, target_name=var, force_load_uv=True)
        assert var in pt_Era5Source.data

        a, b, c = (140, 13, 11), 6.650547848078094, 11.299326952960216  # 6.650457494103541, 11.29947813348796
        assert pt_Era5Source.data[var].shape == a
        assert np.isclose(pt_Era5Source.data[var].mean(), b)
        assert np.isclose(pt_Era5Source.data[var][33, 1, 2], c)

        pt_BoundedEra5Source._sload_wind_speed(height=100, target_name=var, force_load_uv=True)
        assert var in pt_BoundedEra5Source.data

        a, b, c = (140, 6, 6), 7.102551347044989, 12.475086200228056  # 7.102461142186705, 12.475203711050753
        assert pt_BoundedEra5Source.data[var].shape == a
        assert np.isclose(pt_BoundedEra5Source.data[var].mean(), b)
        assert np.isclose(pt_BoundedEra5Source.data[var][33, 1, 2], c)

    for var in ["surface_wind_speed", "wind_speed_at_10m"]:
        pt_Era5Source._sload_wind_speed(height=10, target_name=var, force_load_uv=True)
        assert var in pt_Era5Source.data

        a, b, c = (140, 13, 11), 3.695453610733474, 6.652907948049286  # 3.69537552660054, 6.653035065767388
        assert pt_Era5Source.data[var].shape == a
        assert np.isclose(pt_Era5Source.data[var].mean(), b)
        assert np.isclose(pt_Era5Source.data[var][33, 1, 2], c)

        pt_BoundedEra5Source._sload_wind_speed(height=10, target_name=var, force_load_uv=True)
        assert var in pt_BoundedEra5Source.data

        a, b, c = (140, 6, 6), 3.899670648780753, 7.606956995666662  # 3.8995903495628834, 7.6075014496292
        assert pt_BoundedEra5Source.data[var].shape == a
        assert np.isclose(pt_BoundedEra5Source.data[var].mean(), b)
        assert np.isclose(pt_BoundedEra5Source.data[var][33, 1, 2], c)


def test_Era5Source_sload_elevated_wind_speed(pt_Era5Source, pt_BoundedEra5Source):
    var = "elevated_wind_speed"
    pt_Era5Source.sload_elevated_wind_speed()
    assert var in pt_Era5Source.data

    a, b, c = (140, 13, 11), 6.650457494103541, 11.29947813348796
    assert pt_Era5Source.data[var].shape == a
    assert np.isclose(pt_Era5Source.data[var].mean(), b)
    assert np.isclose(pt_Era5Source.data[var][33, 1, 2], c)

    pt_BoundedEra5Source.sload_elevated_wind_speed()
    assert var in pt_BoundedEra5Source.data

    a, b, c = (140, 6, 6), 7.102461142186705, 12.475203711050753
    assert pt_BoundedEra5Source.data[var].shape == a
    assert np.isclose(pt_BoundedEra5Source.data[var].mean(), b)
    assert np.isclose(pt_BoundedEra5Source.data[var][33, 1, 2], c)


def test_Era5Source_sload_surface_wind_speed(pt_Era5Source, pt_BoundedEra5Source):
    var = "surface_wind_speed"
    pt_Era5Source.sload_surface_wind_speed()
    assert var in pt_Era5Source.data

    a, b, c = (140, 13, 11), 3.69537552660054, 6.653035065767388
    assert pt_Era5Source.data[var].shape == a
    assert np.isclose(pt_Era5Source.data[var].mean(), b)
    assert np.isclose(pt_Era5Source.data[var][33, 1, 2], c)

    pt_BoundedEra5Source.sload_surface_wind_speed()
    assert var in pt_BoundedEra5Source.data

    a, b, c = (140, 6, 6), 3.8995903495628834, 7.6075014496292
    assert pt_BoundedEra5Source.data[var].shape == a
    assert np.isclose(pt_BoundedEra5Source.data[var].mean(), b)
    assert np.isclose(pt_BoundedEra5Source.data[var][33, 1, 2], c)


def test_Era5Source_sload_wind_speed_at_100m(pt_Era5Source, pt_BoundedEra5Source):
    var = "wind_speed_at_100m"
    pt_Era5Source.sload_wind_speed_at_100m()
    assert var in pt_Era5Source.data

    a, b, c = (140, 13, 11), 6.650457494103541, 11.29947813348796
    assert pt_Era5Source.data[var].shape == a
    assert np.isclose(pt_Era5Source.data[var].mean(), b)
    assert np.isclose(pt_Era5Source.data[var][33, 1, 2], c)

    pt_BoundedEra5Source.sload_wind_speed_at_100m()
    assert var in pt_BoundedEra5Source.data

    a, b, c = (140, 6, 6), 7.102461142186705, 12.475203711050753
    assert pt_BoundedEra5Source.data[var].shape == a
    assert np.isclose(pt_BoundedEra5Source.data[var].mean(), b)
    assert np.isclose(pt_BoundedEra5Source.data[var][33, 1, 2], c)


def test_Era5Source_sload_wind_speed_at_10m(pt_Era5Source, pt_BoundedEra5Source):
    var = "wind_speed_at_10m"
    pt_Era5Source.sload_wind_speed_at_10m()
    assert var in pt_Era5Source.data

    a, b, c = (140, 13, 11), 3.69537552660054, 6.653035065767388
    assert pt_Era5Source.data[var].shape == a
    assert np.isclose(pt_Era5Source.data[var].mean(), b)
    assert np.isclose(pt_Era5Source.data[var][33, 1, 2], c)

    pt_BoundedEra5Source.sload_wind_speed_at_10m()
    assert var in pt_BoundedEra5Source.data

    a, b, c = (140, 6, 6), 3.8995903495628834, 7.6075014496292
    assert pt_BoundedEra5Source.data[var].shape == a
    assert np.isclose(pt_BoundedEra5Source.data[var].mean(), b)
    assert np.isclose(pt_BoundedEra5Source.data[var][33, 1, 2], c)


def test_Era5Source_sload_elevated_wind_direction(pt_Era5Source, pt_BoundedEra5Source):
    var = "elevated_wind_direction"
    pt_Era5Source.sload_elevated_wind_direction()
    assert var in pt_Era5Source.data

    a, b, c = (140, 13, 11), 124.40131260688527, 38.39289529099399
    assert pt_Era5Source.data[var].shape == a
    assert np.isclose(pt_Era5Source.data[var].mean(), b)
    assert np.isclose(pt_Era5Source.data[var][33, 1, 2], c)

    pt_BoundedEra5Source.sload_elevated_wind_direction()
    assert var in pt_BoundedEra5Source.data

    a, b, c = (140, 6, 6), 121.82830613552835, 30.048524430066834
    assert pt_BoundedEra5Source.data[var].shape == a
    assert np.isclose(pt_BoundedEra5Source.data[var].mean(), b)
    assert np.isclose(pt_BoundedEra5Source.data[var][33, 1, 2], c)


def test_Era5Source_sload_surface_pressure(pt_Era5Source, pt_BoundedEra5Source):
    var = "surface_pressure"
    pt_Era5Source.sload_surface_pressure()
    assert var in pt_Era5Source.data

    a, b, c = (140, 13, 11), 100394.52450988448, 100029.60295419171
    assert pt_Era5Source.data[var].shape == a
    assert np.isclose(pt_Era5Source.data[var].mean(), b)
    assert np.isclose(pt_Era5Source.data[var][33, 1, 2], c)

    pt_BoundedEra5Source.sload_surface_pressure()
    assert var in pt_BoundedEra5Source.data

    a, b, c = (140, 6, 6), 99801.9903396807, 96837.36686300242
    assert pt_BoundedEra5Source.data[var].shape == a
    assert np.isclose(pt_BoundedEra5Source.data[var].mean(), b)
    assert np.isclose(pt_BoundedEra5Source.data[var][33, 1, 2], c)


def test_Era5Source_sload_surface_air_temperature(pt_Era5Source, pt_BoundedEra5Source):
    var = "surface_air_temperature"
    pt_Era5Source.sload_surface_air_temperature()
    assert var in pt_Era5Source.data

    a, b, c = (140, 13, 11), 1.2324024725541713, 0.7017410809656326
    assert pt_Era5Source.data[var].shape == a
    assert np.isclose(pt_Era5Source.data[var].mean(), b)
    assert np.isclose(pt_Era5Source.data[var][33, 1, 2], c)

    pt_BoundedEra5Source.sload_surface_air_temperature()
    assert var in pt_BoundedEra5Source.data

    a, b, c = (140, 6, 6), 0.9478995030327368, 0.9703039643544003
    assert pt_BoundedEra5Source.data[var].shape == a
    assert np.isclose(pt_BoundedEra5Source.data[var].mean(), b)
    assert np.isclose(pt_BoundedEra5Source.data[var][33, 1, 2], c)


def test_Era5Source_sload_surface_dew_temperature(pt_Era5Source, pt_BoundedEra5Source):
    var = "surface_dew_temperature"
    pt_Era5Source.sload_surface_dew_temperature()
    assert var in pt_Era5Source.data

    a, b, c = (140, 13, 11), -0.29206140549715787, 0.44538560136726346
    assert pt_Era5Source.data[var].shape == a
    assert np.isclose(pt_Era5Source.data[var].mean(), b)
    assert np.isclose(pt_Era5Source.data[var][33, 1, 2], c)

    pt_BoundedEra5Source.sload_surface_dew_temperature()
    assert var in pt_BoundedEra5Source.data

    a, b, c = (140, 6, 6), -0.8116621155333675, 0.9703039643544003
    assert pt_BoundedEra5Source.data[var].shape == a
    assert np.isclose(pt_BoundedEra5Source.data[var].mean(), b)
    assert np.isclose(pt_BoundedEra5Source.data[var][33, 1, 2], c)


def test_Era5Source_sload_direct_horizontal_irradiance(pt_Era5Source, pt_BoundedEra5Source):
    var = "direct_horizontal_irradiance"
    pt_Era5Source.sload_direct_horizontal_irradiance()
    assert var in pt_Era5Source.data

    a, b, c = (140, 13, 11), 13.16289243762006, 0.04272591326639607
    assert pt_Era5Source.data[var].shape == a
    assert np.isclose(pt_Era5Source.data[var].mean(), b)
    assert np.isclose(pt_Era5Source.data[var][33, 1, 2], c)

    pt_BoundedEra5Source.sload_direct_horizontal_irradiance()
    assert var in pt_BoundedEra5Source.data

    a, b, c = (140, 6, 6), 14.355118462680952, 0.0
    assert pt_BoundedEra5Source.data[var].shape == a
    assert np.isclose(pt_BoundedEra5Source.data[var].mean(), b)
    assert np.isclose(pt_BoundedEra5Source.data[var][33, 1, 2], c)


def test_Era5Source_sload_global_horizontal_irradiance(pt_Era5Source, pt_BoundedEra5Source):
    var = "global_horizontal_irradiance"
    pt_Era5Source.sload_global_horizontal_irradiance()
    assert var in pt_Era5Source.data

    a, b, c = (140, 13, 11), 30.996645951783744, 7.583849604785303
    assert pt_Era5Source.data[var].shape == a
    assert np.isclose(pt_Era5Source.data[var].mean(), b)
    assert np.isclose(pt_Era5Source.data[var][33, 1, 2], c)

    pt_BoundedEra5Source.sload_global_horizontal_irradiance()
    assert var in pt_BoundedEra5Source.data

    a, b, c = (140, 6, 6), 32.51753118822121, 5.063020722067934
    assert pt_BoundedEra5Source.data[var].shape == a
    assert np.isclose(pt_BoundedEra5Source.data[var].mean(), b)
    assert np.isclose(pt_BoundedEra5Source.data[var][33, 1, 2], c)


def test_Era5Source_get(pt_Era5Source, pt_BoundedEra5Source):
    var = "direct_horizontal_irradiance"

    pt_Era5Source.sload(var)
    pt_BoundedEra5Source.sload(var)

    pt = (6.03, 50.81)
    s1 = pt_Era5Source.get(var, pt)
    s2 = pt_BoundedEra5Source.get(var, pt)
    assert (s1 == s2).all()
    assert np.isclose(s1.values.mean(), 15.10422070986053)

    pts = [
        (6.03, 50.81),
        (6.44, 50.47),
    ]
    s1 = pt_Era5Source.get(var, pts)
    s2 = pt_BoundedEra5Source.get(var, pts)
    assert (s1 == s2).values.all()
    assert np.isclose(s1.values.mean(), 15.162205877864922)

    pt = (6.03, 50.81)
    s1 = pt_Era5Source.get(var, pt, interpolation="bilinear")
    assert np.isclose(s1.values.mean(), 15.277533860286267)


def test_Era5Source_sload_snow_albedo(pt_Era5Source, pt_BoundedEra5Source):
    var = "snow_albedo"
    pt_Era5Source.sload_snow_albedo()
    assert var in pt_Era5Source.data

    a, b, c = (140, 13, 11), 0.7941267245053446, 0.6624303997529796
    assert pt_Era5Source.data[var].shape == a
    assert np.isclose(pt_Era5Source.data[var].mean(), b)
    assert np.isclose(pt_Era5Source.data[var][33, 1, 2], c)

    pt_BoundedEra5Source.sload_snow_albedo()
    assert var in pt_BoundedEra5Source.data

    a, b, c = (140, 6, 6), 0.7822069000188031, 0.6775980069172599
    assert pt_BoundedEra5Source.data[var].shape == a
    assert np.isclose(pt_BoundedEra5Source.data[var].mean(), b)
    assert np.isclose(pt_BoundedEra5Source.data[var][33, 1, 2], c)


def test_Era5Source_sload_snow_density(pt_Era5Source, pt_BoundedEra5Source):
    var = "snow_density"
    pt_Era5Source.sload_snow_density()
    assert var in pt_Era5Source.data

    a, b, c = (140, 13, 11), 171.71123863464788, 171.71652449980587
    assert pt_Era5Source.data[var].shape == a
    assert np.isclose(pt_Era5Source.data[var].mean(), b)
    assert np.isclose(pt_Era5Source.data[var][33, 1, 2], c)

    pt_BoundedEra5Source.sload_snow_density()
    assert var in pt_BoundedEra5Source.data

    a, b, c = (140, 6, 6), 189.5077115523002, 176.9238246386672
    assert pt_BoundedEra5Source.data[var].shape == a
    assert np.isclose(pt_BoundedEra5Source.data[var].mean(), b)
    assert np.isclose(pt_BoundedEra5Source.data[var][33, 1, 2], c)


def test_Era5Source_sload_snow_depth_water_equivalent(pt_Era5Source, pt_BoundedEra5Source):
    var = "snow_depth_water_equivalent"
    pt_Era5Source.sload_snow_depth_water_equivalent()
    assert var in pt_Era5Source.data

    a, b, c = (140, 13, 11), 0.002338779435814489, 0.005035630903514665
    assert pt_Era5Source.data[var].shape == a
    assert np.isclose(pt_Era5Source.data[var].mean(), b)
    assert np.isclose(pt_Era5Source.data[var][33, 1, 2], c)

    pt_BoundedEra5Source.sload_snow_depth_water_equivalent()
    assert var in pt_BoundedEra5Source.data

    a, b, c = (140, 6, 6), 0.003410525752322981, 0.010071261807028442
    assert pt_BoundedEra5Source.data[var].shape == a
    assert np.isclose(pt_BoundedEra5Source.data[var].mean(), b)
    assert np.isclose(pt_BoundedEra5Source.data[var][33, 1, 2], c)


def test_Era5Source_sload_snowfall_water_equivalent(pt_Era5Source, pt_BoundedEra5Source):
    var = "snowfall_water_equivalent"
    pt_Era5Source.sload_snowfall_water_equivalent()
    assert var in pt_Era5Source.data

    a, b, c = (140, 13, 11), 1.72628057551922e-05, 0.0
    assert pt_Era5Source.data[var].shape == a
    assert np.isclose(pt_Era5Source.data[var].mean(), b)
    assert np.isclose(pt_Era5Source.data[var][33, 1, 2], c)

    pt_BoundedEra5Source.sload_snowfall_water_equivalent()
    assert var in pt_BoundedEra5Source.data

    a, b, c = (140, 6, 6), 3.066569101542149e-05, 0.0
    assert pt_BoundedEra5Source.data[var].shape == a
    assert np.isclose(pt_BoundedEra5Source.data[var].mean(), b)
    assert np.isclose(pt_BoundedEra5Source.data[var][33, 1, 2], c)



def _write_cf_era5_file(path, *, n_times=4):
    """Write a small ERA5 file which uses the CF compliant 'valid_time' axis.

    Returns the time stamps which the file holds.
    """
    import xarray as xr

    times = pd.date_range("2015-01-01", periods=n_times, freq="h")
    lat = np.array([52.0, 51.75, 51.5], dtype="f4")  # descending, as in a real download
    lon = np.array([5.0, 5.25, 5.5], dtype="f4")
    values = np.arange(n_times * lat.size * lon.size, dtype="f4").reshape(n_times, lat.size, lon.size)
    ds = xr.Dataset(
        {"sp": (("valid_time", "latitude", "longitude"), values, {"units": "Pa"})},
        coords={"valid_time": times, "latitude": lat, "longitude": lon},
    )
    ds["valid_time"].encoding = {"units": "hours since 1900-01-01 00:00:00.0", "calendar": "gregorian"}
    ds.to_netcdf(path)
    return times


def test_Era5Source_reads_a_cf_compliant_file(tmp_path):
    """Era5Source must accept both ERA5 download formats, 'time' and 'valid_time'."""
    path = tmp_path / "reanalysis-era5-single-levels.z4.x8.y5.y2015.surface_pressure.nc"
    times = _write_cf_era5_file(path)

    source = Era5Source(str(path), verbose=False)

    assert source.time_name == "valid_time"
    assert source.time_index.equals(pd.DatetimeIndex(times) - pd.Timedelta(minutes=30))

    source.load("sp", "surface_pressure")
    assert source.data["surface_pressure"].shape == (len(times), 3, 3)
