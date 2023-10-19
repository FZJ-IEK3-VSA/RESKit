from reskit import TEST_DATA
from reskit.weather import MerraSource
import pytest
import numpy as np
import pandas as pd
import geokit as gk
import netCDF4 as nc


@pytest.fixture
def pt_MerraSource():
    return MerraSource(TEST_DATA['merra-like'], verbose=False)


@pytest.fixture
def pt_BoundedMerraSource():
    aachenExt = gk.Extent.fromVector(gk._test_data_['aachenShapefile.shp'])
    return MerraSource(TEST_DATA['merra-like'], bounds=aachenExt, index_pad=0, verbose=False)


def test_constants():
    assert gk.util.isRaster(MerraSource.LONG_RUN_AVERAGE_GHI)
    assert gk.util.isRaster(MerraSource.LONG_RUN_AVERAGE_WINDSPEED)


def test_MerraSource___init__():
    raw = nc.Dataset(TEST_DATA['merra-like.nc4'])
    rawLats = raw["lat"][:]
    rawLons = raw["lon"][:]
    rawTimes = pd.DatetimeIndex(
        nc.num2date(raw["time"][:], raw["time"].units,
                    only_use_cftime_datetimes=False, only_use_python_datetimes=True),
        tz="GMT")

    # Unbounded source
    ms = MerraSource(TEST_DATA['merra-like.nc4'], verbose=False)

    # ensure lats, lons and times are okay
    assert (ms.lats == rawLats).all()
    assert (ms.lons == rawLons).all()
    assert (ms.time_index == rawTimes).all()

    # Initialize a MerraSource with Aachen boundaries
    aachenExt = gk.Extent.fromVector(
        gk._test_data_['aachenShapefile.shp']).pad(0.5).fit(0.01)
    aachenLats = np.array([50.0, 50.5, 51.0, 51.5])
    aachenLons = np.array([5.625, 6.250])

    ms = MerraSource(TEST_DATA['merra-like.nc4'],
                     bounds=aachenExt, index_pad=1, verbose=False)

    # ensure lats, lons and times are okay
    assert np.isclose(ms.lats, aachenLats).all()

    assert np.isclose(ms.lons, aachenLons).all()

    assert (ms.time_index == rawTimes).all()


def test_MerraSource_loc_to_index(pt_MerraSource):
    idx = pt_MerraSource.loc_to_index((5.6, 50.1))
    assert idx.yi == 2
    assert idx.xi == 1

    idx = pt_MerraSource.loc_to_index([(5.6, 50.1), (6.3, 50.8), ])
    assert idx[0].yi == 2
    assert idx[1].yi == 4
    assert idx[0].xi == 1
    assert idx[1].xi == 2

    idx = pt_MerraSource.loc_to_index(
        [(5.6, 50.1), (6.3, 50.8), ], as_int=False)
    assert np.isclose(idx[0].yi, 2.200000000000003)
    assert np.isclose(idx[1].yi, 3.5999999999999943)
    assert np.isclose(idx[0].xi, 0.9599999999999994)
    assert np.isclose(idx[1].xi, 2.0799999999999996)


def test_MerraSource_context_area_at_index(pt_MerraSource):
    pt = pt_MerraSource.context_area_at_index(0, 0).Centroid()

    assert np.isclose(pt.GetX(), 5.0)
    assert np.isclose(pt.GetY(), 49.0)


def test_MerraSource_sload_elevated_wind_speed(pt_MerraSource, pt_BoundedMerraSource):
    pt_MerraSource.sload_elevated_wind_speed()

    assert "U50M" in pt_MerraSource.data
    assert "V50M" in pt_MerraSource.data
    assert "elevated_wind_speed" in pt_MerraSource.data

    assert pt_MerraSource.data['elevated_wind_speed'].shape == (71, 7, 5)
    assert np.isclose(
        pt_MerraSource.data['elevated_wind_speed'].mean(), 7.895609595070423)
    assert np.isclose(
        pt_MerraSource.data['elevated_wind_speed'][60, 1, 3], 5.139748)

    pt_BoundedMerraSource.sload_elevated_wind_speed()

    assert "U50M" in pt_BoundedMerraSource.data
    assert "V50M" in pt_BoundedMerraSource.data
    assert "elevated_wind_speed" in pt_BoundedMerraSource.data

    assert pt_BoundedMerraSource.data['elevated_wind_speed'].shape == (
        71, 4, 3)
    assert np.isclose(
        pt_BoundedMerraSource.data['elevated_wind_speed'].mean(), 8.17508378722858)
    assert np.isclose(
        pt_BoundedMerraSource.data['elevated_wind_speed'][60, 1, 2], 5.71493)


def test_MerraSource_sload_surface_wind_speed(pt_MerraSource, pt_BoundedMerraSource):
    pt_MerraSource.sload_surface_wind_speed()
    assert "U2M" in pt_MerraSource.data
    assert "V2M" in pt_MerraSource.data
    assert "surface_wind_speed" in pt_MerraSource.data

    assert pt_MerraSource.data['surface_wind_speed'].shape == (71, 7, 5)
    assert np.isclose(
        pt_MerraSource.data['surface_wind_speed'].mean(), 3.772894397635815)
    assert np.isclose(
        pt_MerraSource.data['surface_wind_speed'][60, 1, 2], 2.9491692)

    pt_BoundedMerraSource.sload_surface_wind_speed()

    assert "U2M" in pt_BoundedMerraSource.data
    assert "V2M" in pt_BoundedMerraSource.data
    assert "surface_wind_speed" in pt_BoundedMerraSource.data

    assert pt_BoundedMerraSource.data['surface_wind_speed'].shape == (71, 4, 3)
    assert np.isclose(
        pt_BoundedMerraSource.data['surface_wind_speed'].mean(), 3.9790956022593895)
    assert np.isclose(
        pt_BoundedMerraSource.data['surface_wind_speed'][60, 1, 2], 3.0793889)


def test_MerraSource_sload_wind_speed_at_2m(pt_MerraSource, pt_BoundedMerraSource):

    pt_MerraSource.sload_wind_speed_at_2m()

    assert "U2M" in pt_MerraSource.data
    assert "V2M" in pt_MerraSource.data
    assert "wind_speed_at_2m" in pt_MerraSource.data

    assert pt_MerraSource.data['wind_speed_at_2m'].shape == (71, 7, 5)
    assert np.isclose(
        pt_MerraSource.data['wind_speed_at_2m'].mean(), 3.772894397635815)
    assert np.isclose(
        pt_MerraSource.data['wind_speed_at_2m'][60, 1, 2], 2.9491692)

    pt_BoundedMerraSource.sload_wind_speed_at_2m()

    assert "U2M" in pt_BoundedMerraSource.data
    assert "V2M" in pt_BoundedMerraSource.data
    assert "wind_speed_at_2m" in pt_BoundedMerraSource.data

    assert pt_BoundedMerraSource.data['wind_speed_at_2m'].shape == (71, 4, 3)
    assert np.isclose(
        pt_BoundedMerraSource.data['wind_speed_at_2m'].mean(), 3.9790956022593895)
    assert np.isclose(
        pt_BoundedMerraSource.data['wind_speed_at_2m'][60, 1, 2], 3.0793889)


def test_MerraSource_sload_wind_speed_at_10m(pt_MerraSource, pt_BoundedMerraSource):
    pt_MerraSource.sload_wind_speed_at_10m()

    assert "U10M" in pt_MerraSource.data
    assert "V10M" in pt_MerraSource.data
    assert "wind_speed_at_10m" in pt_MerraSource.data

    assert pt_MerraSource.data['wind_speed_at_10m'].shape == (71, 7, 5)
    assert np.isclose(
        pt_MerraSource.data['wind_speed_at_10m'].mean(), 5.601739342303823)
    assert np.isclose(
        pt_MerraSource.data['wind_speed_at_10m'][60, 1, 2], 4.2905645)

    pt_BoundedMerraSource.sload_wind_speed_at_10m()

    assert "U10M" in pt_BoundedMerraSource.data
    assert "V10M" in pt_BoundedMerraSource.data
    assert "wind_speed_at_10m" in pt_BoundedMerraSource.data

    assert pt_BoundedMerraSource.data['wind_speed_at_10m'].shape == (71, 4, 3)
    assert np.isclose(
        pt_BoundedMerraSource.data['wind_speed_at_10m'].mean(), 5.872937412888791)
    assert np.isclose(
        pt_BoundedMerraSource.data['wind_speed_at_10m'][60, 1, 2], 4.427459)


def test_MerraSource_sload_wind_speed_at_50m(pt_MerraSource, pt_BoundedMerraSource):
    pt_MerraSource.sload_wind_speed_at_50m()

    assert "U50M" in pt_MerraSource.data
    assert "V50M" in pt_MerraSource.data
    assert "wind_speed_at_50m" in pt_MerraSource.data

    assert pt_MerraSource.data['wind_speed_at_50m'].shape == (71, 7, 5)
    assert np.isclose(
        pt_MerraSource.data['wind_speed_at_50m'].mean(), 7.895609595070423)
    assert np.isclose(
        pt_MerraSource.data['wind_speed_at_50m'][60, 1, 2], 6.225379)

    pt_BoundedMerraSource.sload_wind_speed_at_50m()

    assert "U50M" in pt_BoundedMerraSource.data
    assert "V50M" in pt_BoundedMerraSource.data
    assert "wind_speed_at_50m" in pt_BoundedMerraSource.data

    assert pt_BoundedMerraSource.data['wind_speed_at_50m'].shape == (71, 4, 3)
    assert np.isclose(
        pt_BoundedMerraSource.data['wind_speed_at_50m'].mean(), 8.17508378722858)
    assert np.isclose(
        pt_BoundedMerraSource.data['wind_speed_at_50m'][60, 1, 2], 5.71493)


def test_MerraSource_sload_elevated_wind_direction(pt_MerraSource, pt_BoundedMerraSource):
    var = "elevated_wind_direction"
    pt_MerraSource.sload_elevated_wind_direction()
    assert var in pt_MerraSource.data

    a, b, c = (71, 7, 5), 7.895609595070423, 6.225379
    assert pt_MerraSource.data[var].shape == a
    assert np.isclose(pt_MerraSource.data[var].mean(), b)
    assert np.isclose(pt_MerraSource.data[var][60, 1, 2], c)

    pt_BoundedMerraSource.sload_elevated_wind_direction()
    assert var in pt_BoundedMerraSource.data

    a, b, c = (71, 4, 3), 8.17508378722858, 5.71493
    assert pt_BoundedMerraSource.data[var].shape == a
    assert np.isclose(pt_BoundedMerraSource.data[var].mean(), b)
    assert np.isclose(pt_BoundedMerraSource.data[var][60, 1, 2], c)


def test_MerraSource_sload_surface_wind_direction(pt_MerraSource, pt_BoundedMerraSource):
    var = "surface_wind_direction"
    pt_MerraSource.sload_surface_wind_direction()
    assert var in pt_MerraSource.data

    a, b, c = (71, 7, 5), 3.772894397635815, 2.9491692
    assert pt_MerraSource.data[var].shape == a
    assert np.isclose(pt_MerraSource.data[var].mean(), b)
    assert np.isclose(pt_MerraSource.data[var][60, 1, 2], c)

    pt_BoundedMerraSource.sload_surface_wind_direction()
    assert var in pt_BoundedMerraSource.data

    a, b, c = (71, 4, 3), 3.9790956022593895, 3.0793889
    assert pt_BoundedMerraSource.data[var].shape == a
    assert np.isclose(pt_BoundedMerraSource.data[var].mean(), b)
    assert np.isclose(pt_BoundedMerraSource.data[var][60, 1, 2], c)


def test_MerraSource_sload_wind_direction_at_2m(pt_MerraSource, pt_BoundedMerraSource):
    var = "wind_direction_at_2m"
    pt_MerraSource.sload_wind_direction_at_2m()
    assert var in pt_MerraSource.data

    a, b, c = (71, 7, 5), 3.772894397635815, 2.9491692
    assert pt_MerraSource.data[var].shape == a
    assert np.isclose(pt_MerraSource.data[var].mean(), b)
    assert np.isclose(pt_MerraSource.data[var][60, 1, 2], c)

    pt_BoundedMerraSource.sload_wind_direction_at_2m()
    assert var in pt_BoundedMerraSource.data

    a, b, c = (71, 4, 3), 3.9790956022593895, 3.0793889
    assert pt_BoundedMerraSource.data[var].shape == a
    assert np.isclose(pt_BoundedMerraSource.data[var].mean(), b)
    assert np.isclose(pt_BoundedMerraSource.data[var][60, 1, 2], c)


def test_MerraSource_sload_wind_direction_at_10m(pt_MerraSource, pt_BoundedMerraSource):
    var = "wind_direction_at_10m"
    pt_MerraSource.sload_wind_direction_at_10m()
    assert var in pt_MerraSource.data

    a, b, c = (71, 7, 5), 5.601739342303823, 4.2905645
    assert pt_MerraSource.data[var].shape == a
    assert np.isclose(pt_MerraSource.data[var].mean(), b)
    assert np.isclose(pt_MerraSource.data[var][60, 1, 2], c)

    pt_BoundedMerraSource.sload_wind_direction_at_10m()
    assert var in pt_BoundedMerraSource.data

    a, b, c = (71, 4, 3), 5.872937412888791, 4.427459
    assert pt_BoundedMerraSource.data[var].shape == a
    assert np.isclose(pt_BoundedMerraSource.data[var].mean(), b)
    assert np.isclose(pt_BoundedMerraSource.data[var][60, 1, 2], c)


def test_MerraSource_sload_wind_direction_at_50m(pt_MerraSource, pt_BoundedMerraSource):
    var = "wind_direction_at_50m"
    pt_MerraSource.sload_wind_direction_at_50m()
    assert var in pt_MerraSource.data

    a, b, c = (71, 7, 5), 7.895609595070423, 6.225379
    assert pt_MerraSource.data[var].shape == a
    assert np.isclose(pt_MerraSource.data[var].mean(), b)
    assert np.isclose(pt_MerraSource.data[var][60, 1, 2], c)

    pt_BoundedMerraSource.sload_wind_direction_at_50m()
    assert var in pt_BoundedMerraSource.data

    a, b, c = (71, 4, 3), 8.17508378722858, 5.71493
    assert pt_BoundedMerraSource.data[var].shape == a
    assert np.isclose(pt_BoundedMerraSource.data[var].mean(), b)
    assert np.isclose(pt_BoundedMerraSource.data[var][60, 1, 2], c)


def test_MerraSource_sload_surface_pressure(pt_MerraSource, pt_BoundedMerraSource):
    var = "surface_pressure"
    pt_MerraSource.sload_surface_pressure()
    assert var in pt_MerraSource.data

    a, b, c = (71, 7, 5), 100376.69, 98695.31
    assert pt_MerraSource.data[var].shape == a
    assert np.isclose(pt_MerraSource.data[var].mean(), b)
    assert np.isclose(pt_MerraSource.data[var][60, 1, 2], c)

    pt_BoundedMerraSource.sload_surface_pressure()
    assert var in pt_BoundedMerraSource.data

    a, b, c = (71, 4, 3), 100148.8, 97535.31
    assert pt_BoundedMerraSource.data[var].shape == a
    assert np.isclose(pt_BoundedMerraSource.data[var].mean(), b)
    assert np.isclose(pt_BoundedMerraSource.data[var][60, 1, 2], c)


def test_MerraSource_sload_surface_air_temperature(pt_MerraSource, pt_BoundedMerraSource):
    var = "surface_air_temperature"
    pt_MerraSource.sload_surface_air_temperature()
    assert var in pt_MerraSource.data

    a, b, c = (71, 7, 5), 0.6883948, 0.55477905
    assert pt_MerraSource.data[var].shape == a
    assert np.isclose(pt_MerraSource.data[var].mean(), b)
    assert np.isclose(pt_MerraSource.data[var][60, 1, 2], c)

    pt_BoundedMerraSource.sload_surface_air_temperature()
    assert var in pt_BoundedMerraSource.data

    a, b, c = (71, 4, 3), 0.5205658, -0.0038146973
    assert pt_BoundedMerraSource.data[var].shape == a
    assert np.isclose(pt_BoundedMerraSource.data[var].mean(), b)
    assert np.isclose(pt_BoundedMerraSource.data[var][60, 1, 2], c)


def test_MerraSource_sload_surface_dew_temperature(pt_MerraSource, pt_BoundedMerraSource):
    var = "surface_dew_temperature"
    pt_MerraSource.sload_surface_dew_temperature()
    assert var in pt_MerraSource.data

    a, b, c = (71, 7, 5), -0.056511015, 0.09063721
    assert pt_MerraSource.data[var].shape == a
    assert np.isclose(pt_MerraSource.data[var].mean(), b)
    assert np.isclose(pt_MerraSource.data[var][60, 1, 2], c)

    pt_BoundedMerraSource.sload_surface_dew_temperature()
    assert var in pt_BoundedMerraSource.data

    a, b, c = (71, 4, 3), -0.25283656, -0.85858154
    assert pt_BoundedMerraSource.data[var].shape == a
    assert np.isclose(pt_BoundedMerraSource.data[var].mean(), b)
    assert np.isclose(pt_BoundedMerraSource.data[var][60, 1, 2], c)


def test_MerraSource_sload_global_horizontal_irradiance(pt_MerraSource, pt_BoundedMerraSource):
    var = "global_horizontal_irradiance"
    pt_MerraSource.sload_global_horizontal_irradiance()
    assert var in pt_MerraSource.data

    a, b, c = (71, 7, 5), 26.115427, 25.71875
    assert pt_MerraSource.data[var].shape == a
    assert np.isclose(pt_MerraSource.data[var].mean(), b)
    assert np.isclose(pt_MerraSource.data[var][60, 1, 2], c)

    pt_BoundedMerraSource.sload_global_horizontal_irradiance()
    assert var in pt_BoundedMerraSource.data

    a, b, c = (71, 4, 3), 27.9464, 55.703125
    assert pt_BoundedMerraSource.data[var].shape == a
    assert np.isclose(pt_BoundedMerraSource.data[var].mean(), b)
    assert np.isclose(pt_BoundedMerraSource.data[var][60, 1, 2], c)


def test_MerraSource_get(pt_MerraSource, pt_BoundedMerraSource):
    var = "surface_pressure"

    pt_MerraSource.sload(var)
    pt_BoundedMerraSource.sload(var)

    pt = (5.8, 50.2)
    s1 = pt_MerraSource.get(var, pt)
    s2 = pt_BoundedMerraSource.get(var, pt)
    assert (s1 == s2).all()
    assert np.isclose(s1.values.mean(), 98274.750000)

    pts = [(5.8, 50.2), (6.6, 51.2), (6.1, 50.8), ]
    s1 = pt_MerraSource.get(var, pts)
    s2 = pt_BoundedMerraSource.get(var, pts)
    assert (s1 == s2).values.all()
    assert np.isclose(s1.values.mean(), 100417.23)

    pt = (6.2, 50.2)
    s1 = pt_MerraSource.get(var, pt, interpolation="bilinear")
    # assert (s1==s2).values.all()
    assert np.isclose(s1.values.mean(), 97982.99218133804)
