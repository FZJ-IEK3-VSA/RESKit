from reskit import TEST_DATA
from reskit.weather import Era5Source
import pytest
import numpy as np
import pandas as pd
import geokit as gk
import netCDF4 as nc
from os.path import join


@pytest.fixture
def pt_Era5Source():
    return Era5Source(TEST_DATA["era5-like"], verbose=False)


@pytest.fixture
def pt_BoundedEra5Source():
    aachenExt = gk.Extent.fromVector(gk._test_data_["aachenShapefile.shp"])
    return Era5Source(
        TEST_DATA["era5-like"], bounds=aachenExt, index_pad=1, verbose=False
    )


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
    aachenExt = (
        gk.Extent.fromVector(gk._test_data_["aachenShapefile.shp"]).pad(0.5).fit(0.01)
    )

    ms = Era5Source(
        TEST_DATA["era5-like"], bounds=aachenExt, index_pad=1, verbose=False
    )

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


def test_Era5Source_sload_direct_horizontal_irradiance(
    pt_Era5Source, pt_BoundedEra5Source
):
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


def test_Era5Source_sload_global_horizontal_irradiance(
    pt_Era5Source, pt_BoundedEra5Source
):
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
    #     print(s1.values.mean())

    pts = [
        (6.03, 50.81),
        (6.44, 50.47),
    ]
    s1 = pt_Era5Source.get(var, pts)
    s2 = pt_BoundedEra5Source.get(var, pts)
    assert (s1 == s2).values.all()
    assert np.isclose(s1.values.mean(), 15.162205877864922)
    #     print(s1.values.mean())

    pt = (6.03, 50.81)
    s1 = pt_Era5Source.get(var, pt, interpolation="bilinear")
    assert np.isclose(s1.values.mean(), 15.277533860286267)


#     print(s1.values.mean())
