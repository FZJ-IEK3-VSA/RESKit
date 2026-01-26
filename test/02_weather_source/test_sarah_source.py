from os.path import join

import geokit as gk
import netCDF4 as nc
import numpy as np
import pandas as pd
import pytest

from reskit import TEST_DATA
from reskit.weather import SarahSource


@pytest.fixture
def pt_sarah_source() -> SarahSource:
    return SarahSource(TEST_DATA["sarah-like"], verbose=False)


@pytest.fixture
def pt_bounded_sarah_source() -> SarahSource:
    aachen_ext = gk.Extent.fromVector(gk._test_data_["aachenShapefile.shp"])
    return SarahSource(TEST_DATA["sarah-like"], bounds=aachen_ext, index_pad=1, verbose=False)


def test_sarah_source___init__():
    raw = nc.Dataset(join(TEST_DATA["sarah-like"], "SARAH-DNI.nc"), mode="r")
    raw_lats = raw["lat"][:]
    raw_lons = raw["lon"][:]
    raw_times = pd.DatetimeIndex(
        nc.num2date(
            raw["time"][:],
            raw["time"].units,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        )
    )

    # Unbounded source
    ms = SarahSource(TEST_DATA["sarah-like"], verbose=False)

    # ensure lats, lons and times are okay
    assert (ms.lats == raw_lats).all()
    assert (ms.lons == raw_lons).all()
    assert (ms.time_index == raw_times).all()

    # Initialize a SarahSource with Aachen boundaries
    aachen_ext = gk.Extent.fromVector(gk._test_data_["aachenShapefile.shp"]).pad(0.5).fit(0.01)

    ms = SarahSource(TEST_DATA["sarah-like"], bounds=aachen_ext, index_pad=1, verbose=False)

    # ensure lats, lons and times are okay
    assert np.isclose(ms.lats[0], 49.9)
    assert np.isclose(ms.lats[-1], 51.55)
    assert ms.lats.size == 34

    assert np.isclose(ms.lons[0], 5.4)
    assert np.isclose(ms.lons[-1], 7.0)
    assert ms.lons.size == 33

    assert (ms.time_index == raw_times).all()


def test_sarah_source_loc_to_index(pt_sarah_source: SarahSource, pt_bounded_sarah_source: SarahSource):
    idx = pt_sarah_source.loc_to_index((6.03, 50.81))
    assert idx.yi == 36
    assert idx.xi == 21

    idx = pt_sarah_source.loc_to_index(
        [
            (6.03, 50.81),
            (6.44, 50.47),
        ]
    )
    assert idx[0].yi == 36
    assert idx[1].yi == 29
    assert idx[0].xi == 21
    assert idx[1].xi == 29

    idx = pt_sarah_source.loc_to_index(
        [
            (6.03, 50.81),
            (6.44, 50.47),
        ],
        as_int=False,
    )
    assert np.isclose(idx[0].yi, 36.200000000000045)
    assert np.isclose(idx[1].yi, 29.399999999999977)
    assert np.isclose(idx[0].xi, 20.600000000000005)
    assert np.isclose(idx[1].xi, 28.800000000000008)

    idx = pt_bounded_sarah_source.loc_to_index((6.03, 50.81))
    assert idx.yi == 8
    assert idx.xi == 3

    idx = pt_bounded_sarah_source.loc_to_index(
        [
            (6.03, 50.81),
            (6.44, 50.47),
        ]
    )
    assert idx[0].yi == 8
    assert idx[1].yi == 1
    assert idx[0].xi == 3
    assert idx[1].xi == 11

    idx = pt_bounded_sarah_source.loc_to_index(
        [
            (6.03, 50.81),
            (6.44, 50.47),
        ],
        as_int=False,
    )
    assert np.isclose(idx[0].yi, 8.19996948242192)
    assert np.isclose(idx[1].yi, 1.3999694824218523)
    assert np.isclose(idx[0].xi, 2.599998092651372)
    assert np.isclose(idx[1].xi, 10.799998092651375)


def test_sarah_source_sload_global_horizontal_irradiance(
    pt_sarah_source: SarahSource, pt_bounded_sarah_source: SarahSource
):
    var = "global_horizontal_irradiance"
    pt_sarah_source.sload_global_horizontal_irradiance()
    assert var in pt_sarah_source.data

    a, b, c = (48, 61, 51), 36.67517813136183, 114
    assert pt_sarah_source.data[var].shape == a
    assert np.isclose(pt_sarah_source.data[var].data.mean(), b)
    assert np.isclose(pt_sarah_source.data[var][24, 1, 2], c)

    pt_bounded_sarah_source.sload_global_horizontal_irradiance()
    assert var in pt_bounded_sarah_source.data

    a, b, c = (48, 14, 13), 46.84146062271062, 179
    assert pt_bounded_sarah_source.data[var].shape == a
    assert np.isclose(pt_bounded_sarah_source.data[var].data.mean(), b)
    assert np.isclose(pt_bounded_sarah_source.data[var][24, 1, 2], c)


def test_sarah_source_sload_direct_normal_irradiance(
    pt_sarah_source: SarahSource, pt_bounded_sarah_source: SarahSource
):
    var = "direct_normal_irradiance"
    pt_sarah_source.sload_direct_normal_irradiance()
    assert var in pt_sarah_source.data

    a, b, c = (48, 61, 51), 73.67886799528554, 9
    assert pt_sarah_source.data[var].shape == a
    assert np.isclose(pt_sarah_source.data[var].data.mean(), b)
    assert np.isclose(pt_sarah_source.data[var][24, 1, 2], c)

    pt_bounded_sarah_source.sload_direct_normal_irradiance()
    assert var in pt_bounded_sarah_source.data

    a, b, c = (48, 14, 13), 127.33505036630036, 173
    assert pt_bounded_sarah_source.data[var].shape == a
    assert np.isclose(pt_bounded_sarah_source.data[var].data.mean(), b)
    assert np.isclose(pt_bounded_sarah_source.data[var][24, 1, 2], c)


def test_sarah_source_get(pt_sarah_source: SarahSource, pt_bounded_sarah_source: SarahSource):
    var = "direct_normal_irradiance"

    pt_sarah_source.sload(var)
    pt_bounded_sarah_source.sload(var)

    pt = (6.03, 50.81)
    s1 = pt_sarah_source.get(var, pt)
    s2 = pt_bounded_sarah_source.get(var, pt)
    assert (s1 == s2).all()
    assert np.isclose(s1.values.mean(), 155.22916666666666)

    pts = [
        (6.03, 50.81),
        (6.44, 50.47),
    ]
    s1 = pt_sarah_source.get(var, pts)
    s2 = pt_bounded_sarah_source.get(var, pts)
    assert (s1 == s2).values.all()
    assert np.isclose(s1.values.mean(), 124.28125)

    pt = (6.03, 50.81)
    s1 = pt_sarah_source.get(var, pt, interpolation="bilinear")
    # assert (s1==s2).values.all()
    assert np.isclose(s1.values.mean(), 154.99248551347725)
