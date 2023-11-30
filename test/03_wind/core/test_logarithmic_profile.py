from reskit import TEST_DATA
from reskit.wind.core.logarithmic_profile import (
    roughness_from_clc,
    roughness_from_land_cover_classification,
    roughness_from_land_cover_source,
    roughness_from_levels,
    apply_logarithmic_profile_projection,
)

import numpy as np
import geokit as gk


def test_apply_logarithmic_profile_projection():
    output = apply_logarithmic_profile_projection(
        measured_wind_speed=6, measured_height=10, target_height=50, roughness=0.002
    )
    assert np.isclose(output, 7.133780490541967)

    output = apply_logarithmic_profile_projection(
        measured_wind_speed=np.array([3, 4, 5]),
        measured_height=10,
        target_height=50,
        roughness=np.array([0.17874692, 0.13864688, 0.11328275]),
    )

    assert np.isclose(output[0], 4.19976902)
    assert np.isclose(output[1], 5.50470654)
    assert np.isclose(output[2], 6.79606587)


def test_roughness_from_levels():
    r = roughness_from_levels(
        low_wind_speed=3, low_height=10, high_wind_speed=4, high_height=50
    )
    assert np.isclose(r, 0.08)

    r = roughness_from_levels(
        low_wind_speed=np.array([3, 4, 5]),
        low_height=10,
        high_wind_speed=np.array([4, 5, 6]),
        high_height=50,
    )
    assert np.isclose(r[0], 0.08)
    assert np.isclose(r[1], 0.016)
    assert np.isclose(r[2], 0.0032)


def test_roughness_from_clc():
    loc1 = gk.Location(
        lat=50.370680, lon=5.752684
    )  # grid value: 24 -> code: 312 -> rough: 0.75
    loc2 = gk.Location(
        lat=50.52603, lon=6.10476
    )  # grid value: 36 -> code: 412 -> rough: 0.0005
    loc3 = gk.Location(
        lat=50.59082, lon=5.86483
    )  # grid value: 1 -> code: 111 -> rough: 1.2

    r = roughness_from_clc(clc_path=TEST_DATA["clc-aachen_clipped.tif"], loc=loc1)
    assert np.isclose(r, 0.75)

    r = roughness_from_clc(
        clc_path=TEST_DATA["clc-aachen_clipped.tif"], loc=[loc1, loc2, loc3]
    )
    assert np.isclose(r[0], 0.75)
    assert np.isclose(r[1], 0.0005)
    assert np.isclose(r[2], 1.2)

    r = roughness_from_clc(
        clc_path=TEST_DATA["clc-aachen_clipped.tif"],
        loc=[loc1, loc2, loc3],
        window_range=2,
    )
    assert np.isclose(r[0], 0.7380)
    assert np.isclose(r[1], 0.0005)
    assert np.isclose(r[2], 1.0040)


def test_roughness_from_land_cover_classification():
    output = roughness_from_land_cover_classification(
        classification=110, land_cover_type="cci"
    )
    assert np.isclose(output, 0.03)

    output = roughness_from_land_cover_classification(
        classification=[220, 150, 30], land_cover_type="globCover"
    )
    assert np.isclose(output, [0.0004, 0.05, 0.3]).all()


def test_roughness_from_land_cover_source():
    loc1 = gk.Location(
        lat=50.370680, lon=5.752684
    )  # grid value: 24 -> code: 312 -> rough: 0.75
    loc2 = gk.Location(
        lat=50.52603, lon=6.10476
    )  # grid value: 36 -> code: 412 -> rough: 0.0005
    loc3 = gk.Location(
        lat=50.59082, lon=5.86483
    )  # grid value: 1 -> code: 111 -> rough: 1.2

    r = roughness_from_land_cover_source(
        source=TEST_DATA["ESA_CCI_2018_clip.tif"], loc=loc1, land_cover_type="cci"
    )
    assert np.isclose(r, 0.75)

    r = roughness_from_land_cover_source(
        source=TEST_DATA["ESA_CCI_2018_clip.tif"],
        loc=[loc1, loc2, loc3],
        land_cover_type="cci",
    )
    assert np.isclose(r[0], 0.75)
    assert np.isclose(r[1], 0.03)
    assert np.isclose(r[2], 1.2)
