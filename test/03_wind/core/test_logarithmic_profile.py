from pathlib import Path

import geokit as gk
import numpy as np
import pytest

from reskit import TEST_DATA
from reskit.wind.core.logarithmic_profile import (
    apply_logarithmic_profile_projection,
    roughness_from_clc,
    roughness_from_land_cover_classification,
    roughness_from_land_cover_source,
    roughness_from_levels,
)


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
    r = roughness_from_levels(low_wind_speed=3, low_height=10, high_wind_speed=4, high_height=50)
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


@pytest.mark.parametrize("raster_input", [str, Path, gk.raster.loadRaster], ids=["str", "path", "dataset"])
def test_roughness_from_clc(raster_input):
    # LCCS 70 (tree cover, needleleaved, evergreen, closed to open) -> rough: 0.75
    loc1 = gk.Location(lat=50.370680, lon=5.752684)
    # LCCS 180 (shrub or herbaceous cover, flooded) -> rough: 0.03
    loc2 = gk.Location(lat=50.52603, lon=6.10476)
    # LCCS 190 (urban areas) -> rough: 1.2
    loc3 = gk.Location(lat=50.59082, lon=5.86483)

    clc_path = raster_input(TEST_DATA["clc-aachen_clipped.tif"])
    r = roughness_from_clc(clc_path=clc_path, loc=loc1)
    assert np.isclose(r, 0.75)

    r = roughness_from_clc(clc_path=clc_path, loc=[loc1, loc2, loc3])
    assert np.isclose(r[0], 0.75)
    assert np.isclose(r[1], 0.0005)
    assert np.isclose(r[2], 1.2)

    r = roughness_from_clc(
        clc_path=clc_path,
        loc=[loc1, loc2, loc3],
        window_range=2,
    )
    assert np.isclose(r[0], 0.7380)
    assert np.isclose(r[1], 0.0005)
    assert np.isclose(r[2], 1.0040)


def test_roughness_from_land_cover_classification():
    output = roughness_from_land_cover_classification(classification=110, land_cover_type="cci")
    assert np.isclose(output, 0.03)

    output = roughness_from_land_cover_classification(classification=[220, 150, 30], land_cover_type="globCover")
    assert np.isclose(output, [0.0004, 0.05, 0.3]).all()


@pytest.mark.parametrize("raster_input", [str, Path, gk.raster.loadRaster], ids=["str", "path", "dataset"])
def test_roughness_from_land_cover_source(raster_input):
    # LCCS 70 (tree cover, needleleaved, evergreen, closed to open) -> rough: 0.75
    loc1 = gk.Location(lat=50.370680, lon=5.752684)
    # LCCS 180 (shrub or herbaceous cover, flooded) -> rough: 0.03
    loc2 = gk.Location(lat=50.52603, lon=6.10476)
    # LCCS 190 (urban areas) -> rough: 1.2
    loc3 = gk.Location(lat=50.59082, lon=5.86483)

    source = raster_input(TEST_DATA["ESA_CCI_2015_clip.tif"])
    r = roughness_from_land_cover_source(source=source, loc=loc1, land_cover_type="cci")
    assert np.isclose(r, 0.75)

    r = roughness_from_land_cover_source(
        source=source,
        loc=[loc1, loc2, loc3],
        land_cover_type="cci",
    )
    assert np.isclose(r[0], 0.75)
    assert np.isclose(r[1], 0.03)
    assert np.isclose(r[2], 1.2)
