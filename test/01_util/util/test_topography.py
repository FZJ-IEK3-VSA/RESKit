from reskit.util.topography import visibility_from_topography
from reskit import TEST_DATA
import numpy as np


def test_visibility_from_topography():
    output = visibility_from_topography(lon=6.0, lat=51, elevation_raster=TEST_DATA['DEM-like.tif'])
    assert np.isclose(output['visibility'].values.mean(), 0.2888095238095238)

    output = visibility_from_topography(lon=6.0, lat=51, elevation_raster=TEST_DATA['DEM-like.tif'], eye_level=20)
    assert np.isclose(output['visibility'].values.mean(), 0.5711904761904761)
