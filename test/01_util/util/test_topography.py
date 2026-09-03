import numpy as np
import pytest

from reskit import TEST_DATA
from reskit.util.topography import visibility_from_topography


def test_visibility_from_topography():
    output = visibility_from_topography(lon=6.0, lat=51, elevation_raster=TEST_DATA["DEM-like.tif"])
    assert np.isclose(output["visibility"].values.mean(), 0.27595238095238095) # Changed due to bug in visibility_from_topography

    output = visibility_from_topography(lon=6.0, lat=51, elevation_raster=TEST_DATA["DEM-like.tif"], eye_level=20)
    assert np.isclose(output["visibility"].values.mean(), 0.5376190476190477)


def test_sample_longitudes_use_the_cosine_of_the_latitude():
    # An east-west arc of a given angular length spans a longitude difference of that
    # length divided by cos(latitude), not by sin(latitude).
    lat = 51.0
    output = visibility_from_topography(lon=6.0, lat=lat, elevation_raster=TEST_DATA["DEM-like.tif"])

    lon_span = output["longitude"].values.max() - 6.0
    lat_span = output["latitude"].values.max() - lat

    assert np.isclose(lon_span / lat_span, 1 / np.cos(np.radians(lat)))


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_sample_points_at_the_equator_are_finite():
    # At the equator sin(latitude) is zero. A division by it gave infinite longitudes.
    output = visibility_from_topography(
        lon=0.0,
        lat=0.0,
        elevation_raster=TEST_DATA["DEM-like.tif"],
        base_elevation=100,
    )

    assert np.isfinite(output["longitude"].values).all()
    # at the equator a degree of longitude and a degree of latitude cover the same arc
    assert np.isclose(output["longitude"].values.max(), output["latitude"].values.max())
