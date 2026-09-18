"""The location rasters come from the ``offshore_siting`` collection unless a path is given.

The two functions used to read their default raster from ``reskit/default_paths.yaml``;
now the default is the collection's full variant, resolved through ``reskit.data``,
and the test variant is the bundled German Bight fixture.
"""

from pathlib import Path

import pytest

from reskit import data
from reskit.util import local_values
from reskit.util.errors import ResError
from reskit.util.local_values import catalogued_input, distance_to_coastline, water_depth_from_location

# A point inside the 24 x 24 cell fixture box, and what the fixtures hold there.
LATITUDE, LONGITUDE = 54.4, 6.9
DEPTH_M = 37.0
COAST_KM = 67


@pytest.fixture
def fixtures():
    """The collection's test variant, answered from the bundle without network access."""
    return data.paths("offshore_siting", test=True)


@pytest.fixture
def default_route(monkeypatch, fixtures):
    """Make the catalogue route hand out the fixtures, so the default can be exercised offline."""

    def paths(collection, **kwargs):
        assert collection == local_values.COLLECTION
        assert not kwargs.get("test")  # the default is the full variant
        return fixtures

    monkeypatch.setattr(data, "paths", paths)
    catalogued_input.cache_clear()
    yield
    catalogued_input.cache_clear()


def test_the_test_variant_is_the_bundled_fixture_pair(fixtures):
    assert set(fixtures) == {"water_depth", "coast_distance"}
    assert fixtures["water_depth"].name == "water_depth_northsea.tif"
    assert fixtures["coast_distance"].name == "coast_distance_northsea.tif"
    assert all(path.is_file() for path in fixtures.values())


def test_both_variants_name_the_same_inputs():
    definitions = data._definitions()
    assert definitions.variants("offshore_siting") == ("test", "full")
    assert set(definitions.named_keys("offshore_siting", test=True)) == set(
        definitions.named_keys("offshore_siting", test=False)
    )


def test_water_depth_from_a_given_raster(fixtures):
    assert water_depth_from_location(LATITUDE, LONGITUDE, waterDepthFilePath=fixtures["water_depth"]) == DEPTH_M
    assert water_depth_from_location(LATITUDE, LONGITUDE, waterDepthFilePath=str(fixtures["water_depth"])) == DEPTH_M


def test_water_depth_searches_a_directory_of_tiles(fixtures):
    directory = Path(fixtures["water_depth"]).parent
    assert water_depth_from_location(LATITUDE, LONGITUDE, waterDepthFilePath=directory) == DEPTH_M


def test_distance_to_coast_from_a_given_raster(fixtures):
    assert distance_to_coastline(LATITUDE, LONGITUDE, distancetoCoastFilePath=fixtures["coast_distance"]) == COAST_KM
    assert (
        distance_to_coastline(LATITUDE, LONGITUDE, distancetoCoastFilePath=str(fixtures["coast_distance"])) == COAST_KM
    )


def test_without_a_path_the_collection_is_read(default_route):
    assert water_depth_from_location(LATITUDE, LONGITUDE) == DEPTH_M
    assert distance_to_coastline(LATITUDE, LONGITUDE) == COAST_KM


def test_the_default_is_resolved_once_per_process(monkeypatch, fixtures):
    calls = []

    def paths(collection, **kwargs):
        calls.append(collection)
        return fixtures

    monkeypatch.setattr(data, "paths", paths)
    catalogued_input.cache_clear()
    try:
        for _ in range(3):
            water_depth_from_location(LATITUDE, LONGITUDE)
            distance_to_coastline(LATITUDE, LONGITUDE)
        assert calls == ["offshore_siting", "offshore_siting"]  # once per handle
    finally:
        catalogued_input.cache_clear()


def test_a_default_the_catalogue_cannot_provide_says_so(monkeypatch):
    def paths(collection, **kwargs):
        raise KeyError("unknown dataset 'gebco-2024'")

    monkeypatch.setattr(data, "paths", paths)
    catalogued_input.cache_clear()
    try:
        with pytest.raises(ResError, match="offshore_siting.*water_depth|water_depth.*offshore_siting") as failure:
            water_depth_from_location(LATITUDE, LONGITUDE)
        assert isinstance(failure.value.__cause__, KeyError)
    finally:
        catalogued_input.cache_clear()
