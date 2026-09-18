"""The collections RESKit ships name what the code reads, and the fixtures answer offline."""

import urllib.request

import pytest

ethos_data = pytest.importorskip("ethos_data")

from reskit import data
from reskit.wind import wind_era5_PenaSanchezDunkelWinklerEtAl2025


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("the collections file should be answerable from the bundle, without network access")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)


def test_the_shipped_file_defines_the_collections_the_code_relies_on():
    definitions = data._definitions()
    names = set(definitions.names())
    assert {"offshore_siting", "turbine_library", "wind_era5_PenaSanchezDunkelWinklerEtAl2025", "test_suite"} <= names


def test_turbine_library_names_the_licensed_library():
    named = data._definitions().named_keys("turbine_library", test=False)
    assert named == {"turbines": "reskit-turbine-library"}


def test_offshore_siting_test_variant_is_bundled():
    inputs = data.paths("offshore_siting", test=True)
    assert set(inputs) == {"water_depth", "coast_distance"}
    assert all(path.is_file() for path in inputs.values())


def test_wind_workflow_collection_is_bundled_and_has_matching_variants():
    """The workflow's own name resolves its inputs offline, with stable handles."""
    name = wind_era5_PenaSanchezDunkelWinklerEtAl2025.__name__
    definitions = data._definitions()
    inputs = data.paths(name, test=True)
    expected = {"era5", "gwa_100m", "gwa_50m", "gwa_200m"}
    assert set(inputs) == expected
    assert set(definitions.named_keys(name, test=False)) == expected
    assert inputs["era5"].is_dir()
    assert all(inputs[key].is_file() for key in expected - {"era5"})
    assert all(path.is_relative_to(data.BUNDLE) for path in inputs.values())
