"""The collections RESKit ships name what the code reads, and the fixtures answer offline."""

import urllib.request

import pytest

ethos_data = pytest.importorskip("ethos_data")

from reskit import data


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("the collections file should be answerable from the bundle, without network access")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)


def test_the_shipped_file_defines_the_collections_the_code_relies_on():
    definitions = data._definitions()
    names = set(definitions.names())
    assert {"offshore_siting", "turbine_library", "onshore_wind", "test_suite"} <= names


def test_turbine_library_names_the_licensed_library():
    named = data._definitions().named_keys("turbine_library", test=False)
    assert named == {"turbines": "reskit-turbine-library"}


def test_offshore_siting_test_variant_is_bundled():
    inputs = data.paths("offshore_siting", test=True)
    assert set(inputs) == {"water_depth", "coast_distance"}
    assert all(path.is_file() for path in inputs.values())
