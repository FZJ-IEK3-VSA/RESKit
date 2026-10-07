"""RESKit ships the reskit-test-data fixtures as an ETHOS.Data bundle and reads them from it first."""

import shutil
import urllib.request
import warnings

import pytest

ethos_data = pytest.importorskip("ethos_data")

import reskit
from reskit import data

BUNDLE = data.BUNDLES[0]


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """A fresh handle with no settings file, a catalogue nobody can read and no network."""
    settings = tmp_path / "ethos-data.yaml"
    settings.write_text("{}\n")
    monkeypatch.setenv("ETHOS_DATA_CONFIG", str(settings))
    monkeypatch.setenv("ETHOS_DATA_CATALOG", str(tmp_path / "no-catalogue" / "datacatalog.json"))
    monkeypatch.setenv("ETHOS_DATA_DIR", str(tmp_path / "cache"))
    for name in (data.CATALOG_ENV, "ETHOS_DATA_DOWNLOAD", "ETHOS_RESTRICTED_DIRS", "ETHOS_STAGING_DIR"):
        monkeypatch.delenv(name, raising=False)

    def refuse(*args, **kwargs):
        pytest.fail("the bundled fixtures were not read from the bundle")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    data.handle.cache_clear()
    data._legacy_test_data.cache_clear()
    yield tmp_path
    data.handle.cache_clear()
    data._legacy_test_data.cache_clear()


def test_the_shipped_bundle_verifies_offline(isolated):
    """Every file, description and licence document matches bundle.json, and nothing is unrecorded."""
    assert data.main(["bundle", "verify", str(BUNDLE)]) == 0


def test_the_bundle_holds_the_whole_fixture_family(isolated):
    bundle = ethos_data.load_bundle(BUNDLE)
    assert set(bundle.manifest.families) == {"reskit-test-data"}
    assert all(name.startswith("reskit-test-data/") for name in bundle.names())
    assert set(data.fetch("test_suite")) == set(bundle.resources)


#: The datasets the bundle may hold ahead of the catalogue: placements holds the 2025
#: Bulawayo tables until the catalogue takes them in. Drop the entry once
#: `reskit-data bundle update` has recorded the release that holds them.
AHEAD = {"reskit-test-data/placements"}


def test_the_bundle_is_ahead_of_the_catalogue_only_where_expected(isolated):
    """A bundle ahead of the catalogue warns in every process: nothing but AHEAD may be."""
    assert set(ethos_data.load_bundle(BUNDLE).ahead()) <= AHEAD
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data.paths("test_suite")
    for warning in caught:
        if issubclass(warning.category, ethos_data.BundleAlignmentWarning):
            assert all(name in str(warning.message) for name in ethos_data.load_bundle(BUNDLE).ahead())


def test_a_changed_fixture_is_an_error_never_a_download(isolated, monkeypatch):
    copy = isolated / "bundle"
    shutil.copytree(BUNDLE, copy)
    edited = copy / "data" / "reskit-test-data" / "placements" / "turbine_placements.csv"
    edited.write_text("edited by hand\n")
    monkeypatch.setattr(data, "BUNDLES", (copy,))
    data.handle.cache_clear()

    with pytest.raises(ethos_data.BundleError, match="bundle update"):
        data.paths("example_placements")
    assert edited.read_text() == "edited by hand\n"


def test_the_download_switch_reads_the_fixtures_through_the_catalogue(isolated, monkeypatch):
    """$ETHOS_DATA_DOWNLOAD sends the bundled files to the catalogue route -- here one nobody can read."""
    monkeypatch.setenv("ETHOS_DATA_DOWNLOAD", "1")
    data.handle.cache_clear()
    with pytest.raises(ethos_data.CatalogUnavailable):
        data.paths("test_suite")


def test_the_reskit_catalogue_override_wins_over_the_ethos_data_one(isolated, monkeypatch):
    monkeypatch.setenv(data.CATALOG_ENV, str(isolated / "reskit-catalogue.json"))
    data.handle.cache_clear()
    settings = data.handle().settings
    assert settings.catalog == str(isolated / "reskit-catalogue.json")
    assert settings.catalog_source == "explicit argument"


class TestTheDeprecatedTestData:
    """``reskit.TEST_DATA`` keeps its keys until RESKit 1.0.0, read through the bundle."""

    def test_it_warns(self, isolated):
        with pytest.warns(DeprecationWarning, match="reskit.data.paths"):
            reskit.TEST_DATA

    def test_it_maps_the_old_keys_to_the_bundled_files(self, isolated):
        fixtures = data.paths("test_suite")
        with pytest.warns(DeprecationWarning):
            test_data = reskit.TEST_DATA
        assert test_data["era5"] == test_data["era5-like"] == str(fixtures["era5"])
        assert test_data["gwa100-like.tif"] == str(fixtures["gwa_100m"])
        assert test_data["merra2/merged/merra-like.nc4"] == str(fixtures["merra_merged"])
        assert test_data["era5/2m_temperature.nc"] == str(fixtures["era5"] / "2m_temperature.nc")

    def test_an_ambiguous_file_name_says_where_it_is(self, isolated):
        with pytest.warns(DeprecationWarning):
            test_data = reskit.TEST_DATA
        with pytest.raises(KeyError, match="era5 and era5-csp"):
            test_data["2m_temperature.nc"]
        with pytest.raises(KeyError, match="not a test fixture"):
            test_data["no-such-fixture.tif"]
