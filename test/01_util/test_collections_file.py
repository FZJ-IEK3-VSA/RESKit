"""The collections RESKit ships name what the code reads, and the fixtures answer offline."""

import inspect
import re
import urllib.request

import pytest

ethos_data = pytest.importorskip("ethos_data")

import reskit as rk
from reskit import data
from reskit.wind import wind_era5_PenaSanchezDunkelWinklerEtAl2025

# A workflow collection is named after the function it feeds.
WORKFLOWS = {
    function.__name__: function
    for function in (
        rk.wind.wind_era5_PenaSanchezDunkelWinklerEtAl2025,
        rk.wind.onshore_wind_merra_ryberg2019_europe,
        rk.wind.offshore_wind_merra_caglayan2019,
        rk.solar.openfield_pv_merra_ryberg2019,
        rk.solar.openfield_pv_sarah_unvalidated,
        rk.solar.openfield_pv_era5,
        rk.dac.lt_dac_era5_wenzel2025,
        rk.dac.ht_dac_era5_wenzel2025,
        rk.cooling_heating.air_cooling_wenzel2025,
        rk.cooling_heating.evaporative_cooling_wortmann2025,
        rk.cooling_heating.air_source_heat_pump,
    )
}
# The collections whose test variant -- or whose whole selection -- the bundle holds.
BUNDLED = sorted(WORKFLOWS) + ["offshore_siting", "test_suite", "example_placements"]


@pytest.fixture
def offline(tmp_path, monkeypatch):
    """A fresh handle whose catalogue cannot be read and which has no network.

    Whatever the bundle holds must still be answered: from the bundle, with no
    catalogue index read and no download.
    """
    settings = tmp_path / "ethos-data.yaml"
    settings.write_text("{}\n")
    monkeypatch.setenv("ETHOS_DATA_CONFIG", str(settings))
    monkeypatch.setenv("ETHOS_DATA_CATALOG", str(tmp_path / "no-catalogue" / "datacatalog.json"))
    monkeypatch.setenv("ETHOS_DATA_DIR", str(tmp_path / "cache"))
    for name in (data.CATALOG_ENV, "ETHOS_DATA_DOWNLOAD", "ETHOS_RESTRICTED_DIRS", "ETHOS_STAGING_DIR"):
        monkeypatch.delenv(name, raising=False)

    def refuse(*args, **kwargs):
        pytest.fail("the collections file should be answerable from the bundle, without network access")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    data.handle.cache_clear()
    yield
    data.handle.cache_clear()


def test_the_shipped_file_defines_the_collections_the_code_relies_on():
    names = set(data.handle().names())
    assert {"offshore_siting", "turbine_library", "test_suite", "example_placements"} <= names
    assert set(WORKFLOWS) <= names


def test_turbine_library_names_the_licensed_library():
    assert data.handle().named_keys("turbine_library") == {"turbines": "reskit-turbine-library"}


@pytest.mark.parametrize("name", BUNDLED)
def test_the_test_inputs_are_answered_from_the_bundle_offline(offline, name):
    """The catalogue named here cannot be read, so any answer came from the bundle alone."""
    inputs = data.paths(name, test=True)
    assert inputs, f"{name} names no inputs"
    for handle, path in inputs.items():
        assert path.exists(), f"{name}: {handle} -> {path}"
        assert path.is_relative_to(data.BUNDLES[0]), f"{name}: {handle} is not read from the bundle"


@pytest.mark.parametrize("name", sorted(WORKFLOWS))
def test_a_workflow_collection_names_its_handles_after_the_arguments(name):
    """``era5`` feeds ``era5_path``; a ``gwa_<height>m`` handle may feed ``height_scaling_data``."""
    parameters = inspect.signature(WORKFLOWS[name]).parameters
    for handle in data.handle().named_keys(name, test=True):
        height_scaling = "height_scaling_data" in parameters and re.fullmatch(r"gwa_\d+m", handle)
        assert f"{handle}_path" in parameters or height_scaling, f"{name}: no argument takes {handle!r}"


@pytest.mark.parametrize("name", sorted(WORKFLOWS))
def test_variants_offer_the_same_handles(name):
    handle = data.handle()
    variants = handle.variants(name)
    assert "test" in variants
    if "full" in variants:
        assert set(handle.named_keys(name, test=True)) == set(handle.named_keys(name, test=False))


def test_a_collection_without_full_inputs_refuses_the_full_variant(offline):
    """Silently running on the fixtures instead would be worse than no answer."""
    assert data.handle().variants("openfield_pv_era5") == ("test",)
    with pytest.raises(ethos_data.CollectionError, match="has no full variant"):
        data.paths("openfield_pv_era5")


def test_wind_workflow_collection_is_bundled_and_has_matching_variants(offline):
    """The workflow's own name resolves its inputs offline, with stable handles."""
    name = wind_era5_PenaSanchezDunkelWinklerEtAl2025.__name__
    inputs = data.paths(name, test=True)
    expected = {"era5", "gwa_100m", "gwa_50m", "gwa_200m"}
    assert set(inputs) == expected
    assert set(data.handle().named_keys(name, test=False)) == expected
    assert inputs["era5"].is_dir()
    assert all(inputs[key].is_file() for key in expected - {"era5"})


def test_the_fixture_handles_name_what_the_tests_read(offline):
    fixtures = data.paths("test_suite")
    assert fixtures["era5"].is_dir()
    assert fixtures["merra"].is_dir()
    assert fixtures["gwa_100m"].name == "gwa100-like.tif"
    assert fixtures["aachen"].name == "aachenShapefile.shp"
    # A shapefile handle brings its sidecars along.
    assert all(fixtures["aachen"].with_suffix(suffix).is_file() for suffix in (".dbf", ".shx", ".prj"))
