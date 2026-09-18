"""RESKit answers from its bundled fixtures first; ``download`` sends a request to the catalogue instead."""

import hashlib
import json
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

ethos_data = pytest.importorskip("ethos_data")

from reskit import data

LICENCE = b"You may use these bytes.\n"
FILES = {
    "value.txt": b"bundled value",
    "folder/a.txt": b"bundled a",
    "folder/b.txt": b"bundled b",
    "other.txt": b"only in the catalogue",
}
COLLECTIONS = """\
catalog: {catalog}
collections:
  example:
    include:
      - dataset: trial
        files: ["value.txt", "folder/*"]
    paths:
      input: trial/value.txt
      folder: trial/folder
  sized:
    test:
      include:
        - dataset: trial
          files: ["value.txt"]
      paths:
        input: trial/value.txt
    full:
      include:
        - dataset: trial
          files: ["other.txt"]
      paths:
        input: trial/other.txt
  unnamed:
    include:
      - dataset: trial
        files: ["value.txt"]
"""


def _sha(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _reset():
    """Forget every per-process cache; a test's monkeypatched stand-in has none to clear."""
    for name in ("handle", "_definitions", "_bundle", "bundled"):
        cached = getattr(data, name)
        if hasattr(cached, "cache_clear"):
            cached.cache_clear()


def _write_catalogue(root: Path) -> Path:
    """A local, public catalogue with one dataset ``trial`` holding FILES."""
    dataset = root / "catalogue" / "datasets" / "trial"
    (dataset / "licenses").mkdir(parents=True)
    (dataset / "licenses" / "terms.txt").write_bytes(LICENCE)
    (dataset / "datapackage.json").write_text(
        json.dumps(
            {
                "name": "trial",
                "ethos:access": "public",
                "ethos:visibility": "public",
                "licenses": [
                    {
                        "name": "trial-terms",
                        "path": "https://example.invalid/terms",
                        "ethos:document": "licenses/terms.txt",
                        "ethos:document_sha256": hashlib.sha256(LICENCE).hexdigest(),
                    }
                ],
                "resources": [
                    {
                        "name": name.replace("/", "-").replace(".", "-"),
                        "path": name,
                        "bytes": len(raw),
                        "hash": _sha(raw),
                        "mediatype": "text/plain",
                    }
                    for name, raw in FILES.items()
                ],
            }
        )
    )
    catalogue = root / "catalogue" / "datacatalog.json"
    catalogue.write_text(
        json.dumps(
            {
                "name": "trial-catalogue",
                "ethos:catalog_role": "published",
                "ethos:publication_url": "https://example.invalid/data",
                "datasets": [
                    {
                        "name": "trial",
                        "path": "datasets/trial/datapackage.json",
                        "ethos:access": "public",
                        "ethos:visibility": "public",
                        "ethos:remote_prefix": "trial",
                        "ethos:license_status": "resolved",
                    }
                ],
            }
        )
    )
    return catalogue


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A bundle of ``example`` beside a catalogue whose files are all already in the shared cache.

    Every route can therefore be answered without the network, and the path a
    call returns says which route answered: under ``bundle/`` or under ``cache/``.
    """
    monkeypatch.setattr(ethos_data.config, "load_config", lambda: ({}, {}))
    for name in (
        "ETHOS_DATA_CATALOG",
        "ETHOS_RESTRICTED_DIR",
        "ETHOS_STAGING_DIR",
        "ETHOS_SKIP_UNAVAILABLE",
        "ETHOS_PUBLICATION_URL",
        data.DOWNLOAD_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    cache = tmp_path / "cache"
    monkeypatch.setenv("ETHOS_DATA_DIR", str(cache))

    source = tmp_path / "source"
    for name, raw in FILES.items():
        for base in (source, cache / "trial"):
            (base / name).parent.mkdir(parents=True, exist_ok=True)
            (base / name).write_bytes(raw)
    catalogue = _write_catalogue(tmp_path)
    collections = tmp_path / "collections.yaml"
    collections.write_text(COLLECTIONS.format(catalog=catalogue.as_posix()))
    monkeypatch.setenv(data.CATALOG_ENV, catalogue.as_posix())

    bundle = tmp_path / "bundle"
    ethos_data.export_bundle(
        collections, ["example"], bundle, catalog=catalogue.as_posix(), dataset_roots={"trial": source}
    )

    monkeypatch.setattr(data, "COLLECTIONS_FILE", collections)
    monkeypatch.setattr(data, "BUNDLE", bundle)
    _reset()

    def no_network(*args, **kwargs):
        pytest.fail("the data module attempted network access")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    yield SimpleNamespace(root=tmp_path, bundle=bundle, cache=cache)
    _reset()


def _catalogue_is_off_limits(monkeypatch):
    monkeypatch.setattr(data, "handle", lambda: pytest.fail("the catalogue was consulted"))


def test_bundled_keys_are_answered_without_the_catalogue(workspace, monkeypatch):
    _catalogue_is_off_limits(monkeypatch)
    assert data.path("trial/value.txt") == workspace.bundle / "data" / "trial" / "value.txt"
    assert data.path("trial/value.txt").read_bytes() == FILES["value.txt"]
    assert data.directory("trial/folder") == workspace.bundle / "data" / "trial" / "folder"
    assert data.directory("trial") == workspace.bundle / "data" / "trial"
    assert sorted(data.bundled()) == ["trial/folder/a.txt", "trial/folder/b.txt", "trial/value.txt"]


def test_named_paths_are_answered_without_the_catalogue(workspace, monkeypatch):
    _catalogue_is_off_limits(monkeypatch)
    named = data.paths("example")
    assert named == {
        "input": workspace.bundle / "data" / "trial" / "value.txt",
        "folder": workspace.bundle / "data" / "trial" / "folder",
    }
    assert named.collection == "example"
    assert data.paths("sized", test=True)["input"] == workspace.bundle / "data" / "trial" / "value.txt"


def test_an_exported_collection_is_fetched_without_the_catalogue(workspace, monkeypatch):
    _catalogue_is_off_limits(monkeypatch)
    files = data.fetch("example")
    assert sorted(files) == ["trial/folder/a.txt", "trial/folder/b.txt", "trial/value.txt"]
    assert files.one("a.txt") == workspace.bundle / "data" / "trial" / "folder" / "a.txt"
    assert files.named["folder"] == workspace.bundle / "data" / "trial" / "folder"


def test_other_collections_are_resolved_in_the_catalogue_and_served_from_the_bundle(workspace):
    files = data.fetch("sized", test=True)
    assert dict(files) == {"trial/value.txt": workspace.bundle / "data" / "trial" / "value.txt"}
    assert files.named["input"] == workspace.bundle / "data" / "trial" / "value.txt"
    assert dict(data.fetch("unnamed")) == {"trial/value.txt": workspace.bundle / "data" / "trial" / "value.txt"}
    assert not data.fetch("unnamed").named


def test_what_the_bundle_lacks_comes_from_the_catalogue(workspace):
    assert data.path("trial/other.txt") == workspace.cache / "trial" / "other.txt"
    assert data.paths("sized")["input"] == workspace.cache / "trial" / "other.txt"
    assert data.fetch("sized")["trial/other.txt"] == workspace.cache / "trial" / "other.txt"
    with pytest.raises(ethos_data.CollectionError, match="no named paths"):
        data.paths("unnamed")
    with pytest.raises(KeyError):
        data.path("trial/missing.txt")


@pytest.mark.parametrize("how", ["argument", "environment"])
def test_download_sends_every_request_to_the_catalogue(workspace, monkeypatch, how):
    kwargs = {}
    if how == "argument":
        kwargs = {"download": True}
    else:
        monkeypatch.setenv(data.DOWNLOAD_ENV, "yes")
    trial = workspace.cache / "trial"
    assert data.path("trial/value.txt", **kwargs) == trial / "value.txt"
    assert data.directory("trial/folder", **kwargs) == trial / "folder"
    assert data.paths("example", **kwargs) == {"input": trial / "value.txt", "folder": trial / "folder"}
    assert data.fetch("example", **kwargs)["trial/folder/a.txt"] == trial / "folder" / "a.txt"


def test_the_download_argument_wins_over_the_environment(workspace, monkeypatch):
    monkeypatch.setenv(data.DOWNLOAD_ENV, "1")
    assert data.path("trial/value.txt", download=False) == workspace.bundle / "data" / "trial" / "value.txt"
    monkeypatch.setenv(data.DOWNLOAD_ENV, "off")
    assert data.path("trial/value.txt", download=True) == workspace.cache / "trial" / "value.txt"


def test_an_unreadable_download_setting_is_refused(workspace, monkeypatch):
    monkeypatch.setenv(data.DOWNLOAD_ENV, "maybe")
    with pytest.raises(ValueError, match=data.DOWNLOAD_ENV):
        data.path("trial/value.txt")


def test_an_altered_bundled_file_is_an_error_not_a_download(workspace):
    edited = workspace.bundle / "data" / "trial" / "value.txt"
    edited.write_bytes(b"edited by hand")
    _reset()
    with pytest.raises(ethos_data.BundleError, match="differ"):
        data.path("trial/folder/a.txt")
    assert edited.read_bytes() == b"edited by hand"


def test_without_a_bundle_everything_comes_from_the_catalogue(workspace, monkeypatch):
    monkeypatch.setattr(data, "BUNDLE", workspace.root / "nowhere")
    _reset()
    assert data.bundled() is None
    assert data.path("trial/value.txt") == workspace.cache / "trial" / "value.txt"
    assert data.paths("example")["folder"] == workspace.cache / "trial" / "folder"


def test_the_shipped_bundle_is_complete_and_verified(monkeypatch):
    """The copy in the repository is what the tests and examples run on, so it must verify offline."""
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("the shipped bundle needed the network"))
    _reset()
    bundle = data.BUNDLE
    assert (bundle / "bundle.json").is_file(), (
        f"{bundle} ships no bundle.json. Export it from the pinned catalogue with "
        "`reskit-data bundle export`; docs/how_to/get_input_data.md, 'Refresh the bundle', has the steps."
    )
    assert data.main(["bundle", "verify", str(bundle), "test_suite"]) == 0
    files = data.bundled()
    assert data.directory("reskit-test-data/era5") == bundle / "data" / "reskit-test-data" / "era5"
    assert data.path("reskit-test-data/placements/turbine_placements.csv").is_file()
    # Every fixture in the tree is catalogued: a file added by hand and never
    # described would otherwise pass the tests here and be missing everywhere else.
    root = bundle / "data"
    on_disk = {
        found.relative_to(root).as_posix()
        for found in root.rglob("*")
        if found.is_file() and found.name != "__init__.py" and found.suffix != ".pyc"
    }
    assert on_disk == set(files), sorted(on_disk ^ set(files))
    _reset()
