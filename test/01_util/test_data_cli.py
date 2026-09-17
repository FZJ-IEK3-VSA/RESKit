"""RESKit's thin CLI owns collection workflows and development staging."""

import hashlib
import json
import urllib.request

import pytest

ethos_data = pytest.importorskip("ethos_data")

from reskit import data


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(ethos_data.config, "load_config", lambda: ({}, {}))
    for name in (
        "ETHOS_DATA_CATALOG",
        "RESKIT_DATA_CATALOG",
        "ETHOS_RESTRICTED_DIR",
        "ETHOS_SKIP_UNAVAILABLE",
        "ETHOS_PUBLICATION_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ETHOS_DATA_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("ETHOS_STAGING_DIR", str(tmp_path / "staging"))
    monkeypatch.chdir(tmp_path)

    def no_network(*args, **kwargs):
        pytest.fail("the wrapper attempted network access")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    return tmp_path


def test_help_and_config_work_with_the_shipped_file_offline(workspace, capsys):
    with pytest.raises(SystemExit) as stopped:
        data.main(["--help"])
    assert stopped.value.code == 0
    out = capsys.readouterr().out
    assert "usage: reskit-data" in out
    assert "reskit-data fetch all --test --plan" in out
    assert "staging" in out and "bundle" in out
    assert "--collections" not in out
    assert data.main(["config", "show"]) == 0


@pytest.mark.parametrize(
    ("retired", "replacement"),
    [
        ("list", "reskit-data show"),
        ("info", "reskit-data show <collection>"),
        ("plan", "reskit-data fetch <collection> --plan"),
        ("paths", "reskit-data fetch <collection> --paths"),
        ("path", "ethos-data fetch <key>"),
        ("ls", "ethos-data ls [<key>]"),
    ],
)
def test_the_retired_commands_name_what_replaces_them(
    workspace, capsys, retired, replacement
):
    """reskit-data works in collections; a bare catalogue key is ethos-data's."""
    assert data.main([retired, "onshore_wind"]) == 2
    assert replacement in capsys.readouterr().err


def test_staging_is_available_without_loading_a_catalogue(workspace, capsys):
    source = workspace / "source"
    source.mkdir()
    (source / "value.txt").write_text("new data")
    assert data.main(["staging", "add", "trial", str(source), "--copy"]) == 0
    assert "reskit-data staging remove trial" in capsys.readouterr().out
    assert data.main(["staging", "list", "--new-only"]) == 0
    assert "trial" in capsys.readouterr().out
    assert data.main(["staging", "remove", "trial", "--force"]) == 0
    assert not (workspace / "staging/trial").exists()
    assert (source / "value.txt").read_text() == "new data"


@pytest.mark.parametrize("command", ["link", "unlink", "materialize", "catalog"])
def test_shared_maintenance_stays_in_ethos_data(workspace, command):
    with pytest.raises(SystemExit) as stopped:
        data.main([command, "--help"])
    assert stopped.value.code == 2


def test_collection_fetch_and_catalogue_overrides(workspace, monkeypatch, capsys):
    file = workspace / "collections.yaml"
    file.write_text(
        "catalog: missing.json\ncollections:\n  example:\n"
        "    include:\n      - dataset: trial\n    paths:\n      input: trial/value.txt\n"
    )
    monkeypatch.setattr(data, "COLLECTIONS_FILE", file)
    cached = workspace / "cache/trial/value.txt"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"data")
    (workspace / "datapackage.json").write_text(
        json.dumps(
            {
                "name": "trial",
                "resources": [
                    {
                        "name": "value",
                        "path": "value.txt",
                        "bytes": 4,
                        "hash": "sha256:" + hashlib.sha256(b"data").hexdigest(),
                    }
                ],
            }
        )
    )
    catalog = workspace / "datacatalog.json"
    catalog.write_text(
        json.dumps(
            {
                "ethos:publication_url": "https://example.invalid/data",
                "datasets": [{"name": "trial", "path": "datapackage.json", "ethos:license_status": "resolved"}],
            }
        )
    )
    empty = workspace / "empty.json"
    empty.write_text('{"datasets": []}')
    monkeypatch.setenv("ETHOS_DATA_CATALOG", str(empty))
    monkeypatch.setenv("RESKIT_DATA_CATALOG", str(catalog))
    assert data.main(["fetch", "example"]) == 0
    assert data.main(["fetch", "example", "--paths"]) == 0
    assert f"input\t{cached}" in capsys.readouterr().out
    assert data.main(["show", "example"]) == 0
    assert "input  ->  trial/value.txt" in capsys.readouterr().out
    assert data.main(["--catalog", str(empty), "fetch", "example"]) == 2
    assert "trial" in capsys.readouterr().err


def test_staged_data_is_fetched_through_reskit_collections(workspace, monkeypatch, capsys):
    catalog = workspace / "empty.json"
    catalog.write_text('{"datasets": []}')
    monkeypatch.setenv("RESKIT_DATA_CATALOG", str(catalog))
    file = workspace / "collections.yaml"
    file.write_text(
        "collections:\n  experiment:\n    include:\n      - dataset: trial\n    paths:\n      input: trial/value.txt\n"
    )
    monkeypatch.setattr(data, "COLLECTIONS_FILE", file)
    source = workspace / "source"
    source.mkdir()
    (source / "value.txt").write_text("development")
    assert data.main(["staging", "add", "trial", str(source), "--copy"]) == 0
    capsys.readouterr()
    with pytest.warns(UserWarning, match="staging"):
        assert data.main(["fetch", "experiment", "--paths"]) == 0
    assert capsys.readouterr().out.strip() == f"input\t{workspace / 'staging/trial/value.txt'}"
    assert not (workspace / "cache/trial").exists()
