"""``reskit-data`` is ETHOS.Data's package command, bound to RESKit's collections file and bundle."""

import urllib.request

import pytest

ethos_data = pytest.importorskip("ethos_data")

from reskit import data


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """No settings file, a catalogue nobody can read, a staging root of its own and no network."""
    settings = tmp_path / "ethos-data.yaml"
    settings.write_text("{}\n")
    monkeypatch.setenv("ETHOS_DATA_CONFIG", str(settings))
    monkeypatch.setenv("ETHOS_DATA_CATALOG", str(tmp_path / "no-catalogue" / "datacatalog.json"))
    monkeypatch.setenv("ETHOS_DATA_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("ETHOS_STAGING_DIR", str(tmp_path / "staging"))
    for name in (data.CATALOG_ENV, "ETHOS_DATA_DOWNLOAD", "ETHOS_RESTRICTED_DIRS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)

    def no_network(*args, **kwargs):
        pytest.fail("the command attempted network access")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    return tmp_path


def test_help_and_config_work_offline(workspace, capsys):
    with pytest.raises(SystemExit) as stopped:
        data.main(["--help"])
    assert stopped.value.code == 0
    out = capsys.readouterr().out
    assert "usage: reskit-data" in out
    assert str(data.COLLECTIONS_FILE) in out
    for command in ("show", "fetch", "verify", "bundle", "staging", "config"):
        assert command in out
    assert data.main(["config", "show"]) == 0


def test_a_test_variant_is_shown_and_fetched_from_the_bundle_offline(workspace, capsys):
    assert data.main(["show", "offshore_siting", "--test"]) == 0
    out = capsys.readouterr().out
    assert "offshore_siting [test]" in out
    assert "->  reskit-test-data/gebco/water_depth_northsea.tif" in out

    assert data.main(["fetch", "offshore_siting", "--test", "--paths"]) == 0
    lines = dict(line.split("\t") for line in capsys.readouterr().out.splitlines())
    assert set(lines) == {"water_depth", "coast_distance"}
    for path in lines.values():
        assert path.startswith(str(data.BUNDLES[0]))


@pytest.mark.parametrize("command", ["link", "unlink", "materialize", "catalog"])
def test_shared_maintenance_stays_in_ethos_data(workspace, command):
    with pytest.raises(SystemExit) as stopped:
        data.main([command, "--help"])
    assert stopped.value.code == 2


def test_staging_is_available_without_loading_a_catalogue(workspace, capsys):
    source = workspace / "source"
    source.mkdir()
    (source / "value.txt").write_text("new data")
    assert data.main(["staging", "add", "trial", str(source), "--copy"]) == 0
    assert "reskit-data staging remove trial" in capsys.readouterr().out
    assert data.main(["staging", "list"]) == 0
    assert "trial" in capsys.readouterr().out
    assert data.main(["staging", "remove", "trial", "--force"]) == 0
    assert not (workspace / "staging" / "trial").exists()
    assert (source / "value.txt").read_text() == "new data"
