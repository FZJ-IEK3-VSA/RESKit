import importlib.resources


def test_default_pahts_exists():
    # get path to module
    assert importlib.resources.files("reskit").joinpath("default_paths.yaml").exists()
