import shutil
from pathlib import Path

import numpy as np
import pytest

from reskit.wind.core import turbine_library as module
from reskit.wind.core.turbine_library import BUNDLED_TURBINES, turbine_library


@pytest.fixture
def shipped_library_afterwards():
    """A test may select another library; the shipped one must be in use again afterwards."""
    yield
    turbine_library(BUNDLED_TURBINES)


def test_TurbineLibrary():
    turb = turbine_library().loc["E115_2500"]

    assert turb.Manufacturer == "Enercon"
    assert turb.Capacity == 2500
    assert turb.Usage == "Onshore"
    assert (turb.Hub_Height == [92.5, 149.0]).all()
    assert turb.Rotordiameter == 115
    assert np.isclose(turb.PowerCurve.capacity_factor.sum(), 18.2798)


def test_the_shipped_library_is_read_by_default():
    assert Path(module._selected).resolve() == Path(BUNDLED_TURBINES).resolve()
    assert len(turbine_library()) > 100
    assert turbine_library() is turbine_library()  # parsed once per process


def test_a_directory_becomes_the_library_in_use(tmp_path, shipped_library_afterwards):
    """A path selects that library for every later call, as the retired default_paths.yaml key did."""
    shutil.copy(Path(BUNDLED_TURBINES) / "E115_2500.csv", tmp_path / "E115_2500.csv")

    selected = turbine_library(tmp_path)  # a pathlib.Path is accepted

    assert list(selected.index) == ["E115_2500"]
    assert turbine_library() is selected
    assert turbine_library(str(tmp_path)) is selected  # the same directory is not parsed again

    shipped = turbine_library(BUNDLED_TURBINES)
    assert "E115_2500" in shipped.index and len(shipped) > 100
    assert turbine_library() is shipped


def test_a_directory_without_turbines_is_an_error(tmp_path, shipped_library_afterwards):
    with pytest.raises(FileNotFoundError, match="No turbine definition files"):
        turbine_library(tmp_path)
