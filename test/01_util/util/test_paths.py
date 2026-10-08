"""Path arguments must be accepted as strings and as pathlib.Path objects.

``reskit.data`` returns ``pathlib.Path`` objects, while the workflows tell a file
path apart from a plain value by its type. These tests pin the equivalence of the
two spellings, so that a Path is never silently read as a value.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import reskit as rk
from reskit import data
from reskit.util.paths import as_path_string, is_path_like

# The fixtures as reskit.data hands them out, as Path objects. Each test below
# runs once with their string form and once with the Path itself.
FIXTURES = data.paths("test_suite")


@pytest.mark.parametrize("value", ["a/raster.tif", Path("a/raster.tif")])
def test_is_path_like_accepts_both_spellings(value):
    assert is_path_like(value)


@pytest.mark.parametrize("value", [1.0, 7, None, np.array([1.0, 2.0])])
def test_is_path_like_rejects_plain_values(value):
    assert not is_path_like(value)


@pytest.mark.parametrize("value", ["a/raster.tif", Path("a/raster.tif")])
def test_as_path_string_gives_a_string(value):
    """Both spellings give the same string, written with the platform separator."""
    as_string = as_path_string(value)

    assert isinstance(as_string, str)
    assert Path(as_string) == Path("a/raster.tif")


def test_weather_source_reads_a_path_directory():
    """A directory given as a Path must be scanned like the same string."""
    from_string = rk.weather.MerraSource(str(FIXTURES["merra"]), bounds=[5, 49, 7, 52], verbose=False)
    from_path = rk.weather.MerraSource(FIXTURES["merra"], bounds=[5, 49, 7, 52], verbose=False)

    assert from_path.variables.equals(from_string.variables)


def test_wind_workflow_accepts_path_inputs():
    """The rasters and the weather folder of a workflow may be Path objects."""
    placements = pd.read_csv(FIXTURES["turbine_placements"])
    arguments = dict(
        era5_path=str(FIXTURES["era5"]),
        gwa_100m_path=str(FIXTURES["gwa_100m"]),
        height_scaling_data={
            50: str(FIXTURES["gwa_50m"]),
            200: str(FIXTURES["gwa_200m"]),
        },
    )
    as_paths = dict(
        era5_path=Path(arguments["era5_path"]),
        gwa_100m_path=Path(arguments["gwa_100m_path"]),
        height_scaling_data={height: Path(fp) for height, fp in arguments["height_scaling_data"].items()},
    )

    from_strings = rk.wind.wind_era5_PenaSanchezDunkelWinklerEtAl2025(placements=placements, **arguments)
    from_paths = rk.wind.wind_era5_PenaSanchezDunkelWinklerEtAl2025(placements=placements, **as_paths)

    assert np.allclose(from_paths["capacity_factor"].values, from_strings["capacity_factor"].values)


def test_solar_workflow_accepts_path_inputs():
    """The long-run-average rasters of a solar workflow may be Path objects."""
    placements = pd.read_csv(FIXTURES["module_placements"])
    arguments = dict(
        era5_path=str(FIXTURES["era5"]),
        global_solar_atlas_ghi_path=str(FIXTURES["gsa_ghi"]),
        global_solar_atlas_dni_path=str(FIXTURES["gsa_dni"]),
    )

    from_strings = rk.solar.openfield_pv_era5(placements=placements, **arguments)
    from_paths = rk.solar.openfield_pv_era5(placements=placements, **{k: Path(v) for k, v in arguments.items()})

    assert np.allclose(from_paths["capacity_factor"].values, from_strings["capacity_factor"].values)
