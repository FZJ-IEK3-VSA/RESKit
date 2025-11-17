from reskit.cooling_heating.workflows.workflows import air_cooling_wenzel2025
from reskit.cooling_heating.workflows.workflows import air_source_heat_pump
from reskit import TEST_DATA
import pytest
import numpy as np
import geokit as gk
import os
import pandas as pd
import reskit.weather as rk_weather


@pytest.fixture
def placements() -> pd.DataFrame:
    placements = pd.DataFrame(
        {
            "lon": [5.5, 5.994685, 6.8],
            "lat": [50.797254, 50.794208, 49.5],
            "capacity": [4000, 4000, 4000],
        }
    )
    return placements


def test_air_cooling_wenzel2025(placements: pd.DataFrame):
    gen = air_cooling_wenzel2025(
        placements=placements,
        era5_path=TEST_DATA["era5-like"],
        temperatureCoolant=40,
        designTemperature=20,
    )
    assert np.all(
        np.isclose(
            gen.conversion_factor_electricity.mean(dim="time"),
            [-0.01082566, -0.0107924, -0.01012263],
        )
    )
    assert np.all(
        np.isclose(
            gen.conversion_factor_electricity.min(dim="time"),
            [-0.01364287, -0.01305147, -0.01345005],
        )
    )
    assert np.all(
        np.isclose(
            gen.conversion_factor_electricity.max(dim="time"),
            [-0.00958934, -0.00962442, -0.00878508],
        )
    )
    assert np.all(
        np.isclose(
            gen.conversion_factor_electricity.std(dim="time"),
            [0.00075725, 0.00066738, 0.00079315],
        )
    )


def test_air_source_heat_pump(placements: pd.DataFrame):
    gen = air_source_heat_pump(placements=placements, era5_path=TEST_DATA["era5-like"])
    assert np.all(np.isclose(gen.COP.mean(dim="time"), [1.90161814, 1.90016073, 1.86173579]))
    assert np.all(np.isclose(gen.COP.min(dim="time"), [1.83157454, 1.83375042, 1.778988]))
    assert np.all(np.isclose(gen.COP.max(dim="time"), [2.03473257, 2.01022362, 2.02689851]))
    assert np.all(np.isclose(gen.COP.std(dim="time"), [0.03941226, 0.03536727, 0.04394226]))
