import os

import geokit as gk
import numpy as np
import pandas as pd
import pytest

import reskit.weather as rk_weather
from reskit import TEST_DATA
from reskit.cooling_heating.workflows.workflows import (
    air_cooling_wenzel2025,
    air_source_heat_pump,
    evaporative_cooling_wortmann2025,
)


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
        temperature_coolant=40,
        design_temperature=20,
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


def test_evaporative_cooling_wortmann2025(placements: pd.DataFrame):
    gen = evaporative_cooling_wortmann2025(
        placements=placements,
        era5_path=TEST_DATA["era5-like"],
        temperature_coolant=80,
        heat_transfer_delta=10,
        efficiency_cooling_tower=0.65,
    )
    assert np.all(
        np.isclose(
            gen.conversion_factor_water.mean(dim="time"),
            [-1.14410054, -1.14399711, -1.10596042],
        )
    )
    assert np.all(
        np.isclose(
            gen.conversion_factor_water.min(dim="time"),
            [-1.24571965, -1.23125088, -1.2329343],
        )
    )
    assert np.all(
        np.isclose(
            gen.conversion_factor_water.max(dim="time"),
            [-1.08438261, -1.08912747, -1.02922007],
        )
    )
    assert np.all(
        np.isclose(
            gen.conversion_factor_water.std(dim="time"),
            [0.03379537, 0.03082596, 0.03771017],
        )
    )


def test_air_source_heat_pump(placements: pd.DataFrame):
    gen = air_source_heat_pump(placements=placements, era5_path=TEST_DATA["era5-like"])
    assert np.all(np.isclose(gen.COP.mean(dim="time"), [1.90161814, 1.90016073, 1.86173579]))
    assert np.all(np.isclose(gen.COP.min(dim="time"), [1.83157454, 1.83375042, 1.778988]))
    assert np.all(np.isclose(gen.COP.max(dim="time"), [2.03473257, 2.01022362, 2.02689851]))
    assert np.all(np.isclose(gen.COP.std(dim="time"), [0.03941226, 0.03536727, 0.04394226]))
