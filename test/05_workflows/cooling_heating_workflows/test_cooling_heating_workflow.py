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
        designTemperature=20)
    assert np.all(np.isclose(gen.capacity_factor.mean(dim='time'), [1.13578348, 1.13615734, 1.14453698]))
    assert np.all(np.isclose(gen.capacity_factor.min(dim='time'), [1.1032281 , 1.10971561, 1.10532639]))
    assert np.all(np.isclose(gen.capacity_factor.max(dim='time'), [1.15116857, 1.15071551, 1.16178569]))
    assert np.all(np.isclose(gen.capacity_factor.std(dim='time'), [0.00909699, 0.00808064, 0.00976524]))

def test_air_source_heat_pump(placements: pd.Dataframe):
    gen = air_source_heat_pump(
    placements=placements,
    era5_path=TEST_DATA["era5-like"])
    assert np.all(np.isclose(gen.COP.mean(dim='time'), [1.90161814, 1.90016073, 1.86173579]))
    assert np.all(np.isclose(gen.COP.min(dim='time'), [1.83157454, 1.83375042, 1.778988]))
    assert np.all(np.isclose(gen.COP.max(dim='time'), [2.03473257, 2.01022362, 2.02689851]))
    assert np.all(np.isclose(gen.COP.std(dim='time'), [0.03941226, 0.03536727, 0.04394226]))