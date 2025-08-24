from reskit.dac.workflows.workflows import lt_dac_era5_wenzel2025
from reskit import TEST_DATA
import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def dac_placements() -> pd.DataFrame:
    placements = pd.DataFrame(
        {
            "lon": [5.985195, 5.5, 5.5, 6.3],
            "lat": [50.797254, 50.794208, 50, 51],
            "capacity": [1, 10, 5, 5],  # capacity of the DAC plant to simulate in t/h
        }
    )
    return placements


def test_dac_era5_wenzel2025(dac_placements: pd.DataFrame):
    gen = lt_dac_era5_wenzel2025(
        placements=dac_placements, era5_path=TEST_DATA["era5-like"], model="LT_jajjawi"
    )
    assert np.all(
        np.isclose(
            gen.capacity_factor.mean(dim="time"),
            [1.08517377, 1.07983349, 1.05817328, 1.08709602],
        )
    )
    assert np.all(
        np.isclose(
            gen.capacity_factor.min(dim="time"),
            [1.04869404, 1.04749502, 1.02896871, 1.04984081],
        )
    )
    assert np.all(
        np.isclose(
            gen.capacity_factor.max(dim="time"),
            [1.13374767, 1.1314784, 1.0913033, 1.13395484],
        )
    )
    assert np.all(
        np.isclose(
            gen.capacity_factor.std(dim="time"),
            [0.02654445, 0.02320278, 0.01160597, 0.02647336],
        )
    )
