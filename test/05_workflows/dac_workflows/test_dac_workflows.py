import numpy as np
import pandas as pd
import pytest

from reskit import TEST_DATA
from reskit.dac.workflows.workflows import (
    ht_dac_era5_wenzel2025,
    lt_dac_era5_wenzel2025,
)


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


def test_lt_dac_era5_wenzel2025(dac_placements: pd.DataFrame):
    gen = lt_dac_era5_wenzel2025(placements=dac_placements, era5_path=TEST_DATA["era5-like"], model="LT_jajjawi")
    assert np.all(
        np.isclose(
            gen.capacity_factor.mean(dim="time"),
            [1.08590981, 1.08114436, 1.0581021, 1.08804831],
        )
    )
    assert np.all(
        np.isclose(
            gen.capacity_factor.min(dim="time"),
            [1.05074335, 1.0498154, 1.02946935, 1.05228196],
        )
    )
    assert np.all(
        np.isclose(
            gen.capacity_factor.max(dim="time"),
            [1.12960038, 1.12922649, 1.09326807, 1.12933755],
        )
    )
    assert np.all(
        np.isclose(
            gen.capacity_factor.std(dim="time"),
            [0.02602212, 0.02330062, 0.01166089, 0.02574228],
        )
    )


def test_ht_dac_era5_wenzel2025(dac_placements: pd.DataFrame):
    gen = ht_dac_era5_wenzel2025(placements=dac_placements, era5_path=TEST_DATA["era5-like"], model="HT_okosun")
    assert np.all(
        np.isclose(
            gen.capacity_factor.mean(dim="time"),
            [0.77772581, 0.78001443, 0.74575996, 0.78649318],
        )
    )
    assert np.all(
        np.isclose(
            gen.capacity_factor.min(dim="time"),
            [0.71635856, 0.71888185, 0.68504126, 0.72036108],
        )
    )
    assert np.all(
        np.isclose(
            gen.capacity_factor.max(dim="time"),
            [0.86194148, 0.87629404, 0.85932953, 0.87056588],
        )
    )
    assert np.all(
        np.isclose(
            gen.capacity_factor.std(dim="time"),
            [0.02720932, 0.02935947, 0.03245553, 0.02854373],
        )
    )


def test_lt_dac_rejects_an_unknown_fill_method(dac_placements: pd.DataFrame):
    # the docstring documents a NotImplementedError, the code used an assert
    with pytest.raises(NotImplementedError, match="bogus"):
        lt_dac_era5_wenzel2025(
            placements=dac_placements,
            era5_path=TEST_DATA["era5-like"],
            model="LT_jajjawi",
            fillMethod="bogus",
        )


@pytest.mark.parametrize("fill_method", ["nearest", "offTmin"])
def test_lt_dac_accepts_the_supported_fill_methods(dac_placements: pd.DataFrame, fill_method):
    gen = lt_dac_era5_wenzel2025(
        placements=dac_placements,
        era5_path=TEST_DATA["era5-like"],
        model="LT_jajjawi",
        fillMethod=fill_method,
    )

    assert "capacity_factor" in gen.variables


def test_ht_dac_rejects_an_unknown_model(dac_placements: pd.DataFrame):
    with pytest.raises(NotImplementedError, match="bogus"):
        ht_dac_era5_wenzel2025(
            placements=dac_placements,
            era5_path=TEST_DATA["era5-like"],
            model="bogus",
        )
