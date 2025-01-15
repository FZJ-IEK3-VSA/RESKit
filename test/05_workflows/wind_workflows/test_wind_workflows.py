from reskit.wind.workflows.workflows import (
    onshore_wind_merra_ryberg2019_europe,
    offshore_wind_merra_caglayan2019,
    offshore_wind_era5,
    onshore_wind_era5,
    wind_era5_2023,
)
from reskit import TEST_DATA
import pytest
import numpy as np
import geokit as gk
import pandas as pd


@pytest.fixture
def pt_wind_placements() -> pd.DataFrame:
    df = gk.vector.extractFeatures(TEST_DATA["turbinePlacements.shp"])
    df["hub_height"] = np.linspace(100, 130, df.shape[0])
    df["capacity"] = 3000
    df["rotor_diam"] = 170
    df.loc[::2, "rotor_diam"] = 150
    return df


def test_onshore_wind_merra_ryberg2019_europe(pt_wind_placements: pd.DataFrame):
    gen = onshore_wind_merra_ryberg2019_europe(
        placements=pt_wind_placements,
        merra_path=TEST_DATA["merra-like"],
        gwa_50m_path=TEST_DATA["gwa50-like.tif"],
        clc2012_path=TEST_DATA["clc-aachen_clipped.tif"],
    )

    assert gen.roughness.shape == (560,)
    assert np.isclose(gen.roughness.mean(), 0.21235714)
    assert np.isclose(gen.roughness.min(), 0.005)
    assert np.isclose(gen.roughness.max(), 1.2)
    assert np.isclose(gen.roughness.std(), 0.22275803)

    assert gen.elevated_wind_speed.shape == (71, 560)
    assert np.isclose(gen.elevated_wind_speed.mean(), 7.25949888)
    assert np.isclose(gen.elevated_wind_speed.min(), 1.65055839)
    assert np.isclose(gen.elevated_wind_speed.max(), 14.22762999)
    assert np.isclose(gen.elevated_wind_speed.std(), 2.48011369)

    assert gen.capacity_factor.shape == (71, 560)
    assert np.isclose(gen.capacity_factor.mean(), 0.57601636)
    assert np.isclose(gen.capacity_factor.min(), 0.0)
    assert np.isclose(gen.capacity_factor.max(), 0.99326205)
    assert np.isclose(gen.capacity_factor.std(), 0.35040166)


def test_offshore_wind_merra_caglayan2019(pt_wind_placements):
    # placements, merra_path, output_netcdf_path=None, output_variables=None):
    gen = offshore_wind_merra_caglayan2019(
        placements=pt_wind_placements,
        merra_path=TEST_DATA["merra-like"],
    )

    assert gen.roughness.shape == (560,)
    assert np.isclose(gen.roughness.mean(), 0.0002)
    assert np.isclose(gen.roughness.min(), 0.0002)
    assert np.isclose(gen.roughness.max(), 0.0002)
    assert np.isclose(gen.roughness.std(), 0)

    assert gen.elevated_wind_speed.shape == (71, 560)
    assert np.isclose(gen.elevated_wind_speed.mean(), 9.16905288)
    assert np.isclose(gen.elevated_wind_speed.min(), 1.93876409)
    assert np.isclose(gen.elevated_wind_speed.max(), 15.98714391)
    assert np.isclose(gen.elevated_wind_speed.std(), 3.07147403)

    assert gen.capacity_factor.shape == (71, 560)
    assert np.isclose(gen.capacity_factor.mean(), 0.74947119)
    assert np.isclose(gen.capacity_factor.min(), 0.00045005)
    assert np.isclose(gen.capacity_factor.max(), 0.97282235)
    assert np.isclose(gen.capacity_factor.std(), 0.29063037)
