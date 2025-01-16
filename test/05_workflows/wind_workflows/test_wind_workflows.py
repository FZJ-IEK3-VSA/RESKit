from reskit.wind.workflows.workflows import (
    onshore_wind_merra_ryberg2019_europe,
    offshore_wind_merra_caglayan2019,
    wind_era5_PenaSanchezDunkelWinklerEtAl2025,
    wind_config,
)
from reskit import TEST_DATA
import pytest
import numpy as np
import geokit as gk
import os
import pandas as pd
import reskit.weather as rk_weather
from reskit.wind import DATAFOLDER


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


def test_wind_era5_PenaSanchezDunkelWinklerEtAl2025(pt_wind_placements: pd.DataFrame):
    gen = wind_era5_PenaSanchezDunkelWinklerEtAl2025(
        placements=pt_wind_placements,
        era5_path=TEST_DATA["era5-like"],
        gwa_100m_path=TEST_DATA["gwa100-like.tif"],
        esa_cci_path=TEST_DATA["ESA_CCI_2018_clip.tif"],
        output_netcdf_path=None,
        cf_correction=True,
    )

    assert gen.roughness.shape == (560,)
    assert np.isclose(gen.roughness.mean(), 0.44921429)
    assert np.isclose(gen.roughness.min(), 0.03)
    assert np.isclose(gen.roughness.max(), 1.2)
    assert np.isclose(gen.roughness.std(), 0.55593945)

    assert gen.elevated_wind_speed.shape == (140, 560)
    assert np.isclose(gen.elevated_wind_speed.mean(), 5.86475732)
    assert np.isclose(gen.elevated_wind_speed.min(), 0.26886236)
    assert np.isclose(gen.elevated_wind_speed.max(), 12.86308429)
    assert np.isclose(gen.elevated_wind_speed.std(), 2.18707038)

    assert gen.capacity_factor.shape == (140, 560)
    assert np.isclose(gen.capacity_factor.mean(), 0.32975885)
    assert np.isclose(gen.capacity_factor.min(), 0.0)
    assert np.isclose(gen.capacity_factor.max(), 0.98)
    assert np.isclose(gen.capacity_factor.std(), 0.28939232)


def test_wind_config(pt_wind_placements: pd.DataFrame):
    gen = wind_config(
        placements=pt_wind_placements,
        weather_path=TEST_DATA["era5-like"],
        weather_source_type="ERA5",
        weather_lra_ws_path=rk_weather.Era5Source.LONG_RUN_AVERAGE_WINDSPEED_2008TO2017,
        real_lra_ws_path=TEST_DATA["gwa100-like.tif"],
        real_lra_ws_scaling=1,
        real_lra_ws_spatial_interpolation="average",
        real_lra_ws_nodata_fallback=np.nan,
        landcover_path=TEST_DATA["ESA_CCI_2018_clip.tif"],
        landcover_source_type="cci",
        ws_correction_func=(
            "ws_bins",
            os.path.join(DATAFOLDER, f"ws_correction_factors_PSDW2025.yaml"),
        ),
        cf_correction_factor=os.path.join(
            DATAFOLDER, f"cf_correction_factors_PSDW2025.tif"
        ),
        wake_curve="dena_mean",
        availability_factor=0.98,
        consider_boundary_layer_height=True,
        power_curve_scaling=0.01,
        power_curve_base=0.0,
        convolute_power_curves_args={},
        loss_factor_args={},
        output_variables=None,
        max_batch_size=25000,
        output_netcdf_path=None,
        elevated_wind_speed=None,
    )

    assert gen.roughness.shape == (560,)
    assert np.isclose(gen.roughness.mean(), 0.44921429)
    assert np.isclose(gen.roughness.min(), 0.03)
    assert np.isclose(gen.roughness.max(), 1.2)
    assert np.isclose(gen.roughness.std(), 0.55593945)

    assert gen.elevated_wind_speed.shape == (140, 560)
    assert np.isclose(gen.elevated_wind_speed.mean(), 5.86475732)
    assert np.isclose(gen.elevated_wind_speed.min(), 0.26886236)
    assert np.isclose(gen.elevated_wind_speed.max(), 12.86308429)
    assert np.isclose(gen.elevated_wind_speed.std(), 2.18707038)

    assert gen.capacity_factor.shape == (140, 560)
    assert np.isclose(gen.capacity_factor.mean(), 0.32975885)
    assert np.isclose(gen.capacity_factor.min(), 0.0)
    assert np.isclose(gen.capacity_factor.max(), 0.98)
    assert np.isclose(gen.capacity_factor.std(), 0.28939232)
