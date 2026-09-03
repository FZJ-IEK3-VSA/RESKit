import numpy as np
import pandas as pd
import pytest

from reskit.solar.core.frank_correction import frank_correction_factors

# the month factors and the elevation factors which the model defines
JANUARY_CLOUDY_FACTOR = 0.7776553729824053
JULY_CLOUDY_FACTOR = 0.9350856478115459
CLEAR_SKY_FACTOR_ABOVE_60_DEGREE = 1.0715262914980628
CLEAR_SKY_FACTOR_BELOW_10_DEGREE = 1.0


def _inputs(month, transmissivity, solar_elevation, locations=2, steps=3):
    """Build one small correction input set with a constant transmissivity."""
    times = pd.date_range(f"2020-{month:02d}-15 10:00:00", periods=steps, freq="h")
    dni_extra = np.full((steps, locations), 1000.0)
    ghi = dni_extra * transmissivity
    elevation = np.full((steps, locations), float(solar_elevation))
    return ghi, dni_extra, times, elevation


def test_frank_correction_factors_shape():
    ghi, dni_extra, times, elevation = _inputs(month=1, transmissivity=0.5, solar_elevation=45)

    factors = frank_correction_factors(ghi, dni_extra, times, elevation)

    assert factors.shape == ghi.shape


def test_frank_correction_factors_cloudy_regime():
    # a very low transmissivity drives the sigmoid to zero, so only the month factor remains
    ghi, dni_extra, times, elevation = _inputs(month=1, transmissivity=0.05, solar_elevation=70)

    factors = frank_correction_factors(ghi, dni_extra, times, elevation)

    assert np.allclose(factors, JANUARY_CLOUDY_FACTOR)


def test_frank_correction_factors_clear_sky_regime():
    # a very high transmissivity drives the sigmoid to one, so only the elevation factor remains
    ghi, dni_extra, times, elevation = _inputs(month=7, transmissivity=0.95, solar_elevation=70)

    factors = frank_correction_factors(ghi, dni_extra, times, elevation)

    assert np.allclose(factors, CLEAR_SKY_FACTOR_ABOVE_60_DEGREE)


def test_frank_correction_factors_use_the_month():
    ghi, dni_extra, times, elevation = _inputs(month=7, transmissivity=0.05, solar_elevation=70)

    factors = frank_correction_factors(ghi, dni_extra, times, elevation)

    assert np.allclose(factors, JULY_CLOUDY_FACTOR)
    assert JULY_CLOUDY_FACTOR > JANUARY_CLOUDY_FACTOR


def test_frank_correction_factors_low_sun_keeps_the_clear_sky_factor():
    # below 10 degree elevation the clear sky factor stays at one
    ghi, dni_extra, times, elevation = _inputs(month=7, transmissivity=0.95, solar_elevation=5)

    factors = frank_correction_factors(ghi, dni_extra, times, elevation)

    assert np.allclose(factors, CLEAR_SKY_FACTOR_BELOW_10_DEGREE)


@pytest.mark.parametrize("transmissivity", [0.2, 0.4, 0.5, 0.6, 0.8])
def test_frank_correction_factors_lie_between_both_regimes(transmissivity):
    ghi, dni_extra, times, elevation = _inputs(month=1, transmissivity=transmissivity, solar_elevation=70)

    factors = frank_correction_factors(ghi, dni_extra, times, elevation)

    assert (factors >= JANUARY_CLOUDY_FACTOR).all()
    assert (factors <= CLEAR_SKY_FACTOR_ABOVE_60_DEGREE).all()
