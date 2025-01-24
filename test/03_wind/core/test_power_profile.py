from reskit.wind.core.power_profile import (
    alpha_from_levels,
    apply_power_profile_projection,
)
import numpy as np


def test_apply_power_profile_projection():
    output = apply_power_profile_projection(
        measured_wind_speed=3,
        measured_height=10,
        target_height=50,
        alpha=0.1787469216608008,
    )
    assert np.isclose(output, 4.0)

    output = apply_power_profile_projection(
        measured_wind_speed=np.array([3, 4, 5]),
        measured_height=10,
        target_height=50,
        alpha=np.array([0.17874692, 0.13864688, 0.11328275]),
    )
    assert np.isclose(output[0], 4.0)
    assert np.isclose(output[1], 5.0)
    assert np.isclose(output[2], 6.0)


def test_alphaFromLevels():
    a = alpha_from_levels(
        low_wind_speed=3, low_height=10, high_wind_speed=4, high_height=50
    )
    assert np.isclose(a, 0.1787469216608008)

    a = alpha_from_levels(
        low_wind_speed=np.array([3, 4, 5]),
        low_height=10,
        high_wind_speed=np.array([4, 5, 6]),
        high_height=50,
    )
    assert np.isclose(a[0], 0.17874692)
    assert np.isclose(a[1], 0.13864688)
    assert np.isclose(a[2], 0.11328275)
