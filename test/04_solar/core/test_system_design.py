import numpy as np

from reskit.solar.core.system_design import location_to_tilt

LOCS = [(6.0, 51.0), (7.0, 45.0), (0.0, 0.0)]


def test_location_to_tilt_ryberg2020():
    tilt = location_to_tilt(LOCS, convention="Ryberg2020")

    expected = 42.327719357601396 * np.arctan(1.5 * np.radians(np.abs([51.0, 45.0, 0.0])))
    assert np.allclose(tilt, expected)
    # the default convention gives the same result
    assert np.allclose(location_to_tilt(LOCS), expected)
