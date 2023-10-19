import numpy as np
from reskit.util.air_density import compute_air_density


def test_compute_air_density():
    assert np.isclose(compute_air_density(), 1.2050461878674021)
    assert np.isclose(compute_air_density(
        temperature=30, pressure=105000, relative_humidity=0.3), 1.2021773918072405)
