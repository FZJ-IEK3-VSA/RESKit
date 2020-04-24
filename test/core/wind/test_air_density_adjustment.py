from reskit import TEST_DATA
from reskit.core.wind.air_density_adjustment import apply_air_density_adjustment
import numpy as np


def test_apply_air_density_adjustment():
    assert np.isclose(apply_air_density_adjustment(5, 105000, 0), 5.150675296246491)
    assert np.isclose(apply_air_density_adjustment(np.linspace(3, 4, 10), 105000, 0)[4], 3.548242981858694)
