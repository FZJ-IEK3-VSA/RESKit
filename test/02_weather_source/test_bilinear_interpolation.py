import numpy as np
from scipy.interpolate import RectBivariateSpline

from reskit.weather.nc_source import _bilinear_interpolation


def test_bilinear_interpolation_matches_a_degree_one_spline():
    """The vectorised weights must reproduce the per-time-step spline they replaced."""
    rng = np.random.default_rng(0)
    lats = 50.0 + 0.25 * np.arange(6)
    lons = 6.0 + 0.25 * np.arange(7)
    window = rng.random((12, lats.size, lons.size)) * 20

    # scattered points inside the window, plus one beyond each edge, so that the
    # spline's linear extrapolation is covered as well
    y = np.concatenate([rng.uniform(lats[0], lats[-1], 8), [lats[0] - 0.4, lats[-1] + 0.3]])
    x = np.concatenate([rng.uniform(lons[0], lons[-1], 8), [lons[0] - 0.2, lons[-1] + 0.5]])

    expected = np.stack(
        [RectBivariateSpline(lats, lons, window[ts], kx=1, ky=1)(y, x, grid=False) for ts in range(window.shape[0])]
    )

    assert np.allclose(_bilinear_interpolation(window, lats, lons, y, x), expected, rtol=0, atol=1e-12)
