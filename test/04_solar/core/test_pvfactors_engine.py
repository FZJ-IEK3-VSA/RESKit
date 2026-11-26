import numpy as np
import pandas as pd
import pvlib
import pytest
from pvlib.bifacial.pvfactors import pvfactors_timeseries as pvlib_pvfactors_timeseries

from reskit.solar.core.pvfactors_engine import _block_solve_engine, _vf_calculator, pvfactors_timeseries


def _inputs(tracking, n_pvrows, gcr=0.35, pvrow_height=2.6, index_observed_pvrow=1, surface_azimuth=180.0):
    """Two weeks of hourly daylight inputs at one location, with varying sky and albedo."""
    times = pd.date_range("2018-06-01", "2018-06-14 23:00", freq="h", tz="UTC")
    location = pvlib.location.Location(50.5, 6.5)
    solpos = location.get_solarposition(times)
    clearsky = location.get_clearsky(times)
    day = (solpos["apparent_zenith"] < 95).to_numpy()
    solpos, clearsky = solpos[day], clearsky[day]
    rng = np.random.default_rng(0)
    cloud = rng.uniform(0.2, 1.0, len(solpos))
    if tracking == "fixed":
        surface_tilt, axis_azimuth = 35.0, 270.0
    else:
        tracker = pvlib.tracking.singleaxis(
            solpos["apparent_zenith"], solpos["azimuth"], axis_azimuth=0, max_angle=60, backtrack=True, gcr=gcr
        )
        surface_tilt = tracker["surface_tilt"].fillna(0).to_numpy()
        surface_azimuth = tracker["surface_azimuth"].fillna(90).to_numpy()
        axis_azimuth = 0.0
    return dict(
        solar_azimuth=solpos["azimuth"].to_numpy(),
        solar_zenith=solpos["apparent_zenith"].to_numpy(),
        surface_azimuth=surface_azimuth,
        surface_tilt=surface_tilt,
        axis_azimuth=axis_azimuth,
        timestamps=np.arange(len(solpos)),
        dni=clearsky["dni"].to_numpy() * cloud,
        dhi=clearsky["dhi"].to_numpy() * (2 - cloud),
        gcr=gcr,
        pvrow_height=pvrow_height,
        pvrow_width=4.8,
        albedo=rng.uniform(0.15, 0.6, len(solpos)),
        n_pvrows=n_pvrows,
        index_observed_pvrow=index_observed_pvrow,
    )


# tracking, n_pvrows and other arguments of _inputs()
CASES = [
    ("fixed", 3, {}),
    ("singleaxis", 3, {}),
    ("fixed", 5, {}),
    ("singleaxis", 1, {"index_observed_pvrow": 0}),
    ("singleaxis", 5, {"index_observed_pvrow": 2}),
    ("singleaxis", 3, {"gcr": 0.7}),
    ("fixed", 3, {"surface_azimuth": 0.0}),  # facing away from the sun
    # rows reaching below the ground at steep tracker angles: not physical, but pvfactors
    # computes them, and the calculator must not change its results there either
    ("singleaxis", 3, {"pvrow_height": 1.2}),
]


@pytest.mark.parametrize("tracking, n_pvrows, kwargs", CASES)
def test_pvfactors_timeseries_matches_pvlib(tracking, n_pvrows, kwargs, monkeypatch):
    args = _inputs(tracking, n_pvrows, **kwargs)
    expected = pvlib_pvfactors_timeseries(**args)

    # make sure the block solve itself runs, not the fallback to pvfactors' own run_full_mode
    engine = _block_solve_engine()
    monkeypatch.setattr(
        engine.__mro__[1], "run_full_mode", lambda *a, **k: pytest.fail("fell back to PVEngine.run_full_mode")
    )
    result = pvfactors_timeseries(**args)

    for name, exp, res in zip(("inc_front", "inc_back", "abs_front", "abs_back"), expected, result):
        pd.testing.assert_index_equal(res.index, exp.index)
        np.testing.assert_allclose(res.to_numpy(), exp.to_numpy(), rtol=1e-12, atol=1e-9, err_msg=name)


@pytest.mark.parametrize("tracking, n_pvrows, kwargs", CASES)
def test_vf_calculator_matches_pvfactors(tracking, n_pvrows, kwargs):
    # Equal to rounding, not bit for bit: at the ground points where pvfactors places the shadow
    # edges, the string from a PV row edge just touches the neighbouring row, so whether it
    # counts as obstructed depends on the last bits of arctan2 (or of the calculator's cross
    # product); both string lengths agree to rounding there.
    from pvfactors.geometry.pvarray import OrderedPVArray
    from pvfactors.viewfactors.calculator import VFCalculator

    args = _inputs(tracking, n_pvrows, **kwargs)
    n_steps = len(args["solar_zenith"])
    pvarray = OrderedPVArray.init_from_dict(
        {key: args[key] for key in ("n_pvrows", "axis_azimuth", "pvrow_height", "pvrow_width", "gcr")}
    )
    pvarray.fit(
        args["solar_zenith"],
        args["solar_azimuth"],
        np.full(n_steps, args["surface_tilt"], dtype=float),
        np.full(n_steps, args["surface_azimuth"], dtype=float),
    )
    expected = VFCalculator().build_ts_vf_matrix(pvarray)
    result = _vf_calculator()().build_ts_vf_matrix(pvarray)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-14)
