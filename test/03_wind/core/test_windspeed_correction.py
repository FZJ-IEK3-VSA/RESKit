import numpy as np
import pytest
from pandas import Interval

from reskit.wind.core.windspeed_correction import build_ws_correction_function


def _loop_reference(bins, x):
    """The correction as it was written before the lookup: one sweep per bin, in order."""
    corrected = x.copy()
    for ws_bin, factor in bins.items():
        mask = (x >= ws_bin.left) & (x < ws_bin.right)
        corrected[mask] = x[mask] * (1 - factor)
    return corrected


def test_ws_bins_lookup_matches_the_per_bin_loop():
    rng = np.random.default_rng(0)
    bins = {Interval(i / 2, (i + 1) / 2, closed="right"): 0.05 + i / 100 for i in range(20)}
    # spans the bins and reaches past both ends, where no correction applies
    ws = rng.uniform(-1.0, 12.0, size=(500, 7))

    corrected = build_ws_correction_function("ws_bins", {"ws_bins": dict(bins)})(ws)

    assert np.array_equal(corrected, _loop_reference(bins, ws))


def test_ws_bins_rejects_overlapping_bins():
    """A wind speed in two bins has no single factor, so the table is rejected, not resolved."""
    bins = {
        Interval(0.0, 6.0, closed="right"): 0.1,
        Interval(4.0, 8.0, closed="right"): 0.2,  # overlaps the first
    }

    with pytest.raises(AssertionError, match="ws_bins must not overlap"):
        build_ws_correction_function("ws_bins", {"ws_bins": dict(bins)})
