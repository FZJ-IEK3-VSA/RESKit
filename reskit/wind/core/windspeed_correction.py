# import primary packages
import os

import numpy as np
import yaml

# import third packages
from pandas import Interval


# helper function to generate the actual correction function
def build_ws_correction_function(type, data_dict):
    """
    type: str
        type of correction function
    data_dict: dict, str
        dictionary or json file containing the data needed to
        build the correction function
    """
    if isinstance(data_dict, str):
        assert os.path.isfile(data_dict), f"data_dict is a str but not an existing file: {data_dict}"
        assert os.path.splitext(data_dict)[-1] in [
            ".yaml",
            ".yml",
        ], f"data_dict must be a yaml file if given as str path."
        with open(data_dict, "r") as f:
            data_dict = yaml.load(f, Loader=yaml.FullLoader)
    if type == "polynomial":
        # convert tuple to dict first if needed
        if isinstance(data_dict, (list, tuple)):
            # assume that the polynomial factors a_i*x^^i are sorted (a_n, ..., a_2, a_1, a_0)
            data_dict = {i: v for i, v in enumerate(list(data_dict)[::-1])}
        assert isinstance(data_dict, dict), f"data_dict must be a dict if not given as a tuple of polynomial factors."
        assert all([x % 1 == 0 for x in data_dict.keys()]), (
            f"All data_dict keys must be integers i with values a_i, for all required polynomial factors a_i*x^^i."
        )

        def correction_function(x):
            _func = 0
            for deg, fac in data_dict.items():
                _func = _func + fac * x ** int(deg)
            return _func

        return correction_function
    elif type == "ws_bins":
        assert "ws_bins" in data_dict.keys(), "data_dict must contain key 'ws_bins' with a dict of ws bins and factors."
        if not all(isinstance(ws_bin, Interval) for ws_bin in data_dict["ws_bins"].keys()):
            ws_bins_dict = {}
            for range_str, factor in data_dict["ws_bins"].copy().items():
                left, right = range_str.split("-")
                left = float(left)
                right = float(right) if right != "inf" else np.inf
                ws_bins_dict[Interval(left, right, closed="right")] = factor
            data_dict["ws_bins"] = ws_bins_dict

        # check if all keys are of instance Interval
        assert all(isinstance(ws_bin, Interval) for ws_bin in data_dict["ws_bins"].keys())
        ws_bins_correction = data_dict["ws_bins"]

        # The bins are looked up rather than looped over: one pass of the wind speeds per
        # bin costs a full sweep of the (time x location) array each, and these tables run
        # to several hundred bins. Sorting them by left edge lets a single searchsorted
        # place every wind speed in its bin at once.
        _bins = sorted(ws_bins_correction.items(), key=lambda item: item[0].left)
        _lefts = np.array([ws_bin.left for ws_bin, _ in _bins], dtype=float)
        _rights = np.array([ws_bin.right for ws_bin, _ in _bins], dtype=float)
        _factors = np.array([factor for _, factor in _bins], dtype=float)
        # A wind speed covered by two bins has no single correction factor, so such a table
        # is rejected rather than resolved by the order the bins happen to be written in.
        _overlapping = [(_bins[i][0], _bins[i + 1][0]) for i in np.flatnonzero(_rights[:-1] > _lefts[1:])]
        assert not _overlapping, f"ws_bins must not overlap, but these do: {_overlapping}"

        def correction_function(x):
            # x is numpy array. modify x based on ws_bins

            # the bin each wind speed falls in, clipped so that speeds below the first
            # left edge index a real bin; the where() below leaves those uncorrected,
            # along with any speed falling in a gap between bins or past the last one
            values = np.asarray(x)
            index = np.searchsorted(_lefts, values, side="right") - 1
            np.clip(index, 0, _lefts.size - 1, out=index)
            scale = 1.0 - _factors[index]
            np.copyto(scale, 1.0, where=(values < _lefts[index]) | (values >= _rights[index]))
            return x * scale.astype(values.dtype, copy=False)

            # the bin each wind speed falls in, clipped so that speeds below the first
            # left edge index a real bin; the where() below leaves those uncorrected,
            # along with any speed falling in a gap between bins or past the last one
            values = np.asarray(x)
            index = np.searchsorted(_lefts, values, side="right") - 1
            np.clip(index, 0, _lefts.size - 1, out=index)
            scale = 1.0 - _factors[index]
            np.copyto(scale, 1.0, where=(values < _lefts[index]) | (values >= _rights[index]))
            return x * scale.astype(values.dtype, copy=False)

        return correction_function

    elif type == "ws_double_bins":
        if not all(isinstance(ws_bin, Interval) for ws_bin in data_dict.keys()):
            # convert keys to pd.Interval
            def convert_interval(interval):
                left, right = interval.split("-")
                left = float(left)
                right = float(right) if right != "inf" else np.inf
                return Interval(left, right, closed="right")

            ws_bins_correction = {}
            for mean_ws_bin, mean_ws_bin_dict in data_dict.items():
                mean_ws_bin_interval = convert_interval(mean_ws_bin)
                _mean_ws_bin_dict = {}
                for range_str, factor in mean_ws_bin_dict.copy().items():
                    _mean_ws_bin_dict[convert_interval(range_str)] = factor
                ws_bins_correction[mean_ws_bin_interval] = _mean_ws_bin_dict

        def correction_function(x):
            mean_ws = x.mean(axis=0)

            corrected_x = x.copy()
            for mean_ws_bin, mean_ws_bin_dict in ws_bins_correction.items():
                mask_mean_ws = (mean_ws >= mean_ws_bin.left) & (mean_ws < mean_ws_bin.right)
                for ws_bin, factor in mean_ws_bin_dict.items():
                    mask_hourly_ws = (x >= ws_bin.left) & (x < ws_bin.right)
                    corrected_x[mask_mean_ws & mask_hourly_ws] = x[mask_mean_ws & mask_hourly_ws] * (1 - factor)
            return corrected_x

        return correction_function

    else:
        raise ValueError(f"Invalid ws_correction_func type: {type}. Select from: 'polynomial', 'ws_bins'.")
