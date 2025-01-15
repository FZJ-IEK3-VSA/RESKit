# import primary packages
import numpy as np
import os
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
        assert os.path.isfile(
            data_dict
        ), f"data_dict is a str but not an existing file: {data_dict}"
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
        assert isinstance(
            data_dict, dict
        ), f"data_dict must be a dict if not given as a tuple of polynomial factors."
        assert all(
            [x % 1 == 0 for x in data_dict.keys()]
        ), f"All data_dict keys must be integers i with values a_i, for all required polynomial factors a_i*x^^i."

        def correction_function(x):
            _func = 0
            for deg, fac in data_dict.items():
                _func = _func + fac * x ** int(deg)
            return _func

        return correction_function
    elif type == "ws_bins":
        assert (
            "ws_bins" in data_dict.keys()
        ), "data_dict must contain key 'ws_bins' with a dict of ws bins and factors."
        if not all(
            isinstance(ws_bin, Interval)
            for ws_bin in data_dict["ws_bins"].keys()
        ):
            ws_bins_dict = {}
            for range_str, factor in data_dict["ws_bins"].copy().items():
                left, right = range_str.split("-")
                left = float(left)
                right = float(right) if right != "inf" else np.inf
                ws_bins_dict[Interval(left, right, closed="right")] = factor
            data_dict["ws_bins"] = ws_bins_dict

        # check if all keys are of instance Interval
        assert all(
            isinstance(ws_bin, Interval)
            for ws_bin in data_dict["ws_bins"].keys()
        )
        ws_bins_correction = data_dict["ws_bins"]

        def correction_function(x):
            # x is numpy array. modify x based on ws_bins
            corrected_x = x.copy()
            for ws_bin, factor in ws_bins_correction.items():
                mask = (x >= ws_bin.left) & (x < ws_bin.right)
                corrected_x[mask] = x[mask] * (1 - factor)
            return corrected_x

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
                mask_mean_ws = (mean_ws >= mean_ws_bin.left) & (
                    mean_ws < mean_ws_bin.right
                )
                for ws_bin, factor in mean_ws_bin_dict.items():
                    mask_hourly_ws = (x >= ws_bin.left) & (x < ws_bin.right)
                    corrected_x[mask_mean_ws & mask_hourly_ws] = x[
                        mask_mean_ws & mask_hourly_ws
                    ] * (1 - factor)
            return corrected_x

        return correction_function

    else:
        raise ValueError(
            f"Invalid ws_correction_func type: {type}. Select from: 'polynomial', 'ws_bins'."
        )
