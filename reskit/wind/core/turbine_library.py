import re
from collections import OrderedDict, namedtuple
from glob import glob
from os.path import dirname, join

import numpy as np
import pandas as pd

from reskit.default_paths import DEFAULT_PATHS
from reskit.wind.core.power_curve import PowerCurve

##################################################
# Make a turbine model library
TurbineInfo = namedtuple("TurbineInfo", "profile meta")

range_re = re.compile("([0-9.]{1,})-([0-9.]{1,})")


def parse_turbine(path):
    """
    **internal function**

    Parses over a turbine's data file to get hub height, capacity, rotor diameter and powercurve.

    Used for loading into the TurbineLibrary table
    """
    meta = OrderedDict()
    with open(path) as fin:
        # Meta extraction mode
        while True:
            line = fin.readline()[:-1]

            if line == "" or line[0] == "#":
                continue  # skip blank lines and comment lines
            if "power curve" in line.lower():
                break

            s_line = line.split(",")
            if s_line[0].lower() == "hubheight" or s_line[0].lower() == "hub_height":
                heights = []
                for h in s_line[1:]:
                    h = h.replace('"', "")
                    h = h.strip()
                    h = h.replace(" ", "")

                    try:
                        h = float(h)
                        heights.append(h)
                    except:
                        try:
                            a, b = range_re.search(h).groups()
                            a = int(a)
                            b = int(b)

                            for hh in range(a, b + 1):
                                heights.append(hh)
                        except:
                            raise RuntimeError("Could not understand heights")

                meta["Hub_Height"] = np.array(heights)
            else:
                try:
                    meta[s_line[0].title()] = float(s_line[1])
                except:
                    meta[s_line[0].title()] = s_line[1]

        # Extract power profile
        tmp = pd.read_csv(fin)
        tmp = np.array([(ws, output) for i, ws, output in tmp.iloc[:, :2].itertuples()])
        power = PowerCurve(tmp[:, 0], tmp[:, 1] / tmp[:, 1].max())
    return TurbineInfo(power, meta)


_Turbine_Library = None


def turbine_library() -> pd.DataFrame:
    """
    A dataframe of internally configured wind turbines accessible to later simulations
    """
    global _Turbine_Library

    if _Turbine_Library is None:
        if DEFAULT_PATHS["turbine_library_path"] is None:
            turbine_files = glob(join(dirname(__file__), "data", "turbines", "*.csv"))
        else:
            turbine_files = glob(join(DEFAULT_PATHS["turbine_library_path"], "*.csv"))
        tmp = []
        already_added_models = []
        for f in turbine_files:
            try:
                _parsed = parse_turbine(f)
                model_id = parse_turbine(f)[1]["Model"]
                if model_id in already_added_models:
                    print(model_id, "already in Turbine Library")
                    continue
                else:
                    tmp.append(_parsed)
                    already_added_models.append(model_id)
            except:
                print("failed to parse:", f)

        _Turbine_Library = pd.DataFrame([i.meta for i in tmp])
        _Turbine_Library.set_index("Model", inplace=True)
        _Turbine_Library["PowerCurve"] = [x.profile for x in tmp]

    return _Turbine_Library
