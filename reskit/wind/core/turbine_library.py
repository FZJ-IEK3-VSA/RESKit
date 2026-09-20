import os
import re
import warnings
from collections import OrderedDict, namedtuple
from glob import glob
from os.path import dirname, join

import numpy as np
import pandas as pd

from .power_curve import PowerCurve

##################################################
# Make a turbine model library
TurbineInfo = namedtuple("TurbineInfo", "profile meta")

rangeRE = re.compile("([0-9.]{1,})-([0-9.]{1,})")


def parse_turbine(path):
    """
    **internal function**

    Parses over a turbine's data file to get hub height, capacity, rotor diameter and powercurve.

    Used for loading into the turbine_library table
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

            sLine = line.split(",")
            if sLine[0].lower() == "hubheight" or sLine[0].lower() == "hub_height":
                heights = []
                for h in sLine[1:]:
                    h = h.replace('"', "")
                    h = h.strip()
                    h = h.replace(" ", "")

                    try:
                        h = float(h)
                        heights.append(h)
                    except:
                        try:
                            a, b = rangeRE.search(h).groups()
                            a = int(a)
                            b = int(b)

                            for hh in range(a, b + 1):
                                heights.append(hh)
                        except:
                            raise RuntimeError("Could not understand heights")

                meta["Hub_Height"] = np.array(heights)
            else:
                try:
                    meta[sLine[0].title()] = float(sLine[1])
                except:
                    meta[sLine[0].title()] = sLine[1]

        # Extract power profile
        tmp = pd.read_csv(fin)
        tmp = np.array([(ws, output) for i, ws, output in tmp.iloc[:, :2].itertuples()])
        power = PowerCurve(tmp[:, 0], tmp[:, 1] / tmp[:, 1].max())
    return TurbineInfo(power, meta)


#: The turbine definitions RESKit ships: one CSV per model under ``data/turbines``.
BUNDLED_TURBINES = join(dirname(__file__), "data", "turbines")

# Every library read so far, by directory, and the one in use. A directory is
# parsed once per process; selecting it again is free.
_libraries: dict[str, pd.DataFrame] = {}
_selected: str = BUNDLED_TURBINES


def _read_library(directory):
    """Parse every ``*.csv`` under ``directory`` into a turbine library dataframe."""
    turbineFiles = sorted(glob(join(directory, "*.csv")))
    if not turbineFiles:
        raise FileNotFoundError(f"No turbine definition files (*.csv) found under: {directory}")

    tmp = []
    already_added_models = []
    for f in turbineFiles:
        try:
            _parsed = parse_turbine(f)
            model_id = _parsed.meta["Model"]
            if model_id in already_added_models:
                print(model_id, "already in Turbine Library")
                continue
            else:
                tmp.append(_parsed)
                already_added_models.append(model_id)
        except Exception:
            print("failed to parse:", f)

    library = pd.DataFrame([i.meta for i in tmp])
    library.set_index("Model", inplace=True)
    library["PowerCurve"] = [x.profile for x in tmp]
    return library


def turbine_library(path=None):
    """
    The turbine library: one row per wind turbine model, indexed by model name.

    Each row carries the manufacturer, capacity, usage, hub heights and rotor diameter
    of a model and its :class:`~reskit.wind.core.power_curve.PowerCurve`. Workflows
    resolve a power curve given by name against this library.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        A directory of turbine definition files, one ``*.csv`` per model in the
        format of the files under ``reskit/wind/core/data/turbines``. When given,
        the directory is read and becomes the library in use for the rest of the
        process, so every later call without ``path`` -- including the ones the
        workflows make -- returns it. By default the library RESKit ships is in
        use; pass :data:`BUNDLED_TURBINES` to return to it.

        The larger, licensed library in the ETHOS.Data catalogue is selected with
        ``turbine_library(reskit.data.paths("turbine_library")["turbines"])``; a
        library built from purchased power curves (see the example
        ``1_3_1_process_power_curves_from_thewindpower_net``) with the directory
        it was written to.

    Returns
    -------
    pandas.DataFrame
        The library in use.
    """
    global _selected

    if path is not None:
        _selected = os.path.abspath(os.fspath(path))
    if _selected not in _libraries:
        _libraries[_selected] = _read_library(_selected)
    return _libraries[_selected]


##########################
# DEPRECATED NAMES (#226) #
##########################
# The names below were renamed for PEP 8 in RESKit v0.6.0. Each old name stays
# available as a warning wrapper until v1.0.0. Do not add new code here.


def TurbineLibrary(*args, **kwargs):
    """
    Deprecated alias of :func:`turbine_library`.

    Kept for backward compatibility and scheduled for removal in RESKit v1.0.0.
    Use :func:`turbine_library` instead. All arguments are passed through unchanged.
    """
    warnings.warn(
        "TurbineLibrary() is deprecated and will be removed in RESKit v1.0.0. Use turbine_library() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return turbine_library(*args, **kwargs)
