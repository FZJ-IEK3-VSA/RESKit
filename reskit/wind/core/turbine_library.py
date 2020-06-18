from collections import OrderedDict, namedtuple
import numpy as np
import pandas as pd
from glob import glob
import re
from os.path import join, dirname

from .power_curve import PowerCurve

##################################################
# Make a turbine model library
TurbineInfo = namedtuple('TurbineInfo', 'profile meta')

rangeRE = re.compile("([0-9.]{1,})-([0-9.]{1,})")


def parse_turbine(path):
    """
    Parses over a turbine's data file to get hub height (, capacity ,?) and power values and their corresponding 

    Parameters
    ----------
    path : str
        path to the turbines' data file

    Returns
    -------
    namedtuple
        Turbine's power curve and data in a namedtuple.
    """

    meta = OrderedDict()
    with open(path) as fin:
        # Meta extraction mode
        while True:
            line = fin.readline()[:-1]

            if line == "" or line[0] == "#":
                continue  # skip blank lines and comment lines
            if 'power curve' in line.lower():
                break

            sLine = line.split(',')
            if sLine[0].lower() == "hubheight" or sLine[0].lower() == "hub_height":
                heights = []
                for h in sLine[1:]:
                    h = h.replace("\"", "")
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
        tmp = np.array([(ws, output)
                        for i, ws, output in tmp.iloc[:, :2].itertuples()])
        power = PowerCurve(tmp[:, 0], tmp[:, 1] / meta["Capacity"])
    return TurbineInfo(power, meta)


_Turbine_Library = None


def TurbineLibrary():
    """
    Reads turbine data cvs files

    Returns
    -------
    pandas Dataframe
        joined turbine data files
    """    
    global _Turbine_Library

    if _Turbine_Library is None:
        turbineFiles = glob(join(dirname(__file__), "data", "turbines", "*.csv"))

        tmp = []
        for f in turbineFiles:
            try:
                tmp.append(parse_turbine(f))
            except:
                print("failed to parse:", f)

        _Turbine_Library = pd.DataFrame([i.meta for i in tmp])
        _Turbine_Library.set_index('Model', inplace=True)
        _Turbine_Library['PowerCurve'] = [x.profile for x in tmp]

    return _Turbine_Library
