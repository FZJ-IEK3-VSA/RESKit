import numpy as np
import pandas as pd

from . import ResError


def remove_leap_day(timeseries):
    """Removes leap days from a given timeseries

    Parameters
    ----------
    timeseries : array_like
        The time series data to remove leap days from
          * If something array_like is given, the length must be 8784
          * If a pandas DataFrame or Series is given, time indexes will be used
            directly

    Returns
    -------
    Array

    """
    if isinstance(timeseries, np.ndarray):
        if timeseries.shape[0] == 8760:
            return timeseries
        elif timeseries.shape[0] == 8784:
            times = pd.date_range("01-01-2000 00:00:00",
                                  "12-31-2000 23:00:00", freq="H")
            sel = np.logical_and((times.day == 29), (times.month == 2))
            if len(timeseries.shape) == 1:
                return timeseries[~sel]
            else:
                return timeseries[~sel, :]
        else:
            raise ResError('Cannot handle array shape ' +
                           str(timeseries.shape))

    elif isinstance(timeseries, pd.Series) or isinstance(timeseries, pd.DataFrame):
        times = timeseries.index
        sel = np.logical_and((times.day == 29), (times.month == 2))
        if isinstance(timeseries, pd.Series):
            return timeseries[~sel]
        else:
            return timeseries.loc[~sel]

    else:
        return remove_leap_day(np.array(timeseries))
