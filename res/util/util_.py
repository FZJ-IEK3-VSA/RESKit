import numpy as np
import netCDF4 as nc
import geokit as gk
from geokit import Location, LocationSet, Extent
import ogr, osr
import pandas as pd
from collections import namedtuple, OrderedDict
from scipy.interpolate import splrep, splev
from scipy.stats import norm
from glob import glob
import re
from os.path import join, dirname, basename, splitext
import types
from types import FunctionType
from datetime import datetime as dt

# making an error
class ResError(Exception): pass # this just creates an error that we can use

# Make easy access to latlon projection system
LATLONSRS = gk.srs.EPSG4326
LATLONSRS.__doc__ = "Spatial reference system for latitue and longitude coordinates"

## STAN
def storeTimeseriesAsNc(output, timedata, varmeta={}, keydata=None, keydatameta={}, timeunit="minutes since 1900-01-01 00:00:00"):
    """Create a netCDF4 file from a set of time series arrays.

        Parameters
        ----------
        output : str
            The path to write the netCDF4 file to

        timedata : DataFrame or { <varname>:DataFrame, ... }
            Two dimensional data which will be written to the netCDF4 file
              * All dataframes must share a time-index and columns names
              * If only a single DataFrame is given, a variable name of "var" is 
                assumed

        varmeta : dict or { <varname>:dict, ... }, optional
            Meta data to apply to the time-series variables
              * If timedata is a DataFrame, the varmeta dictionary will be applied 
                directly to the "var" variable
              * Variable names must match names given in timedata
              * Dict keys must be strings, and values must be strings or numerics
              * Example:
                varmeta={ "power_output":
                             { "name":"The power output of each wind turbine",
                               "units":"kWh", }, 
                        }
        
        keydata : DataFrame, optional
            Column-wise data to save for each key
              * Indexes must match the columns in the timedata DataFrames

        keydatameta : { <keyname>:dict, ... }, optional 
            Meta data to apply to the keydata variables
              * Dict keys must be strings, and values must be strings or numerics

        timeunit : str, optional
            The time unit to use when compressing the time index
              * Example: "Minutes since 01-01-1970"

        Returns
        -------
        None
        
    """
    # correct the time data
    if isinstance(timedata, pd.DataFrame):
        timedata = {"var":timedata, }
        varmeta = {"var":varmeta, }

    # Check the input data, just in case
    cols = list(timedata.keys())
    if len(cols)>1:
        for i in range(1,len(cols)):
            if not (timedata[cols[i]].columns == timedata[cols[0]].columns).all():
                raise RuntimeError("timedata columns do not match eachother")

            if not (timedata[cols[i]].index == timedata[cols[0]].index).all():
                raise RuntimeError("timedata indicies do not match eachother")

    # Make an output file
    ds = nc.Dataset(output, mode="w")
    try:
        # Make the dimensions
        ds.createDimension("time", size=timedata[cols[0]].shape[0])
        ds.createDimension("key", size=timedata[cols[0]].shape[1])

        # Make the time variable
        timeV = ds.createVariable("time", "u4", dimensions=("time",), contiguous=True)
        timeV.units = timeunit

        times = timedata[cols[0]].index
        if timedata[cols[0]].index[0].tz is None:
            timeV.tz = "unknown"
        else:
            timeV.tz = timedata[cols[0]].index[0].tzname()
            times = times.tz_localize(None)

        timeV[:] = nc.date2num(times.to_pydatetime(), timeunit)

        # Make the data variables
        for varN, tab in timedata.items():
            var = ds.createVariable(varN, tab.iloc[0,0].dtype, dimensions=("time", "key",), contiguous=True)
            if varN in varmeta and len(varmeta[varN])>0:
                var.setncatts(varmeta[varN])
            var[:] = tab.values

        # Make some key variables, maybe
        if not keydata is None:
            # test if the keys are in the right order
            if not (timedata[cols[0]].columns == keydata.index).all():
                raise RuntimeError("timedata columns do not match keydata indecies")

            for col in keydata.columns:
                dtype = str if keydata[col].dtype == np.dtype("O") else keydata[col].dtype
                var = ds.createVariable(col, dtype, dimensions=( "key",), contiguous=True)
                if col in keydatameta and len(keydatameta[col])>0:
                    var.setncatts(keydatameta[col])
                
                var[:] = keydata[col].values if not dtype is str else keydata[col].values.astype(np.str)
        ds.close()
        
    except Exception as e:
        ds.close() # make sure the ds is closed!
        raise e

    # All done!
    return


## Make basic helper functions
def removeLeapDay(x):
    if isinstance(x, pd.Series) or isinstance(s, pd.DataFrame):
        times = x.index
        sel = np.logical_and((times.day==29), (times.month==2))
        if isinstance(x, pd.Series): return x[~sel]
        else: return x.loc[~sel]

    elif isinstance(x, np.ndarray) and x.shape[0] == 8784:
        times = pd.date_range("01-01-2000 00:00:00", "12-31-2000 23:00:00", freq="H")
        sel = np.logical_and((times.day==29), (times.month==2))
        if len(x.shape)==1: return x[~sel]
        else: return x[~sel,:]

    else:
        return removeLeapDay(np.array(x))

def linearTransition(x, start, stop, invert=False):
    tmp = np.zeros(x.shape)

    s = x<=start
    tmp[s] = 0

    s = (x>start)&(x<=stop)
    tmp[s] = (x[s]-start)/(stop-start)

    s = x>stop
    tmp[s]=1

    if invert: return 1-tmp
    else: return tmp

## Parse Generation File
_SGF = namedtuple("RESGeneration", "capacity capex generation regionName variable capacityUnit capexUnit generationUnit")
def parseRESGenerationFile(f, capacity, generationName="generation"):
    ds = nc.Dataset(f)
    try:
        timeIndex = nc.num2date(ds["time"][:], ds["time"].units)
        CAP = ds["total_capacity"][:]
        COST = ds["total_cost"][:]

        try:
            capacity = list(capacity)
        except:
            capacity = [capacity,]

        def atCapacity(cap):
            s = np.argmin(np.abs(CAP-cap))

            if CAP[s] == cap: 
                gen = ds["generation"][:,s]
                capex = ds["total_cost"][s]
            else:
                if CAP[s] > cap: low, high = s-1,s
                else: low, high = s,s+1

                raw = ds["generation"][:,[low, high]]

                factor = (cap-CAP[low])/(CAP[high]-CAP[low])
                
                gen = raw[:,0]*(1-factor) + raw[:,1]*factor

                lowCost, highCost = ds["total_cost"][[low,high]]
                capex = lowCost*(1-factor) + highCost*factor
            return gen, capex

        generations = pd.DataFrame(index=timeIndex,)
        capexes = []
        for cap in capacity:
            gen,capex = atCapacity(cap)
            generations[cap] = gen
            capexes.append(capex)

    except Exception as e:
        ds.close()
        raise e

    return _SGF(capacity=np.array(capacity), capex=np.array(capexes), generation=generations, 
                regionName=ds["generation"].region, variable=ds["generation"].technology,
                capacityUnit=ds["total_capacity"].unit, capexUnit=ds["total_cost"].unit, 
                generationUnit=ds["generation"].unit)