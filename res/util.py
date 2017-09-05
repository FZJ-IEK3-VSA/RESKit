import numpy as np
import netCDF4 as nc
import geokit as gk
import ogr, osr
import pandas as pd
from collections import namedtuple, OrderedDict
import types

# making an error
class ResError(Exception): pass # this just creates an error that we can use

# Make some type-helpers
LATLONSRS = gk.srs.EPSG4326
Index = namedtuple("Index", "yi xi")
Location = namedtuple("Location", "x y")
def LatLonLocation(lat, lon):
        return Location(x=lon, y=lat)

BoundsNT = namedtuple("Bounds","lonMin latMin lonMax latMax")
class Bounds(BoundsNT):
    #def __init__(s, lonMin, latMin, lonMax, latMax):
    #    BoundsNT.__init__(s, lonMin, latMin, lonMax, latMax)
    def __str__(s):
        out =  "Lat: %.4f  -  %.4f\n"%(s.latMin, s.latMax)
        out += "Lon: %.4f  -  %.4f"%(s.lonMin, s.lonMax)
        return out


def ensureList(a):
    # Ensure loc is a list
    if isinstance(a, list) or isinstance(a, np.ndarray):
        pass    
    elif isinstance(a, types.GeneratorType):
        a = list(a)
    else:
        a = [a, ]
    # Done!
    return a

def ensureGeom(locations):
    if isinstance(locations, list) or isinstance(locations, np.ndarray):
        if isinstance(locations[0], ogr.Geometry): # Check if loc is a list of point
            if not locations[0].GetSpatialReference().IsSame(LATLONSRS):
                locations = gk.geom.transform(locations, toSRS=LATLONSRS)
        elif isinstance(locations[0], Location):
            locations = [gk.geom.point(loc.x, loc.y, srs=LATLONSRS) for loc in locations]
        else:
            raise ResError("Cannot understand location input. Use either a Location or an ogr.Geometry object")

    elif isinstance(locations, types.GeneratorType):
        locations = ensureGeom(list(locations))

    elif isinstance(locations, ogr.Geometry): # Check if loc is a single point
        if not locations.GetSpatialReference().IsSame(LATLONSRS):
            locations = locations.Clone()
            locations.TransformTo(LATLONSRS)

    elif isinstance(locations, Location):
        locations = gk.geom.point(locations.x, locations.y, srs=LATLONSRS)
    elif isinstance(locations, tuple) and len(locations)==2:
        locations = gk.geom.point(locations[0], locations[1], srs=LATLONSRS)
        print("Consider using a Location object. It is safer!")
    else:
        raise ResError("Cannot understand location input. Use either a Location or an ogr.Geometry object")

    # Done!
    return locations

def ensureLoc(locations):
    if isinstance(locations, list) or isinstance(locations, np.ndarray):
        if isinstance(locations[0], ogr.Geometry): # Check if loc is a list of point
            if not locations[0].GetSpatialReference().IsSame(LATLONSRS):
                locations = gk.geom.transform(locations, toSRS=LATLONSRS)
            locations = [Location(x=l.GetX(), y=l.GetY()) for l in locations]
        elif isinstance(locations[0], Location):
            pass
        else:
            raise ResError("Cannot understand location input. Use either a Location or an ogr.Geometry object")

    elif isinstance(locations, types.GeneratorType):
        locations = ensureLoc(list(locations))

    elif isinstance(locations, ogr.Geometry): # Check if loc is a single point
        if not locations.GetSpatialReference().IsSame(LATLONSRS):
            locations = locations.Clone()
            locations.TransformTo(LATLONSRS)
            locations = Location(x=locations.GetX(), y=locations.GetY())

    elif isinstance(locations, Location):
        pass
    elif isinstance(locations, tuple) and len(locations)==2:
        locations = Location(*locations)
        print("Consider using a Location object. It is safer!")
    else:
        raise ResError("Cannot understand location input. Use either a Location or an ogr.Geometry object")

    # Done!
    return locations

## STAN
def storeTimeseriesAsNc(output, timedata, varmeta={}, keydata=None, keydatameta={}, timeunit="minutes since 1900-01-01 00:00:00"):
    """
    Create a netCDF4 file from a set of time series arrays

    Inputs:
        output : str -- An output file path

        timedata
            Pandas-DataFrame -- The time-series data to write
                * Must be time-indexed
                * Will be written with the variable name "var"
            { <varname>:Pandas-DataFrame, } -- A dictionary of variable names to DataFrames
                * All variables will be written to the output file
                * All DataFrames must share the same index and columns

        varmeta : dict -- Optional meta data to apply to the time-series variables
            * If time data is a DataFrame, the varmeta dictionary will be applied directly to the "var" variable
            * Otherwise varmeta needs to be a dictionary of dictionaries
            * Example:
                varmeta = { "power_output":{ "name":"The power output of each turbine",
                                             "units":"kWh", } }
        
        keydata : Pandas-DataFrame -- Optional data to save for each key
            * Must be a pandas DataFrame whose index matches the columns in the timedata DataFrames
            * Could be, for example, the hub height of each turbine or a the model

        keydatameta : dict -- Optional meta data to apply to the key data variables
            * Must be a dictionary of dictionaries

        timeunit : str -- The time unit to use when compressing the time index

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
        timeV[:] = nc.date2num(timedata[cols[0]].index.to_pydatetime(), timeunit)

        # Make the Key variable
        keyV = ds.createVariable("keyID", "u4", dimensions=("key",), contiguous=True)
        keyV.units = ""
        keyV[:] = np.arange(timedata[cols[0]].shape[1])

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
    except Exception as e:
        ds.close() # make sure the ds is closed!
        raise e

    # All done!
    return