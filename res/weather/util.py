import numpy as np
import netCDF4 as nc
import geokit as gk
import ogr, osr
from os.path import join, isfile, dirname, basename
from glob import glob
import pandas as pd
from collections import namedtuple, OrderedDict
from scipy.interpolate import RectBivariateSpline, interp2d

Index = namedtuple("Index", "yi xi")
Location = namedtuple("Location", "x y")

# making an error
class ResWeatherError(Exception): pass # this just creates an error that we can use

# Make a single data container
class GenericElement(object):
    def __init__(s, name, data, times, xCoords, yCoords, coordSRS, meta=None):
        s.data = data
        s.times = times
        s.xCoords = xCoords
        s.yCoords = yCoords

        s._xMin = s.xCoords.min()
        s._xMax = s.xCoords.max()
        s._yMin = s.yCoords.min()
        s._yMax = s.yCoords.max()

        s.coordSRS = coordSRS
        s.name = name

        s.meta = OrderedDict()
        if not meta is None:
            s.meta.update(meta)

        s._interpolation = 'near'

    def setInterpolationMode(s, mode=None):
        if mode is None: mode='near'
        if not mode in ['near', "cubic", "linear"]:
            raise ResWeatherError("mode not recognized")

        s._interpolation = mode


    def __getitem__(s,loc):
        # Check if the location input is a point geometry
        if isinstance(loc, ogr.Geometry):
            if not loc.GetGeometryName()=="POINT":
                raise ResWeatherError("location geometry is not a Point object")
            
            # make sure the input is in the correct reference system
            locSRS = loc.GetSpatialReference()
            if not locSRS is None: 
                if not locSRS.IsSame(s.coordSRS):
                    loc.TransformTo(s.coordSRS)

            # Get x and y coordinates
            x = loc.GetX()
            y = loc.GetY()

        else: # input should be a tuple in the correct SRS
            x = loc[0]
            y = loc[1]
            
        # make sure we're in range
        if not (x>s._xMin and x<s._xMax): raise ResWeatherError("Input exceeds X boundary")
        if not (y>s._yMin and y<s._yMax): raise ResWeatherError("Input exceeds Y boundary")

        # Compute the best indicies
        xDist = x - s.xCoords
        yDist = y - s.yCoords

        totalDist2 = xDist*xDist + yDist*yDist

        yi,xi = np.argwhere(totalDist2 == np.min(totalDist2))[0]

        # Interpolate and return
        if s._interpolation=='near':
            outdata = s.data[:,yi,xi]
        else:
            if s._interpolation=="cubic": 
                width=3
                kwargs = dict()
            elif s._interpolation=="linear": 
                width=1
                kwargs = dict(kx=1, ky=1)
            else:
                raise ResWeatherError("Bad interpolation scheme")

            # Make sure the contained data is sufficient
            if yi<width or yi>s.yCoords.shape[0]-width or xi<width or xi>s.xCoords.shape[0]-width:
                raise ResWeatherError("Insufficient spatial extents for interpolation")

            # Setup interpolator
            yVals = s.yCoords[yi-width:yi+width+1, xi-width:xi+width+1]
            xVals = s.xCoords[yi-width:yi+width+1, xi-width:xi+width+1]

            def interpolate(timeIndex):
                data = s.data[timeIndex, yi-width:yi+width+1, xi-width:xi+width+1]
                
                # THIS CAN BE MADE MUCH FASTER IF I USE RectBivariateSpline, BUT I NEED TO 
                #  FIGURE OUT HOW BEST TO IMPLEMENT IT
                #rbs = RectBivariateSpline(yVals, xVals, data, **kwargs)
                rbs = interp2d( yVals, xVals, data, kind=s._interpolation) # I NEED TO IMPROVE THIS!!!
                return rbs(y,x)[0]

            # do interpolations
            outdata = [interpolate(i) for i in range(s.times.shape[0])]

        return pd.Series(outdata, index=s.times, name=s.name)
        
# make a generic datasource
class GenericSource(object):
    _GWA50_AVERAGE_SOURCE = None
    _GWA100_AVERAGE_SOURCE = None
    _WINDSPEED_NORMALIZER_SOURCE = None
    _WINDSPEED_NORMALIZER_VAR = None

    def __init__(s, timeframe, xCoords, yCoords, coordSRS='latlon'):
        s.source = "unknown"

        # arrange time index
        if isinstance(timeframe,int):
            s.timeStart = pd.Timestamp(year=timeframe, month=1, day=1, hour=0)
            s.timeEnd = pd.Timestamp(year=timeframe, month=12, day=31, hour=23)
        else:
            s.timeStart = pd.Timestamp(timeframe[0])
            s.timeEnd = pd.Timestamp(timeframe[1])

        # arange point coordinates
        s.coordSRS = gk.srs.loadSRS(coordSRS)

        s.xCoords = np.array(xCoords)
        s.yCoords = np.array(yCoords)

        if len(xCoords.shape) == 1: # make it two dimensional
            s.xCoords, s.yCoords = np.meshgrid(s.xCoords, s.yCoords)

        s._xMin = s.xCoords.min()
        s._xMax = s.xCoords.max()
        s._yMin = s.yCoords.min()
        s._yMax = s.yCoords.max()

        # initialize an empty data container
        s.data = OrderedDict()
        s.dataShape = xCoords.shape

    def loc2Index(s, loc):
        # Check if the location input is a point geometry
        if isinstance(loc, ogr.Geometry):
            if not loc.GetGeometryName()=="POINT":
                raise ResWeatherError("location geometry is not a Point object")
            
            # make sure the input is in the correct reference system
            locSRS = loc.GetSpatialReference()
            if not locSRS is None: 
                if not locSRS.IsSame(s.coordSRS):
                    loc.TransformTo(s.coordSRS)

            # Get x and y coordinates
            x = loc.GetX()
            y = loc.GetY()

        else: # input should be a tuple in the correct SRS
            x = loc[0]
            y = loc[1]
            
        # make sure we're in range
        if not (x>=s._xMin and x<=s._xMax): raise ResWeatherError("Input exceeds X boundary")
        if not (y>=s._yMin and y<=s._yMax): raise ResWeatherError("Input exceeds Y boundary")

        # Compute the best indicies
        xDist = x - s.xCoords
        yDist = y - s.yCoords

        totalDist2 = xDist*xDist + yDist*yDist

        yi,xi = np.argwhere(totalDist2 == np.min(totalDist2))[0]

        # done!
        return Index(yi,xi)

    def gwaContextAverage(s, loc, height=50):
        if height == 50:
            return gk.raster.extractValues(s._GWA50_AVERAGE_SOURCE, loc).data
        elif height == 100:
            return gk.raster.extractValues(s._GWA100_AVERAGE_SOURCE, loc).data
        else:
            raise ResWeatherError("height is not 50 or 100")

    def setInterpolationMode(s, mode=None):
        for k,v in s.data.items():
            v.setInterpolationMode(mode)

    def setElement(s, name, data, times, meta=None):
        s.data[name] = GenericElement(name, data, times, xCoords=s.xCoords, yCoords=s.yCoords, coordSRS=s.coordSRS, meta=meta)

    def __getitem__(s, i):
        if isinstance(i, str):
            return s.data[i]
        else:
            output = OrderedDict()
            for k,v in s.data.items():
                output[k] = v[i]

            return pd.DataFrame(output)