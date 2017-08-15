import numpy as np
import netCDF4 as nc
import geokit as gk
import ogr, osr
from os import listdir
from os.path import join, isfile, dirname, basename
from glob import glob
import pandas as pd
from collections import namedtuple, OrderedDict
from scipy.interpolate import RectBivariateSpline, interp2d
import types

from res.util import *

# Make some type-helpers
Index = namedtuple("Index", "yi xi")
Location = namedtuple("Location", "x y")
def LatLonLocation(lat, lon):
        return Location(x=lon, y=lat)
Bounds = namedtuple("Bounds","lonMin latMin lonMax latMax")

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
            if not locations[0].GetSpatialReference().IsSame(gk.srs.EPSG4326):
                locations = gk.geom.transform(locations, toSRS=gk.srs.EPSG4326)
        elif isinstance(locations[0], Location):
            locations = [gk.geom.point(loc.x, loc.y, srs=gk.srs.EPSG4326) for loc in locations]
        else:
            raise ResError("Cannot understand location input. Use either a Location or an ogr.Geometry object")

    elif isinstance(locations, types.GeneratorType):
        locations = ensureGeom(list(locations))

    elif isinstance(locations, ogr.Geometry): # Check if loc is a single point
        if not locations.GetSpatialReference().IsSame(gk.srs.EPSG4326):
            locations = locations.Clone()
            locations.TransformTo(gk.srs.EPSG4326)

    elif isinstance(locations, Location):
        locations = gk.geom.point(locations.x, locations.y, srs=gk.srs.EPSG4326)
    elif isinstance(locations, tuple) and len(locations)==2:
        locations = gk.geom.point(locations[0], locations[1], srs=gk.srs.EPSG4326)
        print("Consider using a Location object. It is safer!")
    else:
        raise ResError("Cannot understand location input. Use either a Location or an ogr.Geometry object")

    # Done!
    return locations

def ensureLoc(locations):
    if isinstance(locations, list) or isinstance(locations, np.ndarray):
        if isinstance(locations[0], ogr.Geometry): # Check if loc is a list of point
            if not locations[0].GetSpatialReference().IsSame(gk.srs.EPSG4326):
                locations = gk.geom.transform(locations, toSRS=gk.srs.EPSG4326)
            locations = [Location(x=l.GetX(), y=l.GetY()) for l in locations]
        elif isinstance(locations[0], Location):
            pass
        else:
            raise ResError("Cannot understand location input. Use either a Location or an ogr.Geometry object")

    elif isinstance(locations, types.GeneratorType):
        locations = ensureLoc(list(locations))

    elif isinstance(locations, ogr.Geometry): # Check if loc is a single point
        if not locations.GetSpatialReference().IsSame(gk.srs.EPSG4326):
            locations = locations.Clone()
            locations.TransformTo(gk.srs.EPSG4326)
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

# context mean
def computeContextMean(source, contextArea, fillnan=True, pixelSize=None, srs=None):
    if pixelSize is None and srs is None:
        ras = gk.raster.rasterInfo(source)
        pixelSize=(ras.dx, ras.dy)
        srs=ras.srs

    # get all GWA values in the area
    rm = gk.RegionMask.fromGeom(contextArea, pixelSize=pixelSize, srs=srs)
    values = rm.warp(source)
    values[values<0] = np.nan # set the no data value so they can be filled properly

    # fill all no data with the closest values
    if fillnan:
        while np.isnan(values[rm.mask]).any():
            tmp = values.copy()
            for yi,xi in np.argwhere(np.isnan(values)):
                vals = np.array([np.nan, np.nan, np.nan, np.nan])
                if yi>0:    vals[0] = values[yi-1,xi]
                if xi>0:    vals[1] = values[yi,xi-1]
                if yi<yN-1: vals[2] = values[yi+1,xi]
                if xi<xN-1: vals[3] = values[yi,xi+1]

                if (~np.isnan(vals)).any():
                    tmp[yi,xi] = np.nanmean(vals)
            values = tmp
    # done!
    return np.nanmean(values[rm.mask])


# make a data handler
class NCSource(object):
    def __init__(s, path, bounds=None, timeName="time", latName="lat", lonName="lon"):
        # set basic vairables 
        s.path = path
        s.ds = nc.Dataset(path)
        s._allLats = s.ds[latName][:]
        s._allLons = s.ds[lonName][:]
        s.variables = s.ds.variables.keys()

        # set lat and lon selections
        if not bounds is None:
            if isinstance(bounds, gk.Extent):
                if not bounds.srs.IsSame(gk.srs.EPSG4326):
                    bounds = bounds.castTo(gk.srs.EPSG4326)
                bounds = Bounds(lonMin=bounds.xMin, latMin=bounds.yMin, lonMax=bounds.xMax, latMax=bounds.yMax )
            elif not isinstance(bounds, Bounds):
                bounds = Bounds(*bounds)
                print("bounds input is not a 'Bounds' or a 'geokit.Extent' type. Using one of these is safer!")
            
            # find slices
            s._lonSel = np.logical_and(s._allLons >= bounds.lonMin, s._allLons <= bounds.lonMax)
            s._latSel = np.logical_and(s._allLats >= bounds.latMin, s._allLats <= bounds.latMax)
        else:
            s._latSel = np.s_[:]
            s._lonSel = np.s_[:]

        s.lats = s._allLats[s._latSel]
        s.lons = s._allLons[s._lonSel]

        # compute time index
        s.timeindex = nc.num2date(s.ds[timeName][:], s.ds[timeName].units)

        # initialize some variables
        s.data = OrderedDict()

    def load(s, variable, name=None, processor=None):
        # read the data
        if not variable in s.ds.variables.keys():
            raise ResError(variable+" not is source")
        tmp = s.ds[variable][:,s._latSel,s._lonSel]

        # process, maybe?
        if not processor is None:
            tmp = processor(tmp)

        # save the data
        if name is None: name = variable
        s.data[variable] = tmp

    def loc2Index(s, loc):
        # Ensure loc is a list
        locations = ensureLoc(ensureList(loc))

        # Get coordinates
        lonCoords = np.array([loc.x for loc in locations])
        latCoords = np.array([loc.y for loc in locations])

        # get closest indecies
        lonIndecies = [np.argmin(np.abs(lon-s.lons)) for lon in lonCoords]
        latIndecies = [np.argmin(np.abs(lat-s.lats)) for lat in latCoords]

        # Make output
        if len(locations)==1:
            return Index(yi=latIndecies[0], xi=lonIndecies[0])
        else:
            return [Index(yi=latIndex, xi=lonIndex) for latIndex,lonIndex in zip(latIndecies, lonIndecies)]

    def get(s, variable, locations, interpolation='near', forceDataFrame=False):
        # Ensure loc is a list
        locations = ensureLoc(ensureList(locations))

        # Do interpolation
        if interpolation == 'near':
            # compute the closest indecies
            indecies = s.loc2Index(locations)
            if isinstance(indecies, Index): indecies = [indecies, ]

            # arrange the output data
            output = [s.data[variable][:, i.yi, i.xi] for i in indecies]

        else:
            raise ResError("No other interpolation schemes are implemented at this time :(")

        # Make output as Series objects
        if forceDataFrame or len(output) > 1:
            return pd.DataFrame(np.column_stack(output), index=s.timeindex, columns=locations)
        else: 
            return pd.Series(output[0], index=s.timeindex, name=locations[0])

    def contextAreaAt(s,location):
        # Ensure we have a Location
        if isinstance(location, ogr.Geometry):
            # test if location is in lat & lon coordinates
            if not location.GetSpatialReference().IsSame(gk.srs.EPSG4326):
                location.TransformTo(gk.srs.EPSG4326)
            location = Location(x=location.GetX(), y=location.GetY())
        elif not isinstance(location, Location):
            raise ResError("Cannot understand location input. Use either a Location or an ogr.Geometry object")

        # Get closest indexes
        index = s.loc2Index(location)
        # get area
        return s._contextAreaAt(index.yi, index.xi)

    def _contextAreaAt(s, latI, lonI):

        # Make and return a box
        lowLat = (s.lats[latI]+s.lats[latI-1])/2
        highLat = (s.lats[latI]+s.lats[latI+1])/2
        lowLon = (s.lons[lonI]+s.lons[lonI-1])/2
        highLon = (s.lons[lonI]+s.lons[lonI+1])/2
        
        return gk.geom.box( lowLon, lowLat, highLon, highLat, srs=gk.srs.EPSG4326 )

    def computeContextMeans(s, source, output=None, fillnan=True):
        # get rastr info
        ras = gk.raster.describe(source)

        # compute all means
        means = np.zeros((s.lats.size, s.lons.size))
        for latI,lat in enumerate(s.lats):
            for lonI,lon in enumerate(s.lons):        
                means[latI,lonI] = computeContextMean(source=source, contextArea=s._contextAreaAt(latI,lonI), 
                                                      pixelSize=(ras.dx, ras.dy), srs=ras.srs, 
                                                      fillnan=fillnan)

        return means

'''
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
'''