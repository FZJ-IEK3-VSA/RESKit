from os import listdir
from os.path import join, isfile, dirname, basename
from glob import glob
from scipy.interpolate import RectBivariateSpline, interp2d
from pickle import load, dump

from res.util import *

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
        yN,xN = values.shape
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
    def __init__(s, path, bounds=None, timeName="time", latName="lat", lonName="lon", dependent_coordinates=False, constantsPath=None):
        # set basic variables 
        s.path = path
        if not path is None:
            if constantsPath is None:
                dsC = nc.Dataset(s.path)
                ds = dsC
            else:
                dsC = nc.Dataset(constantsPath)
                ds = nc.Dataset(s.path)
        
            s._allLats = dsC[latName][:]
            s._allLons = dsC[lonName][:]
            
            s.variables = list(ds.variables.keys())
            s._maximal_lon_difference=10000000
            s._maximal_lat_difference=10000000

            s.dependent_coordinates = dependent_coordinates

            # set lat and lon selections
            if not bounds is None:
                if isinstance(bounds, gk.Extent):
                    if not bounds.srs.IsSame(gk.srs.EPSG4326):
                        bounds = bounds.castTo(gk.srs.EPSG4326)
                    bounds = Bounds(lonMin=bounds.xMin, latMin=bounds.yMin, lonMax=bounds.xMax, latMax=bounds.yMax )
                elif not isinstance(bounds, Bounds):
                    bounds = Bounds(*bounds)
                    print("bounds input is not a 'Bounds' or a 'geokit.Extent' type. Using one of these is safer!")
                s.bounds = bounds

                # find slices
                s._lonSel = (s._allLons >= bounds.lonMin) & (s._allLons <= bounds.lonMax)
                s._latSel = (s._allLats >= bounds.latMin) & (s._allLats <= bounds.latMax)

                if s.dependent_coordinates:
                    selTmp = s._latSel&s._lonSel
                    s._latSel = selTmp.any(axis=1)
                    s._lonSel = selTmp.any(axis=0)

                s._lonStart = np.argmax(s._lonSel)
                s._lonStop = s._lonSel.size-np.argmax(s._lonSel[::-1])
                s._latStart = np.argmax(s._latSel)
                s._latStop = s._latSel.size-np.argmax(s._latSel[::-1])

            else:
                s.bounds = None
                s._lonStart = 0
                s._latStart = 0

                if dependent_coordinates:
                    s._lonStop = s._allLons.shape[1]
                    s._latStop = s._allLons.shape[0]
                else:
                    s._lonStop = s._allLons.size
                    s._latStop = s._allLats.size

            if s.dependent_coordinates:
                s.lats = s._allLats[s._latStart:s._latStop,s._lonStart:s._lonStop]
                s.lons = s._allLons[s._latStart:s._latStop,s._lonStart:s._lonStop]
            else:
                s.lats = s._allLats[s._latStart:s._latStop]
                s.lons = s._allLons[s._lonStart:s._lonStop]

            # compute time index
            s.timeindex = nc.num2date(ds[timeName][:], ds[timeName].units)

            # initialize some variables
            s.data = OrderedDict()

    def __add__(s,o, _shell=None):
        # ensure self and other have the same indexes
        if (s.lats != o.lats).any(): raise ResError("Latitude indexes to not match")
        if (s.lons != o.lons).any(): raise ResError("Longitude indexes to not match")
        #if (s.timeindex != o.timeindex).any(): raise ResError("Longitude indexes to not match")

        # Create an empty NCSource and fill top-level information
        if _shell is None:
            out = NCSource(None)
        else:
            out = _shell

        out.bounds = s.bounds
        out.lats = s.lats
        out.lons = s.lons
        out.timeindex = s.timeindex
        out._maximal_lon_difference = s._maximal_lon_difference
        out._maximal_lat_difference = s._maximal_lat_difference
        out.dependent_coordinates = s.dependent_coordinates

        # Join variables in each object
        out.data = OrderedDict()
        for name,data in s.data.items():
            out.data[name] = data.copy()
        for name,data in o.data.items():
            out.data[name] = data.copy()
        
        out.variables = list(out.data.keys())

        # Done!
        return out
    
    def pickle(s, path):
        with open(path, 'wb') as fo:
            dump(s, fo)

    @staticmethod
    def fromPickle(path):
        with open(path, 'rb') as fo:
            out = load(fo)
        return out

    def load(s, variable, name=None, heightIdx=None, processor=None):
        if s.path is None: raise ResError("Cannot load new variables when path is None")
        # read the data
        ds = nc.Dataset(s.path)
        if not variable in ds.variables.keys():
            raise ResError(variable+" not in source")

        if heightIdx is None:
            tmp = ds[variable][:,s._latStart:s._latStop,s._lonStart:s._lonStop]
        else:
            tmp = ds[variable][:,heightIdx,s._latStart:s._latStop,s._lonStart:s._lonStop]

        # process, maybe?
        if not processor is None:
            tmp = processor(tmp)

        # save the data
        if name is None: name = variable
        s.data[name] = tmp

    def addData(s, name, data):
        # test shape
        if data.shape[0] != s.timeindex.shape[0]: raise ResError("Input data's first dimension does not match the time index")
        if s.dependent_coordinates:
            if data.shape[1] != s.lats.shape[0]: raise ResError("Input data's second dimension does not match the latitude dimension")
            if data.shape[2] != s.lons.shape[1]: raise ResError("Input data's second dimension does not match the longitude dimension")
        else:
            if data.shape[1] != s.lats.shape[0]: raise ResError("Input data's second dimension does not match the latitude dimension")
            if data.shape[2] != s.lons.shape[0]: raise ResError("Input data's second dimension does not match the longitude dimension")

        # Add to data
        s.data[name] = data

    def loc2Index(s, loc, outsideOkay=False):
        # Ensure loc is a list
        locations = Location.ensureLocation(loc, forceAsArray=True)

        # Get coordinates
        lonCoords = np.array([loc.lon for loc in locations])
        latCoords = np.array([loc.lat for loc in locations])

        # get closest indices
        idx = []
        for lat,lon in zip(latCoords, lonCoords):
            # Check the lat distance
            tmpLat = np.abs(lat-s.lats)
            if tmpLat.min() > s._maximal_lat_difference: 
                if not outsideOkay:
                    raise ResError("(%f,%f) are outside the boundaries"%(lat,lon))
                else:
                    idx.append(None)
                    continue

            # Check the lon distance
            tmpLon = np.abs(lon-s.lons)
            if tmpLon.min() > s._maximal_lon_difference: 
                if not outsideOkay:
                    raise ResError("(%f,%f) are outside the boundaries"%(lat,lon))
                else:
                    idx.append(None)
                    continue
            
            # Get the best indices 
            if s.dependent_coordinates:
                dist = np.sqrt(tmpLon*tmpLon+tmpLat*tmpLat)
                latI,lonI = np.unravel_index(np.argmin(dist),dist.shape)
            else:
                lonI = np.argmin(tmpLon)
                latI = np.argmin(tmpLat)
            
            # append
            idx.append( Index(yi=latI,xi=lonI) )

        # Make output
        if len(locations)==1:
            return idx[0]
        else:
            return idx

    def get(s, variable, locations, interpolation='near', forceDataFrame=False, outsideOkay=False):
        # Ensure loc is a list
        locations = Location.ensureLocation(locations, forceAsArray=True)

        # compute the closest indices
        indecies = s.loc2Index(locations, outsideOkay)
        if isinstance(indecies, Index): indecies = [indecies, ]

        # Do interpolation
        if interpolation == 'near':            
            # arrange the output data
            tmp = []
            for i in indecies:
                if not i is None: tmp.append(s.data[variable][:, i.yi, i.xi])
                else: tmp.append( np.array([np.nan,]*s.timeindex.size) ) 
            output = np.column_stack(tmp)
        
        elif interpolation == "cubic" or interpolation == "bilinear":
            if s.dependent_coordinates:
                raise ResError("Interpolation not setup for datasets with dependent lat/lon coordinates")

            ##########
            ## TODO: Update interpolation schemes to handle out-of-bounds indices 
            ##########
            # set some arguments for later use
            if interpolation == "cubic":
                win = 4
                rbsArgs = dict()
            else:
                win = 2
                rbsArgs = dict(kx=1, ky=1)

            # Find the minimal indexes needed
            yiMin = min([i.yi for i in indecies])-win
            yiMax = max([i.yi for i in indecies])+win
            xiMin = min([i.xi for i in indecies])-win
            xiMax = max([i.xi for i in indecies])+win

            # ensure boundaries are okay
            if yiMin < 0 or xiMin < 0 or yiMax > s.lats.size or xiMax > s.lons.size: 
                raise ResError("Insufficient data. Try expanding the boundary of the extracted data")

            # Set up grid
            gridLats = s.lats[yiMin:yiMax+1]
            gridLons = s.lons[xiMin:xiMax+1]
            
            # build output
            lats = [loc.y for loc in locations]
            lons = [loc.x for loc in locations]
            
            output = []
            for ts in range(s.data[variable].shape[0]):
                # set up interpolation
                rbs = RectBivariateSpline(gridLats,gridLons,s.data[variable][ts, yiMin:yiMax+1, xiMin:xiMax+1], **rbsArgs)

                # interpolate for each location
                output.append(rbs(lats, lons, grid=False)) # lat/lon order switched to match index order
     
            output = np.stack(output)

        else:
            raise ResError("Interpolation scheme not one of: 'near', 'cubic', or 'bilinear'")

        # Make output as Series objects
        if forceDataFrame or (len(output.shape)>1 and output.shape[1]>1):
            return pd.DataFrame(output, index=s.timeindex, columns=locations)
        else: 
            try:
                return pd.Series(output[:,0], index=s.timeindex, name=locations[0])
            except:
                return pd.Series(output, index=s.timeindex, name=locations[0])

    def contextAreaAt(s,location):
        # Get closest indexes
        index = s.loc2Index(location)
        # get area
        return s.contextAreaAtIndex(index.yi, index.xi)

    def contextAreaAtIndex(s, latI, lonI):
        if s.dependent_coordinates:
            ctr = np.array([s.lons[latI,lonI],s.lats[latI,lonI]])
            up = np.array([s.lons[latI+1,lonI],s.lats[latI+1,lonI]])
            dw = np.array([s.lons[latI-1,lonI],s.lats[latI-1,lonI]])
            rt = np.array([s.lons[latI,lonI+1],s.lats[latI,lonI+1]])
            lt = np.array([s.lons[latI,lonI-1],s.lats[latI,lonI-1]])

            up_rt = np.array([s.lons[latI+1,lonI+1],s.lats[latI+1,lonI+1]])
            dw_rt = np.array([s.lons[latI-1,lonI+1],s.lats[latI-1,lonI+1]])
            up_lt = np.array([s.lons[latI+1,lonI-1],s.lats[latI+1,lonI-1]])
            dw_lt = np.array([s.lons[latI-1,lonI-1],s.lats[latI-1,lonI-1]])

            # compute points in context
            mid_up_rt = (ctr + up + rt + up_rt)/4
            mid_dw_rt = (ctr + dw + rt + dw_rt)/4
            mid_up_lt = (ctr + up + lt + up_lt)/4
            mid_dw_lt = (ctr + dw + lt + dw_lt)/4

            # return polygon
            return gk.geom.polygon([(mid_up_rt), 
                                    (mid_dw_rt),
                                    (mid_dw_lt),
                                    (mid_up_lt),
                                    (mid_up_rt)], srs=gk.srs.EPSG4326)
        else:
            # Make and return a box
            lowLat = (s.lats[latI]+s.lats[latI-1])/2
            highLat = (s.lats[latI]+s.lats[latI+1])/2
            lowLon = (s.lons[lonI]+s.lons[lonI-1])/2
            highLon = (s.lons[lonI]+s.lons[lonI+1])/2
        
            # return box
            return gk.geom.box( lowLon, lowLat, highLon, highLat, srs=gk.srs.EPSG4326 )

    def computeContextMeans(s, source, fillnan=True):
        # get raster info
        ras = gk.raster.rasterInfo(source)

        # compute all means
        means = np.zeros((s.lats.size, s.lons.size))
        means[:] = np.nan

        for latI in range(1,s.lats.size-1):
            for lonI in range(1,s.lons.size-1):        
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