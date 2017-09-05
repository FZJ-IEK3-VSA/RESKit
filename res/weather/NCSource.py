from os import listdir
from os.path import join, isfile, dirname, basename
from glob import glob
from scipy.interpolate import RectBivariateSpline, interp2d

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
    def __init__(s, path, bounds=None, timeName="time", latName="lat", lonName="lon"):
        # set basic vairables 
        s.path = path
        ds = nc.Dataset(s.path)
        s._allLats = ds[latName][:]
        s._allLons = ds[lonName][:]
        s.variables = list(ds.variables.keys())
        s._maximal_lon_difference=10000000
        s._maximal_lat_difference=10000000

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
            s._lonSel = np.logical_and(s._allLons >= bounds.lonMin, s._allLons <= bounds.lonMax)
            s._latSel = np.logical_and(s._allLats >= bounds.latMin, s._allLats <= bounds.latMax)
        else:
            s.bounds = None
            s._latSel = np.s_[:]
            s._lonSel = np.s_[:]

        s.lats = s._allLats[s._latSel]
        s.lons = s._allLons[s._lonSel]

        # compute time index
        s.timeindex = nc.num2date(ds[timeName][:], ds[timeName].units)

        # initialize some variables
        s.data = OrderedDict()

    def load(s, variable, name=None, processor=None):
        # read the data
        ds = nc.Dataset(s.path)
        if not variable in ds.variables.keys():
            raise ResError(variable+" not in source")
        tmp = ds[variable][:,s._latSel,s._lonSel]

        # process, maybe?
        if not processor is None:
            tmp = processor(tmp)

        # save the data
        if name is None: name = variable
        s.data[name] = tmp


    def loc2Index(s, loc, outsideOkay=False):
        # Ensure loc is a list
        locations = ensureLoc(ensureList(loc))

        # Get coordinates
        lonCoords = np.array([loc.x for loc in locations])
        latCoords = np.array([loc.y for loc in locations])

        # get closest indecies
        idx = []
        for lat,lon in zip(latCoords, lonCoords):
            # get lat index
            tmp = np.abs(lat-s.lats)
            if tmp.min() > s._maximal_lat_difference: 
                if not outsideOkay:
                    raise ResError("(%f,%f) are outside the boundaries"%(lat,lon))
                else:
                    idx.append(None)
                    continue
            latI = np.argmin(tmp)

            # get lon index
            tmp = np.abs(lon-s.lons)
            if tmp.min() > s._maximal_lon_difference: 
                if not outsideOkay:
                    raise ResError("(%f,%f) are outside the boundaries"%(lat,lon))
                else:
                    idx.append(None)
                    continue
            lonI = np.argmin(tmp)
            
            # append
            idx.append( Index(yi=latI,xi=lonI) )

        # Make output
        if len(locations)==1:
            return idx[0]
        else:
            return idx

    def get(s, variable, locations, interpolation='near', forceDataFrame=False, outsideOkay=False):
        # Ensure loc is a list
        locations = ensureLoc(ensureList(locations))

        # compute the closest indecies
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
            ##########
            ## TODO: Update interpolation schemes to handle out-of-bounds indecies 
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
                raise ResError("Insuffecient data. Try expanding the boundary of the extracted data")

            # Set up grid
            gridLats = s.lats[yiMin:yiMax+1]
            gridLons = s.lons[xiMin:xiMax+1]
            
            # build output
            lats = [loc.y for loc in locations]
            lons = [loc.x for loc in locations]
            
            output = []
            for ts in range(s.data[variable].shape[0]):
                # set up interpolator
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
        return s.contextAreaAtIndex(index.yi, index.xi)

    def contextAreaAtIndex(s, latI, lonI):

        # Make and return a box
        lowLat = (s.lats[latI]+s.lats[latI-1])/2
        highLat = (s.lats[latI]+s.lats[latI+1])/2
        lowLon = (s.lons[lonI]+s.lons[lonI-1])/2
        highLon = (s.lons[lonI]+s.lons[lonI+1])/2
        
        return gk.geom.box( lowLon, lowLat, highLon, highLat, srs=gk.srs.EPSG4326 )

    def computeContextMeans(s, source, fillnan=True):
        # get rastr info
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