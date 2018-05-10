from os import listdir
from os.path import join, isfile, dirname, basename
from glob import glob
from scipy.interpolate import RectBivariateSpline, interp2d
from pickle import load, dump

from res.util import *

# context mean
def computeContextMean(source, contextArea, fillnan=True, pixelSize=None, srs=None):
    """Compute the mean of raster data found in the context of an area. Usually this is used for one of the weather data sources in order to compute the, for example, mean elevation around each grid cell.

    Inputs:
        source : 
            str - The path to a raster file to read from
            gdal.Dataset - The raster file to read from in memory
            * Must contain the entire context area (nan values are okay)

        contextArea : 
            ogr.Geometry - The context area

        fillnan : 
            T/F - Dictates whether nan values are filled using the 'closest real value' approach

        pixelSize : 
            float - The pixel size to use wile sampling the raster file
            * Normally this should be left as 'None', indicating that the raster's inherent resolution should be used

        srs : The spatial reference system to use while processing
            int - A EPSG reference integer
            str - One of GeoKit's common srs keys 
            osr.SpatialReference - An SRS object
            * Normally this should be left as 'None', indicating that the raster's inherent SRS should be used
    """


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
Index = namedtuple("Index", "yi xi")
class NCSource(object):
    def _loadDS(s, path):
        if isinstance(path, str):
            return nc.Dataset(path)
        elif isinstance(path, list):
            return nc.MFDataset( path, aggdim=s.timeName)
        else:
            raise ResError("Could not understand data source input. Must be a path or a list of paths")

    def __init__(s, path, bounds=None, padFactor=0, timeName="time", latName="lat", lonName="lon", dependent_coordinates=False, constantsPath=None, timeBounds=None, _maxLonDiff=10000000, _maxLatDiff=10000000):
        # set basic variables 
        s.path = path
        s.timeName = timeName
        if not path is None:
            if constantsPath is None:
                dsC = s._loadDS(s.path)
                ds = dsC
            else:
                dsC = s._loadDS(constantsPath)
                ds = s._loadDS(s.path)
        
            s._allLats = dsC[latName][:]
            s._allLons = dsC[lonName][:]
            
            s.variables = list(ds.variables.keys())
            s._maximal_lon_difference=_maxLonDiff
            s._maximal_lat_difference=_maxLatDiff

            s.dependent_coordinates = dependent_coordinates

            # set lat and lon selections
            if not bounds is None:
                if isinstance(bounds, gk.Extent):
                    lonMin,latMin,lonMax,latMax = bounds.castTo(LATLONSRS).xyXY

                elif isinstance(bounds, gk.Location):
                    lonMin=bounds.lon
                    latMin=bounds.lat
                    lonMax=bounds.lon
                    latMax=bounds.lat

                elif isinstance(bounds, gk.LocationSet):
                    bounds = bounds.getBounds()

                elif isinstance(bounds, Bounds):
                    lonMin = bounds.lonMin
                    latMin = bounds.latMin
                    lonMax = bounds.lonMax
                    latMax = bounds.latMax
                else:
                    try:
                        lon,lat = bounds

                        lonMin=lon
                        latMin=lat
                        lonMax=lon
                        latMax=lat

                    except:
                        lonMin,latMin,lonMax,latMax = bounds
                
                # Add padding to the boundaries
                s.bounds = Bounds(lonMin = lonMin - s._maximal_lon_difference*padFactor,
                                  latMin = latMin - s._maximal_lat_difference*padFactor,
                                  lonMax = lonMax + s._maximal_lon_difference*padFactor,
                                  latMax = latMax + s._maximal_lat_difference*padFactor,)

                # find slices
                s._lonSel = (s._allLons >= s.bounds.lonMin) & (s._allLons <= s.bounds.lonMax)
                s._latSel = (s._allLats >= s.bounds.latMin) & (s._allLats <= s.bounds.latMax)

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
            timeindex = nc.num2date(ds[s.timeName][:], ds[s.timeName].units)
            
            if timeBounds is None:
                s._timeSel = np.s_[:]
            else:
                timeStart = pd.Timestamp(timeBounds[0])
                timeEnd = pd.Timestamp(timeBounds[1])
                s._timeSel = (timeindex >= timeStart) & (timeindex <= timeEnd)

            s.timeindex = timeindex[s._timeSel]

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
        """Save the source as a pickle file, so it can be quickly reopened later"""
        with open(path, 'wb') as fo:
            dump(s, fo)

    @staticmethod
    def fromPickle(path):
        """Load a source from a pickle file"""
        with open(path, 'rb') as fo:
            out = load(fo)
        return out

    def load(s, variable, name=None, heightIdx=None, processor=None):
        """Load a variable into the source's data container

        Inputs:
            variable : str - The variable within the original NC file to load
            
            name : str - The name to give this variable in the data container
                * If left as 'None', the original variable name is maintained

            heightIdx : idx - The height index to use
                * If the variable is 4D (time, height, lat, lon), use this to select the level which is extracted

            processor : function - An optional processing function to, for example, convert units
                * Ex. If the NC file has temperature in Kelvin and you need degrees C:
                    processor = lambda x: x+273.15
        """
        if s.path is None: raise ResError("Cannot load new variables when path is None")
        # read the data
        ds = s._loadDS(s.path)
        if not variable in ds.variables.keys():
            raise ResError(variable+" not in source")

        if heightIdx is None:
            tmp = ds[variable][s._timeSel,s._latStart:s._latStop,s._lonStart:s._lonStop]
        else:
            tmp = ds[variable][s._timeSel,heightIdx,s._latStart:s._latStop,s._lonStart:s._lonStop]

        # process, maybe?
        if not processor is None:
            tmp = processor(tmp)

        # save the data
        if name is None: name = variable
        s.data[name] = tmp

    def addData(s, name, data):
        """Manually add a variable to the data container

        Inputs:
            name : str - The name of the new variable

            data : np.ndarray - A 3 dimensional matrix with shape (timeN, latN, lonN)
        """
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
        """Returns the closest X and Y indexes corresponding to a given location or set of locations

        * If a single location is given, a single index is returned
        * If multiple locations are given, a list of indexes is returned which match to the order of locations

        Inputs:
            loc : The location(s) to search for
                - geokit.Location - Preferred location identifier for a single location
                - [ geokit.location, ] - Preferred location identifier for a multiple locations
                * Can be anything else which is understood by goekit.Location.load

            outsideOkay : T/F - Determines if points which are outside the source's lat/lon grid are allowed
                * If True, points outside this space will return as None
        """
        # Ensure loc is a list
        locations = LocationSet(loc)

        # Get coordinates
        lonCoords = locations.lons
        latCoords = locations.lats

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
        if locations.count==1:
            return idx[0]
        else:
            return idx

    def _timeindex(s): return s.timeindex
    def get(s, variable, locations, interpolation='near', forceDataFrame=False, outsideOkay=False):
        """
        Retrieve a variable from the source's data container at the given location(s)

        * Fetches the complete time series corresponding to the given location(s)
        * If a single location is given, a pandas.Series object is returned (index is time)
        * If multiple locations are given, a pandas.DataFrame object is returned (index is time, columns are locations)
            - Column order will always match the order of locations

        Inputs:
            variable : str - The variable within the data container to extract

            locations : The location(s) to search for
                - geokit.Location - Preferred location identifier for a single location
                - [ geokit.location, ] - Preferred location identifier for a multiple locations
                * Can be anything else which is understood by geokit.Location.load

            interpolation : str - The interpolation method to use
                * 'near' => For each location, extract the time series at the closest lat/lon index
                * 'bilinear' => For each location, use the time series of the surrounding four index locations to create an estimated time series at the given location
                    - Uses the bilinear interpolation scheme
                * 'cubic' => For each location, use the time series of the surrounding 16 index locations to create an estimated time series at the given location
                    - Uses the cubic interpolation scheme

            forceDataFrame : T/F - Instructs the returned value to take the form of a DataFrame regardless of how many locations are specified

            outsideOkay : T/F - Determines if points which are outside the source's lat/lon grid are allowed
                * If True, points outside this space will a time series of NaN values

        """
        # Ensure loc is a list
        locations = LocationSet(locations)

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
            lats = [loc.lat for loc in locations]
            lons = [loc.lon for loc in locations]
            
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

    def __getattr__(s, v):
        var, locs = v
        return s.get(var, locs)

    def contextAreaAt(s,location):
        """Compute the sources-index's context area surrounding the given location"""
        # Get closest indexes
        index = s.loc2Index(location)
        # get area
        return s.contextAreaAtIndex(index.yi, index.xi)

    def contextAreaAtIndex(s, latI, lonI):
        """Compute the context area surrounding the a specified index"""
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
        """Compute the context means of a source at all lat/lon indexes"""
        # get raster info
        ras = gk.raster.rasterInfo(source)

        # compute all means
        means = np.zeros((s.lats.size, s.lons.size))
        means[:] = np.nan

        for latI in range(1,s.lats.size-1):
            for lonI in range(1,s.lons.size-1):        
                means[latI,lonI] = computeContextMean(source=source, contextArea=s._contextAreaAt(latI,lonI), pixelSize=(ras.dx, ras.dy), srs=ras.srs, 
                    fillnan=fillnan)

        return means
