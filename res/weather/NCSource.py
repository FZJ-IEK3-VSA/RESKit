from os import listdir
from os.path import join, isfile, dirname, basename
from glob import glob
from scipy.interpolate import RectBivariateSpline, interp2d
from pickle import load, dump

from res.util.util_ import *

Bounds = namedtuple("Bounds", "lonMin latMin lonMax latMax")

# make a data handler
Index = namedtuple("Index", "yi xi")
class NCSource(object):
    """THE NCSource object manages weather data from a generic netCDF4 file source"""
    def _loadDS(s, path):
        if isinstance(path, str):
            return nc.Dataset(path)
        elif isinstance(path, list):
            return nc.MFDataset( path, aggdim=s.timeName)
        else:
            raise ResError("Could not understand data source input. Must be a path or a list of paths")

    def __init__(s, source, bounds=None, padFactor=0, timeName="time", latName="lat", lonName="lon", timeBounds=None, _maxLonDiff=10000000, _maxLatDiff=10000000):
        """Initialize a generic netCDF4 file source

        Note
        ----
        Generally not intended for normal use. Look into MerraSource, CordexSource, or CosmoSource

        Parameters
        ----------
        path : str
            The path to the main data file

        bounds : Anything acceptable to geokit.Extent.load(), optional
            The boundaries of the data which is needed
              * Usage of this will help with memory mangement
              * If None, the full dataset is loaded in memory
              
        padFactor : numeric, optional
            The padding to apply to the boundaries 
              * Useful in case of interpolation
              
        timeName : str, optional
            The name of the time parameter in the netCDF4 dataset
              
        latName : str, optional
            The name of the latitude parameter in the netCDF4 dataset
              
        lonName : str, optional
            The name of the longitude parameter in the netCDF4 dataset

        timeBounds : tuple of length 2, optional
            Used to employ a slice of the time dimension
              * Expect two pandas Timestamp objects> The first indicates the point
                to start collecting data, and the second indicates the end

        """
        # Collect all variable information
        s.variables = OrderedDict()
        if not isinstance(source, list): source = [source, ]

        expectedShape = OrderedDict()

        units = []
        names = []

        for src in source:
            ds = nc.Dataset(src)
            for var in ds.variables:
                if not var in s.variables:
                    s.variables[var] = src
                    expectedShape[var] = ds[var].shape

                    try: unit = ds[var].units
                    except: unit = "Unknown"
                    
                    try: name = ds[var].standard_name
                    except: name = "Unknown"

                    names.append(name)
                    units.append(unit)

                else:
                    if ds[var].shape != expectedShape[var]:
                        raise ResError("Variable %s does not match expected shape %s. From %s"%(var, expectedShape[var], src))
        
        tmp = pd.DataFrame(columns=["name","units","path",], index=s.variables.keys())
        tmp["name"] = names
        tmp["units"] = units
        tmp["shape"] = [expectedShape[v] for v in tmp.index]
        tmp["path"] = [s.variables[v] for v in tmp.index]
        s.variables = tmp
        
        # set basic variables
        s.timeName = timeName
        lonVar = s[lonName]
        latVar = s[latName]
    
        s._allLats = latVar[:]
        s._allLons = lonVar[:]
        
        s._maximal_lon_difference=_maxLonDiff
        s._maximal_lat_difference=_maxLatDiff

        if len(s._allLats.shape)==1 and len(s._allLons.shape)==1:
            s.dependent_coordinates = False
        elif  len(s._allLats.shape)==2 and len(s._allLons.shape)==2:
            s.dependent_coordinates = True
        else:
            raise ResError("latitude and longitude shapes are not usable")

        # set lat and lon selections
        if not bounds is None:
            s.extent = gk.Extent.load(bounds).castTo(gk.srs.EPSG4326).pad(padFactor)
            s.bounds = Bounds(*s.extent.xyXY)

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

            if s.dependent_coordinates:
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
        timeVar = s[timeName]
        timeindex = nc.num2date(timeVar[:], timeVar.units)
        
        if timeBounds is None:
            s._timeSel = np.s_[:]
        else:
            timeStart = pd.Timestamp(timeBounds[0])
            timeEnd = pd.Timestamp(timeBounds[1])
            s._timeSel = (timeindex >= timeStart) & (timeindex <= timeEnd)

        s.timeindex = timeindex[s._timeSel]

        # initialize some variables
        s.data = OrderedDict()
    
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
        """Load a variable into the source's data table

        Note
        ----
        Generally not intended for normal use. Look into MerraSource, CordexSource, or CosmoSource

        Parameters
        ----------
        variable : str
            The variable within the initialized datasources to load
              * The variable must either be of dimension (time, lat, lon) or 
                (time, height, lat, lon)

        name : str, optional
            The name to give this variable in the loaded data table
              * If None, the name of the original variable is kept

        heightIdx : int, optional
            The Height index to extract if the original variable has the height
            dimension

        processor : func, optional
            A function to process the loaded data before loading it into the 
            the loaded data table
              * This function must take a single matrix argument with dimensions 
                (time, lat, lon), and must return a matrix of the same shape
              * Example:If the NC file has temperature in Kelvin and you need C:
                  processor = lambda x: x+273.15

        """
        
        # read the data
        var = s[variable]

        if heightIdx is None:
            tmp = var[s._timeSel,s._latStart:s._latStop,s._lonStart:s._lonStop]
        else:
            tmp = var[s._timeSel,heightIdx,s._latStart:s._latStop,s._lonStart:s._lonStop]

        # process, maybe?
        if not processor is None:
            tmp = processor(tmp)

        # save the data
        if name is None: name = variable
        s.data[name] = tmp

    def addData(s, name, data):
        """Manually add a variable to the loaded data table

        Parameters
        ----------
            name : str 
                The name of the new variable

            data : np.ndarray
                A 3 dimensional matrix with shape (time, lat, lon)
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
        """Returns the closest X and Y indexes corresponding to a given location 
        or set of locations

        Parameters
        ----------
            loc : Anything acceptable by geokit.LocationSet
                The location(s) to search for
                  * A single tuple with (lon, lat) is acceptable, or a list of such 
                    tuples
                  * A single point geometry (as long as it has an SRS), or a list
                    of geometries is okay
                  * geokit,Location, or geokit.LocationSet are best, though

            outsideOkay : bool, optional
                Determines if points which are outside the source's lat/lon grid
                are allowed
                * If True, points outside this space will return as None
                * If False, an error is raised 

        Returns
        -------
        If a single location is given: tuple 
            * Format: (yIndex, xIndex)
            * y index can be accessed with '.yi'
            * x index can be accessed with '.xi'

        If multiple locations are given: list
            * Format: [ (yIndex1, xIndex1), (yIndex2, xIndex2), ...]
            * Order matches the given order of locations
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

    def get(s, variable, locations, interpolation='near', forceDataFrame=False, outsideOkay=False):
        """
        Retrieve complete time series for a variable from the source's loaded data 
        table at the given location(s)

        Parameters
        ----------
            variable : str
                The variable within the data container to extract

            locations : Anything acceptable by geokit.LocationSet
                The location(s) to search for
                  * A single tuple with (lon, lat) is acceptable, or a list of such 
                    tuples
                  * A single point geometry (as long as it has an SRS), or a list
                    of geometries is okay
                  * geokit,Location, or geokit.LocationSet are best, though

            interpolation : str, optional
                The interpolation method to use
                  * 'near' => For each location, extract the time series at the 
                    closest lat/lon index
                  * 'bilinear' => For each location, use the time series of the 
                    surrounding +/- 1 index locations to create an estimated time 
                    series at the given location using a biliear scheme
                  * 'cubic' => For each location, use the time series of the 
                    surrounding +/- 2 index locations to create an estimated time 
                    series at the given location using a cubic scheme

            forceDataFrame : bool, optional
                Instructs the returned value to take the form of a DataFrame 
                regardless of how many locations are specified


            outsideOkay : bool, optional
                Determines if points which are outside the source's lat/lon grid
                are allowed
                * If True, points outside this space will return as None
                * If False, an error is raised 
        
        Returns
        -------

        If a single location is given: pandas.Series
          * Indexes match to times
        
        If multiple locations are given: pandas.DataFrame
          * Indexes match to times
          * Columns match to the given order of locations
        
        """
        # Ensure loc is a list
        locations = LocationSet(locations)

        # compute the closest indices
        indecies = s.loc2Index(locations, outsideOkay)
        if isinstance(indecies, Index): indecies = [indecies, ]

        # Do interpolation
        if interpolation == 'oldnear':            
            # arrange the output data
            tmp = []
            for i in indecies:
                if not i is None: tmp.append(s.data[variable][:, i.yi, i.xi])
                else: tmp.append( np.array([np.nan,]*s.timeindex.size) ) 
            output = np.column_stack(tmp)

        elif interpolation == 'near':
            # arrange the output data
            indexSet = set(indecies)
            allYI = np.array([i.yi for i in indecies])
            allXI = np.array([i.xi for i in indecies])
            output = np.zeros((s.timeindex.size, locations.count))

            for i in indexSet:
                if i is None: continue
                
                ysel = allYI==i.yi
                xsel = allXI==i.xi
                sel = np.logical_and(xsel, ysel)
                output[:,sel] = s.data[variable][:, i.yi, i.xi].reshape((s.timeindex.size, 1))
        
        elif interpolation == "cubic" or interpolation == "bilinear":
            # set some arguments for later use
            if interpolation == "cubic":
                win = 4
                rbsArgs = dict()
            else:
                win = 2
                rbsArgs = dict(kx=1, ky=1)

            # Set up interpolation arrays
            yiMin = min([i.yi for i in indecies])-win
            yiMax = max([i.yi for i in indecies])+win
            xiMin = min([i.xi for i in indecies])-win
            xiMax = max([i.xi for i in indecies])+win

            # ensure boundaries are okay
            if s.dependent_coordinates:
                if yiMin < 0 or xiMin < 0 or yiMax > s.lats.shape[0] or xiMax > s.lons.shape[1]: 
                    raise ResError("Insufficient data. Try expanding the boundary of the extracted data")
            else:
                if yiMin < 0 or xiMin < 0 or yiMax > s.lats.size or xiMax > s.lons.size: 
                    raise ResError("Insufficient data. Try expanding the boundary of the extracted data")

            ##########
            ## TODO: Update interpolation schemes to handle out-of-bounds indices 
            ##########
            
            if s.dependent_coordinates: # do interpolations in 'index space'                
                from scipy.interpolate import interp1d
                
                gridYVals = np.arange(yiMin, yiMax+1)
                gridXVals = np.arange(xiMin, xiMax+1)

                def getIntermediateIndex(loc, i):
                    fy = interp1d(s.lats[:,i.xi], np.arange(s.lats.shape[0]), kind='cubic')
                    fx = interp1d(s.lons[i.yi,:], np.arange(s.lats.shape[1]), kind='cubic')
                    return Index( fy(loc.lat), fx(loc.lon) )

                tmp = [getIntermediateIndex(loc,i) for loc,i in zip(locations, indecies)]
                yInterp = [i.yi for i in tmp]
                xInterp = [i.xi for i in tmp]
                
            else: # do interpolation in the expected 'coordinate space'
                gridYVals = s.lats[yiMin:yiMax+1]
                gridXVals = s.lons[xiMin:xiMax+1]
                
                yInterp = [loc.lat for loc in locations]
                xInterp = [loc.lon for loc in locations]
            
            # Do interpolation
            output = []
            for ts in range(s.data[variable].shape[0]):
                # set up interpolation
                rbs = RectBivariateSpline(gridYVals,gridXVals,s.data[variable][ts, yiMin:yiMax+1, xiMin:xiMax+1], **rbsArgs)

                # interpolate for each location
                output.append(rbs(yInterp, xInterp, grid=False)) # lat/lon order switched to match index order
     
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

    def __getitem__(s, v): 
        try:
            return nc.Dataset(s.variables["path"][v])[v]
        except KeyError as e:
            pass # pass to avoid horrible pandas trace
        raise KeyError(str(v))

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
