from os import listdir
from os.path import join, isfile, dirname, basename, isdir
from glob import glob
from scipy.interpolate import RectBivariateSpline, interp2d, bisplrep, bisplev, interp1d
from pickle import load, dump
from collections import defaultdict

from reskit.util.util_ import *

# make a data handler
Index = namedtuple("Index", "yi xi")


class NCSource(object):
    """The NCSource object manages weather data from a generic set of netCDF4 
    file sources"""

    WIND_SPEED_HEIGHT_FOR_WIND_ENERGY = None
    WIND_SPEED_HEIGHT_FOR_SOLAR_ENERGY = None
    LONG_RUN_AVERAGE_WINDSPEED = None
    LONG_RUN_AVERAGE_WINDDIR = None
    LONG_RUN_AVERAGE_GHI = None
    LONG_RUN_AVERAGE_DNI = None
    MAX_LON_DIFFERENCE = None
    MAX_LAT_DIFFERENCE = None

    def _loadDS(s, path):
        if isinstance(path, str):
            return nc.Dataset(path, keepweakref=True)
        # elif isinstance(path, list):
        #     return nc.MFDataset( path, aggdim=s.timeName)
        else:
            raise ResError(
                "Could not understand data source input. Must be a path or a list of paths")

    def __init__(s, source, bounds=None, indexPad=0, timeName="time", latName="lat", lonName="lon", tz=None, _maxLonDiff=0.6, _maxLatDiff=0.6, verbose=True, forwardFill=True, flip_lat=False, flip_lon=False):
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
              * The actual extent of the loaded data depends on the source's 
                available data

        padExtent : numeric, optional
            The padding to apply to the boundaries 
              * Useful in case of interpolation
              * Units are in longitudinal degrees

        timeName : str, optional
            The name of the time parameter in the netCDF4 dataset

        latName : str, optional
            The name of the latitude parameter in the netCDF4 dataset

        lonName : str, optional
            The name of the longitude parameter in the netCDF4 dataset

        tz: str; optional
            Applies the indicated timezone onto the time axis
            * For example, use "GMT" for unadjusted time


        """
        # Collect sources
        def addSource(src):
            out = []
            if isinstance(src, list):
                for s in src:
                    out.extend(addSource(s))
            elif isinstance(src, str):
                if isfile(src):  # Assume its an NC file
                    out.extend([src, ])
                elif isdir(src):  # Assume its a directory of NC files
                    for s in glob(join(src, "*.nc")):
                        out.append(s)
                    for s in glob(join(src, "*.nc4")):
                        out.append(s)
                else:  # Assume we were given a glob string
                    for s in glob(src):
                        out.extend(addSource(s))
            return out
        sources = addSource(source)
        if len(sources) == 0:
            raise ResError("No '.nc' or '.nc4' files found")
        sources.sort()

        # Collect all variable information
        s.variables = OrderedDict()
        s.fill = forwardFill
        expectedShape = OrderedDict()

        units = []
        names = []

        for src in sources:
            if verbose:
                print(src)
            ds = nc.Dataset(src, keepweakref=True)
            for var in ds.variables:
                if not var in s.variables:
                    s.variables[var] = src
                    expectedShape[var] = ds[var].shape

                    try:
                        unit = ds[var].units
                    except:
                        unit = "Unknown"

                    try:
                        name = ds[var].standard_name
                    except:
                        name = "Unknown"

                    names.append(name)
                    units.append(unit)

                else:
                    if ds[var].shape[1:] != expectedShape[var][1:]:
                        raise ResError("Variable %s does not match expected shape %s. From %s" % (
                            var, expectedShape[var], src))
            ds.close()

        tmp = pd.DataFrame(
            columns=["name", "units", "path", ], index=s.variables.keys())
        tmp["name"] = names
        tmp["units"] = units
        tmp["shape"] = [expectedShape[v] for v in tmp.index]
        tmp["path"] = [s.variables[v] for v in tmp.index]
        s.variables = tmp

        # set basic variables
        ds = nc.Dataset(s.variables["path"][latName], keepweakref=True)
        s._allLats = ds[latName][:]
        ds.close()

        ds = nc.Dataset(s.variables["path"][lonName], keepweakref=True)
        s._allLons = ds[lonName][:]
        ds.close()

        s._maximal_lon_difference = _maxLonDiff
        s._maximal_lat_difference = _maxLatDiff

        if len(s._allLats.shape) == 1 and len(s._allLons.shape) == 1:
            s.dependent_coordinates = False
            s._lonN = s._allLons.size
            s._latN = s._allLats.size
        elif len(s._allLats.shape) == 2 and len(s._allLons.shape) == 2:
            s.dependent_coordinates = True
            s._lonN = s._allLons.shape[1]
            s._latN = s._allLats.shape[0]
        else:
            raise ResError("latitude and longitude shapes are not usable")

        # set lat and lon selections
        if not bounds is None:
            s.bounds = gk.Extent.load(bounds).castTo(4326).xyXY
            if abs(s.bounds[0] - s.bounds[2]) <= 0.625:
                s.bounds = s.bounds[0] - \
                    0.3125, s.bounds[1], s.bounds[2] + 0.3125, s.bounds[3]
            if abs(s.bounds[1] - s.bounds[3]) <= 0.5:
                s.bounds = s.bounds[0], s.bounds[1] - \
                    0.25, s.bounds[2], s.bounds[3] + 0.25

            # find slices which contains our extent
            if s.dependent_coordinates:
                left = s._allLons < s.bounds[0]
                right = s._allLons > s.bounds[2]
                if (left | right).all():
                    left[:, :-1] = np.logical_and(left[:, 1:], left[:, :-1])
                    right[:, 1:] = np.logical_and(right[:, 1:], right[:, :-1])

                bot = s._allLats < s.bounds[1]
                top = s._allLats > s.bounds[3]
                if (top | bot).all():
                    top[:-1, :] = np.logical_and(top[1:, :], top[:-1, :])
                    bot[1:, :] = np.logical_and(bot[1:, :], bot[:-1, :])

                s._lonStart = np.argmin(
                    (bot | left | top).all(0)) - 1 - indexPad
                s._lonStop = s._lonN - \
                    np.argmin((bot | top | right).all(0)[::-1]) + 1 + indexPad
                s._latStart = np.argmin(
                    (bot | left | right).all(1)) - 1 - indexPad
                s._latStop = s._latN - \
                    np.argmax((left | top | right).all(1)[::-1]) + 1 + indexPad

            else:
                tmp = np.logical_and(
                    s._allLons >= s.bounds[0], s._allLons <= s.bounds[2])
                s._lonStart = np.argmax(tmp) - 1
                s._lonStop = s._lonStart + 1 + \
                    np.argmin(tmp[s._lonStart + 1:]) + 1

                tmp = np.logical_and(
                    s._allLats >= s.bounds[1], s._allLats <= s.bounds[3])
                s._latStart = np.argmax(tmp) - 1
                s._latStop = s._latStart + 1 + \
                    np.argmin(tmp[s._latStart + 1:]) + 1

                s._lonStart = max(0, s._lonStart - indexPad)
                s._lonStop = min(s._allLons.size - 1, s._lonStop + indexPad)
                s._latStart = max(0, s._latStart - indexPad)
                s._latStop = min(s._allLats.size - 1, s._latStop + indexPad)

                # s._lonStart = np.argmin(s._allLons < s.bounds[0]) - 1 - indexPad
                # s._lonStop = np.argmax(s._allLons > s.bounds[2]) + indexPad
                # s._latStart = np.argmin(s._allLats < s.bounds[1]) - 1 - indexPad
                # s._latStop = np.argmax(s._allLats > s.bounds[3]) + indexPad

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

        # Read working lats/lon
        s._flip_lat = flip_lat
        s._flip_lon = flip_lon

        if s.dependent_coordinates:
            s.lats = s._allLats[s._latStart:s._latStop, s._lonStart:s._lonStop]
            s.lons = s._allLons[s._latStart:s._latStop, s._lonStart:s._lonStop]

            if flip_lat:
                s.lats = s.lats[::-1, :]
            if flip_lon:
                s.lons = s.lons[:, ::-1]
        else:
            s.lats = s._allLats[s._latStart:s._latStop]
            s.lons = s._allLons[s._lonStart:s._lonStop]

            if flip_lat:
                s.lats = s.lats[::-1]
            if flip_lon:
                s.lons = s.lons[::-1]

        s.extent = gk.Extent(s.lons.min(), s.lats.min(),
                             s.lons.max(), s.lats.max(), srs=gk.srs.EPSG4326)

        # compute time index
        s.timeName = timeName

        ds = nc.Dataset(s.variables["path"][timeName], keepweakref=True)
        timeVar = ds[timeName]
        timeindex = nc.num2date(
            timeVar[:], timeVar.units, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
        ds.close()

        s._timeindex_raw = pd.DatetimeIndex(timeindex)
        if not tz is None:
            s.timeindex = s._timeindex_raw.tz_localize(tz)
        else:
            s.timeindex = s._timeindex_raw

        # initialize the data container
        s.data = OrderedDict()

    def varInfo(s, var):
        """Prints more information about the given parameter"""
        try:
            ds = nc.Dataset(s.variables["path"][var], keepweakref=True)
            print(ds[var])
            ds.close()
            return
        except KeyError as e:
            pass  # pass to avoid horrible pandas trace
        raise KeyError(str(v))

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

    def sload(self, *variables):
        """Load standard variables
        """
        for var in variables:
            if hasattr(self, "sload_" + var):
                getattr(self, "sload_" + var)()
            else:
                raise RuntimeError(
                    var + " is not an acceptable key for this weather source")

    def load(s, variable, name=None, heightIdx=None, processor=None, overwrite=False):
        """Load a variable into the source's data table

        Parameters
        ----------
        variable : str
            The variable within the currated datasources to load
              * The variable must either be of dimension (time, lat, lon) or 
                (time, height, lat, lon)

        name : str; optional
            The name to give this variable in the loaded data table
              * If None, the name of the original variable is kept

        heightIdx : int; optional
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
        if name is None:
            name = variable

        if not overwrite and name in s.data:
            # nothing to do...
            return

        # read the data
        ds = nc.Dataset(s.variables["path"][variable], keepweakref=True)
        var = ds[variable]

        if heightIdx is None:
            tmp = var[:, s._latStart:s._latStop, s._lonStart:s._lonStop]
        else:
            tmp = var[:, heightIdx, s._latStart:s._latStop,
                      s._lonStart:s._lonStop]

        # process, maybe?
        if not processor is None:
            tmp = processor(tmp)

        # forward fill the last time step since it can sometimes be missing
        if not tmp.shape[0] == s._timeindex_raw.shape[0]:
            if not s.fill:
                raise ResError("Time mismatch with variable %s. Expected %d, got %d" % (
                    variable, s.timeindex.shape[0], tmp.shape[0]))

            lastTimeIndex = nc.num2date(
                ds[s.timeName][-1], ds[s.timeName].units)

            if not lastTimeIndex in s._timeindex_raw:
                raise ResError(
                    "Filling is only intended to fill the last missing hour")
            tmp = np.append(tmp, tmp[np.newaxis, -1, :, :], axis=0)

        # save the data
        if not s._flip_lat and not s._flip_lon:
            s.data[name] = tmp
        elif s._flip_lat and not s._flip_lon:
            s.data[name] = tmp[:, ::-1, :]
        elif not s._flip_lat and s._flip_lon:
            s.data[name] = tmp[:, :, ::-1]
        elif s._flip_lat and s._flip_lon:
            s.data[name] = tmp[:, ::-1, ::-1]

        # Clean up
        ds.close()

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
        if data.shape[0] != s.timeindex.shape[0]:
            raise ResError(
                "Input data's first dimension does not match the time index")
        if s.dependent_coordinates:
            if data.shape[1] != s.lats.shape[0]:
                raise ResError(
                    "Input data's second dimension does not match the latitude dimension")
            if data.shape[2] != s.lons.shape[1]:
                raise ResError(
                    "Input data's second dimension does not match the longitude dimension")
        else:
            if data.shape[1] != s.lats.shape[0]:
                raise ResError(
                    "Input data's second dimension does not match the latitude dimension")
            if data.shape[2] != s.lons.shape[0]:
                raise ResError(
                    "Input data's second dimension does not match the longitude dimension")

        # Add to data
        s.data[name] = data

    @staticmethod
    def _loc2IndexRect(lat_step, lon_step):
        def func(s, loc, outsideOkay=False, asInt=True):
            """Returns the closest X and Y indexes corresponding to a given location 
            or set of locations

            Parameters
            ----------
            loc : Anything acceptable by geokit.LocationSet
                The location(s) to search for
                * A single tuple with (lon, lat) is acceptable, or a list of such tuples
                * A single point geometry (as long as it has an SRS), or a list
                  of geometries is okay
                * geokit,Location, or geokit.LocationSet are best!

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

            # get closest indices
            latI = (locations.lats - s.lats[0]) / lat_step
            lonI = (locations.lons - s.lons[0]) / lon_step

            # Check for out of bounds
            s = (latI < 0) | (latI >= s._latN) | (lonI < 0) | (lonI >= s._lonN)
            if s.any():
                if not outsideOkay:
                    print("The following locations are out of bounds")
                    print(locations[s])
                    raise ResError("Locations are outside the boundaries")

            # As int?
            if asInt:
                latI = np.round(latI).astype(int)
                lonI = np.round(lonI).astype(int)

            # Make output
            if locations.count == 1:
                if s[0] is True:
                    return None
                else:
                    return Index(yi=latI[0], xi=lonI[0])
            else:
                return [None if ss else Index(yi=y, xi=x) for ss, y, x in zip(s, latI, lonI)]
        return func

    def loc2Index(s, loc, outsideOkay=False, asInt=True):
        """Returns the closest X and Y indexes corresponding to a given location 
        or set of locations

        Parameters
        ----------
        loc : Anything acceptable by geokit.LocationSet
            The location(s) to search for
            * A single tuple with (lon, lat) is acceptable, or a list of such tuples
            * A single point geometry (as long as it has an SRS), or a list
              of geometries is okay
            * geokit,Location, or geokit.LocationSet are best!

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

        # get closest indices
        idx = []
        for lat, lon in zip(locations.lats, locations.lons):
            # Check the distance
            latDist = lat - s.lats
            lonDist = lon - s.lons

            # Get the best indices
            if s.dependent_coordinates:
                dist = lonDist * lonDist + latDist * latDist
                latI, lonI = np.unravel_index(np.argmin(dist), dist.shape)

                latDists = []
                if latI < s._latN - 1:
                    latDists.append(
                        (s.lats[latI + 1, lonI] - s.lats[latI, lonI]))
                if latI > 0:
                    latDists.append(
                        (s.lats[latI, lonI] - s.lats[latI - 1, lonI]))
                latDistI = latDist[latI, lonI] / np.mean(latDists)

                lonDists = []
                if lonI < s._lonN - 1:
                    lonDists.append(
                        (s.lons[latI, lonI + 1] - s.lons[latI, lonI]))
                if lonI > 0:
                    lonDists.append(
                        (s.lons[latI, lonI] - s.lons[latI, lonI - 1]))
                lonDistI = lonDist[latI, lonI] / np.mean(lonDists)

            else:
                lonI = np.argmin(np.abs(lonDist))
                latI = np.argmin(np.abs(latDist))

                latDists = []
                if latI < s._latN - 1:
                    latDists.append((s.lats[latI + 1] - s.lats[latI]))
                if latI > 0:
                    latDists.append((s.lats[latI] - s.lats[latI - 1]))
                latDistI = latDist[latI] / np.mean(latDists)

                lonDists = []
                if lonI < s._latN - 1:
                    lonDists.append((s.lons[lonI + 1] - s.lons[lonI]))
                if lonI > 0:
                    lonDists.append((s.lons[lonI] - s.lons[lonI - 1]))
                lonDistI = lonDist[lonI] / np.mean(lonDists)

            # Check for out of bounds
            if np.abs(latDistI) > s._maximal_lat_difference or np.abs(lonDistI) > s._maximal_lon_difference:
                if not outsideOkay:
                    raise ResError(
                        "(%f,%f) are outside the boundaries" % (lat, lon))
                else:
                    idx.append(None)
                    continue

            # As int?
            if not asInt:
                latI = latI + latDistI
                lonI = lonI + lonDistI

            # append
            idx.append(Index(yi=latI, xi=lonI))

        # Make output
        if locations.count == 1:
            return idx[0]
        else:
            return idx

    def get(s, variable, locations, interpolation='near', forceDataFrame=False, outsideOkay=False, _indicies=None):
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

        # Get the indicies
        if _indicies is None:
            # compute the closest indices
            if not s.dependent_coordinates or interpolation == 'near':
                asInt = True
            else:
                asInt = False
            indicies = s.loc2Index(locations, outsideOkay, asInt=asInt)
        else:
            # Assume indicies match locations
            indicies = _indicies

        if isinstance(indicies, Index):
            indicies = [indicies, ]

        # Do interpolation
        if interpolation == 'near':
            # arrange the output data
            tmp = []
            for i in indicies:
                if not i is None:
                    tmp.append(s.data[variable][:, i.yi, i.xi])
                else:
                    tmp.append(np.array([np.nan, ] * s.timeindex.size))
            output = np.column_stack(tmp)

        elif interpolation == "cubic" or interpolation == "bilinear":
            # set some arguments for later use
            if interpolation == "cubic":
                win = 4
                rbsArgs = dict()
            else:
                win = 2
                rbsArgs = dict(kx=1, ky=1)

            # Set up interpolation arrays
            yiMin = np.round(min([i.yi for i in indicies]) - win).astype(int)
            yiMax = np.round(max([i.yi for i in indicies]) + win).astype(int)
            xiMin = np.round(min([i.xi for i in indicies]) - win).astype(int)
            xiMax = np.round(max([i.xi for i in indicies]) + win).astype(int)

            # ensure boundaries are okay
            if yiMin < 0 or xiMin < 0 or yiMax > s._latN or xiMax > s._lonN:
                raise ResError(
                    "Insufficient data. Try expanding the boundary of the extracted data")

            ##########
            # TODO: Update interpolation schemes to handle out-of-bounds indices
            ##########

            if s.dependent_coordinates:  # do interpolations in 'index space'
                if isinstance(indicies[0][0], int):
                    raise ResError(
                        "Index must be float type for interpolation")

                gridYVals = np.arange(yiMin, yiMax + 1)
                gridXVals = np.arange(xiMin, xiMax + 1)

                yInterp = [i.yi for i in indicies]
                xInterp = [i.xi for i in indicies]

            else:  # do interpolation in the expected 'coordinate space'
                gridYVals = s.lats[yiMin:yiMax + 1]
                gridXVals = s.lons[xiMin:xiMax + 1]

                yInterp = [loc.lat for loc in locations]
                xInterp = [loc.lon for loc in locations]

            # Do interpolation
            output = []
            for ts in range(s.data[variable].shape[0]):
                # set up interpolation
                rbs = RectBivariateSpline(
                    gridYVals, gridXVals, s.data[variable][ts, yiMin:yiMax + 1, xiMin:xiMax + 1], **rbsArgs)

                # interpolate for each location
                # lat/lon order switched to match index order
                output.append(rbs(yInterp, xInterp, grid=False))

            output = np.stack(output)

        else:
            raise ResError(
                "Interpolation scheme not one of: 'near', 'cubic', or 'bilinear'")

        # Make output as Series objects
        if forceDataFrame or (len(output.shape) > 1 and output.shape[1] > 1):
            return pd.DataFrame(output, index=s.timeindex, columns=locations)
        else:
            try:
                return pd.Series(output[:, 0], index=s.timeindex, name=locations[0])
            except:
                return pd.Series(output, index=s.timeindex, name=locations[0])

    def contextAreaAt(s, location):
        """Compute the sources-index's context area surrounding the given location"""
        # Get closest indexes
        index = s.loc2Index(location)
        # get area
        return s.contextAreaAtIndex(index.yi, index.xi)

    def contextAreaAtIndex(s, latI, lonI):
        """Compute the context area surrounding the a specified index"""
        if s.dependent_coordinates:
            ctr = np.array([s.lons[latI, lonI], s.lats[latI, lonI]])
            up = np.array([s.lons[latI + 1, lonI], s.lats[latI + 1, lonI]])
            dw = np.array([s.lons[latI - 1, lonI], s.lats[latI - 1, lonI]])
            rt = np.array([s.lons[latI, lonI + 1], s.lats[latI, lonI + 1]])
            lt = np.array([s.lons[latI, lonI - 1], s.lats[latI, lonI - 1]])

            up_rt = np.array([s.lons[latI + 1, lonI + 1],
                              s.lats[latI + 1, lonI + 1]])
            dw_rt = np.array([s.lons[latI - 1, lonI + 1],
                              s.lats[latI - 1, lonI + 1]])
            up_lt = np.array([s.lons[latI + 1, lonI - 1],
                              s.lats[latI + 1, lonI - 1]])
            dw_lt = np.array([s.lons[latI - 1, lonI - 1],
                              s.lats[latI - 1, lonI - 1]])

            # compute points in context
            mid_up_rt = (ctr + up + rt + up_rt) / 4
            mid_dw_rt = (ctr + dw + rt + dw_rt) / 4
            mid_up_lt = (ctr + up + lt + up_lt) / 4
            mid_dw_lt = (ctr + dw + lt + dw_lt) / 4

            # return polygon
            return gk.geom.polygon([(mid_up_rt),
                                    (mid_dw_rt),
                                    (mid_dw_lt),
                                    (mid_up_lt),
                                    (mid_up_rt)], srs=gk.srs.EPSG4326)
        else:
            # Make and return a box
            lowLat = (s.lats[latI] + s.lats[latI - 1]) / 2
            highLat = (s.lats[latI] + s.lats[latI + 1]) / 2
            lowLon = (s.lons[lonI] + s.lons[lonI - 1]) / 2
            highLon = (s.lons[lonI] + s.lons[lonI + 1]) / 2

            # return box
            return gk.geom.box(lowLon, lowLat, highLon, highLat, srs=gk.srs.EPSG4326)
