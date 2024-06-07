from os.path import join, isfile, isdir
from glob import glob
from scipy.interpolate import RectBivariateSpline
from collections import namedtuple, OrderedDict
import netCDF4 as nc
import numpy as np
import geokit as gk
import pandas as pd

from ..util import ResError


# make a data handler
Index = namedtuple("Index", "yi xi")


class NCSource(object):
    """The NCSource object manages weather data from a generic set of netCDF4 file sources

    If furthermore allows access a number of common functionalities and constants which are
    often encountered when simulating renewable energy technologies

    Note:
    -----
    Various constants can be set for a given weather source which can impact later simulation workflows.
        Note that not all weather sources will have all of these constants available. Also more may be
        implemented besides (so be sure to check the DocString for the source you intend to use).

    These constants include:

        MAX_LON_DIFFERENCE
            The maximum logitude difference to accept between a grid cell and the coordinates you would
                like to extract data for

        MAX_LAT_DIFFERENCE
            The maximum latitude difference to accept between a grid cell and the coordinates you would
                like to extract data for

        WIND_SPEED_HEIGHT_FOR_WIND_ENERGY
            The suggested altitude of wind speed data to use for wind-energy simulations

        WIND_SPEED_HEIGHT_FOR_SOLAR_ENERGY
            The suggested altitude of wind speed data to use for wind-energy simulations

        LONG_RUN_AVERAGE_WINDSPEED
            A path to a raster file with the long-time average wind speed in each grid cell
            * Can be used in wind energy simulations
            * Calculated at the height specified in `WIND_SPEED_HEIGHT_FOR_WIND_ENERGY`
            * Time range included in the long run averaging depends on the data source

        LONG_RUN_AVERAGE_WINDDIR
            A path to a raster file with the long-time average wind direction in each grid cell
            * Can be used in wind energy simulations
            * Calculated at the height specified in `WIND_SPEED_HEIGHT_FOR_WIND_ENERGY`
            * Time range included in the long run averaging depends on the data source

        LONG_RUN_AVERAGE_GHI
            A path to a raster file with the long-time average global horizontal irradiance
                in each grid cell
            * Can be used in solar energy simulations
            * Calculated at the surface
            * Time range included in the long run averaging depends on the data source

        LONG_RUN_AVERAGE_DNI
            A path to a raster file with the long-time average direct normal irradiance
                in each grid cell
            * Can be used in solar energy simulations
            * Calculated at the surface
            * Time range included in the long run averaging depends on the data source


    See Also:
    ---------
    reskit.weather.MerraSource
    reskit.weather.SarahSource
    reskit.weather.Era5Source
    """

    WIND_SPEED_HEIGHT_FOR_WIND_ENERGY = None
    WIND_SPEED_HEIGHT_FOR_SOLAR_ENERGY = None
    LONG_RUN_AVERAGE_WINDSPEED = None
    LONG_RUN_AVERAGE_WINDDIR = None
    LONG_RUN_AVERAGE_GHI = None
    LONG_RUN_AVERAGE_DNI = None
    MAX_LON_DIFFERENCE = None
    MAX_LAT_DIFFERENCE = None

    def __init__(
        self,
        source,
        bounds=None,
        index_pad=0,
        time_name="time",
        lat_name="lat",
        lon_name="lon",
        tz=None,
        _max_lon_diff=0.6,
        _max_lat_diff=0.6,
        verbose=True,
        forward_fill=True,
        flip_lat=False,
        flip_lon=False,
        time_offset_minutes=None,
        time_index_from=None,
    ):
        """Initialize a generic netCDF4 file source


        Note:
        -----
        Generally not intended for normal use. Look into MerraSource, CordexSource, or CosmoSource


        Parameters:
        -----------
        path : str or list of strings
            The path to the main data file(s) to load

            If multiple files are given, or if a directory of netCDF4 files is given, then it is assumed
            that all files ending with the extension '.nc' or '.nc4' should be managed by this object.
            * Be sure that all the netCDF4 files given share the same time and spatial dimensions!

        bounds : Anything acceptable to geokit.Extent.load(), optional
            The boundaries of the data which is needed
              * Usage of this will help with memory mangement
              * If None, the full dataset is loaded in memory
              * The actual extent of the loaded data depends on the source's
                available data

        index_pad : int, optional
            The padding to apply to the boundaries
              * Useful in case of interpolation
              * Units are in longitudinal degrees

        time_name : str, optional
            The name of the time parameter in the netCDF4 dataset

        lat_name : str, optional
            The name of the latitude parameter in the netCDF4 dataset

        lon_name : str, optional
            The name of the longitude parameter in the netCDF4 dataset

        tz : str, optional
            Applies the indicated timezone onto the time axis
            * For example, use "GMT" for unadjusted time

        verbose : bool, optional
            If True, then status outputs are printed when searching for and reading weather data

        forward_fill : bool, optional
            If True, then missing data in the weather file is forward-filled
            * Generally, there should be no missing data at all. This option is only intended to
                catch the rare scenarios where one or two timesteps are missing

        flip_lat : bool, optional
            If True, flips the latitude dimension when reading weather data from the source
            * Should only be given if latitudes are given in descending order

        flip_lon : bool, optional
            If True, flips the longitude dimension when reading weather data from the source
            * Should only be given if longitudes are given in descending order

        time_offset_minutes : numeric, optional
            If not none, adds the specific offset in minutes to the timesteps read from the weather file


        See Also:
        ---------
        MerraSource
        SarahSource
        Era5Source
        """

        # Collect sources
        def addSource(src):
            out = []
            if isinstance(src, list):
                for s in src:
                    out.extend(addSource(s))
            elif isinstance(src, str):
                if isfile(src):  # Assume its an NC file
                    out.extend(
                        [
                            src,
                        ]
                    )
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
            raise ResError(f"No '.nc' or '.nc4' files found for tile base path: {source}")
        sources.sort()

        # Collect all variable information
        self.variables = OrderedDict()
        self.fill = forward_fill
        expectedShape = OrderedDict()

        units = []
        names = []

        for src in sources:
            if verbose:
                print(src)
            ds = nc.Dataset(src, keepweakref=True)
            for var in ds.variables:
                if not var in self.variables:
                    self.variables[var] = src
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
                        raise ResError(
                            "Variable %s does not match expected shape %s. From %s"
                            % (var, expectedShape[var], src)
                        )
            ds.close()

        tmp = pd.DataFrame(
            columns=[
                "name",
                "units",
                "path",
            ],
            index=self.variables.keys(),
        )
        tmp["name"] = names
        tmp["units"] = units
        tmp["shape"] = [expectedShape[v] for v in tmp.index]
        tmp["path"] = [self.variables[v] for v in tmp.index]
        self.variables = tmp

        # choose source for the time step extraction
        if not time_index_from == None:
            assert (
                time_index_from in self.variables.index
            ), f'ERA_5-key {time_index_from} not known. Check variable "time_index_from" and folder {source}'
            self.variables["path"][time_name] = self.variables["path"][time_index_from]

        # set basic variables
        ds = nc.Dataset(self.variables["path"][lat_name], keepweakref=True)
        self._allLats = ds[lat_name][:]
        ds.close()

        ds = nc.Dataset(self.variables["path"][lon_name], keepweakref=True)
        self._allLons = ds[lon_name][:]
        ds.close()

        self._maximal_lon_difference = _max_lon_diff
        self._maximal_lat_difference = _max_lat_diff

        if len(self._allLats.shape) == 1 and len(self._allLons.shape) == 1:
            self.dependent_coordinates = False
            self._lonN = self._allLons.size
            self._latN = self._allLats.size
        elif len(self._allLats.shape) == 2 and len(self._allLons.shape) == 2:
            self.dependent_coordinates = True
            self._lonN = self._allLons.shape[1]
            self._latN = self._allLats.shape[0]
        else:
            raise ResError("latitude and longitude shapes are not usable")

        # set lat and lon selections
        if bounds is not None:
            self.bounds = gk.Extent.load(bounds).castTo(4326)
            if abs(self.bounds.xMin - self.bounds.xMax) <= self.MAX_LON_DIFFERENCE:
                self.bounds = gk.Extent(
                    self.bounds.xMin - self.MAX_LON_DIFFERENCE / 2,
                    self.bounds.yMin,
                    self.bounds.xMax + self.MAX_LON_DIFFERENCE / 2,
                    self.bounds.yMax,
                    srs=gk.srs.EPSG4326,
                )

            if abs(self.bounds.yMin - self.bounds.yMax) <= self.MAX_LAT_DIFFERENCE:
                self.bounds = gk.Extent(
                    self.bounds.xMin,
                    self.bounds.yMin - self.MAX_LAT_DIFFERENCE / 2,
                    self.bounds.xMax,
                    self.bounds.yMax + self.MAX_LAT_DIFFERENCE / 2,
                    srs=gk.srs.EPSG4326,
                )

            # find slices which contains our extent
            if self.dependent_coordinates:
                left = self._allLons < self.bounds.xMin
                right = self._allLons > self.bounds.xMax
                if (left | right).all():
                    left[:, :-1] = np.logical_and(left[:, 1:], left[:, :-1])
                    right[:, 1:] = np.logical_and(right[:, 1:], right[:, :-1])

                bot = self._allLats < self.bounds.yMin
                top = self._allLats > self.bounds.yMax
                if (top | bot).all():
                    top[:-1, :] = np.logical_and(top[1:, :], top[:-1, :])
                    bot[1:, :] = np.logical_and(bot[1:, :], bot[:-1, :])

                self._lonStart = np.argmin((bot | left | top).all(0)) - 1 - index_pad
                self._lonStop = (
                    self._lonN
                    - np.argmin((bot | top | right).all(0)[::-1])
                    + 1
                    + index_pad
                )
                self._latStart = np.argmin((bot | left | right).all(1)) - 1 - index_pad
                self._latStop = (
                    self._latN
                    - np.argmax((left | top | right).all(1)[::-1])
                    + 1
                    + index_pad
                )

            else:
                tmp = np.logical_and(
                    self._allLons >= self.bounds.xMin, self._allLons <= self.bounds.xMax
                )
                self._lonStart = np.argmax(tmp) - 1
                self._lonStop = (
                    self._lonStart + 1 + np.argmin(tmp[self._lonStart + 1 :]) + 1
                )

                tmp = np.logical_and(
                    self._allLats >= self.bounds.yMin, self._allLats <= self.bounds.yMax
                )
                self._latStart = np.argmax(tmp) - 1
                self._latStop = (
                    self._latStart + 1 + np.argmin(tmp[self._latStart + 1 :]) + 1
                )

                self._lonStart = max(0, self._lonStart - index_pad)
                self._lonStop = min(self._allLons.size, self._lonStop + index_pad)
                self._latStart = max(0, self._latStart - index_pad)
                self._latStop = min(self._allLats.size, self._latStop + index_pad)

        else:
            self.bounds = None
            self._lonStart = 0
            self._latStart = 0

            if self.dependent_coordinates:
                self._lonStop = self._allLons.shape[1]
                self._latStop = self._allLons.shape[0]
            else:
                self._lonStop = self._allLons.size
                self._latStop = self._allLats.size

        # Read working lats/lon
        self._flip_lat = flip_lat
        self._flip_lon = flip_lon

        if self.dependent_coordinates:
            self.lats = self._allLats[
                self._latStart : self._latStop, self._lonStart : self._lonStop
            ]
            self.lons = self._allLons[
                self._latStart : self._latStop, self._lonStart : self._lonStop
            ]

            if flip_lat:
                self.lats = self.lats[::-1, :]
            if flip_lon:
                self.lons = self.lons[:, ::-1]
        else:
            self.lats = self._allLats[self._latStart : self._latStop]
            self.lons = self._allLons[self._lonStart : self._lonStop]

            if flip_lat:
                self.lats = self.lats[::-1]
            if flip_lon:
                self.lons = self.lons[::-1]

        self.extent = gk.Extent(
            self.lons.min(),
            self.lats.min(),
            self.lons.max(),
            self.lats.max(),
            srs=gk.srs.EPSG4326,
        )

        # compute time index
        self.time_name = time_name

        ds = nc.Dataset(self.variables["path"][time_name], keepweakref=True)
        timeVar = ds[time_name]
        timeindex = nc.num2date(
            timeVar[:],
            timeVar.units,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        )
        ds.close()

        if time_offset_minutes is not None:
            from datetime import timedelta

            timeindex = [t + timedelta(minutes=time_offset_minutes) for t in timeindex]

        self._timeindex_raw = pd.DatetimeIndex(timeindex)
        if not tz is None:
            self.time_index = self._timeindex_raw.tz_localize(tz)
        else:
            self.time_index = self._timeindex_raw

        # initialize the data container
        self.data = OrderedDict()

    def var_info(self, var):
        """Prints more information about the given variable

        Parameters:
        -----------
        var : str
            The variable to get more information about


        Returns:
        --------
        None


        Note:
        -----
        You can access a list of all available variables by printing the member ".variables"

        """
        assert var in self.variables.index
        ds = nc.Dataset(self.variables["path"][var], keepweakref=True)
        print(ds[var])
        ds.close()

    def to_pickle(self, path):
        """Save the source as a pickle file, so it can be quickly reopened later

        Parameters:
        -----------
        path : str
            The path to write the output file at

        Returns:
        --------
        None
        """
        from pickle import dump

        with open(path, "wb") as fo:
            dump(self, fo)

    @staticmethod
    def from_pickle(path):
        """Load an NCSource source from a pickle file

        Parameters:
        -----------
        path : str
            The path to read from


        Returns:
        --------
        NCSource
        """
        from pickle import load

        with open(path, "rb") as fo:
            out = load(fo)
        return out

    def list_standard_variables(self):
        """Prints the standard variable loaders available to this weather source"""
        for var in dir(self):
            if var[:6] == "sload_":
                print(var)

    def sload(self, *variables):
        """Load standard variables into the source's data library

        Parameters:
        -----------
        *variables : str
            The standard variables to read from the weather source

        Returns:
        --------
        None


        Raises:
        --------
        RuntimeError
            If the given standard variable name is not known to the weather source


        Note:
        -----
        The names of the standard variable do not refer to the names of the data within the source.
            Instead, they refer to common plain-english names which are translated to the source-
            specific names within the associated standard-loader function

        You can see which standard loaders are are available for the weather source by seeing the
            class methods starting with the name "sload_"

        Common variable names include:
            elevated_wind_speed          -> The wind speed at `WIND_SPEED_HEIGHT_FOR_WIND_ENERGY`
            surface_wind_speed           -> The wind speed at `WIND_SPEED_HEIGHT_FOR_SOLAR_ENERGY`
            wind_speed_at_Xm             -> The wind speed at X meters above the surface
            elevated_wind_direction      -> The wind direction at `WIND_SPEED_HEIGHT_FOR_WIND_ENERGY`
            surface_wind_direction       -> The wind direction at `WIND_SPEED_HEIGHT_FOR_SOLAR_ENERGY`
            wind_direction_at_Xm         -> The wind direction at X meters above the surface
            surface_pressure             -> The pressure at the surface
            surface_air_temperature      -> The air temperature at the surface
            surface_dew_temperature      -> The dew-point temperature at the surface
            global_horizontal_irradiance -> The global horizontal irradiance at the surface
            direct_normal_irradiance     -> The direct normal irradiance at the surface
            direct_horzontal_irradiance  -> The direct irradiance at the surface on a horizontal plane


        See Also:
        ---------
        NCSource.load( variable, name, height_index, processor )
            - For more configurable data loading into the weather source

        """
        for var in variables:
            if hasattr(self, "sload_" + var):
                getattr(self, "sload_" + var)()
            else:
                raise RuntimeError(
                    var + " is not an acceptable key for this weather source"
                )

    def load(
        self, variable, name=None, height_idx=None, processor=None, overwrite=False
    ):
        """Load a variable into the source's data table

        Parameters:
        -----------
        variable : str
            The variable within the currated datasources to load
              * The variable must either be of dimension (time, lat, lon) or
                (time, height, lat, lon)

        name : str, optional
            The name to give this variable in the data library
              * If None, the name of the original variable is kept

        height_idx : int; optional
            The height index to extract if the original variable has the height
            dimension

        processor : func, optional
            A function to process the loaded data before loading it into the
            the data library
              * This function must take a single matrix argument with dimensions
                (time, lat, lon), and must return a matrix of the same shape
              * Example:If the NC file has temperature in Kelvin and you need C:
                  processor = lambda x: x+273.15

        overwrite : bool, optional
            If False, then this function will exit early if the desired variable name
                already exists within the data library. Otherwise, any pre-existing data
                is overwritten


        Returns:
        --------
        None


        See Also:
        ---------
        sload( variable )
            - For loading standard variables into the weather source using pre-configured calls
                to 'load'

        """
        if name is None:
            name = variable

        if not overwrite and name in self.data:
            # nothing to do...
            return

        # read the data
        assert variable in self.variables.index
        ds = nc.Dataset(self.variables["path"][variable], keepweakref=True)
        var = ds[variable]

        if height_idx is None:
            tmp = var[:, self._latStart : self._latStop, self._lonStart : self._lonStop]
        else:
            tmp = var[
                :,
                height_idx,
                self._latStart : self._latStop,
                self._lonStart : self._lonStop,
            ]

        # process, maybe?
        if processor is not None:
            tmp = processor(tmp)

        # forward fill the last time step since it can sometimes be missing
        if not tmp.shape[0] == self._timeindex_raw.shape[0]:
            if not self.fill:
                raise ResError(
                    "Time mismatch with variable %s. Expected %d, got %d"
                    % (variable, self.time_index.shape[0], tmp.shape[0])
                )

            lastTimeIndex = nc.num2date(
                ds[self.time_name][-1], ds[self.time_name].units
            )

            if not lastTimeIndex in self._timeindex_raw:
                raise ResError("Filling is only intended to fill the last missing step")
            tmp = np.append(tmp, tmp[np.newaxis, -1, :, :], axis=0)

        # save the data
        if not self._flip_lat and not self._flip_lon:
            self.data[name] = tmp
        elif self._flip_lat and not self._flip_lon:
            self.data[name] = tmp[:, ::-1, :]
        elif not self._flip_lat and self._flip_lon:
            self.data[name] = tmp[:, :, ::-1]
        elif self._flip_lat and self._flip_lon:
            self.data[name] = tmp[:, ::-1, ::-1]

        # Clean up
        ds.close()

    @staticmethod
    def _loc_to_index_rect(lat_step, lon_step):
        def func(self, loc, outside_okay=False, as_int=True):
            """Returns the closest X and Y indexes corresponding to a given location
            or set of locations

            Parameters:
            -----------
            loc : Anything acceptable by geokit.LocationSet
                The location(s) to search for
                * A single tuple with (lon, lat) is acceptable, or a list of such tuples
                * A single point geometry (as long as it has an SRS), or a list
                  of geometries is okay
                * geokit,Location, or geokit.LocationSet are best!

            outside_okay : bool, optional
                Determines if points which are outside the source's lat/lon grid
                are allowed
                * If True, points outside this space will return as None
                * If False, an error is raised

            Returns:
            --------
            If a single location is given: tuple
                * Format: (yIndex, xIndex)
                * y index can be accessed with '.yi'
                * x index can be accessed with '.xi'

            If multiple locations are given: list
                * Format: [ (yIndex1, xIndex1), (yIndex2, xIndex2), ...]
                * Order matches the given order of locations

            """
            # Ensure loc is a list
            locations = gk.LocationSet(loc)

            # get closest indices
            latI = (locations.lats - self.lats[0]) / lat_step
            lonI = (locations.lons - self.lons[0]) / lon_step

            # Check for out of bounds
            oob = (latI < 0) | (latI >= self._latN) | (lonI < 0) | (lonI >= self._lonN)
            if oob.any():
                if not outside_okay:
                    print("The following locations are out of bounds")
                    print(locations[oob])
                    raise ResError("Locations are outside the boundaries")

            # As int?
            if as_int:
                latI = np.round(latI).astype(int)
                lonI = np.round(lonI).astype(int)

            # Make output
            if locations.count == 1:
                if oob[0] is True:
                    return None
                else:
                    return Index(yi=latI[0], xi=lonI[0])
            else:
                return [
                    None if _oob else Index(yi=y, xi=x)
                    for _oob, y, x in zip(oob, latI, lonI)
                ]

        return func

    def loc_to_index(self, loc, outside_okay=False, as_int=True):
        """Returns the closest X and Y indexes corresponding to a given location
        or set of locations


        Parameters:
        -----------
        loc : Anything acceptable by geokit.LocationSet
            The location(s) to search for
            * A single tuple with (lon, lat) is acceptable, or a list of such tuples
            * A single point geometry (as long as it has an SRS), or a list
              of geometries is okay
            * geokit,Location, or geokit.LocationSet are best!

        outside_okay : bool, optional
            Determines if points which are outside the source's lat/lon grid
            are allowed
            * If True, points outside this space will return as None
            * If False, an error is raised


        Returns:
        --------
        If a single location is given: tuple
            * Format: (yIndex, xIndex)
            * y index can be accessed with '.yi'
            * x index can be accessed with '.xi'

        If multiple locations are given: list
            * Format: [ (yIndex1, xIndex1), (yIndex2, xIndex2), ...]
            * Order matches the given order of locations


        Note:
        -----
        The default form of this function (which is the one used here) is not very efficient, ultimately
            leading to much longer look-up than they otherwise need to be. When the weather source has
            grid cells on a regular lat/lon grid then a more efficient form of this function can be
            configured using the function generator "_loc_to_index_rect". In these instances, this is
            the recommended function to use.

        For example, if the weather source uses a latitude spacing of 0.5, and a longitude spacing of
            0.625, then the function generator can be used like:

            > source.loc_to_index = source._loc_to_index_rect(lat_step=0.5, lon_step=0.625)

        """
        # Ensure loc is a list
        locations = gk.LocationSet(loc)

        # get closest indices
        idx = []
        for lat, lon in zip(locations.lats, locations.lons):
            # Check the distance
            latDist = lat - self.lats
            lonDist = lon - self.lons

            # Get the best indices
            if self.dependent_coordinates:
                dist = lonDist * lonDist + latDist * latDist
                latI, lonI = np.unravel_index(np.argmin(dist), dist.shape)

                latDists = []
                if latI < self._latN - 1:
                    latDists.append((self.lats[latI + 1, lonI] - self.lats[latI, lonI]))
                if latI > 0:
                    latDists.append((self.lats[latI, lonI] - self.lats[latI - 1, lonI]))
                latDistI = latDist[latI, lonI] / np.mean(latDists)

                lonDists = []
                if lonI < self._lonN - 1:
                    lonDists.append((self.lons[latI, lonI + 1] - self.lons[latI, lonI]))
                if lonI > 0:
                    lonDists.append((self.lons[latI, lonI] - self.lons[latI, lonI - 1]))
                lonDistI = lonDist[latI, lonI] / np.mean(lonDists)

            else:
                lonI = np.argmin(np.abs(lonDist))
                latI = np.argmin(np.abs(latDist))

                latDists = []
                if latI < self._latN - 1:
                    latDists.append((self.lats[latI + 1] - self.lats[latI]))
                if latI > 0:
                    latDists.append((self.lats[latI] - self.lats[latI - 1]))
                latDistI = latDist[latI] / np.mean(latDists)

                lonDists = []
                if lonI < self._latN - 1:
                    lonDists.append((self.lons[lonI + 1] - self.lons[lonI]))
                if lonI > 0:
                    lonDists.append((self.lons[lonI] - self.lons[lonI - 1]))
                lonDistI = lonDist[lonI] / np.mean(lonDists)

            # Check for out of bounds
            if (
                np.abs(latDistI) > self._maximal_lat_difference
                or np.abs(lonDistI) > self._maximal_lon_difference
            ):
                if not outside_okay:
                    raise ResError("(%f,%f) are outside the boundaries" % (lat, lon))
                else:
                    idx.append(None)
                    continue

            # As int?
            if not as_int:
                latI = latI + latDistI
                lonI = lonI + lonDistI

            # append
            idx.append(Index(yi=latI, xi=lonI))

        # Make output
        if locations.count == 1:
            return idx[0]
        else:
            return idx

    def get(
        self,
        variable,
        locations,
        interpolation="near",
        force_as_data_frame=False,
        outside_okay=False,
        _indicies=None,
    ):
        """
        Retrieve a time series for a variable from the source's data library at the given location(s)

        Can also use various interpolation schemes (e.g. near, bilinear, or cubic)

        Parameters:
        -----------
            variable : str
                The variable within the data library to extract

            locations : Anything acceptable by geokit.LocationSet.load( )
                The location(s) to search for

                * geokit.Location, or geokit.LocationSet are best
                * A single tuple with (lon, lat) is acceptable, or a list of such tuples
                * A single point geometry (as long as it has an SRS), or a list of geometries

            interpolation : str, optional
                The interpolation method to use

                * 'near' => For each location, extract the time series from the source's
                closest lat/lon index
                * 'bilinear' => For each location, use the time series of the source's
                surrounding +/- 1 index locations to create an estimated time
                series at the given location using a biliear interpolation scheme
                * 'cubic' => For each location, use the time series of the source's
                surrounding +/- 2 index locations to create an estimated time
                series at the given location using a cubic scheme

            force_as_data_frame : bool, optional
                If True, instructs the returned value to always take the form of a
                Pandas DataFrame regardless of how many locations are specified

            outside_okay : bool, optional
                Determines if points which are outside the source's lat/lon grid
                are allowed
                * If True, points outside this space will return as None
                * If False, an error is raised

        Returns:
        --------

        If a single location is given: pandas.Series
          * Indexes match to the source's time dimension

        If multiple locations are given (or if `force_as_data_frame` is True): pandas.DataFrame
          * Indexes match to the source's time dimension
          * Columns match to the given order of locations

        """
        # Ensure loc is a list
        locations = gk.LocationSet(locations)

        # Get the indicies
        if _indicies is None:
            # compute the closest indices
            if not self.dependent_coordinates or interpolation == "near":
                as_int = True
            else:
                as_int = False
            indicies = self.loc_to_index(locations, outside_okay, as_int=as_int)
        else:
            # Assume indicies match locations
            indicies = _indicies

        if isinstance(indicies, Index):
            indicies = [
                indicies,
            ]

        # Do interpolation
        if interpolation == "near":
            # arrange the output data
            tmp = []
            for i in indicies:
                if not i is None:
                    tmp.append(self.data[variable][:, i.yi, i.xi])
                else:
                    tmp.append(
                        np.array(
                            [
                                np.nan,
                            ]
                            * self.time_index.size
                        )
                    )
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
            for i in indicies:
                if i is None:

                else:
                    xi = i.xi
                    yi = i.yi
            yiMin = np.round(min([i.yi for i in indicies]) - win).astype(int)
            yiMax = np.round(max([i.yi for i in indicies]) + win).astype(int)
            xiMin = np.round(min([i.xi for i in indicies]) - win).astype(int)
            xiMax = np.round(max([i.xi for i in indicies]) + win).astype(int)

            # ensure boundaries are okay
            if yiMin < 0 or xiMin < 0 or yiMax > self._latN or xiMax > self._lonN:
                raise ResError(
                    "Insufficient data. Try expanding the boundary of the extracted data"
                )

            ##########
            # TODO: Update interpolation schemes to handle out-of-bounds indices
            ##########

            if self.dependent_coordinates:  # do interpolations in 'index space'
                if isinstance(indicies[0][0], int):
                    raise ResError("Index must be float type for interpolation")

                gridYVals = np.arange(yiMin, yiMax + 1)
                gridXVals = np.arange(xiMin, xiMax + 1)

                yInterp = [i.yi for i in indicies]
                xInterp = [i.xi for i in indicies]

            else:  # do interpolation in the expected 'coordinate space'
                gridYVals = self.lats[yiMin : yiMax + 1]
                gridXVals = self.lons[xiMin : xiMax + 1]

                yInterp = [loc.lat for loc in locations]
                xInterp = [loc.lon for loc in locations]

            # Do interpolation
            output = []
            for ts in range(self.data[variable].shape[0]):
                # set up interpolation
                rbs = RectBivariateSpline(
                    gridYVals,
                    gridXVals,
                    self.data[variable][ts, yiMin : yiMax + 1, xiMin : xiMax + 1],
                    **rbsArgs,
                )

                # interpolate for each location
                # lat/lon order switched to match index order
                output.append(rbs(yInterp, xInterp, grid=False))

            output = np.stack(output)

        else:
            raise ResError(
                "Interpolation scheme not one of: 'near', 'cubic', or 'bilinear'"
            )

        # Make output as Series objects
        if force_as_data_frame or (len(output.shape) > 1 and output.shape[1] > 1):
            return pd.DataFrame(output, index=self.time_index, columns=locations)
        else:
            try:
                return pd.Series(output[:, 0], index=self.time_index, name=locations[0])
            except:
                return pd.Series(output, index=self.time_index, name=locations[0])
