from ..NCSource import *

# Define constants


class MerraSource(NCSource):

    ELEVATED_WIND_SPEED_HEIGHT = 50
    SURFACE_WIND_SPEED_HEIGHT = 2

    LONG_RUN_AVERAGE_WINDSPEED = join(
        dirname(__file__),
        "data",
        "merra_average_windspeed_50m-shifted.tif")

    LONG_RUN_AVERAGE_GHI = join(
        dirname(__file__),
        "data",
        "Emerra_average_SWGDN_1994-2015_globe.tif")

    MAX_LON_DIFFERENCE = 0.5
    MAX_LAT_DIFFERENCE = 0.5

    def __init__(s, source, bounds=None, indexPad=5, **kwargs):
        """Initialize a Merra2 style netCDF4 file source


        Parameters
        ----------
        source : str
            The path to the main data file

        bounds : Anything acceptable to geokit.Extent.load(), optional
            The boundaries of the data which is needed
              * Usage of this will help with memory mangement
              * If None, the full dataset is loaded in memory

        padExtent : numeric, optional
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

        NCSource.__init__(s, source=source, bounds=bounds, timeName="time", latName="lat", lonName="lon",
                          indexPad=indexPad, _maxLonDiff=s.MAX_LON_DIFFERENCE, _maxLatDiff=s.MAX_LAT_DIFFERENCE,
                          tz="GMT", **kwargs)

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
        latI = (locations.lats - s.lats[0]) / 0.5
        lonI = (locations.lons - s.lons[0]) / 0.625

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

    def contextAreaAtIndex(s, latI, lonI):
        """Compute the context area surrounding the a specified index"""
        # Make and return a box
        lowLat = s.lats[latI] - 0.25
        highLat = s.lats[latI] + 0.25
        lowLon = s.lons[lonI] - 0.3125
        highLon = s.lons[lonI] + 0.3125

        return gk.geom.box(lowLon, lowLat, highLon, highLat, srs=gk.srs.EPSG4326)

    # def loadWindSpeed(s, height=50, winddir=False):
    #     """Load the U and V wind speed data at the specified height, and compute
    #     the overall windspeed and winddir

    #     Parameters
    #     ----------
    #     height : int, optional
    #         The height value to load, given in meters above ground
    #           * Options are 2, 10, 50
    #           * Maps to a vaiable named 'windspeed'

    #     winddir : bool, optional
    #         If True, the wind direction is calculated and saved under a variable
    #         named 'winddir'
    #     """
    #     # read raw data
    #     s.load("U%dM" % height)
    #     s.load("V%dM" % height)

    #     # read the data
    #     uData = s.data["U%dM" % height]
    #     vData = s.data["V%dM" % height]

    #     # combine into a single time series matrix
    #     speed = np.sqrt(uData * uData + vData * vData)  # total speed
    #     s.data["windspeed"] = speed

    #     if winddir:
    #         direction = np.arctan2(vData, uData) * \
    #             (180 / np.pi)  # total direction
    #         s.data["winddir"] = direction

    # def loadRadiation(s):
    #     """Load the SWGDN variable into the data table with the name 'ghi'
    #     """
    #     s.load("SWGDN", name="ghi")

    # def loadTemperature(s, which='air', height=2):
    #     """Load air temperature variables

    #     The name of the variable loaded into the data table depends on the type
    #     of temperature chosen:
    #       * If which='air' -> 'air_temp' is created
    #       * If which='dew' -> 'dew_temp' is created
    #       * If which='wet' -> 'wet_temp' is created

    #     Parameters:
    #     -----------
    #     which : str, optional
    #         The specific type of air temperature to read
    #         * Can be: air, dew, or wet

    #     height : int, optional
    #         The height in meters to load
    #         * Options are: 2, 10? and 50?
    #     """
    #     if which.lower() == 'air':
    #         varName = "T%dM" % height
    #     elif which.lower() == 'dew':
    #         varName = "T%dMDEW" % height
    #     elif which.lower() == 'wet':
    #         varName = "T%dMWET" % height
    #     else:
    #         raise ResMerraError("sub group '%s' not understood" % which)

    #     # load
    #     s.load(varName, name=which + "_temp", processor=lambda x: x - 273.15)

    # def loadPressure(s):
    #     """Load the PS Merra variable into the data table with the name 'pressure'"""
    #     s.load("PS", name='pressure')

    # def loadSet_PV(s, verbose=False, _clockstart=None, _header=""):
    #     """Load basic PV power simulation variables

    #       * 'windspeed' from U2M and V2M
    #       * 'dni' from SWGDN
    #       * 'air_temp' from T2M
    #       * 'pressure' from PS
    #     """
    #     if verbose:
    #         from datetime import datetime as dt
    #         if _clockstart is None:
    #             _clockstart = dt.now()
    #         print(_header, "Loading wind speeds at: +%.2fs" %
    #               (dt.now() - _clockstart).total_seconds())

    #     s.loadWindSpeed(height=2)
    #     del s.data["U2M"]
    #     del s.data["V2M"]

    #     if verbose:
    #         print(_header, "Loading ghi at: +%.2fs" %
    #               (dt.now() - _clockstart).total_seconds())
    #     s.load("SWGDN", "ghi")
    #     # s.loadRadiation()
    #     if verbose:
    #         print(_header, "Loading temperature at: +%.2fs" %
    #               (dt.now() - _clockstart).total_seconds())
    #     s.loadTemperature('air', height=2)
    #     s.loadTemperature('dew', height=2)

    #     if verbose:
    #         print(_header, "Loading pressure at: +%.2fs" %
    #               (dt.now() - _clockstart).total_seconds())
    #     s.loadPressure()

    #     if verbose:
    #         print(_header, "Done loading data at: +%.2fs" %
    #               (dt.now() - _clockstart).total_seconds())

    # def loadSet_Wind(s):
    #     """Load basic Wind power simulation variables

    #       * 'windspeed' from U50M and V50M
    #     """
    #     s.loadWindSpeed(height=50)
    #     del s.data["U50M"]
    #     del s.data["V50M"]

    # STANDARD LOADERS
    def _load_uv(self, height):
        U = "U{}M".format(height)
        V = "V{}M".format(height)

        self.load(U)
        self.load(V)

        return self.data[U], self.data[V]

    def _load_wind_speed(self, height):
        uData, vData = self._load_uv(height=height)
        return np.sqrt(uData * uData + vData * vData)  # total speed

    def sload_elevated_wind_speed(self):
        self.data["elevated_wind_speed"] = self._load_wind_speed(
            height=self.ELEVATED_WIND_SPEED_HEIGHT)

    def sload_surface_wind_speed(self):
        self.data["surface_wind_speed"] = self._load_wind_speed(
            height=self.SURFACE_WIND_SPEED_HEIGHT)

    def sload_wind_speed_at_2m(self):
        self.data["wind_speed_at_2m"] = self._load_wind_speed(2)

    def sload_wind_speed_at_10m(self):
        self.data["wind_speed_at_10m"] = self._load_wind_speed(10)

    def sload_wind_speed_at_50m(self):
        self.data["wind_speed_at_50m"] = self._load_wind_speed(50)

    def _load_wind_dir(self, height):
        uData, vData = self._load_uv(height=height)
        return np.arctan2(vData, uData) * (180 / np.pi)  # total direction

    def sload_elevated_wind_direction(self):
        self.data["elevated_wind_speed"] = self._load_wind_speed(
            height=self.ELEVATED_WIND_SPEED_HEIGHT)

    def sload_surface_wind_direction(self):
        self.data["surface_wind_speed"] = self._load_wind_speed(
            height=self.SURFACE_WIND_SPEED_HEIGHT)

    def sload_wind_direction_at_2m(self):
        self.data["wind_speed_at_2m"] = self._load_wind_speed(2)

    def sload_wind_direction_at_10m(self):
        self.data["wind_speed_at_10m"] = self._load_wind_speed(10)

    def sload_wind_direction_at_50m(self):
        self.data["wind_speed_at_50m"] = self._load_wind_speed(50)

    def sload_surface_pressure(self):
        return self.load("PS", name='surface_pressure')

    def sload_surface_air_temperature(self):
        return self.load("T2M", name="surface_air_temperature", processor=lambda x: x - 273.15)

    def sload_surface_dew_temperature(self):
        return self.load("T2MDEW", name="surface_dew_temperature", processor=lambda x: x - 273.15)

    def sload_global_horizontal_irradiance(self):
        return self.load("SWGDN", name="global_horizontal_irradiance")
