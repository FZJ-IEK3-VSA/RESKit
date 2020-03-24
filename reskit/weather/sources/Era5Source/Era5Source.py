from ..NCSource import *

# Define constants

class Era5Source(NCSource):

    LONG_RUN_AVERAGE_WINDSPEED_100M = join(
        dirname(__file__), "data", "ERA5_wind_speed_100m_mean.tiff")
    LONG_RUN_AVERAGE_WINDDIR_100M = join(
        dirname(__file__), "data", "ERA5_wind_direction_100m_mean.tiff")
    LONG_RUN_AVERAGE_GHI = join(
        dirname(__file__), "data", "ERA5_surface_solar_radiation_downwards_mean.tiff")
    LONG_RUN_AVERAGE_DNI = join(
        dirname(__file__), "data", "ERA5_total_sky_direct_solar_radiation_at_surface_mean.tiff")
    
    MAX_LON_DIFFERENCE = 0.26
    MAX_LAT_DIFFERENCE = 0.26

    def __init__(s, source, bounds=None, indexPad=5, **kwargs):
        """Initialize a ERA5 style netCDF4 file source


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

        NCSource.__init__(s, source=source, bounds=bounds, timeName="time", latName="latitude", lonName="longitude",
                          indexPad=indexPad, _maxLonDiff=s.MAX_LON_DIFFERENCE, _maxLatDiff=s.MAX_LAT_DIFFERENCE,
                          tz=None, flip_lat=True, **kwargs)

    loc2Index = NCSource._loc2IndexRect(0.25, 0.25)
    
    def loadWindSpeed(s, height=100, winddir=False):
        """TODO:UPDATE
        Load the U and V wind speed data at the specified height, and compute
        the overall windspeed and winddir

        Parameters
        ----------
        height : int, optional
            The height value to load, given in meters above ground
              * Options are 2, 10, 50
              * Maps to a vaiable named 'windspeed'

        winddir : bool, optional
            If True, the wind direction is calculated and saved under a variable
            named 'winddir'
        """
        # read raw data
        assert height in [10,100], "Acceptable heights are 10 and 100"
        
        s.load("ws%d" % height, "windspeed")
        if winddir: 
            s.load("wd%d" % height, "winddir")

    def loadRadiation(s):
        """TODO:UPDATE
        Load the SWGDN variable into the data table with the name 'ghi'
        """
        s.load("ssrd", name="ghi")
        s.load("fdir", name="dni_flat")

    def loadTemperature(s, air=True, dew=False, height=2):
        """TODO:UPDATE
        Load air temperature variables 

        The name of the variable loaded into the data table depends on the type
        of temperature chosen:
          * If which='air' -> 'air_temp' is created
          * If which='dew' -> 'dew_temp' is created
          * If which='wet' -> 'wet_temp' is created

        Parameters:
        -----------
        which : str, optional
            The specific type of air temperature to read
            * Can be: air, dew, or wet

        height : int, optional
            The height in meters to load
            * Options are: 2, 10? and 50?
        """
        assert height in [2], "Acceptable heights are [2, ]"
        if air:
            s.load("t%dm"%height, name="air_temp", processor=lambda x: x-273.15)
        if dew:
            s.load("d%dm"%height, name="dew_temp", processor=lambda x: x-273.15)
            
    def loadPressure(s):
        """Load the 'sp' variable into the data table with the name 'pressure'"""
        s.load("sp", name='pressure')

    def loadSet_PV(s, verbose=False, _clockstart=None, _header=""):
        """TODO:UPDATE
        Load basic PV power simulation variables

          * 'windspeed' from U2M and V2M
          * 'dni' from SWGDN 
          * 'air_temp' from T2M
          * 'pressure' from PS
        """
        if verbose:
            from datetime import datetime as dt
            if _clockstart is None:
                _clockstart = dt.now()
            print(_header, "Loading wind speeds at: +%.2fs" %
                  (dt.now()-_clockstart).total_seconds())

        s.loadWindSpeed(height=10)

        if verbose:
            print(_header, "Loading ghi at: +%.2fs" %
                  (dt.now()-_clockstart).total_seconds())
        s.loadRadiation()
        
        if verbose:
            print(_header, "Loading temperature at: +%.2fs" %
                  (dt.now()-_clockstart).total_seconds())
        s.loadTemperature(air=True, dew=True, height=2)

        if verbose:
            print(_header, "Loading pressure at: +%.2fs" %
                  (dt.now()-_clockstart).total_seconds())
        s.loadPressure()

        if verbose:
            print(_header, "Done loading data at: +%.2fs" %
                  (dt.now()-_clockstart).total_seconds())

    def loadSet_Wind(s):
        """TODO:UPDATE
        Load basic Wind power simulation variables

          * 'windspeed' from U50M and V50M
        """
        s.loadWindSpeed(height=100, winddir=True)
