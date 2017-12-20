from ..NCSource import *

## Define constants
class TrySource(NCSource):
    """
    Open a netCDF4 source which is in the TRY domain (from DWD)

    Standard variables are:
        TT  - air temperature at 2m, x0.1 [1/10 °C]
        N   - cloud_cover
        TD  - dew_point
        RH  - humidity
        PRED- sea level pressure          [hPa]
        SID - radiation_direct            [Wh/m2]
        SDL - radiation_downwelling
        SIS - radiation_global            [Wh/m2]
        SOL - radiation_upwelling        
        X   - vapor_pressure
        DD  - wind_direction
        FF  - wind speed at 10m, x0.1     [m/s]
    """
    
    GWA50_CONTEXT_MEAN_SOURCE = None
    GWA100_CONTEXT_MEAN_SOURCE = None

    MAX_LAT_DIFFERENCE = 0.01
    MAX_LON_DIFFERENCE = 0.02

    def __init__(s, path, bounds=None):

        if not bounds is None:
            if isinstance(bounds, gk.Extent):
                bounds.pad( (s.MAX_LON_DIFFERENCE, s.MAX_LAT_DIFFERENCE) )
            else:
                if isinstance(bounds, Bounds):
                    lonMin = bounds.lonMin
                    latMin = bounds.latMin
                    lonMax = bounds.lonMax
                    latMax = bounds.latMax
                else:
                    print("Consider using a Bounds object or a gk.Extent object. They are safer!")
                    lonMin,latMin,lonMax,latMax = bounds
                    
                bounds = Bounds(lonMin = lonMin - s.MAX_LON_DIFFERENCE,
                                latMin = latMin - s.MAX_LAT_DIFFERENCE,
                                lonMax = lonMax + s.MAX_LON_DIFFERENCE,
                                latMax = latMax + s.MAX_LAT_DIFFERENCE,)

        NCSource.__init__(s, path=path, bounds=bounds, timeName="time", latName="lat", lonName="lon", dependent_coordinates=True)

        # set maximal differences
        s._maximal_lon_difference=s.MAX_LON_DIFFERENCE
        s._maximal_lat_difference=s.MAX_LAT_DIFFERENCE

    def __add__(s,o):
        out = TrySource(None)
        return NCSource.__add__(s, o, _shell=out)

    def loadWindSpeed(s):
        """Load windspeed at 10m"""
        s.load("FF", name="windspeed")

    def loadRadiation(s):
        """Load ghi and dni"""
        s.load("SIS", name="ghi")
        s.load("SID", name="dni")

    def loadTemperature(s, which='air'):
        """Temperature variable loader"""
        if which == 'air':
            s.load("temperature", name="air_temp")

    def loadPressure(s): 
        """Pressure variable loader"""
        s.load("SLP", name="pressure", processor=lambda x: x*100)

    def loadSet_PV(s):
        """Load basic PV simulating variables"""
        s.loadWindSpeed()
        s.loadRadiation()
        s.loadTemperature('air')
        s.loadPressure()
