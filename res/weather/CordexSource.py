from .NCSource import *

## Define constants
class CordexSource(NCSource):
    """
    Open a netCDF4 source which is at the EURO-CORDEX EUR-11 domain

    Standard variables are:
        clt   - cloud cover                                    [] 
        dpas  - 2m dew point temperature                       [K]
        hurs  - 2m relative humidity                           []
        huss  - 2m specific humidity                           [kg kg-1]
        pr    - total (convective + large scale) precipitation [kg m-2 s-1]
        prsn  - snowfall flux                                  [kg m-2 s-1]
        ps    - surface pressure                               [Pa]
        rlen  - roughness length                               [m]
        rsds  - surface downwelling shortwave radiation        [W m-2] 
        rsdt  - top of atmosphere incident shortwave radiation [W m-2]
        tas   - 2m temperature                                 [K]
        uas   - 10m u-velocity                                 [m s-1]
        vas   - 10m v-velocity                                 [m s-1]
        glat  - geographical latitude                          [deg N]
        glon  - geographical longitude                         [deg E]
        orog  - surface orography                              [m]
        sftlf - lang area fraction                             []
    """
    
    GWA50_CONTEXT_MEAN_SOURCE = None
    GWA100_CONTEXT_MEAN_SOURCE = None
    LONG_RUN_AVERAGE_50M_SOURCE = None

    MAX_LON_DIFFERENCE=0.0625
    MAX_LAT_DIFFERENCE=0.0625

    def __init__(s, path, bounds=None,):

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
        out = CordexSource(None)
        return NCSource.__add__(s, o, _shell=out)

    def loadWindSpeed(s, vName="vas", uName="uas" ):
        # read raw data
        s.load(vName, heightIdx=0)
        s.load(uName, heightIdx=0)

        # read the data
        uData = s.data[uName]
        vData = s.data[vName]

        # combine into a single time series matrix
        speed = np.sqrt(uData*uData+vData*vData) # total speed
        direction = np.arctan2(vData,uData)*(180/np.pi)# total direction
        
        # done!
        s.data["windspeed"] = speed
        s.data["winddir"] = direction

    def loadRadiation(s, ghiName="rsds"):
        # read raw data
        s.load(ghiName, name="ghi")

    def loadTemperature(s, which='air', processor=lambda x: x-273.15):
        """Temperature variable loader"""
        if which.lower() == 'air': varName = "tas"
        elif which.lower() == 'dew': varName = "dpas"
        else: raise ResMerraError("sub group '%s' not understood"%which)

        # load
        s.load(varName, name=which+"_temp", processor=processor)

    def loadPressure(s): s.load("ps", name='pressure')