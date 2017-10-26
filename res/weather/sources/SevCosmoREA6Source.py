from ..NCSource import *

## Define constants
class SevCosmoREA6Source(NCSource):
    """
    Handles the sources Sev created from the COSMO-REA6 dataset (cannot handle the orginal sources because they're whack)
    """
    
    GWA50_CONTEXT_MEAN_SOURCE = None
    GWA100_CONTEXT_MEAN_SOURCE = None

    MAX_LON_DIFFERENCE = 0.125 # a LARGE ooverestimate of how much space should be inbetween a given point and the nearest index
    MAX_LAT_DIFFERENCE = 0.125 # a LARGE ooverestimate of how much space should be inbetween a given point and the nearest index

    def __init__(s, path, constantsPath, bounds=None):

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

        NCSource.__init__(s, path=path, bounds=bounds, timeName="time", constantsPath=constantsPath,
                          latName="lat", lonName="lon", dependent_coordinates=True)

        # set maximal differences
        s._maximal_lon_difference=s.MAX_LON_DIFFERENCE
        s._maximal_lat_difference=s.MAX_LAT_DIFFERENCE
        
    #def __add__(s,o):
    #    out = CordexSource(None)
    #    return NCSource.__add__(s, o, _shell=out)

    def loadWindSpeed(s, height=100):
            # Check if height is on of the heights we already have
            # The 3 known heights should always be 10, 100, and 120
            if height == 10:
                  s.load("wspd", name="windspeed", heightIdx=0)
            elif height == 100:
                  s.load("wspd", name="windspeed", heightIdx=1)
            elif height == 120:
                  s.load("wspd", name="windspeed", heightIdx=2)
            else:
                # projection is required
                if height <= 60:
                    lowIndex = 0
                    highIndex = 1

                    lowHeight = 10
                    highHeight = 100
                else:
                    lowIndex = 1
                    highIndex = 2

                    lowHeight = 100
                    highHeight = 120

                
                s.load("wspd", name="lowWspd", heightIdx=lowIndex)
                s.load("wspd", name="highWspd", heightIdx=highIndex)

                lowData = s.data["lowWspd"]
                highData = s.data["highWspd"]

                alpha = np.log(highData/lowData)/np.log(highHeight/lowHeight)

                s.data["windspeed"] = lowData*np.power(height/lowHeight, alpha)

                del s.data["lowWspd"]
                del s.data["highWspd"]

    #def loadRadiation(s, ghiName="rsds"):
    #    # read raw data
    #    s.load(ghiName, name="ghi")

    #def loadTemperature(s, which='air', processor=lambda x: x-273.15):
    #    """Temperature variable loader"""
    #    if which.lower() == 'air': varName = "tas"
    #    elif which.lower() == 'dew': varName = "dpas"
    #    else: raise ResMerraError("sub group '%s' not understood"%which)
    #
    #    # load
    #    s.load(varName, name=which+"_temp", processor=processor)

    #def loadPressure(s): s.load("ps", name='pressure')