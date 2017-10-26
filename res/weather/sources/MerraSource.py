from ..NCSource import *

## Define constants
class MerraSource(NCSource):
    
    GWA50_CONTEXT_MEAN_SOURCE = join(dirname(__file__),"..","..","data","gwa50_mean_over_merra.tif")
    GWA100_CONTEXT_MEAN_SOURCE = join(dirname(__file__),"..","..","data","gwa100_mean_over_merra.tif")
    LONG_RUN_AVERAGE_50M_SOURCE = join(dirname(__file__),"..","..","data","merra_average_windspeed_50m.tif")

    MAX_LON_DIFFERENCE=0.3125
    MAX_LAT_DIFFERENCE=0.25

    def __init__(s, path, bounds=None,):

        if not bounds is None:
            if isinstance(bounds, gk.Extent):
                bounds.castTo(LATLONSRS).pad( (s.MAX_LON_DIFFERENCE, s.MAX_LAT_DIFFERENCE) )
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

        NCSource.__init__(s, path=path, bounds=bounds, timeName="time", latName="lat", lonName="lon")

        # set maximal differences
        s._maximal_lon_difference=s.MAX_LON_DIFFERENCE
        s._maximal_lat_difference=s.MAX_LAT_DIFFERENCE

    def __add__(s,o):
        out = MerraSource(None)
        return NCSource.__add__(s, o, _shell=out)

    def contextAreaAtIndex(s, latI, lonI):
        print("USING MERRA VERSION!")

        # Make and return a box
        lowLat = s.lats[latI]-0.25
        highLat = s.lats[latI]+0.25
        lowLon = s.lons[lonI]-0.3125
        highLon = s.lons[lonI]+0.3125
        
        return gk.geom.box( lowLon, lowLat, highLon, highLat, srs=gk.srs.EPSG4326 )

    def loadWindSpeed(s, height=50 ):
        # read raw data
        s.load("U%dM"%height)
        s.load("V%dM"%height)

        # read the data
        uData = s.data["U%dM"%height]
        vData = s.data["V%dM"%height]

        # combine into a single time series matrix
        speed = np.sqrt(uData*uData+vData*vData) # total speed
        direction = np.arctan2(vData,uData)*(180/np.pi)# total direction
        
        # done!
        s.data["windspeed"] = speed
        s.data["winddir"] = direction

    def loadRadiation(s):
        # read raw data
        s.load("SWGNT", name="ghi")
        s.load("SWGDN", name="dni")

    def loadTemperature(s, which='air', height=2):
        """Temperature variable loader"""
        if which.lower() == 'air': varName = "T%dM"%height
        elif which.lower() == 'dew': varName = "T%dMDEW"%height
        elif which.lower() == 'wet': varName = "T%dMWET"%height
        else: raise ResMerraError("sub group '%s' not understood"%which)

        # load
        s.load(varName, name=which+"_temp", processor=None)

    def loadPressure(s): s.load("PS", name='pressure')


