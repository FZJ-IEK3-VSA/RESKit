from .NCSource import *

## Define constants
class MerraSource (NCSource):
    
    GWA50_CONTEXT_MEAN_SOURCE = join(dirname(__file__),"..","..","data","gwa50_mean_over_merra.tif")
    GWA100_CONTEXT_MEAN_SOURCE = join(dirname(__file__),"..","..","data","gwa100_mean_over_merra.tif")
    LONG_RUN_AVERAGE_50M_SOURCE = join(dirname(__file__),"..","..","data","merra_average_windspeed_50m.tif")

    def __init__(s, path, bounds=None,):
        NCSource.__init__(s, path=path, bounds=bounds, timeName="time", latName="lat", lonName="lon")

        s._maximal_lon_difference=0.3125
        s._maximal_lat_difference=0.25

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


