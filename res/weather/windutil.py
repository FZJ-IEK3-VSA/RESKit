from .util import *

def fixLoc(loc):
    ## Ensure location is okay
    if isinstance(loc, ogr.Geometry): # Check if loc is a single point
        if not loc.GetSpatialReference().IsSame(gk.srs.EPSG4326):
            loc = loc.TransformTo(gk.srs.EPSG4326) # make sure the point is in lat-lon
    elif isinstance(loc, tuple) and len(loc)==2: # Check if loc is a single tuple
        lon,lat = loc
        loc = gk.geom.makePoint(float(lon), float(lat), srs=gk.srs.EPSG4326)
    else: # assume loc is iterable and so lets make it into a series of points
        loc = [gk.geom.makePoint(float(lon), float(lat), srs=gk.srs.EPSG4326) for lon,lat in loc]

    return loc

def spatialAdjustmentByGWA(ws, targetLoc, gwa, gwaMean=None, longRunAverage=None, contextArea=None):
    # check inputs
    if (gwaMean is None and longRunAverage is None and contextArea is None) or \
       (not gwaMean is None and not longRunAverage is None and not contextArea is None) :
        raise RuntimeError("Exactly one of gwaMean, longRunAverage or contextArea must be defined")

    ## Ensure location is okay
    targetLoc = fixLoc(targetLoc)
    
    # Get the local gwa value
    gwaLocValue = gk.raster.extractValues(gwa, targetLoc).data

    # Apply methods
    if not contextArea is None: # The user has chosen the GWA relative method, but the GWA average in the context area
                                #  needs to be determined
        if isinstance(contextArea, tuple): # context area is a set lat/lon boundaries, make them into a geometry 
            contextArea = gk.geom.makeBox(contextArea, srs=gk.srs.EPSG4326)

        if not isinstance(contextArea, ogr.Geometry) or contextArea.GetSpatialReference() is None:
            raise RuntimeError("contextArea must be an OGR Geometry object with an SRS or a set of lat/lon boundaries")  
        
        # get all GWA values in the area
        rm = gk.RegionMask.fromGeom(contextArea, pixelSize=0.01, srs=gk.srs.EPSG4326)
        gwaValues = rm.warp(gwa)
        gwaValues[gwaValues==-999] = np.nan # set the no data value so they can be filled properly

        # fill all no data with the closest values
        while np.isnan(gwaValues[rm.mask]).any():
            tmp = gwaValues.copy()
            for yi,xi in np.argwhere(np.isnan(gwaValues)):
                vals = np.array([np.nan, np.nan, np.nan, np.nan])
                if yi>0:    vals[0] = gwaValues[yi-1,xi]
                if xi>0:    vals[1] = gwaValues[yi,xi-1]
                if yi<yN-1: vals[2] = gwaValues[yi+1,xi]
                if xi<xN-1: vals[3] = gwaValues[yi,xi+1]

                if (~np.isnan(vals)).any():
                    tmp[yi,xi] = np.nanmean(vals)
            gwaValues = tmp

        # get mean
        gwaMean = gwaValues[rm.mask].mean()

    if not gwaMean is None : # the user has chosen the GWA-relative method
        if isinstance(gwaMean, str): # A path to a raster dataset has been given
            gwaMean = gk.raster.extractValues(gwaMean, targetLoc).data
        return ws * (gwaLocValue / gwaMean)

    else: # the user has chosen the GWA-normalized method
        if isinstance(longRunAverage, str): # A path to a raster dataset has been given
            longRunAverage = gk.raster.extractValues(longRunAverage, targetLoc).data
        return ws * (gwaLocValue / longRunAverage)


def projectByLogLaw(windspeed, measuredHeight, targetHeight, roughness):
    return windspeed * np.log(targetHeight/roughness) / np.log(measuredHeight/roughness)

def projectByPowerLaw(windspeed, measuredHeight, targetHeight, alpha):
    return windspeed * np.power(targetHeight/measuredHeight, alpha)
    
def projectByGWA(windspeed, targetHeight, measuredHeight, gwaDir, loc, method="log", value=0):
    # determine method and project
    if method == 'log':
        roughness = computeRoughnessFromGWA(gwaDir, loc)[value]
        wsTarget = projectByLogLaw(windspeed, measuredHeight, targetHeight, roughness)
    elif method == 'power':
        alpha = computeAlphaFromGWA(gwaDir, loc)[value]
        wsTarget = projectByPowerLaw(windspeed, measuredHeight, targetHeight, alpha)

    # done!
    return wsTarget

def projectByCLC(windspeed, targetHeight, measuredHeight, clcPath, loc):
    roughness = computeRoughnessFromCLC(loc, clcPath)
    wsTarget = projectByLogLaw(windspeed, measuredHeight, targetHeight, roughness)
    
    # done!
    return wsTarget


def computeRoughnessFromGWA(gwaDir, loc):
    ## Ensure location is okay
    loc = fixLoc(loc)

    # Get the GWA averages
    GWA_files = [join(gwaDir, "WS_050m_global_wgs84_mean_trimmed.tif"),
                 join(gwaDir, "WS_100m_global_wgs84_mean_trimmed.tif"),
                 join(gwaDir, "WS_200m_global_wgs84_mean_trimmed.tif")]

    for f in GWA_files: 
        if not isfile(f): 
            raise ResWeatherError("Could not find file: "+f)

    try:
        gwaAverage50 =  gk.raster.extractValues(GWA_files[0], loc, noDataOkay=False).data
        gwaAverage100 = gk.raster.extractValues(GWA_files[1], loc, noDataOkay=False).data
        gwaAverage200 = gk.raster.extractValues(GWA_files[2], loc, noDataOkay=False).data

    except gk.util.GeoKitRasterError as e:
        if str(e) == "No data values found in extractValues with 'noDataOkay' set to False":
            raise ResWeatherError("The given point does not appear to have valid data in the Global Wind Atlas dataset")
        else:
            raise e

    # Interpolate gwa average to desired height
    r_50_100 = np.exp( (gwaAverage100 * np.log(50) - gwaAverage50 * np.log(100) )/(gwaAverage100 - gwaAverage50) )
    r_100_200 = np.exp( (gwaAverage200 * np.log(100) - gwaAverage100 * np.log(200) )/(gwaAverage200 - gwaAverage100) )
    r_50_200 = np.exp( (gwaAverage200 * np.log(50) - gwaAverage50 * np.log(200) )/(gwaAverage200 - gwaAverage50) )

    # done!
    return (r_50_100, r_100_200, r_50_200)

def computeAlphaFromGWA(gwaDir, loc):
    ## Ensure location is okay
    loc = fixLoc(loc)

    # Get the GWA averages
    GWA_files = [join(gwaDir, "WS_050m_global_wgs84_mean_trimmed.tif"),
                 join(gwaDir, "WS_100m_global_wgs84_mean_trimmed.tif"),
                 join(gwaDir, "WS_200m_global_wgs84_mean_trimmed.tif")]

    for f in GWA_files: 
        if not isfile(f): 
            raise ResWeatherError("Could not find file: "+f)

    try:
        gwaAverage50 =  gk.raster.extractValues(GWA_files[0], loc, noDataOkay=False).data
        gwaAverage100 = gk.raster.extractValues(GWA_files[1], loc, noDataOkay=False).data
        gwaAverage200 = gk.raster.extractValues(GWA_files[2], loc, noDataOkay=False).data

    except gk.util.GeoKitRasterError as e:
        if str(e) == "No data values found in extractValues with 'noDataOkay' set to False":
            raise ResWeatherError("The given point does not appear to have valid data in the Global Wind Atlas dataset")
        else:
            raise e

    # Interpolate gwa average to desired height
    a_50_100 = np.log(gwaAverage50/gwaAverage100)/np.log(50.0/100)
    a_100_200 = np.log(gwaAverage100/gwaAverage200)/np.log(100.0/200)
    a_50_200 = np.log(gwaAverage50/gwaAverage200)/np.log(50.0/200)
       
    # done!
    return a_50_100, a_100_200, a_50_200


############################################################################
## See CLC codes at: http://uls.eionet.europa.eu/CLC2000/classes/
## Roughnesses defined primarily from :
## Title -- ROUGHNESS LENGTH CLASSIFICATION OF CORINE LAND COVER CLASSES
## Authors -- Julieta Silva, Carla Ribeiro, Ricardo Guedes
clcCodeToRoughess = OrderedDict()
clcCodeToRoughess[111] = 1.2 # Continuous urban fabric
clcCodeToRoughess[311] = 0.75 # Broad-leaved forest
clcCodeToRoughess[312] = 0.75 # Coniferous-leaved forest
clcCodeToRoughess[313] = 0.75 # Mixed-leaved forest
clcCodeToRoughess[141] = 0.6 # Green urban areas
clcCodeToRoughess[324] = 0.6 # Transitional woodland/shrub
clcCodeToRoughess[334] = 0.6 # Burnt areas
clcCodeToRoughess[112] = 0.5 # Discontinous urban fabric
clcCodeToRoughess[133] = 0.5 # Construction sites
clcCodeToRoughess[121] = 0.5 # Industrial or commercial units
clcCodeToRoughess[142] = 0.5 # Sport and leisure facilities
clcCodeToRoughess[123] = 0.5 # Port areas
clcCodeToRoughess[242] = 0.3 # Agro-forestry areas
clcCodeToRoughess[243] = 0.3 # Complex cultivation patterns
clcCodeToRoughess[244] = 0.3 # Land principally occupied by agriculture, with significant areas of natural vegetation
clcCodeToRoughess[241] = 0.1 # Annual crops associated with permanent crops
clcCodeToRoughess[221] = 0.1 # Fruit trees and berry plantations
clcCodeToRoughess[222] = 0.1 # Vineyard
clcCodeToRoughess[223] = 0.1 # Olive groves
clcCodeToRoughess[122] = 0.075 # Road and rail networks and associated land
clcCodeToRoughess[211] = 0.05 # Non-irrigated arable land
clcCodeToRoughess[212] = 0.05 # Permanently irrigated land
clcCodeToRoughess[213] = 0.05 # Rice fields
clcCodeToRoughess[411] = 0.05 # Inland marshes
clcCodeToRoughess[421] = 0.05 # Salt marshes
clcCodeToRoughess[321] = 0.03 # Sclerophylous vegetation
clcCodeToRoughess[322] = 0.03 # Moors and heathland
clcCodeToRoughess[323] = 0.03 # Natural grassland
clcCodeToRoughess[231] = 0.03 # Pastures
clcCodeToRoughess[131] = 0.005 # Dump sites
clcCodeToRoughess[132] = 0.005 # Mineral extraction sites
clcCodeToRoughess[124] = 0.005 # Airports
clcCodeToRoughess[332] = 0.005 # Bare rock
clcCodeToRoughess[333] = 0.005 # Sparsely vegetated areas
clcCodeToRoughess[335] = 0.001 # Glaciers and perpetual snow
clcCodeToRoughess[422] = 0.0005 # Peatbogs
clcCodeToRoughess[412] = 0.0005 # Salines
clcCodeToRoughess[423] = 0.0005 # Intertidal flats
clcCodeToRoughess[331] = 0.0003 # Beaches, dunes, and sand plains
clcCodeToRoughess[511] = 0.001 # Water courses # SUSPICIOUS
clcCodeToRoughess[512] = 0.0005 # Water bodies # SUSPISCIOUS
clcCodeToRoughess[523] = 0.0005 # Costal lagoons # SUSPISCIOUS
clcCodeToRoughess[522] = 0.0008 # Estuaries # SUSPISCIOUS
clcCodeToRoughess[521] = 0.0002 # Sea and ocean # SUSPISCIOUS

_clcGridToCode_v2006 = OrderedDict()
_clcGridToCode_v2006[1] = 111
_clcGridToCode_v2006[2] = 112
_clcGridToCode_v2006[3] = 121
_clcGridToCode_v2006[4] = 122
_clcGridToCode_v2006[5] = 123
_clcGridToCode_v2006[6] = 124
_clcGridToCode_v2006[7] = 131
_clcGridToCode_v2006[8] = 132
_clcGridToCode_v2006[9] = 133
_clcGridToCode_v2006[10] = 141
_clcGridToCode_v2006[11] = 142
_clcGridToCode_v2006[12] = 211
_clcGridToCode_v2006[13] = 212
_clcGridToCode_v2006[14] = 213
_clcGridToCode_v2006[15] = 221
_clcGridToCode_v2006[16] = 222
_clcGridToCode_v2006[17] = 223
_clcGridToCode_v2006[18] = 231
_clcGridToCode_v2006[19] = 241
_clcGridToCode_v2006[20] = 242
_clcGridToCode_v2006[21] = 243
_clcGridToCode_v2006[22] = 244
_clcGridToCode_v2006[23] = 311
_clcGridToCode_v2006[24] = 312
_clcGridToCode_v2006[25] = 313
_clcGridToCode_v2006[26] = 321
_clcGridToCode_v2006[27] = 322
_clcGridToCode_v2006[28] = 323
_clcGridToCode_v2006[29] = 324
_clcGridToCode_v2006[30] = 331
_clcGridToCode_v2006[31] = 332
_clcGridToCode_v2006[32] = 333
_clcGridToCode_v2006[33] = 334
_clcGridToCode_v2006[34] = 335
_clcGridToCode_v2006[35] = 411
_clcGridToCode_v2006[36] = 412
_clcGridToCode_v2006[37] = 421
_clcGridToCode_v2006[38] = 422
_clcGridToCode_v2006[39] = 423
_clcGridToCode_v2006[40] = 511
_clcGridToCode_v2006[41] = 512
_clcGridToCode_v2006[42] = 521
_clcGridToCode_v2006[43] = 522
_clcGridToCode_v2006[44] = 523

def computeRoughnessFromCLC(s, loc, clcPath):
    ## Ensure location is okay
    loc = fixLoc(loc)

    ## Get pixels values from clc
    clcGridValue = gk.raster.extractValues(clcPath, loc, noDataOkay=False).data

    ## Get the associated
    return clcCodeToRoughess[_clcGridToCode_v2006[clcGridValue]]
