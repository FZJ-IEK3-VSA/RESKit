from .util import *

def spatialAdjustmentByGWA(ws, targetLoc, gwa, gwaMean=None, longRunAverage=None, contextArea=None):
    # check inputs
    if (gwaMean is None and longRunAverage is None and contextArea is None) or 
       (not gwaMean is None and not longRunAverage is None and not contextArea is None) :
        raise RuntimeError("Exactly one of gwaMean, longRunAverage or contextArea must be defined")
    
    # Get the local gwa value
    gwaLocValue = gk.raster.extractValues(gwa, loc=targetLoc).data

    # Apply methods
    if not contextArea is None: # The user has chosen the GWA relative method, but the GWA average in the context area
                                #  needs to be determined
        if isinstance(contextArea, tuple): # context area is a set lat/lon boundaries, make them into a geometry 
            contextArea = gk.geom.makeBox(contextArea, gk.srs.EPSG4326)

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

    if not gwaMean is None or : # the user has chosen the GWA-relative method
        elif isinstance(gwaMean, str): # A path to a raster dataset has been given
            gwaMean = gk.raster.extractValues(gwaMean, loc=targetLoc).data

        return ws * (gwaLocValue / gwaMean)

    else: # the user has chosen the GWA-normalized method
        return ws * (gwaLocValue / longRunAverage)


def projectByGWA(ws, targetHeight, measuredHeight, gwaDir, loc, method="log", value=0):
    # determine method and project
    if method == 'log':
        roughness = computeRoughnessFromGWA(gwaDir, loc)[value]
        wsTarget = projectByLogLaw(ws, measuredHeight, targetHeight, roughness)
    elif method == 'power':
        roughness = computeAlphaFromGWA(gwaDir, loc)[value]
        wsTarget = projectByPowerLaw(ws, measuredHeight, targetHeight, alpha)

    # done!
    return wsTarget


def computeRoughnessFromGWA(gwaDir, loc):
    ## Ensure location is okay
    if not pt is None:
        if not pt.GetSpatialReference().IsSame(gk.srs.EPSG4326):
            pt = pt.TransformTo(gk.srs.EPSG4326) # make sure the point is in lat-lon
    else:
        pt = gk.geom.makePoint(lon, lat, srs=gk.srs.EPSG4326)

    # Get the GWA averages
    GWA_files = [join(gwaDir, "WS_050m_global_wgs84_mean_trimmed.tif"),
                 join(gwaDir, "WS_100m_global_wgs84_mean_trimmed.tif"),
                 join(gwaDir, "WS_200m_global_wgs84_mean_trimmed.tif")]

    for f in GWA_files: 
        if not isfile(f): 
            raise ResWeatherError("Could not find file: "+f)

    try:
        gwaAverage50 =  gk.raster.extractValues(GWA_files[0], pt, noDataOkay=False).data
        gwaAverage100 = gk.raster.extractValues(GWA_files[1], pt, noDataOkay=False).data
        gwaAverage200 = gk.raster.extractValues(GWA_files[2], pt, noDataOkay=False).data

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
    if not pt is None:
        if not pt.GetSpatialReference().IsSame(gk.srs.EPSG4326):
            pt = pt.TransformTo(gk.srs.EPSG4326) # make sure the point is in lat-lon
    else:
        pt = gk.geom.makePoint(lon, lat, srs=gk.srs.EPSG4326)

    # Get the GWA averages
    GWA_files = [join(gwaDir, "WS_050m_global_wgs84_mean_trimmed.tif"),
                 join(gwaDir, "WS_100m_global_wgs84_mean_trimmed.tif"),
                 join(gwaDir, "WS_200m_global_wgs84_mean_trimmed.tif")]

    for f in GWA_files: 
        if not isfile(f): 
            raise ResWeatherError("Could not find file: "+f)

    try:
        gwaAverage50 =  gk.raster.extractValues(GWA_files[0], pt, noDataOkay=False).data
        gwaAverage100 = gk.raster.extractValues(GWA_files[1], pt, noDataOkay=False).data
        gwaAverage200 = gk.raster.extractValues(GWA_files[2], pt, noDataOkay=False).data

    except gk.util.GeoKitRasterError as e:
        if str(e) == "No data values found in extractValues with 'noDataOkay' set to False":
            raise ResWeatherError("The given point does not appear to have valid data in the Global Wind Atlas dataset")
        else:
            raise e

    # Interpolate gwa average to desired height
    a_50_100 = np.log(gwaAverage50/gwaAverage100)/np.log(50.0/100)
    a_100_200 = np.log(gwaAverage100/gwaAverage200)/np.log(100.0/200)
    a_50_100 = np.log(gwaAverage50/gwaAverage200)/np.log(50.0/200)
       
    # done!
    return alpha

def computeRoughnessFromLandCover(s, source, sourceDefinitions="clc"):
    pass

def projectByLogLaw(windspeed, measuredHeight, targetHeight, roughness):
    return windspeed * np.log(targetHeight/roughness) / np.log(measuredHeight/roughness)

def projectByPowerLaw(windspeed, measuredHeight, targetHeight, alpha):
    return windspeed * np.power(targetHeight/measuredHeight, alpha)

"""
def oldMethod():
    ## Ensure location is okay
    if not pt is None:
        if not pt.GetSpatialReference().IsSame(gk.srs.EPSG4326):
            pt = pt.TransformTo(gk.srs.EPSG4326) # make sure the point is in lat-lon
    else:
        pt = gk.geom.makePoint(lon, lat, srs=gk.srs.EPSG4326)

    # Ensure we have the right data in our MERRA source
    varname = source._WINDSPEED_NORMALIZER_VAR
    if not varname in source.data.keys():
        # Load the wind speeds at 50 meters
        raise ResWeatherError("'%s' has not been pre-loaded"%varname)
    
    # Extract the data we need
    windSpeed = source[varname][pt]

    # Get the total average
    lraSource = source._WINDSPEED_NORMALIZER_SOURCE
    longRunAverage = gk.raster.extractValues(lraSource, pt, noDataOkay=True).data
    if np.isnan(longRunAverage):
        print("WARNING: could not find the long-run average windspeed value at the given location, defaulting to the average of the current time series")
        longRunAverage = windSpeed.mean()

    # Do normalization
    windSpeedNormalized = windSpeed / longRunAverage

    ###################
    ## Scale the normalized time series by the GWA average (at location)
    # Get the GWA averages
    GWA_files = [join(gwaDir, "WS_050m_global_wgs84_mean_trimmed.tif"),
                 join(gwaDir, "WS_100m_global_wgs84_mean_trimmed.tif"),
                 join(gwaDir, "WS_200m_global_wgs84_mean_trimmed.tif")]

    for f in GWA_files: 
        if not isfile(f): 
            raise ResWeatherError("Could not find file: "+f)

    try:
        gwaAverage50 =  gk.raster.extractValues(GWA_files[0], pt, noDataOkay=False).data
        gwaAverage100 = gk.raster.extractValues(GWA_files[1], pt, noDataOkay=False).data
        gwaAverage200 = gk.raster.extractValues(GWA_files[2], pt, noDataOkay=False).data

    except gk.util.GeoKitRasterError as e:
        if str(e) == "No data values found in extractValues with 'noDataOkay' set to False":
            raise ResWeatherError("The given point does not appear to have valid data in the Global Wind Atlas dataset")
        else:
            raise e

    # Interpolate gwa average to desired height
    if targetHeight == 50:
        gwaAverage = gwaAverage50
    elif targetHeight < 100:
        alpha = np.log(gwaAverage50/gwaAverage100)/np.log(50.0/100)
        gwaAverage = gwaAverage50 * np.power(targetHeight/50, alpha)
    elif targetHeight == 100:
        gwaAverage = gwaAverage100
    elif targetHeight > 100 and targetHeight <200:
        alpha = np.log(gwaAverage100/gwaAverage200)/np.log(100/200)
        gwaAverage = gwaAverage100 * np.power(targetHeight/100, alpha)
    elif targetHeight == 200:
        gwaAverage = gwaAverage200
    else:
        raise ResWeatherError("Wind speed cannot be extrapolated above 200m")
    
    # Rescale normalized wind speeds
    windSpeedScaled = gwaAverage * windSpeedNormalized

    # ALL DONE!
    return windSpeedScaled
"""