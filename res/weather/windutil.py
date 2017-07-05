from .util import *

def interpolateAndProjectByGWA(source, gwaDir, targetHeight, lat=None, lon=None, pt=None, **kwargs):
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

def computeAlphaFromGWA(s, gwaDir, lat=None, lon=None, pt=None):
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
    if height < 100:
        alpha = np.log(gwaAverage50/gwaAverage100)/np.log(50.0/100)
    else:
        alpha = np.log(gwaAverage100/gwaAverage200)/np.log(100/200)
       
    # done!
    return alpha

def computeRoughnessFromLandCover(s, source, sourceDefinitions="clc"):
    pass

def projectByLogLaw(windspeed, measuredHeight, targetHeight, roughness):
    return windspeed * np.log(targetHeight/roughness) / np.log(measuredHeight/roughness)

def projectByPowerLaw(windspeed, measuredHeight, targetHeight, alpha):
    return windspeed * np.power(targetHeight/measuredHeight, alpha)