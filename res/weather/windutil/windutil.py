from ..NCSource import *

################################################################################
## Pressure adjustment to wind speed
def airDensity( temperature=20, pressure=101325, relativeHumidity=0, dewPointTemperature=None ):
    """Follows the apprach of "Revised formula for the density of moist air (CIPM-2007)" by A Picard, R S Davis, M Glaser and K Fujii"""

    if relativeHumidity is None and dewPointTemperature is None:
        relativeHumidity = 0

    t = temperature
    T = 273.15+t
    p = pressure
    
    A = 1.2378847e-5
    B = -1.9121316e-2
    C = 33.93711047
    D = -6.3431645e3
    

    a_ = 1.00062
    b_ = 3.14e-8
    y_ = 5.6e-7

    if not dewPointTemperature is None:
        Td = dewPointTemperature + 273.15
        psv = np.exp(A*np.power(Td, 2) + B*Td + C + D/Td)
        f = a_ + b_*p + y_*np.power(dewPointTemperature,2)
        xv = f * psv / p
    else:
        psv = np.exp(A*np.power(T, 2) + B*T + C + D/T)
        f = a_ + b_*p + y_*np.power(t,2)
        xv = relativeHumidity * f * psv / p

    a0 = 1.58123e-6
    a1 = -2.9331e-8
    a2 = 1.1043e-10
    b0 = 5.707e-6 
    b1 = -2.051e-8
    c0 = 1.9898e-4 
    c1 = -2.376e-6
    d = 1.83e-11 
    e = -0.765e-8 

    Z = 1 - (p/T) * (a0-a1*t+a2*np.power(t,2)  + (b0+b1*t)*xv + (c0+c1*t)*np.power(xv,2)) + np.power(p/T ,2)*(d+e*np.power(xv,2))


    Ma = 28.96546e-3
    Mv = 18.01528e-3
    R = 8.314472

    airden = p * Ma / (Z*R*T) * ( 1-xv*(1-(Mv/Ma)))

    return airden


def densityAdjustment(windspeed, pressure, temperature, height=0):
    """
    * Density calculation from ideal gas
    * Projection using barometric equation
    * Density correction from assuming equal energy in the wind 
     - Suggested by IEC IEC61400-12

    Parameters:
    ===========
    windspeed : The wind speeds to adjust

    pressure : The pressure at the surface, in Pa

    temperature : Air temperature at the surface, in C

    height : The height to project the air density to, in meters
    """
    g0 = 9.80665 # Gravitational acceleration [m/s2]
    Ma =  0.0289644 # Molar mass of dry air [kg/mol]
    R = 8.3144598 # Universal gas constant [N·m/(mol·K)]
    rhoSTD =  1.225 # Standard air density [kg/m3]

    temperature = (temperature+273.15)
    
    # Get surface density
    # NOTE: I could use the equation from above, but its basically exactly equal 
    #       to ideal gas for humidity=0, and humidity does not have a significant 
    #       impact until high temperatures are considered
    
    rho = pressure*Ma / ( R * temperature)

    # Project rho to the desired height
    if not height is None:
        rho = rho * np.exp((-g0 * Ma * height) / (R * temperature))

    # Adjust wind speeds to standard-air-density-equivalent
    wsAdjusted = np.power(rho/rhoSTD, 1/3) * windspeed

    # Done!
    return wsAdjusted



################################################################################
## Spatial adjustment methods
def adjustLraToGwa( windspeed, targetLoc, gwa, longRunAverage, windspeedSourceName="windspeed"):
    """Adjust a timeseries of wind speed values to the average suggested by 
    Global Wind Atlas at a specific location by comparing against a given 
    long run average of the timeseries

    Uses this equation for each target location:
    .. math::
        ws_{adj} = ws_{raw} * GWA_{target} / LRA

      Where:
        * $ws_{adj}$ -> The output adjusted windspeed
        * $ws_{raw}$ -> The raw windspeed
        * $GWA_{target}$ -> The Global Wind Atlas average windspeed value value 
          at the target location
        * $LRA$ -> The long run average of the raw windspeed timeseries
    
    Example use case:
      When you have wind speeds from a weather dataset (like MERRA), and the raw 
      windspeeds for some index need to be adjusted to a specific location.

    Parameters:
    -----------
    windspeed : numpy.ndarray or NCSource
        The raw windspeeds to be adjusted
        * If an array is given with a single dimension, it is assumed to represent 
          timeseries values for a single location
        * If multidimensional array is given, the assumed dimensional context is 
          (time, locations), and 'targetLoc' must be an iterable with the same 
          length as the 'locations' dimension
        * If an NCSource is given, windspeeds are extracted from the source for 
          each target location, under the variable name specified by
          'windspeedSourceName'

    targetLoc : Anything acceptable by geokit.LocationSet
        The location(s) to adjust the wind speeds to
          * A single tuple with (lon, lat) is acceptable, or an iterable of such 
            tuples
          * A single point geometry (as long as it has an SRS), or an iterable
            of geometries is okay
          * geokit,Location, or geokit.LocationSet are best, though
    
    gwa : str
        The path to the Global Wind Atlas raster file
          * WARNING: Be sure you are using the appropriate height, since GWA 
            gives average windspeeds at 50, 100, and 200 meters
            (If you are adjusting wind speeds from a MerraSource, you want the 
            50 meter GWA version...)

    longRunAverage : numeric or numpy.ndarray or str
        The long run average of the raw windspeed time series
          * If only a single target location is desired, a single LRA value is
            expected
          * If multiple target locations are desired, an array of LRA values 
            for each target is expected
          * A path to a raster file containing LRA values can be given as a 
            string, from which the LRA value for each target location is extracted
    
    windspeedSourceName : str, optional
        The name of the variable to extract from the given NCSource (or derivative)
          * Only useful if the 'windspeed' input is an NCSource
    """ 


    ## Ensure location is okay
    targetLoc = LocationSet(targetLoc)
    multi = targetLoc.count>1

    # Get the local gwa value
    gwaLocValue = np.array(gk.raster.extractValues(gwa, targetLoc).data)
    if multi: gwaLocValue = gwaLocValue.reshape((1,gwaLocValue.size))
    else: gwaLocValue = gwaLocValue[0]
    
    # Get the long run average value
    if isinstance(longRunAverage, str): # A path to a raster dataset has been given
        longRunAverage = np.array(gk.raster.extractValues(longRunAverage, targetLoc).data)
        if multi: longRunAverage = longRunAverage.reshape((1,longRunAverage.size))
        else: longRunAverage = longRunAverage[0]
    else: # A simple number or array has been given
        if multi: # expect an array
            longRunAverage = np.array(longRunAverage) # turns longRunAverage into an array or a scalar
            longRunAverage = longRunAverage.reshape((1,longRunAverage.size))

    # apply adjustment
    if isinstance(windspeed, NCSource):
        windspeed = windspeed.get(windspeedSourceName, targetLoc)

    if multi and  isinstance(windspeed, pd.DataFrame): # reshape so that pandas will distribute properly
        gwaLocValue = gwaLocValue[0,:]
        longRunAverage = longRunAverage[0,:]

    return windspeed * (gwaLocValue / longRunAverage)

def adjustContextMeanToGwa( windspeed, targetLoc, gwa, contextMean=None, windspeedSourceName="windspeed", **kwargs):
    """Adjust a timeseries of wind speed values to the average suggested by 
    Global Wind Atlas at a specific location by comparing against the average
    of Global Wind Atlas in a surrounding contextual area

    Uses this equation for each target location:
    .. math::
        ws_{adj} = ws_{raw} * GWA_{target} / contextMean

      Where:
        * $ws_{adj}$ -> The output adjusted windspeed
        * $ws_{raw}$ -> The raw windspeed
        * $GWA_{target}$ -> The Global Wind Atlas average windspeed value value 
          at the target location
        * $contextMean$ -> The contextual average of GWA windspeed values
    
    Example use case:
        When you have wind speeds from a weather dataset (like MERRA), and the raw 
        windspeeds for some index need to be adjusted to a specific location.

    Parameters:
    -----------
    windspeed : numpy.ndarray or NCSource
        The raw windspeeds to be adjusted
        * If an array is given with a single dimension, it is assumed to represent 
          timeseries values for a single location
        * If multidimensional array is given, the assumed dimensional context is 
          (time, locations), and 'targetLoc' must be an iterable with the same 
          length as the 'locations' dimension
        * If an NCSource is given, windspeeds are extracted from the source for 
          each target location, under the variable name specified by
          'windspeedSourceName'

    targetLoc : Anything acceptable by geokit.LocationSet
        The location(s) to adjust the wind speeds to
          * A single tuple with (lon, lat) is acceptable, or an iterable of such 
            tuples
          * A single point geometry (as long as it has an SRS), or an iterable
            of geometries is okay
          * geokit,Location, or geokit.LocationSet are best, though
    
    gwa : str
        The path to the Global Wind Atlas raster file
          * WARNING: Be sure you are using the appropriate height, since GWA 
            gives average windspeeds at 50, 100, and 200 meters
            (If you are adjusting wind speeds from a MerraSource, you want the 
            50 meter GWA version...)

    contextMean : numeric or numpy.ndarray or str
        The  average of the GWA windspeeds in each location's contextual area
          * If only a single target location is desired, a single contextMean value
            is expected
          * If multiple target locations are desired, an array of contextMean 
            values for each target is expected
          * A path to a raster file containing contextMean values can be given as
            a string, from which the contextMean value for each target location 
            is extracted
    
    windspeedSourceName : str, optional
        The name of the variable to extract from the given NCSource (or derivative)
          * Only useful if the 'windspeed' input is an NCSource
    """ 
    ## Ensure location is okay
    targetLoc = LocationSet(targetLoc)
    multi = targetLoc.count>1

    # Get the local gwa value
    gwaLocValue = np.array(gk.raster.extractValues(gwa, targetLoc).data) # results in a (1 X number_of_locations) matrix
    if multi: gwaLocValue = gwaLocValue.reshape((1,gwaLocValue.size))
    else: gwaLocValue = gwaLocValue[0]

    # Get the gwa contextual mean value
    if contextMean is None: # the contexts needs to be computed
        # this only works when windspeed is an NCSource object
        if not isinstance(windspeed, NCSource):
            raise ResError("contextMean must be provided when windspeed is not a Source")
        print("Autocomputation of contextual mean is currently untested")
        contextMean = np.array([computeContextMean(gwa, windspeed.contextAreaAt(loc), **kwargs) for loc in targetLoc])
        if multi: contextMean = contextMean.reshape((1,contextMean.size))
        else: contextMean = contextMean[0]

    elif isinstance(contextMean, str): # A path to a raster dataset has been given to read the means from
        contextMean = np.array(gk.raster.extractValues(contextMean, targetLoc).data) # results in a (1 X number_of_locations) matrix
        if multi: contextMean = contextMean.reshape((1,contextMean.size))
        else: contextMean = contextMean[0]

    else: # A simple number or array has been given
        if multi: # expect an array
            contextMean = np.array(contextMean) # turns contextMean into an array or a scalar
            contextMean = contextMean.reshape((1,contextMean.size))

    # apply adjustment    
    if isinstance(windspeed, NCSource):
        windspeed = windspeed.get(windspeedSourceName, targetLoc)

    if multi and isinstance(windspeed, pd.DataFrame):
        gwaLocValue = gwaLocValue[0,:]
        contextMean = contextMean[0,:]

    return windspeed * (gwaLocValue / contextMean)

################################################################################
## Vertical projection methods
def projectByLogLaw( measuredWindspeed, measuredHeight, targetHeight, roughness, displacement=0, stability=0):
    """Estimates windspeed at target height ($h_t$) based off a measured windspeed 
    ($u_m$) at a known measurement height ($h_m$) subject to the surface roughness 
    ($z$), displacement height ($d$), and stability ($S$)

    * Begins with the semi-empirical log wind profile ($a$ stands for any height):
        $ u_a = \\frac{u_*}{\\kappa}[ln(\\frac{h_a - d}{z}) + S] $

    * Solves for $u_t$ based off known values:
        $ u_t = u_m * \\frac{ln((h_t - d)/z}) + S]}{ln((h_m - d)/z}) + S]} $
    
    * Simplifications:
        - stability -> 0 under "neutral stability conditions"

    Parameters:
    -----------
    measuredWindspeed : numpy.ndarray
        The raw windspeeds to be adjusted
        * If an array is given with a single dimension, it is assumed to represent 
          timeseries values for a single location
        * If multidimensional array is given, the assumed dimensional context is 
          (time, locations), and 'targetLoc' must be an iterable with the same 
          length as the 'locations' dimension
        
    measuredHeight : numeric or numpy.ndarray
        The measurement height of the raw windspeeds
        * If an array is given for measuredWindspeed with a single dimension, a 
          single value is expected for measuredHeight
        * If multidimensional array is given for measuredWindspeed, an array of
          values is expected for measuredHeight. One value for each wind speed
          timeseries

    targetHeight : numeric or numpy.ndarray
        The height to project each wind speed timeseries to
        * If a numeric value is given, all windspeed timeseries will be projected
          to this height
        * If an array is given for targetHeight, each value must match to one
          wind speed timeseries in measuredWindspeed

    roughness : numeric or numpy.ndarray
        The roughness value used to project each wind speed timeseries
        * If a numeric value is given, all windspeed timeseries will be projected
          using this roughness value
        * If an array is given, each value must match to one wind speed timeseries
          in measuredWindspeed

    displacement : numeric or numpy.ndarray, optional
        The displacement value used to project each wind speed timeseries
        * If a numeric value is given, all windspeed timeseries will be projected
          using this displacement value
        * If an array is given, each value must match to one wind speed timeseries
          in measuredWindspeed

    stability : numeric or numpy.ndarray, optional
        The stability value used to project each wind speed timeseries
        * If a numeric value is given, all windspeed timeseries will be projected
          using this stability value
        * If an array is given, each value must match to one wind speed timeseries
          in measuredWindspeed
        
    """
    return measuredWindspeed * (np.log( (targetHeight-displacement)/roughness)+stability) / (np.log((measuredHeight-displacement)/roughness)+stability)

def projectByPowerLaw( measuredWindspeed, measuredHeight, targetHeight, alpha=1/7):
    """Estimates windspeed at target height ($h_t$) based off a measured windspeed 
    ($u_m$) at a known measurement height ($h_m$) subject to the scaling factor ($a$)

    $ u_t = u_m * (\\frac{h_t}{h_m})^a $

    
    Parameters:
    -----------
    measuredWindspeed : numpy.ndarray
        The raw windspeeds to be adjusted
        * If an array is given with a single dimension, it is assumed to represent 
          timeseries values for a single location
        * If multidimensional array is given, the assumed dimensional context is 
          (time, locations), and 'targetLoc' must be an iterable with the same 
          length as the 'locations' dimension
        
    measuredHeight : numeric or numpy.ndarray
        The measurement height of the raw windspeeds
        * If an array is given for measuredWindspeed with a single dimension, a 
          single value is expected for measuredHeight
        * If multidimensional array is given for measuredWindspeed, an array of
          values is expected for measuredHeight. One value for each wind speed
          timeseries

    targetHeight : numeric or numpy.ndarray
        The height to project each wind speed timeseries to
        * If a numeric value is given, all windspeed timeseries will be projected
          to this height
        * If an array is given for targetHeight, each value must match to one
          wind speed timeseries in measuredWindspeed

    alpha : numeric or numpy.ndarray, optional
        The alpha value used to project each wind speed timeseries
        * If a numeric value is given, all windspeed timeseries will be projected
          using this alpha value
        * If an array is given, each value must match to one wind speed timeseries
          in measuredWindspeed
        * The default 1/7 value corresponds to neutral stability conditions

    """
    return measuredWindspeed * np.power(targetHeight/measuredHeight, alpha)

################################################################################
## Alpha computers

def alphaFromLevels( lowWindSpeed, lowHeight, highWindSpeed, highHeight):
    """Solves for the scaling factor ($a$) given two windspeeds with known heights

    $ a = log(\\frac{u_m}{u_t}) / log(\\frac{h_m}{h_t}) $

    Parameters:
    -----------
    lowWindspeed : numeric or numpy.ndarray
        The measured windspeed at the 'lower height'
    
    lowHeight : numeric or numpy.ndarray
        The measured height at the 'lower height'

    highWindspeed : numeric or numpy.ndarray
        The measured windspeed at the 'lower height'
    
    highHeight : numeric or numpy.ndarray
        The measured height at the 'lower height'
        
    """
    return np.log(lowWindSpeed/highWindSpeed)/np.log(lowHeight/highHeight)

def alphaFromGWA( gwaDir, loc, pairID=1, _structure="WS_%03dm_global_wgs84_mean_trimmed.tif"):
    """Estimates the scaling factor ($a$) at a given location by taking
    two height values from the Global Wind Atlas datasets. 

    * Height options are 50m, 100m, and 200m
    * Solves:
        $ a = log(\\frac{u_1}{u_2}) / log(\\frac{h_1}{h_2}) $

    Parameters:
    -----------
    gwaDir : str
        The path to the directory containing Global Wind Atlas files
        * Files must the name structure of "WS_[HEIGHT]_global_wgs84_mean_trimmed.tif"
    
    loc : numeric or numpy.ndarray
        The measured height at the 'lower height'

    highWindspeed : numeric or numpy.ndarray
        The measured windspeed at the 'lower height'
    
    highHeight : numeric or numpy.ndarray
        The measured height at the 'lower height'
        
    """
    ## Ensure location is okay
    loc = LocationSet(loc)

    # Get the GWA averages
    GWA_files = [join(gwaDir, _structure%(50)),
                 join(gwaDir, _structure%(100)),
                 join(gwaDir, _structure%(200))]

    for f in GWA_files: 
        if not isfile(f): 
            raise ResError("Could not find file: "+f)

    if pairID==0 or pairID==2: gwaAverage50  = gk.raster.interpolateValues(GWA_files[0], loc)
    if pairID==0 or pairID==1: gwaAverage100 = gk.raster.interpolateValues(GWA_files[1], loc)
    if pairID==1 or pairID==2: gwaAverage200 = gk.raster.interpolateValues(GWA_files[2], loc)

    # Compute alpha
    if pairID==0: out = alphaFromLevels(gwaAverage50,50,gwaAverage100,100)
    if pairID==1: out = alphaFromLevels(gwaAverage100,100,gwaAverage200,200)
    if pairID==2: out = alphaFromLevels(gwaAverage50,50,gwaAverage200,200)

    # done!
    if out.size==1: return out[0]
    else: return pd.Series(out,index=loc)

################################################################################
## Roughness computers
def roughnessFromLevels(lowWindSpeed, lowHeight, highWindSpeed, highHeight):
    """Computes a roughness factor from two windspeed values at two distinct heights

    Parameters:
    -----------
    lowWindspeed : numeric or np.ndarray
        The measured wind speed at the lower height

    lowHeight : numeric or np.ndarray
        The lower height

    highWindspeed : numeric or np.ndarray
        The measured wind speed at the higher height

    highHeight : numeric or np.ndarray
        The higher height
    """

    return np.exp( (highWindSpeed * np.log(lowHeight) - lowWindSpeed * np.log(highHeight) )/(highWindSpeed - lowWindSpeed) )

def roughnessFromGWA(gwaDir, loc, pairID=1, _structure="WS_%03dm_global_wgs84_mean_trimmed.tif"):
    """Computes a roughness factor from two windspeed values found at the same 
    location, but different heights, in the Global Wind Atlas datasets

    Parameters:
    -----------
    gwaDir : str
        The directory containing global wind atlas files
          * The expected file names are: "WS_%03dm_global_wgs84_mean_trimmed.tif"
          * This can be changed with the '_structure' input

    loc : Anything acceptable to geokit.LocationSet
        The locations for which roughness should be calculated

    pairID : int
        An id indicating which two Global Wind Atlas files should be used to in
          the computation:
            0 -> 50m and 100m
            1 -> 100m and 200m
            2 -> 50m and 200m

    _structure : str; optional
        The filename structure to expect
        * Must accept a single integer formatting input
    """
    ## Ensure location is okay
    loc = LocationSet(loc)

    # Get the GWA averages
    GWA_files = [join(gwaDir, _structure%(50)),
                 join(gwaDir, _structure%(100)),
                 join(gwaDir, _structure%(200))]

    for f in GWA_files: 
        if not isfile(f): 
            raise ResWeatherError("Could not find file: "+f)

    if pairID==0 or pairID==2: gwaAverage50  = gk.raster.interpolateValues(GWA_files[0], loc)
    if pairID==0 or pairID==1: gwaAverage100 = gk.raster.interpolateValues(GWA_files[1], loc)
    if pairID==1 or pairID==2: gwaAverage200 = gk.raster.interpolateValues(GWA_files[2], loc)

    # Interpolate gwa average to desired height
    if pairID==0: out = roughnessFromLevels(gwaAverage50,50,gwaAverage100,100)
    if pairID==1: out = roughnessFromLevels(gwaAverage100,100,gwaAverage200,200)
    if pairID==2: out = roughnessFromLevels(gwaAverage50,50,gwaAverage200,200)

    # done!
    if out.size==1: return out[0]
    else: return pd.Series(out,index=loc)

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
clcCodeToRoughess[521] = 0.0005 # Costal lagoons # SUSPISCIOUS
clcCodeToRoughess[522] = 0.0008 # Estuaries # SUSPISCIOUS
clcCodeToRoughess[523] = 0.0002 # Sea and ocean # SUSPISCIOUS

clcGridToCode_v2006 = OrderedDict()
clcGridToCode_v2006[1] = 111
clcGridToCode_v2006[2] = 112
clcGridToCode_v2006[3] = 121
clcGridToCode_v2006[4] = 122
clcGridToCode_v2006[5] = 123
clcGridToCode_v2006[6] = 124
clcGridToCode_v2006[7] = 131
clcGridToCode_v2006[8] = 132
clcGridToCode_v2006[9] = 133
clcGridToCode_v2006[10] = 141
clcGridToCode_v2006[11] = 142
clcGridToCode_v2006[12] = 211
clcGridToCode_v2006[13] = 212
clcGridToCode_v2006[14] = 213
clcGridToCode_v2006[15] = 221
clcGridToCode_v2006[16] = 222
clcGridToCode_v2006[17] = 223
clcGridToCode_v2006[18] = 231
clcGridToCode_v2006[19] = 241
clcGridToCode_v2006[20] = 242
clcGridToCode_v2006[21] = 243
clcGridToCode_v2006[22] = 244
clcGridToCode_v2006[23] = 311
clcGridToCode_v2006[24] = 312
clcGridToCode_v2006[25] = 313
clcGridToCode_v2006[26] = 321
clcGridToCode_v2006[27] = 322
clcGridToCode_v2006[28] = 323
clcGridToCode_v2006[29] = 324
clcGridToCode_v2006[30] = 331
clcGridToCode_v2006[31] = 332
clcGridToCode_v2006[32] = 333
clcGridToCode_v2006[33] = 334
clcGridToCode_v2006[34] = 335
clcGridToCode_v2006[35] = 411
clcGridToCode_v2006[36] = 412
clcGridToCode_v2006[37] = 421
clcGridToCode_v2006[38] = 422
clcGridToCode_v2006[39] = 423
clcGridToCode_v2006[40] = 511
clcGridToCode_v2006[41] = 512
clcGridToCode_v2006[42] = 521
clcGridToCode_v2006[43] = 522
clcGridToCode_v2006[44] = 523

def roughnessFromCLC(clcPath, loc, winRange=0):
    """Estimates a roughness factor by the prominent land cover at given locations
    given by the Corine Land Cover dataset.
    
    * Roughness suggestions from [1], and are given below

    Parameters:
    -----------
    clcPath : str
        The path to the Corine Land Cover file on disk

    loc : Anything acceptable to geokit.LocationSet
        The locations for which roughness should be estimated

    winRange : int; optional
        An extra number of pixels to extract around the indicated locations
          * A winRange of 0 means only the CLC pixel value for each location is
            returned
          * A winRange of 1 means an extra pixel is extracted around each location
            in all directions. Leading to a 3x3 matrix of roughness values
          * Use this if you need to do some operation on the roughnesses found
            around the indicated location

    Sources:
    --------
    1: Silva et al.

    Roughness Values:
    -----------------
        Continuous urban fabric : 1.2 
        Broad-leaved forest : 0.75 
        Coniferous-leaved forest : 0.75 
        Mixed-leaved forest : 0.75 
        Green urban areas : 0.6 
        Transitional woodland/shrub : 0.6 
        Burnt areas : 0.6 
        Discontinous urban fabric : 0.5 
        Construction sites : 0.5 
        Industrial or commercial units : 0.5 
        Sport and leisure facilities : 0.5 
        Port areas : 0.5 
        Agro-forestry areas : 0.3 
        Complex cultivation patterns : 0.3 
        Land principally occupied by agriculture, with significant areas of natural vegetation : 0.3 
        Annual crops associated with permanent crops : 0.1 
        Fruit trees and berry plantations : 0.1 
        Vineyard : 0.1 
        Olive groves : 0.1 
        Road and rail networks and associated land : 0.075 
        Non-irrigated arable land : 0.05 
        Permanently irrigated land : 0.05 
        Rice fields : 0.05 
        Inland marshes : 0.05 
        Salt marshes : 0.05 
        Sclerophylous vegetation : 0.03 
        Moors and heathland : 0.03 
        Natural grassland : 0.03 
        Pastures : 0.03 
        Dump sites : 0.005 
        Mineral extraction sites : 0.005 
        Airports : 0.005 
        Bare rock : 0.005 
        Sparsely vegetated areas : 0.005 
        Glaciers and perpetual snow : 0.001 
        Peatbogs : 0.0005 
        Salines : 0.0005 
        Intertidal flats : 0.0005 
        Beaches, dunes, and sand plains : 0.0003 
        Water courses # SUSPICIOUS : 0.001 
        Water bodies # SUSPISCIOUS : 0.0005 
        Costal lagoons # SUSPISCIOUS : 0.0005 
        Estuaries # SUSPISCIOUS : 0.0008 
        Sea and ocean # SUSPISCIOUS : 0.0002 

    """
    ## Ensure location is okay
    loc = LocationSet(loc)

    ## Get pixels values from clc
    clcGridValues = gk.raster.interpolateValues(clcPath, loc, winRange=winRange, noDataOkay=True)

    ## make output array
    if winRange>0:
        outputs = []
        for v in clcGridValues:
            # Treat nodata as ocean
            v[np.isnan(v)] = 44
            v[ v>44 ] = 44 
            v = v.astype(int)
        
            values, counts = np.unique( v, return_counts=True )
            
            total = 0
            for val,cnt in zip(values,counts):
                total += cnt * clcCodeToRoughess[clcGridToCode_v2006[ val ]]

            outputs.append(total/counts.sum())
    else:
        # Treat nodata as ocean
        clcGridValues[np.isnan(clcGridValues)] = 44
        clcGridValues[ clcGridValues>44 ] = 44 
        clcGridValues = clcGridValues.astype(int)

        ## Get the associated
        outputs = [clcCodeToRoughess[clcGridToCode_v2006[ val ]] for val in clcGridValues]

    ## Done!
    if len(outputs)==1: return outputs[0]
    else: return outputs


############################################################################
## Defined primarily from :
## Title -- ROUGHNESS LENGTH CLASSIFICATION OF Global Wind Atlas
## Authors -- DTU
globCoverCodeToRoughess = OrderedDict()
# GlobCover Number
globCoverCodeToRoughess[210] = 0.0002 # Water Bodies # changed by Me from 0.0 to 0.0002
globCoverCodeToRoughess[220] = 0.0004 # Permanant Snow and ice
globCoverCodeToRoughess[200] = 0.005 # Bare areas
globCoverCodeToRoughess[140] = 0.03 # Grasslands, savannas or lichens/mosses
globCoverCodeToRoughess[150] = 0.05 # Sparse vegetation
globCoverCodeToRoughess[11] = 0.1 # Croplands
globCoverCodeToRoughess[14] = 0.1 # Croplands
globCoverCodeToRoughess[130] = 0.1 # Shrubland
globCoverCodeToRoughess[180] = 0.2 # Wetlands
globCoverCodeToRoughess[20] = 0.3 # Mosaic natural vegetation/cropland
globCoverCodeToRoughess[30] = 0.3 # Mosaic natural vegetation/cropland
globCoverCodeToRoughess[160] = 0.5 # Flooded forest
globCoverCodeToRoughess[120] = 0.5 # Mosaic grassland/forest
globCoverCodeToRoughess[170] = 0.6 # Flooded forest or shrubland
globCoverCodeToRoughess[190] = 1.0 # Urban Areas
globCoverCodeToRoughess[40] = 1.5 # Forests
globCoverCodeToRoughess[50] = 1.5 # Forests
globCoverCodeToRoughess[60] = 1.5 # Forests
globCoverCodeToRoughess[70] = 1.5 # Forests
globCoverCodeToRoughess[90] = 1.5 # Forests
globCoverCodeToRoughess[100] = 1.5 # Forests
globCoverCodeToRoughess[110] = 1.5 # Forests

# Modis Number for "no data" points of GlobCover (mostly in areas North of 60°)
modisCodeToRoughess = OrderedDict()
modisCodeToRoughess[0] = 0.0 # Water Bodies
modisCodeToRoughess[15] = 0.0004 # Permanant Snow and ice
modisCodeToRoughess[16] = 0.005 # Bare areas
modisCodeToRoughess[10] = 0.03 # Grasslands, savannas or lichens/mosses
modisCodeToRoughess[12] = 0.1 # Croplands
modisCodeToRoughess[6] = 0.1 # Shrubland
modisCodeToRoughess[7] = 0.1 # Shrubland
modisCodeToRoughess[11] = 0.2 # Wetlands
modisCodeToRoughess[14] = 0.3 # Mosaic natural vegetation/cropland
modisCodeToRoughess[9] = 0.5 # Mosaic grassland/forest
modisCodeToRoughess[13] = 1.0 # Urban Areas
modisCodeToRoughess[1] = 1.5 # Forests
modisCodeToRoughess[2] = 1.5 # Forests
modisCodeToRoughess[3] = 1.5 # Forests
modisCodeToRoughess[4] = 1.5 # Forests
modisCodeToRoughess[5] = 1.5 # Forests
modisCodeToRoughess[8] = 1.5 # Forests

############################################################################
## CCI Landcover  classification by ESA and the Climate Change Initiative 
## ABOUT: https://www.esa-landcover-cci.org/?q=node/1
## Roughnesses defined due to the comparison with CLC and globCover
cciCodeToRoughess = OrderedDict()
# CCI LC Number
cciCodeToRoughess [210] = 0.0002 # Water bodies
cciCodeToRoughess [220] = 0.001 # Permanent snow and ice
cciCodeToRoughess [200] = 0.005 # Bare areas
cciCodeToRoughess [201] = 0.005 # Consolidated bare areas
cciCodeToRoughess [202] = 0.005 # Unconsolidated bare areas
cciCodeToRoughess [150] = 0.005 # Sparse vegetation (tree, shrub, herbaceous cover) (<15%)
cciCodeToRoughess [152] = 0.005 # Sparse shrub (<15%)
cciCodeToRoughess [153] = 0.005 # Sparse herbaceous cover (<15%)
cciCodeToRoughess [10] = 0.03 # Cropland, rainfed
cciCodeToRoughess [11] = 0.03 # Herbaceous cover
cciCodeToRoughess [120] = 0.03 # Shrubland
cciCodeToRoughess [121] = 0.03 # Shrubland evergreen #barely exists, only near water bodies, ocean
cciCodeToRoughess [122] = 0.03 # Shrubland deciduous #barely exists, only near water bodies, ocean
cciCodeToRoughess [12] = 0.3 # Tree or shrub cover
cciCodeToRoughess [110] = 0.03 # Mosaic herbaceous cover (>50%) / tree and shrub (<50%)
cciCodeToRoughess [40] = 0.03 # Mosaic natural vegetation (tree, shrub, herbaceous cover) (>50%) / cropland (<50%)
cciCodeToRoughess [180] = 0.03 # Shrub or herbaceous cover, flooded, fresh/saline/brakish water
cciCodeToRoughess [130] = 0.03 # Grassland
cciCodeToRoughess [140] = 0.03 # Lichens and mosses
cciCodeToRoughess [170] = 0.1 # Tree cover, flooded, saline water (areas around river deltas and ocean) 
cciCodeToRoughess [20] = 0.1 # Cropland, irrigated or post-flooding
cciCodeToRoughess [30] = 0.1 # Mosaic cropland (>50%) / natural vegetation (tree, shrub, herbaceous cover) (<50%)
cciCodeToRoughess [160] = 0.5 # Tree cover, flooded, fresh or brakish water, barely exists
cciCodeToRoughess [100] = 0.75 # Mosaic tree and shrub (>50%) / herbaceous cover (<50%)
cciCodeToRoughess [50] = 0.75 # Tree cover, broadleaved, evergreen, closed to open (>15%)
cciCodeToRoughess [60] = 0.75 # Tree cover, broadleaved, deciduous, closed to open (>15%)
cciCodeToRoughess [61] = 0.75 # Tree cover, broadleaved, deciduous, closed (>40%)
cciCodeToRoughess [62] = 0.75 # Tree cover, broadleaved, deciduous, open (15-40%)
cciCodeToRoughess [70] = 0.75 # Tree cover, needleleaved, evergreen, closed to open (>15%)
cciCodeToRoughess [71] = 0.75 # Tree cover, needleleaved, evergreen, closed (>40%)
cciCodeToRoughess [72] = 0.75 # Tree cover, needleleaved, evergreen, open (15-40%)
cciCodeToRoughess [80] = 0.75 # Tree cover, needleleaved, deciduous, closed to open (>15%)
cciCodeToRoughess [81] = 0.75 # Tree cover, needleleaved, deciduous, closed (>40%)
cciCodeToRoughess [82] = 0.75 # Tree cover, needleleaved, deciduous, open (15-40%)
cciCodeToRoughess [90] = 0.75 # Tree cover, mixed leaf type (broadleaved and needleleaved)
cciCodeToRoughess [190] = 1.2 # Urban areas

def roughnessFromLandCover(num, lctype='clc'):
    """
    landCover can be 'clc', 'clc-code', globCover', 'modis', or 'cci'
    """
    if lctype=='clc': 
        # fix no data values
        num[num<0] = 44
        num[num>44] = 44
        num[np.isnan(num)] = 44

        # set source
        source = lambda x: clcCodeToRoughess[clcGridToCode_v2006[x]]
    elif lctype=='clc-code': source = lambda x: clcCodeToRoughess[x]
    elif lctype=='globCover': source = lambda x: globCoverCodeToRoughess[x]
    elif lctype=='modis': source = lambda x: modisCodeToRoughess[x]
    elif lctype=='cci' : source = lambda x: cciCodeToRoughess[x]
    else: 
        raise ResError("invalid input")

    converter = np.vectorize(source)
    return converter(num)
