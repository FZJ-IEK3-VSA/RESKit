from ..NCSource import *

################################################################################
## Spatial adjustment methods

def adjustLraToGwa( windspeed, targetLoc, gwa, longRunAverage, windspeedSourceName="windspeed"):
    ## Ensure location is okay
    targetLoc = ensureList(ensureGeom(targetLoc))
    multi = len(targetLoc)>1

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
        if multi: # reshape so that pandas will distribute properly
            gwaLocValue = gwaLocValue[0,:]
            longRunAverage = longRunAverage[0,:]

    return windspeed * (gwaLocValue / longRunAverage)

def adjustContextMeanToGwa( windspeed, targetLoc, gwa, contextMean=None, windspeedSourceName="windspeed", **kwargs):
    ## Ensure location is okay
    targetLoc = ensureList(ensureGeom(targetLoc))
    multi = len(targetLoc)>1

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
        if multi: # reshape so that pandas will distribute properly
            gwaLocValue = gwaLocValue[0,:]
            contextMean = contextMean[0,:]

    return windspeed * (gwaLocValue / contextMean)

################################################################################
## Vertical projection methods

def projectByLogLaw( measuredWindspeed, measuredHeight, targetHeight, roughness, displacement=0, stability=0):
    """Estimates windspeed at target height ($h_t$) based off a measured windspeed ($u_m$) at a known measurement height ($h_m$) subject to the surface roughness ($z$), displacement height ($d$), and stability ($S$)

    * Begins with the semi-empirical log wind profile ($a$ stands for any height):
        $ u_a = \frac{u_*}{\kappa}[ln(\frac{h_a - d}{z}) + S]$

    * Solves for $u_t$ based off known values:
        $ u_t = u_m * \frac{ln((h_t - d)/z}) + S]}{ln((h_m - d)/z}) + S]}
    
    * Simplifications:
        - stability -> 0 under "neutral stability conditions"
    """
    return measuredWindspeed * (np.log( (targetHeight-displacement)/roughness)+stability) / (np.log((measuredHeight-displacement)/roughness)+stability)

def projectByPowerLaw( measuredWindspeed, measuredHeight, targetHeight, alpha):
    return measuredWindspeed * np.power(targetHeight/measuredHeight, alpha)

################################################################################
## Alpha computers

def alphaFromLevels( lowWindSpeed, lowHeight, highWindSpeed, highHeight):
    return np.log(lowWindSpeed/highWindSpeed)/np.log(lowHeight/highHeight)

def alphaFromGWA( gwaDir, loc, pairID=1, _structure="WS_%03dm_global_wgs84_mean_trimmed.tif"):
    ## Ensure location is okay
    loc = ensureList(loc)
    locGeoms = ensureGeom(loc)

    # Get the GWA averages
    GWA_files = [join(gwaDir, _structure%(50)),
                 join(gwaDir, _structure%(100)),
                 join(gwaDir, _structure%(200))]

    for f in GWA_files: 
        if not isfile(f): 
            raise ResError("Could not find file: "+f)

    if pairID==0 or pairID==2: gwaAverage50  = gk.raster.extractValues(GWA_files[0], locGeoms).data.values
    if pairID==0 or pairID==1: gwaAverage100 = gk.raster.extractValues(GWA_files[1], locGeoms).data.values
    if pairID==1 or pairID==2: gwaAverage200 = gk.raster.extractValues(GWA_files[2], locGeoms).data.values

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
    return np.exp( (highWindSpeed * np.log(lowHeight) - lowWindSpeed * np.log(highHeight) )/(highWindSpeed - lowWindSpeed) )

def roughnessFromGWA(gwaDir, loc, pairID=1, _structure="WS_%03dm_global_wgs84_mean_trimmed.tif"):
    ## Ensure location is okay
    loc = ensureList(loc)
    locGeoms = ensureGeom(loc)

    # Get the GWA averages
    GWA_files = [join(gwaDir, _structure%(50)),
                 join(gwaDir, _structure%(100)),
                 join(gwaDir, _structure%(200))]

    for f in GWA_files: 
        if not isfile(f): 
            raise ResWeatherError("Could not find file: "+f)

    if pairID==0 or pairID==2: gwaAverage50  = gk.raster.extractValues(GWA_files[0], locGeoms).data.values
    if pairID==0 or pairID==1: gwaAverage100 = gk.raster.extractValues(GWA_files[1], locGeoms).data.values
    if pairID==1 or pairID==2: gwaAverage200 = gk.raster.extractValues(GWA_files[2], locGeoms).data.values

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
clcCodeToRoughess[523] = 0.0005 # Costal lagoons # SUSPISCIOUS
clcCodeToRoughess[522] = 0.0008 # Estuaries # SUSPISCIOUS
clcCodeToRoughess[521] = 0.0002 # Sea and ocean # SUSPISCIOUS

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

def roughnessFromCLC(clcPath, loc):
    ## Ensure location is okay
    loc = ensureList(loc)
    locGeoms = ensureGeom(loc)

    ## Get pixels values from clc (assume nodata is ocean)
    clcGridValues = gk.raster.extractValues(clcPath, locGeoms, noDataOkay=True).data.values
    clcGridValues[np.isnan(clcGridValues)] = 42
    clcGridValues = clcGridValues.astype(int)

    ## Get the associated
    outputs = [clcCodeToRoughess[clcGridToCode_v2006[ int(val) ]] for val in clcGridValues]

    ## Done!
    if len(outputs)==1: return outputs[0]
    else: return pd.Series(outputs,index=loc)


############################################################################
## Defined primarily from :
## Title -- ROUGHNESS LENGTH CLASSIFICATION OF Global Wind Atlas
## Authors -- DTU
globCoverCodeToRoughess = OrderedDict()
# GlobCover Number
globCoverCodeToRoughess[210] = 0.0 # Water Bodies
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

def clcGridToRough(grid):
    if grid <= 0 or grid > 44: grid = 42 # assume ocean if grid value is unknown
    code = clcGridToCode_v2006[grid]
    rough = clcCodeToRoughess[code]
    return rough

clcGridToCodeFunc = np.vectorize( lambda x: clcGridToCode_v2006[x] )
clcCodeToRoughFunc = np.vectorize( lambda x: clcCodeToRoughess[x] )
def roughnessFromCLCValues( num, gridValues=True ):
    num = np.array(num) # be sure we have an array

    # convert to grid values, if needed
    if gridValues:
        # assume unknown data means ocean
        num[num<=0] = 44
        num[num>44] = 44
        num[np.isnan(num)] = 44

        num = clcGridToCodeFunc(num)

    # convert to roughness
    rough - clcCodeToRoughFunc(num)

    # done
    return rough


def roughnessFromLandCover(num, landCover='clc'):
    """
    landCover can be 'clc', 'globCover', or 'modis'
    """
    if landCover=='clc': source = lambda x: clcCodeToRoughess[x]
    elif landCover=='clc-grid': source = clcGridToRough
    elif landCover=='globCover': source = lambda x: globCoverCodeToRoughess[x]
    elif landCover=='modis': source = lambda x: modisCodeToRoughess[x]
    else: 
        raise ResError("invalid input")

    converter = np.vectorize(source)
    return converter(num)

    if isinstance(num,int):
        return source(num)
    if isinstance(num, np.ndarray) or isinstance(num,list):
        return np.array([source(int(x)) for x in num])
    if isinstance(num, pd.Series) or isinstance(num, pd.DataFrame):
        return num.apply( lambda x: source(int(x)))
    else: 
        raise ResError("invalid input")