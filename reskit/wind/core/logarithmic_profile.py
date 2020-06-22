import numpy as np
import geokit as gk
from collections import OrderedDict

from ...util import ResError


def apply_logarithmic_profile_projection(measured_wind_speed, measured_height, target_height, roughness, displacement=0, stability=0):
    """
    Estimates wind speeds at a target height based on measured wind speeds values at a known measurement height.
    Estimation subject to surface roughness, displacement height, and a stability factor.

    Parameters
    ----------
    measured_wind_speed : multidimentional array-like
        The wind speeds in m/s that is going to be adjusted 
        If an array is given with a single dimension, it is assumed to represent timeseries values for a single location
        If multidimensional array is given, the assumed dimensional context is (time, locations), and 'targetLoc' must be an iterable with the same length as the 'locations' dimension
    
    measured_height : multidimentional array-like
        The measurement height in m of the raw windspeeds.
        If an array is given for measured_wind_speed with a single dimension, a single value is expected for measured_height
        If multidimensional array is given for measured_wind_speed, an array of values is expected for measured_height. One value for each wind speed timeseries

    target_height : numeric or array-like
        The height in m to project each wind speed timeseries to.
        If an array is given for target_height, each value must match to one wind speed timeseries in measured_wind_speed

    roughness : numeric or array-like
        The roughness value used to project each wind speed timeseries.
        If an array is given, each value must match to one wind speed timeseries in measured_wind_speed.

    displacement : numeric or array-like, optional
        The displacement value used to project each wind speed timeseries, by default 0.
        If an array is given, each value must match to one wind speed timeseries in measured_wind_speed.

    stability : numeric or array-like, optional
        The stability value used to project each wind speed timeseries, by default 0.
        If an array is given, each value must match to one wind speed timeseries in measured_wind_speed.

    Returns
    -------
    multidimentional array-like
        Windspeed in m/s at target height
    """    

    return measured_wind_speed * (np.log((target_height - displacement) / roughness) + stability) / (np.log((measured_height - displacement) / roughness) + stability)


def roughness_from_levels(low_wind_speed, low_height, high_wind_speed, high_height):
    """
    Computes a roughness factor from two windspeed values at two distinct heights.

    Parameters
    ----------
    low_wind_speed : numeric or np.ndarray
        The measured wind speed in m/s at the lower height.
    low_height : numeric or np.ndarray
        The lower height in m.
    high_wind_speed : numeric or np.ndarray
        The measured wind speed in m/s at the higher height.
    high_height : numeric or np.ndarray
        The higher height in m.

    Returns
    -------
    numeric or arrray-like
        Roughness factor
    """ 

    return np.exp((high_wind_speed * np.log(low_height) - low_wind_speed * np.log(high_height)) / (high_wind_speed - low_wind_speed))


############################################################################
# See CLC codes at: https://land.copernicus.eu/pan-european/corine-land-cover/clc-2000 #Link updated 
# Roughnesses defined primarily from: Silva, J., Ribeiro, C., & Guedes, R. (2007). Roughness length classification of corine land cover classes. European Wind Energy Conference and Exhibition 2007, EWEC 2007.

clcCodeToRoughess = OrderedDict()
clcCodeToRoughess[111] = 1.2  # Continuous urban fabric
clcCodeToRoughess[311] = 0.75  # Broad-leaved forest
clcCodeToRoughess[312] = 0.75  # Coniferous-leaved forest
clcCodeToRoughess[313] = 0.75  # Mixed-leaved forest
clcCodeToRoughess[141] = 0.6  # Green urban areas
clcCodeToRoughess[324] = 0.6  # Transitional woodland/shrub
clcCodeToRoughess[334] = 0.6  # Burnt areas
clcCodeToRoughess[112] = 0.5  # Discontinous urban fabric
clcCodeToRoughess[133] = 0.5  # Construction sites
clcCodeToRoughess[121] = 0.5  # Industrial or commercial units
clcCodeToRoughess[142] = 0.5  # Sport and leisure facilities
clcCodeToRoughess[123] = 0.5  # Port areas
clcCodeToRoughess[242] = 0.3  # Agro-forestry areas
clcCodeToRoughess[243] = 0.3  # Complex cultivation patterns
clcCodeToRoughess[244] = 0.3  # Land principally occupied by agriculture, with significant areas of natural vegetation
clcCodeToRoughess[241] = 0.1  # Annual crops associated with permanent crops
clcCodeToRoughess[221] = 0.1  # Fruit trees and berry plantations
clcCodeToRoughess[222] = 0.1  # Vineyard
clcCodeToRoughess[223] = 0.1  # Olive groves
clcCodeToRoughess[122] = 0.075  # Road and rail networks and associated land
clcCodeToRoughess[211] = 0.05  # Non-irrigated arable land
clcCodeToRoughess[212] = 0.05  # Permanently irrigated land
clcCodeToRoughess[213] = 0.05  # Rice fields
clcCodeToRoughess[411] = 0.05  # Inland marshes
clcCodeToRoughess[421] = 0.05  # Salt marshes
clcCodeToRoughess[321] = 0.03  # Sclerophylous vegetation
clcCodeToRoughess[322] = 0.03  # Moors and heathland
clcCodeToRoughess[323] = 0.03  # Natural grassland
clcCodeToRoughess[231] = 0.03  # Pastures
clcCodeToRoughess[131] = 0.005  # Dump sites
clcCodeToRoughess[132] = 0.005  # Mineral extraction sites
clcCodeToRoughess[124] = 0.005  # Airports
clcCodeToRoughess[332] = 0.005  # Bare rock
clcCodeToRoughess[333] = 0.005  # Sparsely vegetated areas
clcCodeToRoughess[335] = 0.001  # Glaciers and perpetual snow
clcCodeToRoughess[422] = 0.0005  # Peatbogs
clcCodeToRoughess[412] = 0.0005  # Salines
clcCodeToRoughess[423] = 0.0005  # Intertidal flats
clcCodeToRoughess[331] = 0.0003  # Beaches, dunes, and sand plains
clcCodeToRoughess[511] = 0.001  # Water courses # SUSPICIOUS
clcCodeToRoughess[512] = 0.0005  # Water bodies # SUSPISCIOUS
clcCodeToRoughess[521] = 0.0005  # Costal lagoons # SUSPISCIOUS
clcCodeToRoughess[522] = 0.0008  # Estuaries # SUSPISCIOUS
clcCodeToRoughess[523] = 0.0002  # Sea and ocean # SUSPISCIOUS

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


def roughness_from_clc(clc_path, loc, window_range=0):
    """
    Estimates a roughness factor according to suggestions by Silva et al. [1] by the prominent land cover at given locations according to the Corine Land Cover dataset [2].

    Parameters
    ----------
    clc_path : str
        The path to the Corine Land Cover raster file on the disk.
    loc : Location or ogr.Geometry or str or tuple (lat,lon)
        The locations for which roughness should be estimated.
    window_range : int; optional
        An extra number of pixels to extract around the indicated locations, by default 0.
        A window_range of 0 means only the CLC pixel value for each location is returned, A window_range of 1 means an extra pixel is extracted around each location in all directions. Leading to a 3x3 matrix of roughness values
        Use this if you need to do some operation on the roughnesses found around the indicated location

    Returns
    -------
    float
        Roughness lengths factors
    
    Sources
    --------
        [1] Silva, J., Ribeiro, C., & Guedes, R. (2007). Roughness length classification of corine land cover classes. European Wind Energy Conference and Exhibition 2007, EWEC 2007.


    Roughness Values
    ----------------

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
    # Ensure location is okay
    loc = gk.LocationSet(loc)

    # Get pixels values from clc
    clcGridValues = gk.raster.interpolateValues(clc_path, loc, winRange=window_range, noDataOkay=True)
    print(clcGridValues)

    # make output array
    if window_range > 0:
        outputs = []
        for v in clcGridValues:
            # Treat nodata as ocean
            v[np.isnan(v)] = 44
            v[v > 44] = 44
            v = v.astype(int)

            values, counts = np.unique(v, return_counts=True)

            total = 0
            for val, cnt in zip(values, counts):
                total += cnt * clcCodeToRoughess[clcGridToCode_v2006[val]]

            outputs.append(total / counts.sum())
    else:
        # Treat nodata as ocean
        clcGridValues[np.isnan(clcGridValues)] = 44
        clcGridValues[clcGridValues > 44] = 44
        clcGridValues = clcGridValues.astype(int)

        # Get the associated
        outputs = [clcCodeToRoughess[clcGridToCode_v2006[val]] for val in clcGridValues]

    # Done!
    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs


############################################################################
# Defined primarily from [2] DTU Wind Energy. (2019). Gloabal Wind Atlas. https://globalwindatlas.info/
globCoverCodeToRoughess = OrderedDict()
# GlobCover Number
globCoverCodeToRoughess[210] = 0.0002  # Water Bodies # changed by Me from 0.0 to 0.0002
globCoverCodeToRoughess[220] = 0.0004  # Permanant Snow and ice
globCoverCodeToRoughess[200] = 0.005  # Bare areas
globCoverCodeToRoughess[140] = 0.03  # Grasslands, savannas or lichens/mosses
globCoverCodeToRoughess[150] = 0.05  # Sparse vegetation
globCoverCodeToRoughess[11] = 0.1  # Croplands
globCoverCodeToRoughess[14] = 0.1  # Croplands
globCoverCodeToRoughess[130] = 0.1  # Shrubland
globCoverCodeToRoughess[180] = 0.2  # Wetlands
globCoverCodeToRoughess[20] = 0.3  # Mosaic natural vegetation/cropland
globCoverCodeToRoughess[30] = 0.3  # Mosaic natural vegetation/cropland
globCoverCodeToRoughess[160] = 0.5  # Flooded forest
globCoverCodeToRoughess[120] = 0.5  # Mosaic grassland/forest
globCoverCodeToRoughess[170] = 0.6  # Flooded forest or shrubland
globCoverCodeToRoughess[190] = 1.0  # Urban Areas
globCoverCodeToRoughess[40] = 1.5  # Forests
globCoverCodeToRoughess[50] = 1.5  # Forests
globCoverCodeToRoughess[60] = 1.5  # Forests
globCoverCodeToRoughess[70] = 1.5  # Forests
globCoverCodeToRoughess[90] = 1.5  # Forests
globCoverCodeToRoughess[100] = 1.5  # Forests
globCoverCodeToRoughess[110] = 1.5  # Forests

# Modis Number for "no data" points of GlobCover (mostly in areas North of 60°)
modisCodeToRoughess = OrderedDict()
modisCodeToRoughess[0] = 0.0  # Water Bodies
modisCodeToRoughess[15] = 0.0004  # Permanant Snow and ice
modisCodeToRoughess[16] = 0.005  # Bare areas
modisCodeToRoughess[10] = 0.03  # Grasslands, savannas or lichens/mosses
modisCodeToRoughess[12] = 0.1  # Croplands
modisCodeToRoughess[6] = 0.1  # Shrubland
modisCodeToRoughess[7] = 0.1  # Shrubland
modisCodeToRoughess[11] = 0.2  # Wetlands
modisCodeToRoughess[14] = 0.3  # Mosaic natural vegetation/cropland
modisCodeToRoughess[9] = 0.5  # Mosaic grassland/forest
modisCodeToRoughess[13] = 1.0  # Urban Areas
modisCodeToRoughess[1] = 1.5  # Forests
modisCodeToRoughess[2] = 1.5  # Forests
modisCodeToRoughess[3] = 1.5  # Forests
modisCodeToRoughess[4] = 1.5  # Forests
modisCodeToRoughess[5] = 1.5  # Forests
modisCodeToRoughess[8] = 1.5  # Forests

############################################################################
# CCI Landcover  classification by ESA and the Climate Change Initiative [3] European Space Agency. (2014). ESA Climate Change Initiative. https://www.esa-landcover-cci.org/?q=node/1
# Roughnesses defined due to the comparison with CLC and globCover
cciCodeToRoughess = OrderedDict()
# CCI LC Number
cciCodeToRoughess[210] = 0.0002  # Water bodies
cciCodeToRoughess[220] = 0.001  # Permanent snow and ice
cciCodeToRoughess[200] = 0.005  # Bare areas
cciCodeToRoughess[201] = 0.005  # Consolidated bare areas
cciCodeToRoughess[202] = 0.005  # Unconsolidated bare areas
cciCodeToRoughess[150] = 0.005  # Sparse vegetation (tree, shrub, herbaceous cover) (<15%)
cciCodeToRoughess[152] = 0.005  # Sparse shrub (<15%)
cciCodeToRoughess[153] = 0.005  # Sparse herbaceous cover (<15%)
cciCodeToRoughess[10] = 0.03  # Cropland, rainfed
cciCodeToRoughess[11] = 0.03  # Herbaceous cover
cciCodeToRoughess[120] = 0.03  # Shrubland
cciCodeToRoughess[121] = 0.03  # Shrubland evergreen #barely exists, only near water bodies, ocean
cciCodeToRoughess[122] = 0.03  # Shrubland deciduous #barely exists, only near water bodies, ocean
cciCodeToRoughess[12] = 0.3  # Tree or shrub cover
cciCodeToRoughess[110] = 0.03  # Mosaic herbaceous cover (>50%) / tree and shrub (<50%)
cciCodeToRoughess[40] = 0.03  # Mosaic natural vegetation (tree, shrub, herbaceous cover) (>50%) / cropland (<50%)
cciCodeToRoughess[180] = 0.03  # Shrub or herbaceous cover, flooded, fresh/saline/brakish water
cciCodeToRoughess[130] = 0.03  # Grassland
cciCodeToRoughess[140] = 0.03  # Lichens and mosses
cciCodeToRoughess[170] = 0.1  # Tree cover, flooded, saline water (areas around river deltas and ocean)
cciCodeToRoughess[20] = 0.1  # Cropland, irrigated or post-flooding
cciCodeToRoughess[30] = 0.1  # Mosaic cropland (>50%) / natural vegetation (tree, shrub, herbaceous cover) (<50%)
cciCodeToRoughess[160] = 0.5  # Tree cover, flooded, fresh or brakish water, barely exists
cciCodeToRoughess[100] = 0.75  # Mosaic tree and shrub (>50%) / herbaceous cover (<50%)
cciCodeToRoughess[50] = 0.75  # Tree cover, broadleaved, evergreen, closed to open (>15%)
cciCodeToRoughess[60] = 0.75  # Tree cover, broadleaved, deciduous, closed to open (>15%)
cciCodeToRoughess[61] = 0.75  # Tree cover, broadleaved, deciduous, closed (>40%)
cciCodeToRoughess[62] = 0.75  # Tree cover, broadleaved, deciduous, open (15-40%)
cciCodeToRoughess[70] = 0.75  # Tree cover, needleleaved, evergreen, closed to open (>15%)
cciCodeToRoughess[71] = 0.75  # Tree cover, needleleaved, evergreen, closed (>40%)
cciCodeToRoughess[72] = 0.75  # Tree cover, needleleaved, evergreen, open (15-40%)
cciCodeToRoughess[80] = 0.75  # Tree cover, needleleaved, deciduous, closed to open (>15%)
cciCodeToRoughess[81] = 0.75  # Tree cover, needleleaved, deciduous, closed (>40%)
cciCodeToRoughess[82] = 0.75  # Tree cover, needleleaved, deciduous, open (15-40%)
cciCodeToRoughess[90] = 0.75  # Tree cover, mixed leaf type (broadleaved and needleleaved)
cciCodeToRoughess[190] = 1.2  # Urban areas


def roughness_from_land_cover_classification(classification, land_cover_type='clc'):
    """
    Estimates roughness value from a given land cover classification raster file.

    Parameters
    ----------
    classification : int
        land cover classification
    land_cover_type : str, optional
        Accepted arguments are 'clc', 'clc-code', 'globCover', 'modis', or 'cci', by default 'clc'

    Returns
    -------
    int or array_like
        Roughness lengnth value(s)

    """
    if land_cover_type == 'clc':
        # fix no data values
        classification[classification < 0] = 44
        classification[classification > 44] = 44
        classification[np.isnan(classification)] = 44

        # set source
        def source(x): return clcCodeToRoughess[clcGridToCode_v2006[x]]
    elif land_cover_type == 'clc-code':
        def source(x): return clcCodeToRoughess[x]
    elif land_cover_type == 'globCover':
        def source(x): return globCoverCodeToRoughess[x]
    elif land_cover_type == 'modis':
        def source(x): return modisCodeToRoughess[x]
    elif land_cover_type == 'cci':
        def source(x): return cciCodeToRoughess[x]
    else:
        raise ResError("invalid input")

    converter = np.vectorize(source)
    return converter(classification)


def roughness_from_land_cover_source(source, loc, land_cover_type='clc'):
    """
    Estimate roughness value from a given land cover source

    Parameters
    ----------
    source : str
        The path to the Corine Land Cover raster file on the disk.
    loc : Location or ogr.Geometry or str or tuple (lat,lon)
        The locations for which roughness should be estimated.
    land_cover_type : str, optional
        Accepted arguments are 'clc', 'clc-code', 'globCover', 'modis', or 'cci', by default 'clc'
    Returns
    -------
    float
        Roughness lengnth value
    
    """
    loc = gk.LocationSet(loc)
    classifications = gk.raster.interpolateValues(source, loc, noDataOkay=False)

    return roughness_from_land_cover_classification(classifications, land_cover_type=land_cover_type)
