from reskit.wind.economic.scale_Offshore_Capex import *
import numpy as np
from reskit.default_paths import DEFAULT_PATHS
from reskit.parameters.parameters import OffshoreParameters

# GPS Coordiantes for location in North Sea
lon = 6.2160
lat = 53.7170


# def test_defaultPaths():
#     assert (
#         DEFAULT_PATHS["waterdepthFile"] is not None
#     ), "waterdepthFile should not be None"
#     assert (
#         DEFAULT_PATHS["distancetoCoast"] is not None
#     ), "waterdepthFile should not be None"


def test_distanceToCoastLine():
    # load tiff file
    band, transformer, transforming = loadDistanceBand()
    # get distance to loaction
    calcuated_distance = distanceToCoastline(lat, lon, band, transformer, transforming)

    assert np.isclose(
        calcuated_distance, 23
    ), "the distance-to-coatslibe-calculation does not work properly"


def test_waterDepthFromLocation():
    waterdepth_exact = 22
    depth = waterDepthFromLocation(
        lat, lon, waterDepthFolderPath=DEFAULT_PATHS.get("waterdepthFile")
    )
    assert np.isclose(
        depth, waterdepth_exact
    ), "the distanceToCoatsLineFile is not working correct"


def test_calculateOffshoreCapex():
    comparedCAPEX = 2943.80819956
    calculatedCAPEX = calculateOffshoreCapex(
        inputCapex=3000,
        capacity=14000,
        hubHeight=150,
        waterDepth=25,
        coastDistance=25,
        rotorDiam=230,
        techYear=2050,
        shareTurb=0.449,
        shareFound=0.204,
        shareCable=0.181,
        shareOverhead=0.166,
        maxMonopileDepth=25,
        maxJacketDepth=55,
        litValueAvgDepth=17,
        litValueAvgDistCoast=27,
        baseCap=13000,
        baseHubHeight=150,
        baseRotorDiam=250,
        defaultOffshoreParamsFp=None,
    )

    assert np.isclose(calculatedCAPEX, comparedCAPEX)


def test_getRatedCostFromWaterDepth():

    test_value = getRatedCostFromWaterDepth(17, 25, 55)
    assert np.isclose(test_value, 431693), "equation is changed"
    assert getRatedCostFromWaterDepth(17) == getRatedCostFromWaterDepth(
        -17
    ), "negative avlues are handled incorrect"


def test_getCableCost():
    short = getCableCost(10, 14000, variableCostFactor=1.35, fixedCost=0)
    long = getCableCost(50, 14000, variableCostFactor=1.35, fixedCost=0)
    small = getCableCost(10, 10000, variableCostFactor=1.35, fixedCost=0)
    large = getCableCost(10, 14000, variableCostFactor=1.35, fixedCost=0)

    assert short < long, "equaiton is wrong in cable cost calculations"
    assert small < large, "equaiton is wrong in cable cost calculations"
