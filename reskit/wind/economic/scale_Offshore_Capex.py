# %%
# M.Stargardt - 10.07.2025
import os
import pickle
import glob
import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer
import numpy as np

from reskit.default_paths import DEFAULT_PATHS
from reskit.parameters.parameters import OffshoreParameters


# %%


def waterDepthFromLocation(
    latitude,
    longitude,
    waterDepthFolderPath=DEFAULT_PATHS.get("waterdepthFile"),
):
    """
    Returns the water depth (in meters) at a given geographic location (latitude and longitude).

    Args:
        latitude (float): Latitude in decimal degrees.
        longitude (float): Longitude in decimal degrees.
        waterDepthFolderPath (str, optional): Path to the folder containing water depth .tif files.

    Returns:
        float: Water depth at the specified location in meters (positive value). Returns None if not found.
    """
    depthFiles = glob.glob(os.path.join(waterDepthFolderPath, "*.tif"))
    resultDepth = getRasterValueFromTifs(depthFiles, latitude, longitude)

    return abs(resultDepth) if resultDepth is not None else None


# %% function to calculate the distance to the coastline
# if you want to execute the distance to coastline more often, please separete the loading of the taserband to increase execution time


def loadDistanceBand(path=DEFAULT_PATHS.get("distancetoCoast")):
    """
    Loads the raster band and sets up the coordinate transformer.

    Args:
        path (str, optional): File path to the distance-to-coast raster.

    Returns:
        ndarray: Raster band values.
        Transformer: Coordinate transformer.
        Affine: Raster transform object for coordinate conversion.
    """
    src = rasterio.open(path)
    band = src.read(1)
    transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
    return band, transformer, src.transform


def distanceToCoastline(latitude, longitude, band, transformer, transformFunc):
    """
    Computes the distance to the coastline from a given geographic point.

    Args:
        latitude (float): Latitude in decimal degrees.
        longitude (float): Longitude in decimal degrees.
        band (ndarray): Raster band values.
        transformer (Transformer): Coordinate transformer.
        transformFunc (Affine): Raster transform object.

    Returns:
        float or None: Distance in kilometers, or None if point is out of bounds or an error occurs.
    """
    x, y = transformer.transform(longitude, latitude)
    try:
        row, col = rowcol(transformFunc, x, y)
        if 0 <= row < band.shape[0] and 0 <= col < band.shape[1]:
            return band[row, col]
        else:
            print(f"Out of bounds for Lat: {latitude}, Lon: {longitude}")
    except Exception as e:
        print(f"Error at Lat: {latitude}, Lon: {longitude}: {e}")
    return None


# %%
def calculateOffshoreCapex(
    capacity,
    hubHeight,
    waterDepth,
    coastDistance,
    rotorDiam,
    techYear=2050,
    shareTurb=0.449,
    shareFound=0.204,
    shareCable=0.181,
    shareOverhead=0.166,
    maxMonopileDepth=25,
    maxJacketDepth=55,
    litValueAvgDepth=17,
    litValueAvgDistCoast=27,
    inputCapex=None,
    baseCap=None,
    baseHubHeight=None,
    baseRotorDiam=None,
    defaultOffshoreParamsFp=None,
):
    """
    Scales a generic offshore CAPEX value based on water depth and distance to shore.

    The function splits the total input CAPEX into major cost components (turbine, foundation,
    cable, overhead) and scales each individually based on project-specific parameters.

    Args:
        capacity (float): Turbine rated capacity in MW.
        hubHeight (float): Hub height in meters.
        waterDepth (float): Site-specific water depth in meters.
        coastDistance (float): Distance from site to nearest coast in kilometers.
        rotorDiam (float): Rotor diameter in meters.
        techYear (int, optional): Year of the applied technology. Default is 2050.
        shareTurb (float, optional): Share of turbine cost in total CAPEX. Default is 0.449.
        shareFound (float, optional): Share of foundation cost. Default is 0.204.
        shareCable (float, optional): Share of cable/connection cost. Default is 0.181.
        shareOverhead (float, optional): Share of overhead/miscellaneous costs. Default is 0.166.
        maxMonopileDepth (float, optional): Maximum depth for monopile foundations. Default is 25.
        maxJacketDepth (float, optional): Maximum depth for jacket foundations. Default is 55.
        litValueAvgDepth (float, optional): Reference depth in CAPEX literature. Default is 17.
        litValueAvgDistCoast (float, optional): Reference coast distance. Default is 27.
        inputCapex (float, optional): Reference CAPEX per kW (€/kW). Loaded from CSV if not provided.
        baseCap (float, optional): Reference turbine capacity. Loaded from CSV if not provided.
        baseHubHeight (float, optional): Reference hub height. Loaded from CSV if not provided.
        baseRotorDiam (float, optional): Reference rotor diameter. Loaded from CSV if not provided.
        defaultOffshoreParamsFp (str, optional): Filepath to offshore turbine parameters CSV.

    Returns:
        float: Adjusted offshore wind CAPEX in €/kW for the given configuration.
    """
    assert np.isclose(
        shareTurb + shareFound + shareCable + shareOverhead, 1.0, rtol=1e-9
    ), "Sum of all cost shares must equal 1"

    assert (
        0 < maxMonopileDepth < 55
    ), "Maximum depth for monopile foundation must be between 0 and 55 m"

    assert (
        55 <= maxJacketDepth < 100
    ), "Maximum depth for jacket foundation must be between 55 and 100 m"

    assert (
        maxMonopileDepth < maxJacketDepth
    ), "Jacket depth must be greater than monopile depth"

    params = OffshoreParameters(fp=defaultOffshoreParamsFp, year=techYear)

    if baseCap is None:
        baseCap = params.base_capacity
        print("baseCap is taken from overall techno-economic file")
    if baseHubHeight is None:
        baseHubHeight = params.base_hub_height
        print("baseHubHeight is taken from overall techno-economic file")
    if baseRotorDiam is None:
        baseRotorDiam = params.base_rotor_diam
        print("baseRotorDiam is taken from overall techno-economic file")
    if inputCapex is None:
        inputCapex = params.base_capex_per_capacity
        print("inputCapex is taken from overall techno-economic file")

    turbineCostBase = inputCapex * shareTurb
    foundCostBase = inputCapex * shareFound
    cableCostBase = inputCapex * shareCable
    overheadCostBase = inputCapex * shareOverhead

    # Scale turbine cost
    turbineCostNew = onshoreTcc(
        capacity,
        hubHeight,
        rotorDiam,
        gdpEscalator=1,
        bladeMaterialEscalator=1,
        blades=3,
    )
    turbineCostReference = onshoreTcc(
        baseCap,
        baseHubHeight,
        baseRotorDiam,
        gdpEscalator=1,
        bladeMaterialEscalator=1,
        blades=3,
    )
    costRatioTurbine = turbineCostNew / turbineCostReference
    newTurbineCost = turbineCostBase * costRatioTurbine

    # Scale foundation cost
    depthBaseCost = getRatedCostFromWaterDepth(litValueAvgDepth)
    depthPlantCost = getRatedCostFromWaterDepth(waterDepth)
    costRatioFoundation = depthPlantCost / depthBaseCost
    newFoundationCost = foundCostBase * costRatioFoundation

    # Scale cable cost
    cableRatio = getCableCost(coastDistance, capacity) / getCableCost(
        litValueAvgDistCoast, baseCap
    )
    newCableCost = cableCostBase * cableRatio

    # Combine all costs
    totalOffshoreCapex = (
        newTurbineCost + newFoundationCost + newCableCost + overheadCostBase
    )

    return totalOffshoreCapex


# %%
def getRasterValueFromTifs(tiffPaths, latitude, longitude):
    """
    Retrieves the raster value from a list of .tif files at a given geographic point.

    Args:
        tiffPaths (list of str): Paths to the .tif files.
        latitude (float): Latitude in decimal degrees.
        longitude (float): Longitude in decimal degrees.

    Returns:
        float or None: The value from the raster at the location, or None if not found.
    """
    for tifPath in tiffPaths:
        with rasterio.open(tifPath) as src:
            bounds = src.bounds
            if (
                bounds.left <= longitude <= bounds.right
                and bounds.bottom <= latitude <= bounds.top
            ):
                try:
                    for val in src.sample([(longitude, latitude)]):
                        return val[0]
                except Exception as e:
                    print(f"Error reading from {tifPath}: {e}")
                    continue
    return None


# %%
def getRatedCostFromWaterDepth(depth, allowNegative=True):
    """
    Estimates the rated cost of offshore wind turbine foundations based on water depth.

    Args:
        depth (float): Water depth at the installation site (in meters).
        allowNegative (bool): Whether negative values are allowed. If False, raises ValueError if depth < 0.

    Returns:
        float: Rated cost in €/kW.

    Reference:
        Rogeau et al. (2023), Renewable and Sustainable Energy Reviews.
    """
    if not allowNegative and depth < 0:
        raise ValueError("Depth must not be negative when not allowNegative")

    if depth < 25:
        c1, c2, c3 = 181, 552, 370
    elif depth <= 55:
        c1, c2, c3 = 103, -2043, 478
    else:
        c1, c2, c3 = 0, 697, 1223

    return c1 * depth**2 + c2 * depth + c3 * 1000


# %%
def getCableCost(distance, capacity, variableCostFactor=1.35, fixedCost=0):
    """
    Calculates the cost for connecting an offshore wind power plant to the coastline.

    Args:
        distance (float): Distance to coastline in kilometers.
        capacity (float): Power plant's capacity in MW.
        variableCostFactor (float, optional): Cost multiplier in €/kW/km (default = 1.35).
        fixedCost (float, optional): Fixed connection cost (default = 0).

    Returns:
        float: Total cable connection cost in monetary units.

    Reference:
        Rogeau et al. (2023), "Review and modeling of offshore wind CAPEX",
        Renewable and Sustainable Energy Reviews, DOI: 10.1016/j.rser.2023.113699
    """
    assert distance > 0, "distance must be larger tan 0"
    assert capacity > 0, " turbine capacity must be larger than 0"
    assert variableCostFactor > 0, "cost factor must be larger tan 0"
    assert fixedCost >= 0, "fixed Cost must be postive or 0"

    variableCost = variableCostFactor * distance * capacity
    cableCost = fixedCost + variableCost

    return cableCost


def onshoreTcc(cp, hh, rd, gdpEscalator=None, bladeMaterialEscalator=None, blades=None):
    """
    Calculates the turbine capital cost (TCC) of a 3-blade onshore wind turbine based on
    capacity, hub height, and rotor diameter according to the model by Fingersh et al.

    Args:
        cp (float): Turbine capacity in kW.
        hh (float): Hub height in meters.
        rd (float): Rotor diameter in meters.
        gdpEscalator (float, optional): Labor cost escalator. Defaults to 1.
        bladeMaterialEscalator (float, optional): Blade material cost escalator. Defaults to 1.
        blades (int, optional): Number of blades. Defaults to 3.

    Returns:
        float: Turbine capital cost (TCC) in monetary units.

    Reference:
        Fingersh et al. (2006), NREL. https://www.nrel.gov/docs/fy07osti/40566.pdf
    """
    if gdpEscalator is None or bladeMaterialEscalator is None or blades is None:
        offshoreParams = OffshoreParameters()
        gdpEscalator = offshoreParams.gdp_escalator
        bladeMaterialEscalator = offshoreParams.blade_material_escalator
        blades = offshoreParams.blades

    rr = rd / 2
    sa = np.pi * rr * rr

    singleBladeMass = 0.4948 * np.power(rr, 2.53)
    singleBladeCost = (
        (0.4019 * np.power(rr, 3) - 21051) * bladeMaterialEscalator
        + 2.7445 * np.power(rr, 2.5025) * gdpEscalator
    ) * (1 - 0.28)

    hubMass = 0.945 * singleBladeMass + 5680.3
    hubCost = hubMass * 4.25

    pitchSystemCost = 2.28 * (0.2106 * np.power(rd, 2.6578))
    noseConeMass = 18.5 * rd - 520.5
    noseConeCost = noseConeMass * 5.57

    lowSpeedShaftCost = 0.01 * np.power(rd, 2.887)
    bearingMass = (rd * 8 / 600 - 0.033) * 0.0092 * np.power(rd, 2.5)
    bearingCost = 2 * bearingMass * 17.6

    breakCouplingCost = 1.9894 * cp - 0.1141
    generatorCost = cp * 219.33
    electronicsCost = cp * 79
    yawSystemCost = 2 * (0.0339 * np.power(rd, 2.964))
    mainframeMass = 1.228 * np.power(rd, 1.953)
    mainframeCost = 627.28 * np.power(rd, 0.85)
    platformAndRailingMass = 0.125 * mainframeMass
    platformAndRailingCost = platformAndRailingMass * 8.7

    electricalConnectionCost = cp * 40
    hydraulicAndCoolingSystemCost = cp * 12
    nacelleCost = 11.537 * cp + 3849.7
    towerMass = 0.2694 * sa * hh + 1779
    towerCost = towerMass * 1.5

    turbineCapitalCost = (
        singleBladeCost * blades
        + hubCost
        + pitchSystemCost
        + noseConeCost
        + lowSpeedShaftCost
        + bearingCost
        + breakCouplingCost
        + generatorCost
        + electronicsCost
        + yawSystemCost
        + mainframeCost
        + platformAndRailingCost
        + electricalConnectionCost
        + hydraulicAndCoolingSystemCost
        + nacelleCost
        + towerCost
    )

    return turbineCapitalCost
