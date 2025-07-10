# %%

import os
import pickle
import glob
import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer
import numpy as np
from onshore_cost_model import onshore_tcc


# %%


def waterDepthFromLocation(
    lat,
    lon,
    waterDepthFolderPath="/benchtop/shared_data/General_Bathymetric_Chart_of_the_Oceans_GEBCO/GEBCO_2020/GEBCO_tiles/",
):
    """
    Returns the water depth (in meters) at a given geographic location (latitude and longitude).


    Args:
        lat (float): Latitude in decimal degrees.
        lon (float): Longitude in decimal degrees.

    Returns:
        float: Water depth at the specified location in meters (positive value).
               Returns None if no valid depth is found.
    """

    depthFiles = glob.glob(os.path.join(waterDepthFolderPath, "*.tif"))
    resultDepth = getRasterValueFromTifs(depthFiles, lat, lon)

    return abs(resultDepth) if resultDepth is not None else None


# %% function to calculate the distance to the coastline


def loadDistanceBand(
    path="/benchtop/projects/2021-m-stargardt-phd/02_GHR_2025/01_offshoreTiffs/GMT_intermediate_coast_distance_01d.tif",
):
    """
    Load the raster band and set up the coordinate transformer.
    Returns:
        band (ndarray): Raster band values.
        transformer (Transformer): Coordinate transformer.
        transform (Affine): Raster transform object for converting coordinates.
    """
    src = rasterio.open(path)
    band = src.read(1)
    transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
    return band, transformer, src.transform


def distanceToCoastline(lat, lon, band, transformer, transformfunc):
    """
    Compute the distance to coastline from given lat/lon in km.
    """
    x, y = transformer.transform(lon, lat)
    try:
        row, col = rowcol(transformfunc, x, y)
        if 0 <= row < band.shape[0] and 0 <= col < band.shape[1]:
            return band[row, col]
        else:
            print(f"Out of bounds for Lat: {lat}, Lon: {lon}")
    except Exception as e:
        print(f"Error at Lat: {lat}, Lon: {lon}: {e}")
    return None


# %%
def calculateOffshoreCapex(
    InputCapex,
    capacity,
    hubheight,
    waterdepth,
    coastDistance,
    rotordiam,
    shareTurb=0.449,
    shareFound=0.204,
    shareCable=0.181,
    shareOverhead=0.166,
    maxMonopolDepth=25,
    maxJacketDepth=55,
):
    """
    Scale a generic offshore CAPEX value based on water depth and distance to shore.

    The function splits the total input CAPEX into major cost components (turbine, foundation,
    cable, overhead) and scales each individually based on project-specific parameters such as
    turbine size, water depth, and coastline distance.

    Args:
        InputCapex (float): Total CAPEX per kW for a reference offshore wind plant (€/kW).
        capacity (float): Turbine capacity in MW.
        hubheight (float): Hub height in meters.
        waterdepth (float): Site-specific water depth in meters.
        coastDistance (float): Distance from site to nearest coast in kilometers.
        rotordiam (float): Rotor diameter in meters.
        shareTurb (float): Share of turbine cost in total CAPEX (default = 0.449).
        shareFound (float): Share of foundation cost in total CAPEX (default = 0.204).
        shareCable (float): Share of cable/connection cost in total CAPEX (default = 0.181).
        shareOverhead (float): Share of overhead/miscellaneous costs in total CAPEX (default = 0.166).
        maxMonopolDepth (float): Max depth suitable for monopile foundations (default = 25 m).
        maxJacketDepth (float): Max depth suitable for jacket foundations (default = 55 m).

    Returns:
        float: Adjusted total offshore CAPEX (€/kW) scaled for given project parameters.
    """

    assert np.isclose(
        shareTurb + shareFound + shareCable + shareOverhead, 1.0, rtol=1e-9
    ), "Sum of all cost shares must equal 1"
    assert (
        0 < maxMonopolDepth < 60
    ), "Maximum Depth for Monopile Foundation must be between 0 and 50 m"
    assert (
        60 <= maxJacketDepth < 100
    ), "Maximum Depth for Jacket Foundation must be between 0 and 50 m"
    assert (
        maxMonopolDepth < maxJacketDepth
    ), " Maximum Depth for Jacket Foundation must be larger than maximum  depth for Monopile Foundation"

    averageDepthLiterature = 17  # m
    averageCoastDistance = 27  # km

    TurbineCostBase = InputCapex * shareTurb
    FoundCostbase = InputCapex * shareFound
    CableCostBase = InputCapex * shareCable
    OverheadCostBase = InputCapex * shareOverhead

    # adapting each cost share
    # Turbine cost are adapted to Severin's onshore cost approach

    # ToDo

    # 9.7 MW capacity as standard (see literautre) hubheight=137m rotor diam=216m
    baseCap = 9.7
    baseHubHeight = 137  # literature
    baserotorDiameter = 216  # literature

    # Scaling new turbines cost according to their dimesniosn and acccording to severins calculations
    TurbineCostNew = onshore_tcc(
        capacity,
        hubheight,
        rotordiam,
        gdp_escalator=1,
        blade_material_escalator=1,
        blades=3,
    )
    TurbineCostRefernce = onshore_tcc(
        baseCap,
        baseHubHeight,
        baserotorDiameter,
        gdp_escalator=1,
        blade_material_escalator=1,
        blades=3,
    )

    costRatioTurbine = TurbineCostNew / TurbineCostRefernce
    NewTurbineCost = TurbineCostBase * costRatioTurbine

    # Found Cost Base
    # Adapting the New foundation costs
    DeptBaseCost = getRatedCostfromWaterdepth(
        averageDepthLiterature
    )  # base depth of CAPEX
    DeptPlantCost = getRatedCostfromWaterdepth(
        waterdepth
    )  # depth of calcualted power plant
    CostRatio = DeptPlantCost / DeptBaseCost
    NewFoundationCost = FoundCostbase * CostRatio

    # Cable cost and connection cost
    # applicaton of cost for DC connection from power plant to coast as scaling factor
    ratioCable = getCableCost(coastDistance, capacity) / getCableCost(
        averageCoastDistance, baseCap
    )

    NewCableCost = CableCostBase * ratioCable
    # Summing new cost components to new OffshoreCapex
    # OverheadCost are assumed to be constant

    TotalOffshoreCapEx = (
        NewTurbineCost + NewFoundationCost + NewCableCost + OverheadCostBase
    )

    return TotalOffshoreCapEx


# %%
def getRasterValueFromTifs(tiffsPath, latitude, longitude):
    for tifPath in tiffsPath:
        with rasterio.open(tifPath) as src:
            bounds = src.bounds  # left, bottom, right, top

            # Check if the point is inside this raster
            if (bounds.left <= longitude <= bounds.right) and (
                bounds.bottom <= latitude <= bounds.top
            ):
                try:
                    # Use safer sampling method
                    for val in src.sample([(longitude, latitude)]):

                        return val[0]

                except Exception as e:
                    print(f"Error reading from {tifPath}: {e}")
                    continue
    return None  # Not found in any tile


# %%
def getRatedCostfromWaterdepth(depth, allowNegative=True):
    """
    Estimate the rated cost of offshore wind turbine foundations based on water depth.

    Args:
        depth (float): Water depth at the installation site (in meters).
        allowNegative (bool): Whether negative depth values (e.g., for land) are allowed.
                              If False, raises ValueError when depth < 0.

    Returns:
        float: Rated cost in €/kW for the specified water depth.

    Reference:
        Rogeau et al. (2023), "Review and modeling of offshore wind CAPEX",
        Renewable and Sustainable Energy Reviews, DOI: 10.1016/j.rser.2023.113699
    """
    if (not allowNegative) and depth < 0:
        raise ValueError("Depth must not be negative when not allowNegative")
    if depth < 25:
        c1 = 181
        c2 = 552
        c3 = 370
    elif depth >= 25 and depth <= 55:
        c1 = 103
        c2 = -2043
        c3 = 478
    else:
        c1 = 0
        c2 = 697
        c3 = 1223

    ratedCost = c1 * (depth**2) + c2 * depth + c3 * 1000

    return ratedCost


# %%
def getCableCost(distance, capacity):
    """A function to get the cost for connecting a off shore windpower plant to the coastline.

    Parameters
    ----------
    distance :  float
                distance to caostline in km
    capacity :  float
                powerplant's capacity in MW

    ____________

    Reference:
    Rogeau et al. (2023), "Review and modeling of offshore wind CAPEX",
    Renewable and Sustainable Energy Reviews, DOI: 10.1016/j.rser.2023.113699
    """

    FixeCost = 0

    variableCost = 1.35 * distance * (capacity * 1e6)
    cableCost = FixeCost + variableCost

    return cableCost
