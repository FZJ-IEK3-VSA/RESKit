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
from reskit.parameters.parameters import  OffshoreParameters


# %%


def waterDepthFromLocation(
    lat,
    lon,
    waterDepthFolderPath=DEFAULT_PATHS.get("waterdepthFile"),
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
# if you want to execute the distance to coastline more often, please separete the loading of the taserband to increase execution time


def loadDistanceBand(
    path=DEFAULT_PATHS.get("distancetoCoast"),
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
    
    capacity,
    hubheight,
    waterdepth,
    coastDistance,
    rotordiam,
    tech_year=2050,
    shareTurb=0.449,
    shareFound=0.204,
    shareCable=0.181,
    shareOverhead=0.166,
    maxMonopolDepth=25,
    maxJacketDepth=55,
    litValueAvgDepth=17,
    litValueAvgDistCoast=27,
    InputCapex=None,
    baseCap=None,
    baseHubHeight=None,
    baseRotorDiam=None,
    defaultOffshoreParamsFp=None,
):
    """
    Scale a generic offshore CAPEX value based on water depth and distance to shore.

    The function splits the total input CAPEX into major cost components (turbine, foundation,
    cable, overhead) and scales each individually based on project-specific parameters such as
    turbine size, water depth, and coastline distance.
    Parameters:
    -----------
    capacity : float
        Turbine rated capacity in MW.

    hubheight : float
        Hub height in meters.

    waterdepth : float
        Site-specific water depth in meters.

    coastDistance : float
        Distance from site to nearest coast in kilometers.

    rotordiam : float
        Rotor diameter in meters.
    
        techyear: int
        specifies the year of the technologie that is applied. (default = 2030)

    shareTurb : float, optional
        Share of turbine cost in total CAPEX (default = 0.449).

    shareFound : float, optional
        Share of foundation cost in total CAPEX (default = 0.204).

    shareCable : float, optional
        Share of cable/connection cost in total CAPEX (default = 0.181).

    shareOverhead : float, optional
        Share of overhead/miscellaneous costs in total CAPEX (default = 0.166).

    maxMonopolDepth : float, optional
        Maximum depth (in meters) for monopile foundations (default = 25).

    maxJacketDepth : float, optional
        Maximum depth (in meters) for jacket foundations (default = 55).

    litValueAvgDepth : float, optional
        Literature-based average depth used in reference CAPEX (default = 17).

    litValueAvgDistCoast : float, optional
        Literature-based average distance to coast for reference CAPEX (default = 27).

    InputCapex : float, optional
        Reference total CAPEX per kW (€/kW). If not provided, it defaults to the value
        from the OffshoreParameters CSV for the given year.

    baseCap : float, optional
        Reference turbine capacity in MW. If not provided, it's loaded from OffshoreParameters.

    baseHubHeight : float, optional
        Reference hub height in meters. If not provided, it's loaded from OffshoreParameters.

    baseRotorDiam : float, optional
        Reference rotor diameter in meters. If not provided, it's loaded from OffshoreParameters.

    defaultOffshoreParamsFp : str, optional
        Optional filepath to a CSV containing default offshore turbine parameters.
        If not provided, a built-in RESKit default is used.

    Returns:
    --------
    float
        Adjusted offshore wind plant CAPEX in €/kW for the given configuration.
    """

    assert np.isclose(
        shareTurb + shareFound + shareCable + shareOverhead, 1.0, rtol=1e-9
    ), "Sum of all cost shares must equal 1"
    assert (
        0 < maxMonopolDepth < 55
    ), "Maximum Depth for Monopile Foundation must be between 0 and 55 m"
    assert (
        55 <= maxJacketDepth < 100
    ), "Maximum Depth for Jacket Foundation must be between 0 and 55 m"
    assert (
        maxMonopolDepth < maxJacketDepth
    ), " Maximum Depth for Jacket Foundation must be larger than maximum  depth for Monopile Foundation"

    # Loading tech-specific parameters
    params = OffshoreParameters(fp=defaultOffshoreParamsFp, year=tech_year)

    #falling back to standard values if no refernce values are given for the calculation

    if baseCap is None:
        baseCap = params.base_capacity
        print('baseCap is taken from overall techno-economic file')
    if baseHubHeight is None:
        baseHubHeight = params.base_hub_height
        print('baseHubHeight is taken from overall techno-economic file')
    if baseRotorDiam is None:
        baseRotorDiam = params.base_rotor_diam
        print('baseRotorDiam is taken from overall techno-economic file')
    
    if InputCapex is None:
        InputCapex = params.base_capex_per_capacity
        print('InputCapes is taken from overall techno-economic file')
    



    TurbineCostBase = InputCapex * shareTurb
    FoundCostbase = InputCapex * shareFound
    CableCostBase = InputCapex * shareCable
    OverheadCostBase = InputCapex * shareOverhead

    # scaling each cost share regarding thecurrent wint turbine settings and the reference settings


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
        baseRotorDiam,
        gdp_escalator=1,
        blade_material_escalator=1,
        blades=3,
    )

    costRatioTurbine = TurbineCostNew / TurbineCostRefernce
    NewTurbineCost = TurbineCostBase * costRatioTurbine
    

    # Found Cost Base
    # Adapting the New foundation costs
    DeptBaseCost = getRatedCostfromWaterdepth(
        litValueAvgDepth
    )  # base depth of CAPEX
    DeptPlantCost = getRatedCostfromWaterdepth(
        waterdepth
    )  # depth of calcualted power plant
    CostRatio = DeptPlantCost / DeptBaseCost
    NewFoundationCost = FoundCostbase * CostRatio

    # Cable cost and connection cost
    # applicaton of cost for DC connection from power plant to coast as scaling factor
    ratioCable = getCableCost(coastDistance, capacity) / getCableCost(
        litValueAvgDistCoast, baseCap
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
def getCableCost(distance, capacity,variableCostFactor=1.35, fixedCost=0):
    """A function to get the cost for connecting a off shore windpower plant to the coastline.

    Parameters
    ----------
    distance :                      float
                                    distance to caostline in km
    capacity :                      float
                                    powerplant's capacity in MW
    variableCostFactor (optional):  float
                                    by default=1.35 in Euro_2022/W/km
                                    Cost factor used to scale distance to shore and waterdepth into site-specific cable cost

    ____________

    Reference:
    [1] Rogeau et al. (2023), "Review and modeling of offshore wind CAPEX",
    Renewable and Sustainable Energy Reviews, DOI: 10.1016/j.rser.2023.113699
    """
    #assert....

    variableCost = variableCostFactor* distance * capacity 
    cableCost = fixedCost + variableCost

    return cableCost


def onshore_tcc(
    cp, hh, rd, gdp_escalator=None, blade_material_escalator=None, blades=None
):
    """
    A function to determine the turbine capital cost (TCC) of a 3 blade standar onshore wind turbine based capacity, hub height and rotor diameter values according to the cost model by Fingersh et al. [1].

    Parameters
    ----------
    cp : numeric or array-like
        Turbine's capacity in kW
    hh : numeric or array-like
        Turbine's hub height in m
    rd : numeric or array-like
        Turbine's rotor diamter in m
    gdp_escalator : int, optional
        Labor cost escalator, by default 1
    blade_material_escalator : int, optional
        Blade material cost escalator, by default 1
    blades : int, optional
        Number of blades, by default 3

    Returns
    -------
    numeric or array-like
        Turbine's turbine capital cost (TCC) in monetary units.

    References
    ---------
    [1] Fingersh, L., Hand, M., & Laxson, A. (2006). Wind Turbine Design Cost and Scaling Model. NREL. https://www.nrel.gov/docs/fy07osti/40566.pdf

    """
    # initialize OnshoreParameters class and feed with custom param values
    if gdp_escalator is None or blade_material_escalator is None or  blades is None:
        O
        offshoreParams = OffshoreParameters()
        
        gdp_escalator=offshoreParams.gdp_escalator
        blade_material_escalator=offshoreParams.blade_material_escalator
        blades=offshoreParams.blades


    rr = rd / 2
    sa = np.pi * rr * rr

    # Blade Cost
    singleBladeMass = 0.4948 * np.power(rr, 2.53)
    singleBladeCost = (
        (0.4019 * np.power(rr, 3) - 21051) * blade_material_escalator
        + 2.7445 * np.power(rr, 2.5025) * gdp_escalator
    ) * (1 - 0.28)

    # Hub
    hubMass = 0.945 * singleBladeMass + 5680.3
    hubCost = hubMass * 4.25

    # Pitch and bearings
    # pitchBearingMass = 0.1295 * (singleBladeMass * blades) + 491.31
    # pitchSystemMass = pitchBearingMass*1.328+555
    pitchSystemCost = 2.28 * (0.2106 * np.power(rd, 2.6578))

    # Spinner and nosecone
    noseConeMass = 18.5 * rd - 520.5
    noseConeCost = noseConeMass * 5.57

    # Low Speed Shaft
    # lowSpeedShaftMass = 0.0142 * np.power(rd, 2.888)
    lowSpeedShaftCost = 0.01 * np.power(rd, 2.887)

    # Main bearings
    bearingMass = (rd * 8 / 600 - 0.033) * 0.0092 * np.power(rd, 2.5)
    bearingCost = 2 * bearingMass * 17.6

    # Gearbox
    # Gearbox not included for direct drive turbines

    # Break, coupling, and others
    breakCouplingCost = 1.9894 * cp - 0.1141
    # breakCouplingMass = breakCouplingCost/10

    # Generator (Assuming direct drive)
    # generatorMass = 6661.25 * np.power(lowSpeedShaftTorque, 0.606) # wtf is the torque?
    generatorCost = cp * 219.33

    # Electronics
    electronicsCost = cp * 79

    # Yaw drive and bearing
    # yawSystemMass = 1.6*(0.0009*np.power(rd, 3.314))
    yawSystemCost = 2 * (0.0339 * np.power(rd, 2.964))

    # Mainframe (Assume direct drive)
    mainframeMass = 1.228 * np.power(rd, 1.953)
    mainframeCost = 627.28 * np.power(rd, 0.85)

    # Platform and railings
    platformAndRailingMass = 0.125 * mainframeMass
    platformAndRailingCost = platformAndRailingMass * 8.7

    # Electrical Connections
    electricalConnectionCost = cp * 40

    # Hydraulic and Cooling systems
    # hydraulicAndCoolingSystemMass = 0.08 * cp
    hydraulicAndCoolingSystemCost = cp * 12

    # Nacelle Cover
    nacelleCost = 11.537 * cp + 3849.7
    # nacelleMass = nacelleCost/10

    # Tower
    towerMass = 0.2694 * sa * hh + 1779
    towerCost = towerMass * 1.5

    # Add up the turbine capital cost
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
