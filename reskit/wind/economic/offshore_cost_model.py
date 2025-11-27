import numpy as np
import os
import pickle
import glob
import geokit as gk
import warnings
import math


from reskit.default_paths import DEFAULT_PATHS
from reskit.parameters.parameters import OffshoreParameters
from reskit.util.local_values import *
from .onshore_cost_model import onshore_tcc
from reskit.parameters.parameters import OffshoreParameters


# %%
def calculateOffshoreCapex(
    baseCapex,
    capacity,
    rotorDiam,
    hubHeight,
    waterDepth,
    coastDistance,
    shareTurb=0.449,
    shareFound=0.204,
    shareCable=0.181,
    shareOverhead=0.166,
    maxMonopileDepth=25,
    maxJacketDepth=55,
    baseDepth=17,
    baseDistCoast=27,
    baseWFSize=106858,
    baseCap=None,
    baseHubHeight=None,
    baseRotorDiam=None,
    defaultOffshoreParamsFp=None,
    techYear=2050,
):
    """
    Scales a generic offshore CAPEX value based on water depth and distance to shore by taking capacity, hubheight and rotor diameter of a base case. If no base case is given, a default base case is applied.

    Parameters
    ----------
    baseCapex : float
        Reference CAPEX per kW (cost unit/kW) that should be scaled. base CApex must be given in €/kW to enable correct scaling.
    capacity : float
        Turbine rated capacity in kW.
    rotorDiam : float
        Rotor diameter in meters.
    hubHeight : float
        Hub height in meters.
    waterDepth : float
        Site-specific water depth in meters.
    coastDistance : float
        Distance from site to nearest coast in kilometers.
    shareTurb : float, optional
        Share of turbine cost in total CAPEX in the baseline turbine reference case. Default is 0.449.
    shareFound : float, optional
        Share of foundation costin total CAPEX in the baseline turbine reference case. Default is 0.204.
    shareCable : float, optional
        Share of cable/connection cost in total CAPEX in the baseline turbine reference case. Default is0.181.
    shareOverhead : float, optional
        Share of overhead/miscellaneous costs in total CAPEX in the baseline turbine reference case. Default is 0.166.
    maxMonopileDepth : float, optional
        Maximum depth for monopile foundations, by default 25.
    maxJacketDepth : float, optional
        Maximum depth for jacket foundations, by default 55.
    baseDepth : float, optional
        Reference depth in CAPEX literature, by default 17.
    baseDistCoast : float, optional
        Reference coast distance, by default 27.
    baseWFSize : int, optional
        The average wind farm size in kW, by default 106858.
    baseCap : float, optional
        Reference turbine capacity. Loaded from CSV if not provided.
    baseHubHeight : float, optional
        Reference hub height. Loaded from CSV if not provided.
    baseRotorDiam : float, optional
        Reference rotor diameter. Loaded from CSV if not provided.
    defaultOffshoreParamsFp : str, optional
        Filepath to offshore turbine parameters CSV.
    techYear : int, optional
        Year of the applied technology, by default 2050.

    Returns
    -------
    float
        Adjusted offshore wind CAPEX per kW for the given configuration. The cost unit is the same as the baseCapex.
    """
    assert np.isclose(
        shareTurb + shareFound + shareCable + shareOverhead, 1.0, rtol=1e-9
    ), "Sum of all cost shares must equal 1"

    assert 0 < maxMonopileDepth < 55, (
        "Maximum depth for monopile foundation must be between 0 and 55 m"
    )

    assert 55 <= maxJacketDepth < 100, (
        "Maximum depth for jacket foundation must be between 55 and 100 m"
    )

    assert maxMonopileDepth < maxJacketDepth, (
        "Jacket depth must be greater than monopile depth"
    )

    if any(_arg is None for _arg in [baseCap, baseHubHeight, baseRotorDiam, baseCapex]):
        params = OffshoreParameters(fp=defaultOffshoreParamsFp, year=techYear)
    elif not all(_arg is None for _arg in [defaultOffshoreParamsFp, techYear]):
        raise ValueError(
            "techYear and defaultOffshoreParamsFp are expected to be None if "
            "baseCap, baseHubHeight, baseRotorDiam and baseCapex are provided explicitly."
        )

    if baseCap is None:
        baseCap = params.base_capacity
        print("baseCap is taken from overall techno-economic file")
    if baseHubHeight is None:
        baseHubHeight = params.base_hub_height
        print("baseHubHeight is taken from overall techno-economic file")
    if baseRotorDiam is None:
        baseRotorDiam = params.base_rotor_diam
        print("baseRotorDiam is taken from overall techno-economic file")
    if baseCapex is None:
        baseCapex = params.base_capex_per_capacity
        print("inputCapex is taken from overall techno-economic file")

    turbineCostBase = baseCapex * shareTurb
    foundCostBase = baseCapex * shareFound
    cableCostBase = baseCapex * shareCable
    overheadCostBase = baseCapex * shareOverhead

    # Scale turbine cost
    turbinePlantCost = onshoreTcc(
        capacity,
        hubHeight,
        rotorDiam,
        gdpEscalator=1,
        bladeMaterialEscalator=1,
        blades=3,
    )
    turbineBaseCost = onshoreTcc(
        baseCap,
        baseHubHeight,
        baseRotorDiam,
        gdpEscalator=1,
        bladeMaterialEscalator=1,
        blades=3,
    )
    scalingFactorTurbine = turbinePlantCost / turbineBaseCost
    ScaledTurbineCost = turbineCostBase * scalingFactorTurbine

    # Scale foundation cost
    depthBaseCost = getRatedCostFromWaterDepth(
        baseDepth, maxMonopileDepth, maxJacketDepth
    )
    depthPlantCost = getRatedCostFromWaterDepth(
        waterDepth, maxMonopileDepth, maxJacketDepth
    )
    scalingFactorFoundation = depthPlantCost / depthBaseCost
    scaledFoundationCost = foundCostBase * scalingFactorFoundation

    # Scale cable cost
    convertercost_onshore = (
        getConverterStationCost(
            capacity=baseWFSize, waterDepth=None, voltageType="dc", maxJacketDepth=55
        )
        * capacity
        / baseWFSize
    )
    convertercost_offshore = (
        getConverterStationCost(
            capacity=baseWFSize,
            waterDepth=waterDepth,
            voltageType="dc",
            maxJacketDepth=55,
        )
        * capacity
        / baseWFSize
    )
    convertercost_total = convertercost_onshore + convertercost_offshore
    scalingFactorCable = getCableCost(
        distance=coastDistance, capacity=capacity, fixedCost=convertercost_total
    ) / getCableCost(
        distance=baseDistCoast, capacity=baseCap, fixedCost=convertercost_total
    )
    scaledCableCost = cableCostBase * scalingFactorCable

    # Combine all costs
    offshoreCapexNoOverhead = ScaledTurbineCost + scaledFoundationCost + scaledCableCost

    scaledOverheadCost = offshoreCapexNoOverhead * (shareOverhead / (1 - shareOverhead))

    totalOffshoreCapex = (
        ScaledTurbineCost + scaledFoundationCost + scaledCableCost + scaledOverheadCost
    )

    return totalOffshoreCapex


# %%
def getRatedCostFromWaterDepth(
    depth,
    maxMonopileDepth=25,
    maxJacketDepth=55,
):
    """
    Estimates the rated cost of offshore wind turbine foundations based on water depth.

    Parameters
    ----------
    depth : float
        Water depth at the installation site (in meters).
    maxMonopileDepth : float, optional
        Threshold depth for monopile foundations, by default 25.
    maxJacketDepth : float, optional
        Threshold depth for jacket foundations, by default 55.

    Returns
    -------
    float
        Rated cost in €_2023/kW.

    References
    ----------
    Rogeau et al. (2023), Renewable and Sustainable Energy Reviews.
    """
    depth = abs(depth)

    if depth < maxMonopileDepth:
        c1, c2, c3 = 181, 552, 370
    elif depth <= maxJacketDepth:
        c1, c2, c3 = 103, -2043, 478
    else:
        c1, c2, c3 = 0, 697, 1223

    return c1 * depth**2 + c2 * depth + c3 * 1000


# %%
def getCableCost(
    distance,
    capacity,
    variableCostFactor=1.350,
    fixedCost=0,
):
    """
    Calculates the cost for connecting an offshore wind power plant to the coastline.

    Parameters
    ----------
    distance : float
        Distance to coastline in kilometers.
    capacity : float
        Power plant's capacity in kW.
    variableCostFactor : float, optional
        Cost multiplier in €/kW/km, by default 1.350
    fixedCost : float, optional
        Fixed connection cost in the respective currency unit. Defaults to 0 [€]

    Returns
    -------
    float
        Total cable connection cost in monetary units.

    References
    ----------
    Rogeau et al. (2023), "Review and modeling of offshore wind CAPEX",
    Renewable and Sustainable Energy Reviews, DOI: 10.1016/j.rser.2023.113699
    """
    assert distance >= 0, "distance must be larger or equal to 0"
    assert capacity > 0, " turbine capacity must be larger than 0"
    assert variableCostFactor > 0, "cost factor must be larger tan 0"
    assert fixedCost >= 0, "fixed Cost must be postive or 0"

    variableCost = variableCostFactor * distance * capacity
    cableCost = fixedCost + variableCost

    return cableCost


def getPlatformCost(
    capacity,
    applicationType,
    waterDepth,
    foundationType=None,
    maxJacketDepth=55,
    convention="RogeauEtAl2023",
):
    """
    Returns the cost of an offshore foundation for offshore substations or
    electrolysis depending on application type, water depth and installed
    capacity.

    capacity : int, float
        The installed capacity in [kW] for the respective application.
    applicationType : str
        The type of application that shall be installed on the platform,
        e.g. "ac" (substation), "dc" (substation) or (offshore) "electrolysis".
    waterDepth : int
        The location water depth in [m].
    maxJacketDepth : int, optional
        The max. possible jacket foundation depth in [m]. By default 55 m
        following [1].
    convention : str, optional
        The convention by which the foundation cost shall be determined,
        e.g. "RogeauEtAl2023" based on the equations in [1].

    [1] Rogeau, Antoine; Vieubled, Julien; Coatpont, Matthieu de; Affonso
    Nobrega, Pedro; Erbs, Guillaume; Girard, Robin (2023): Techno-economic
    evaluation and resource assessment of hydrogen production through
    offshore wind farms: A European perspective. In Renewable and
    Sustainable Energy Reviews 187, p. 113699. DOI: 10.1016/j.rser.2023.113699.
    """
    assert isinstance(waterDepth, (int, float)) and waterDepth > 0, (
        f"waterDepth must be an int or float > 0 m"
    )
    if convention == "RogeauEtAl2023":
        # platform cost factors per type, see table (5)
        RCPF_factors = {
            "jacket": {"c2": 233, "c3": 47},
            "floating": {"c2": 87, "c3": 68},
        }
        UCPF_factors = {
            "jacket": {"c2": 309, "c3": 62},
            "floating": {"c2": 116, "c3": 91},
        }
        assert sorted(RCPF_factors.keys()) == sorted(UCPF_factors.keys())  # make sure

        # relative theoretical power based on power densities/footprints, eq.(9)
        powerDensity_factors = {
            "dc": capacity / 1000,
            "ac": 0.5 * capacity / 1000,
            "electrolysis": 2 * capacity / 1000,
        }

        # get and check foundation type
        if foundationType is None:
            foundationType = "jacket" if waterDepth <= maxJacketDepth else "floating"
        elif not all(
            [foundationType in _dict for _dict in [RCPF_factors, UCPF_factors]]
        ):
            raise ValueError(
                f"foundationType must be in: {', '.join(RCPF_factors.keys())}"
            )
        elif waterDepth > maxJacketDepth and foundationType == "jacket":
            warnings.warn(
                f"waterDepth ({waterDepth} m) exceeds maxJacketDepth ({maxJacketDepth} m) but 'jacket' is enforced as foundationType."
            )

        # check voltage type
        if not all([applicationType in _dict for _dict in [powerDensity_factors]]):
            raise ValueError(
                f"applicationType must be in: {', '.join(powerDensity_factors.keys())}"
            )

        # get RCPF and UCPF for equation () as per equation ()
        RCPF = (
            RCPF_factors[foundationType]["c2"] * waterDepth
            + RCPF_factors[foundationType]["c3"] * 10**3
        )
        UCPF = (
            UCPF_factors[foundationType]["c2"] * waterDepth
            + UCPF_factors[foundationType]["c3"] * 10**3
        )

        # eq.(9) get relative theoretical power, based on power densities/footprints, see [1]
        P_wf_ = powerDensity_factors[applicationType]

        # calculate final platform cost as the sum of capacity-dependent and fixed cost as per eq. (8)
        ECPF = RCPF * P_wf_ + UCPF

        return ECPF
    else:
        raise NotImplementedError(f"convention '{convention}' is not implemented.")


def getConverterStationCost(
    capacity,
    waterDepth,
    voltageType="dc",
    maxJacketDepth=55,
    convention="RogeauEtAl2023",
):
    """
    Calculates the cost of an onshore or offshore converter station for AC or DC
    connections of wind farms.

    capacity : int
        Converter station power in [kW].
    waterDepth : int
        The water depth in case of an offshore substation, does then include the
        platform cost of either jacket or floating type depending on depth. If
        None, an onshore substation without platform will be assumed.
    voltageType : str, optional
        If the substation is for "ac"or "dc" connections, by default "dc".
    maxJacketDepth : int, optional
        The max. possible jacket foundation depth in [m]. By default 55 m
        following [1].
    convention : str, optional
        The convention by which the foundation cost shall be determined,
        e.g. "RogeauEtAl2023" based on the equations in [1].

    [1] Rogeau, Antoine; Vieubled, Julien; Coatpont, Matthieu de; Affonso
    Nobrega, Pedro; Erbs, Guillaume; Girard, Robin (2023): Techno-economic
    evaluation and resource assessment of hydrogen production through
    offshore wind farms: A European perspective. In Renewable and
    Sustainable Energy Reviews 187, p. 113699. DOI: 10.1016/j.rser.2023.113699.
    """
    if convention == "RogeauEtAl2023":
        # calculate electrical powerstation cost based on equation (10) and table 6
        RCPS = {"ac": 22.87, "dc": 102.93}  # EUR/kW
        UCPS = {"ac": 3.1750000, "dc": 7.060000}  # EUR
        assert sorted(RCPS.keys()) == sorted(UCPS.keys())  # make sure
        if not voltageType in RCPS:
            raise ValueError(
                f"unknown voltageType, select from: {', '.join(RCPS.keys())}"
            )

        ECPS = RCPS[voltageType] * capacity + UCPS[voltageType] * 10**3

        if waterDepth is None:
            # an onshore station, no additional platform cost
            ECPF = 0
        else:
            # get platform cost from separate function
            ECPF = getPlatformCost(
                capacity=capacity,
                applicationType=voltageType,
                waterDepth=waterDepth,
                foundationType=None,
                maxJacketDepth=maxJacketDepth,
                convention=convention,
            )

    return ECPS + ECPF


def onshoreTcc(
    cp,
    hh,
    rd,
    gdpEscalator=None,
    bladeMaterialEscalator=None,
    blades=None,
):
    """
    Calculates the turbine capital cost (TCC) of a 3-blade onshore wind turbine.

    Parameters
    ----------
    cp : float
        Turbine capacity in kW.
    hh : float
        Hub height in meters.
    rd : float
        Rotor diameter in meters.
    gdpEscalator : float, optional
        Labor cost escalator, by default taken from OffshoreParameters.
    bladeMaterialEscalator : float, optional
        Blade material cost escalator, by default taken from OffshoreParameters.
    blades : int, optional
        Number of blades, by default taken from OffshoreParameters.

    Returns
    -------
    float
        Turbine capital cost (TCC) in monetary units.

    References
    ----------
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

    # hub
    hubMass = 0.945 * singleBladeMass + 5680.3
    hubCost = hubMass * 4.25

    # Pitch and bearings
    pitchSystemCost = 2.28 * (0.2106 * np.power(rd, 2.6578))

    # Spinner and nosecone
    noseConeMass = 18.5 * rd - 520.5
    noseConeCost = noseConeMass * 5.57

    # Low Speed Shaft
    lowSpeedShaftCost = 0.01 * np.power(rd, 2.887)

    # Main bearings
    bearingMass = (rd * 8 / 600 - 0.033) * 0.0092 * np.power(rd, 2.5)
    bearingCost = 2 * bearingMass * 17.6

    # Gearbox
    # Gearbox not included for direct drive turbines

    # Break, coupling, and others
    breakCouplingCost = 1.9894 * cp - 0.1141

    # Generator (Assuming direct drive)
    generatorCost = cp * 219.33

    # Electronics
    electronicsCost = cp * 79

    # Yaw drive and bearing
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
    hydraulicAndCoolingSystemCost = cp * 12

    # Nacelle Cover
    nacelleCost = 11.537 * cp + 3849.7

    # Tower
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


def installationCostTurbine(
    waterDepth=17,  # m
    maxJacketDepth=55,  # m
    vesselCapacity=7,  # units/lift
    portDistance=17,  # km
    vesselSpeed=18.5,  # km/h
    loadingTime=30,  # h
    installationTime=90,
    numberOfWindTurbines=2,  # numberofWIndturbines in WIndpark
    dayRate=40000,  # €/d,
    turbineCapacity=15,  # MW
    vesselCapacityFixedFoundationTurbines=40,  # MW transported per lift by Rougeau et al.
):
    """
    Calculates the installation cost per turbine for an offshore wind farm, based on
    water depth, vessel logistics, loading and installation times, and turbine capacity.

    Parameters
    ----------
    waterDepth : float, optional
        Water depth at the installation site in meters. Determines whether turbines
        use fixed foundations or anchorage systems. Default is 17 m.
    maxJacketDepth : float, optional
        Maximum feasible water depth for jacket-based fixed foundations. If exceeded,
        anchorage-based platforms are assumed. Default is 55 m.
    vesselCapacity : int, optional
        Number of turbines that can be transported per lift for anchorage cases.
        Overridden internally for fixed-foundation installations. Default is 7 units/lift.
    portDistance : float, optional
        Round-trip transit distance between port and installation site in kilometers.
        Default is 17 km.
    vesselSpeed : float, optional
        Vessel transit speed in km/h. Default is 18.5 km/h.
    loadingTime : float, optional
        Loading time per lift in hours. Default is 30 h; overridden internally.
    installationTime : float, optional
        Installation time per turbine in hours. Default is 90 h; overridden internally.
    numberOfWindTurbines : int, optional
        Total number of turbines to install in the wind farm. Default is 2.
    dayRate : float, optional
        Vessel day rate in euros/day. Default is €40,000/day; overridden internally.
    turbineCapacity : float, optional
        Rated power per turbine in MW. Default is 15 MW.
    vesselCapacityFixedFoundationTurbines : float, optional
        Maximum turbine capacity (MW) that can be transported in one lift for fixed foundations,
        based on Rougeau et al. The vessel capacity is computed as:
        floor(vesselCapacityFixedFoundationTurbines / turbineCapacity).
        Default is 40 MW.

    Returns
    -------
    float
        Installation cost per turbine, in euros.

    """
    assert 0 <= waterDepth, (
        f"waterdepth is {waterDepth}, but must not be smaller than 0."
    )

    if waterDepth <= maxJacketDepth:
        platformType = "fixed"
    else:
        platformType = "anchorages"

    if platformType == "fixed":
        VesselSpeed = 18.5
        loadingTime = 24
        installationTime = 144
        dayRate = 200000
        vesselCapacity = math.floor(
            vesselCapacityFixedFoundationTurbines / turbineCapacity
        )

    else:
        VesselSpeed = 18.5
        loadingTime = 30
        installationTime = 90
        dayRate = 40000
        vesselCapacity = 7

    # calculate the number of requried lifts for the whole windpark

    liftNumber = numberOfWindTurbines / vesselCapacity
    turbineInstallationCost = (
        liftNumber * ((2 * portDistance / VesselSpeed) + loadingTime)
        + installationTime * numberOfWindTurbines
    ) * (dayRate / 24)  #  € installation Cost for whole windpark

    installedPowerPerWindpark = numberOfWindTurbines * turbineCapacity

    turbineInstallationCostperTurbine = (
        turbineInstallationCost / numberOfWindTurbines
    )  # Euro/Turbine
    turbineInstallationCostperCapacity = (
        turbineInstallationCost / installedPowerPerWindpark
    )  # Euro/MW

    return turbineInstallationCostperTurbine


def installationCostPlattform(
    waterDepth=17,  # m
    maxJacketDepth=55,  # m
    vesselCapacity=3,  # units/lift
    portDistance=17,  # km
    vesselSpeed=18.5,  # km/h
    loadingTime=30,  # h
    installationTime=90,
    numberOfPlatforms=1,  # numberofWIndturbines in WIndpark
    dayRate=40000,  # €/d,
    turbineCapacity=15,  # MW
    numberOfWindTurbines=3,
):
    """
    Calculates the installation cost per turbine for offshore platforms (fixed or floating),
    based on site conditions, vessel operations, and logistical parameters.

    Parameters
    ----------
    waterDepth : float, optional
        Water depth at the installation site in meters. Determines whether a fixed
        platform (jacket) or a floating anchorage system is used. Default is 17 m.
    maxJacketDepth : float, optional
        Maximum feasible water depth for jacket foundations. If exceeded, the
        platform type switches to anchorage-based. Default is 55 m.
    vesselCapacity : int, optional
        Number of platform units that can be transported per vessel trip. Default is 3.
        This value is overridden internally depending on platform type.
    portDistance : float, optional
        Distance from port to the installation site in kilometers. Default is 17 km.
    vesselSpeed : float, optional
        Vessel transit speed in km/h. Default is 18.5 km/h.
    loadingTime : float, optional
        Time required to load one vessel trip in hours. Default is 30 h.
        This value is overridden internally depending on platform type.
    installationTime : float, optional
        Time required to install one platform in hours. Default is 90 h.
        This value is overridden internally depending on platform type.
    numberOfPlatforms : int, optional
        Total number of platforms to be installed (usually equal to the number
        of turbines). Default is 1.
    dayRate : float, optional
        Vessel day rate in euros/day. Default is €40,000/day.
        This value is overridden internally depending on platform type.
    turbineCapacity : float, optional
        Rated power of one turbine in MW. Provided for compatibility; not used
        in calculations. Default is 15 MW.
    numberOfWIndTurbines : int, optional
        Number of turbines in the farm. Provided for compatibility; not used
        in calculations. Default is 3.

    Returns
    -------
    float
        Installation cost per turbine in euros.


    """

    assert 0 <= waterDepth, (
        f"waterdepth is {waterDepth}, but must not be smaller than 0."
    )

    if waterDepth <= maxJacketDepth:
        platformType = "fixed"
    else:
        platformType = "anchorages"

    if platformType == "fixed":
        vesselSpeed = 18.5
        loadingTime = 24
        installationTime = 96
        dayRate = 200000
        vesselCapacity = 1

    else:
        vesselSpeed = 18.5
        loadingTime = 30
        installationTime = 90
        dayRate = 40000
        vesselCapacity = 3

    # calculate the number of requried lifts for the whole windpark

    liftNumber = numberOfPlatforms / vesselCapacity
    platformInstallationCost = (
        liftNumber * ((2 * portDistance / vesselSpeed) + loadingTime)
        + installationTime * numberOfPlatforms
    ) * (dayRate / 24)  #  € installation Cost for whole windpark

    platformInstallationCostperTurbine = (
        platformInstallationCost / numberOfWindTurbines
    )  # Euro/Turbine

    platformInstallationCostperCapacity = platformInstallationCost / (
        numberOfWindTurbines * turbineCapacity
    )  # Euro/MW

    return platformInstallationCostperTurbine


# TODO considering how to include cable cost.... seems to be highly complicated and is implmened wrongly here so far
def installationCostCables(
    lengthOfCables=17000,  # m
    voltageType="dc",
):
    """
    Calculates the installation cost of offshore cables based on cable length and voltage type.

    Parameters
    ----------
    lengthOfCables : float, optional
        Total length of cables to be installed, in meters.
        Default is 17,000 m.
    voltageType : str, optional
        Voltage type of the cable system. Must be one of:
        - "dc"          (direct current export cables)
        - "ac"          (alternating current export cables)
        - "interarray"  (inter-array cables within the wind farm)
        Default is "dc".

    Returns
    -------
    float
        Total cable installation cost in monetary units.

    """

    assert voltageType in ["dc", "ac", "interarray"], (
        "voltatgeType must be ac, dc, or interarray"
    )

    if voltageType == "dc":
        linearInstallationCost = 1

    if voltageType == "ac":
        linearInstallationCost = 1

    if voltageType == "interarray":
        linearInstallationCost = 1

    CableInstallationCost = linearInstallationCost * lengthOfCables

    return CableInstallationCost


def installationPipelineCost(pipelineType="export", lengthOfPipe=17000):
    """
    Calculates the installation cost of an offshore pipeline based on pipeline type and length.

    Parameters
    ----------
    pipelineType : str, optional
        Type of pipeline being installed. Must be either "infield" or "export".
        Determines the lay rate (m/day) and vessel day rate (€/day).
        Default is "export".
    lengthOfPipe : float, optional
        Total length of the pipeline in meters. Default is 17,000 meters.

    Returns
    -------
    float
        Total pipeline installation cost in euros.

    -------
    Source:    [1] Rogeau, Antoine; Vieubled, Julien; Coatpont, Matthieu de; Affonso
    Nobrega, Pedro; Erbs, Guillaume; Girard, Robin (2023): Techno-economic
    evaluation and resource assessment of hydrogen production through
    offshore wind farms: A European perspective. In Renewable and
    Sustainable Energy Reviews 187, p. 113699. DOI: 10.1016/j.rser.2023.113699.
    """

    assert pipelineType in ["infield", "export"], (
        "pipelineType must be in ['infield','export]"
    )

    if pipelineType == "infield":
        layRate = 7000  # meter/day
        dayrateVessel = 400000  # euro/day

    if pipelineType == "export":
        layRate = 4000  # meter/day
        dayrateVessel = 700000  # euro/day
    PipelineInstallationCOst = lengthOfPipe / layRate * dayrateVessel

    return PipelineInstallationCOst
