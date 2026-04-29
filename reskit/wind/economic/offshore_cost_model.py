import numpy as np
import warnings

from reskit.parameters.parameters import OffshoreParameters
from reskit.util.local_values import *
from reskit.wind.economic.onshore_cost_model import onshore_tcc


# %%
def calculateSpecificOffshoreCapex(
    baseSpecCapex,
    capacity,
    rotorDiam,
    hubHeight,
    waterDepth: float | int,
    coastDistance: float | int,
    portDistance: float | int = None,
    maxMonopileDepth=25,
    maxJacketDepth=55,
    maxJacketDepthPlatform=150,
    baseDepth=17,
    baseDistCoast=27,
    baseWFSize=180000,
    baseCap=None,
    baseHubHeight=None,
    baseRotorDiam=None,
    defaultOffshoreParamsFp=None,
    techYear=2050,
    voltageType="optimal",
):
    """
    Scales a generic offshore CAPEX value based on water depth and distance to shore by taking capacity, hubheight and rotor diameter of a base case. If no base case is given, a default base case is applied.

    Parameters
    ----------
    baseSpecCapex : float
        Reference custom CAPEX per kW [cost unit/kW] that should be scaled,
        incl. overhead.
    capacity : float
        Turbine rated capacity in [kW].
    rotorDiam : float
        Rotor diameter in [m].
    hubHeight : float
        Hub height in [m].
    waterDepth : float
        Site-specific water depth in [m].
    coastDistance : float
        Distance from site to nearest coast in [km].
    portDistance : float
        Distance from site to nearest base port suitable as installation and maintenance base in [km].
    maxMonopileDepth : float, optional
        Maximum depth for monopile foundations in  [m], by default 25.
    maxJacketDepth : float, optional
        Maximum depth for jacket foundations in [m], by default 55.
    maxJacketDepthPlatform : float, optional
        Maximum depth for installing offshore platforms (e.g. for offshore converter stations) in [m], by default 150.
    baseDepth : float, optional
        Reference depth in [m], by default 17.
    baseDistCoast : float, optional
        Reference coast distance in [km], by default 27.
    baseWFSize : int, optional
        The average wind farm size in [kW], by default 180000 (based on average
        extracted from processed theWindPower.net database v2025/07).
    baseCap : float, optional
        Reference turbine capacity in [kW]. Loaded from CSV if not provided.
    baseHubHeight : float, optional
        Reference hub height in [m]. Loaded from CSV if not provided.
    baseRotorDiam : float, optional
        Reference rotor diameter in [m]. Loaded from CSV if not provided.
    defaultOffshoreParamsFp : str, optional
        Filepath to offshore turbine parameters CSV.
    techYear : int, optional
        Year of the applied technology, by default 2050.

    Returns
    -------
    float
        Adjusted offshore wind CAPEX per kW for the given configuration. The cost unit is the same as the baseSpecCapex.
    """
    # CHECK AND PREPROCESS INPUTS

    # scalar inputs
    assert isinstance(maxMonopileDepth, int), "maxMonopileDepth must be an integer value"
    assert maxMonopileDepth < maxJacketDepth, "Jacket depth must be greater than monopile depth"
    # vectorized inputs
    waterDepth = np.atleast_1d(waterDepth)
    assert (waterDepth >= 0).all(), "waterDepth must be an integer, float or np.ndarray with all values >= 0"
    coastDistance = np.atleast_1d(coastDistance)
    assert (coastDistance >= 0).all(), "coastDistance must be an integer, float or np.ndarray with all values >= 0"
    if portDistance is None:
        portDistance = coastDistance
    else:
        portDistance = np.atleast_1d(portDistance)
    assert (portDistance >= 0).all(), "portDistance must be an integer, float or np.ndarray with all values >= 0"
    assert (portDistance >= coastDistance).all(), "portDistance cannot be smaller than coastDistance for any location."

    # convert all other location-specific parameters to a np array for vectorized handling in subfunctions
    capacity = np.atleast_1d(capacity)
    hubHeight = np.atleast_1d(hubHeight)
    rotorDiam = np.atleast_1d(rotorDiam)
    # voltageType = voltageType

    # GET TURBINE DEFAULT PARAMETERS IF NEEDED

    if any(_arg is None for _arg in [baseCap, baseHubHeight, baseRotorDiam, baseSpecCapex]):
        params = OffshoreParameters(fp=defaultOffshoreParamsFp, year=techYear)
    elif defaultOffshoreParamsFp is not None:
        raise ValueError(
            "defaultOffshoreParamsFp is expected to be None if baseCap, "
            "baseHubHeight, baseRotorDiam and baseSpecCapex are provided explicitly."
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
    if baseSpecCapex is None:
        baseSpecCapex = params.base_capex_per_capacity
        print("inputCapex is taken from overall techno-economic file")

    # PREPARE TURBINE COST FUNCTIONS

    # Note: Rogeau et al. do not differentiate between different turbine sizes/designs, only have avg. spec. capex and avg. capacity per year
    # therefore use turbine cost scaling developed for onshore (RELATIVE effects depend on mechanics and geometry and can be considered similar enough)
    # But absolute values need to be corrected to offshore level to be able to add offshore installation cost
    # calculate a correction factor between Rogeau et al's annual CAPEX (for the turbine = machine only!) and the onshore_tcc value for a turbine with Rogeau's annual capacity
    RogeauEtAlTurbineData = {
        2020: (8000, 1500),
        2030: (15000, 1200),
        2050: (20000, 1000),
    }  # (capacity [kW], spec.CAPEX [EUR/kW])
    assert min(RogeauEtAlTurbineData.keys()) <= techYear <= max(RogeauEtAlTurbineData.keys()), (
        f"techYear {techYear} is outside of the range considered by Rogeau et al.: 2020-2050"
    )
    yearBefore = max((y for y in RogeauEtAlTurbineData.keys() if y <= techYear))
    yearAfter = min((y for y in RogeauEtAlTurbineData.keys() if y >= techYear))
    capacityRogeau = RogeauEtAlTurbineData[yearBefore][0]  # use the yearBefore value, is the same as yearAfter
    capexRogeau = RogeauEtAlTurbineData[yearBefore][1]
    if yearAfter > yearBefore:
        capacityRogeau = capacityRogeau + (
            RogeauEtAlTurbineData[yearAfter][0] - RogeauEtAlTurbineData[yearBefore][0]
        ) * (techYear - yearBefore) / (yearAfter - yearBefore)
        capexRogeau = capexRogeau + (RogeauEtAlTurbineData[yearAfter][1] - RogeauEtAlTurbineData[yearBefore][1]) * (
            techYear - yearBefore
        ) / (yearAfter - yearBefore)
    # Rogeau et al provide only capacity per year, rotor diam and hub height need to be estimated for a typical turbine
    # assume constant spec. power of 350 W/m² (typical offshore) and hub height of 30.5m + rotor radius (see dissertation Winkler)
    rotordiamRogeau = np.sqrt((capacityRogeau * 1000 / 350) / (np.pi)) * 2  # spec power 350 W/m2
    hubheightRogeau = 30.5 + rotordiamRogeau / 2  # in mtrs
    # now calculate the correction factor to align onshore_tcc with Rogeau's value for that year (contains also currency conversion/inflation)
    offshoreCorrfacRogeau = capexRogeau / (
        onshore_tcc(
            cp=capacityRogeau,
            hh=hubheightRogeau,
            rd=rotordiamRogeau,
            gdp_escalator=1,
            blade_material_escalator=1,
            blades=3,
        )
        / capacityRogeau
    )

    # define a turbine installation cost function based on Rogeau et al. section 3.2.1
    # Define the function inside this function so it cannot be used separately by others, since it is not really related to the other fixed-cost components.
    def _getSpecificTurbineInstallCost(_depth):
        # enforce the same shape for all variables
        _depth = np.asarray(_depth, dtype=float)
        if _depth.size == 1:
            _depth = np.full_like(capacity, _depth.item())
        elif _depth.shape != capacity.shape:
            raise ValueError("depth must be scalar or same shape as capacity")
        _portDistance = np.asarray(portDistance, dtype=float)
        if _portDistance.size == 1:
            _portDistance = np.full_like(capacity, _portDistance.item())
        elif _portDistance.shape != capacity.shape:
            raise ValueError("portDistance must be scalar or same shape as capacity")
        # determine fixed vs floating locations
        _depth = np.broadcast_to(_depth, capacity.shape)
        isFixed = _depth <= maxJacketDepth
        # initiate a container for installation cost
        instCost = np.zeros(_depth.shape, dtype=float)
        # first deal with fixed foundation locations, calculate normalized per 1 windturbine foundation
        if np.any(isFixed):
            a = 40.0 * 1000 / capacity[isFixed]  # parameter will be a location-specific array, first value in MW
            instCost[isFixed] = ((1 / a) * (2.0 * _portDistance[isFixed] / 18.5 + 24.0) + 144.0 * 1) * (200.0 / 24.0)
        if np.any(~isFixed):
            # now deal with floating foundations
            # floating has 2 terms, so define params as np.arrays of len 2
            A = np.array([[0.3], [7.0]])
            B = np.array([[7.5], [18.5]])
            C = np.array([[5.0], [30.0]])
            D = np.array([[0.0], [90.0]])
            E = np.array([[2.5], [40.0]])
            # then apply the function to each "column" of the params separately and then add up
            instCost[~isFixed] = (((1 / A) * (2.0 * _portDistance[~isFixed][None, :] / B + C*1) + D) * (E / 24.0)).sum(
                axis=0
            )
        # make installation cost specific and convert to EUR/kW, Rogeau et al provide cost in kEUR2022 [1]
        return (instCost * 1000 / capacity).reshape(_depth.shape)

    # CALCULATE LOCATION-SPECIFIC COST BY SCALING THE INDIVIDUAL TURBINE COMPONENTS

    # get the component cost contributions and total default cost as per Rogeau for the given reference (base) turbine
    # consists of cost for a) turbine, b) foundation, c) cable connection - plus overhead
    # the turbine cost consist of the turbine (machine with tower with design impacts based on onshore calculations, scaled to offshore Rogeau cost level) and installation
    turbineBaseCostDefault = (
        offshoreCorrfacRogeau
        * onshore_tcc(  # without installation cost
            cp=baseCap,
            hh=baseHubHeight,
            rd=baseRotorDiam,
            gdp_escalator=1,
            blade_material_escalator=1,
            blades=3,
        )
        / baseCap
    )
    foundationBaseCostDefault = getOffshoreTurbineFoundationCost(  # without installation cost
        depth=baseDepth,
        maxMonopileDepth=maxMonopileDepth,
        maxJacketDepth=maxJacketDepth,
        year=techYear,
    )
    turbAndFoundInstallBaseCostDefault = _getSpecificTurbineInstallCost(baseDepth)
    connectionBaseCostDefault = getSpecificOffshoreConnectionCost(
        capacity=baseCap,
        waterDepth=baseDepth,
        coastDistance=baseDistCoast,
        voltageType=voltageType,
        baseWFSize=baseWFSize,
        maxJacketDepthPlatform=maxJacketDepthPlatform,
        year=techYear,
    )[0]  # returns tuple for optimal voltage with min. cost as first and voltage type as second entry
    # get the sum of the 3 components
    totalBaseCostDefault = (
        turbineBaseCostDefault
        + foundationBaseCostDefault
        + turbAndFoundInstallBaseCostDefault
        + connectionBaseCostDefault
    )
    # calculate the scaling factor (incl. currency) from Rogeau's base cost to actual given base cost
    customScalingFactor = (
        baseSpecCapex / totalBaseCostDefault
    )  # custom CAPEX / base case CAPEX (default, here Rogeau et al.)

    # now calculate the plant-specific default values as per Rogeau et al.
    turbinePlantCostDefault = (
        offshoreCorrfacRogeau
        * onshore_tcc(
            cp=capacity,
            hh=hubHeight,
            rd=rotorDiam,
            gdp_escalator=1,
            blade_material_escalator=1,
            blades=3,
        )
        / capacity
    )
    foundationPlantCostDefault = getOffshoreTurbineFoundationCost(
        depth=waterDepth,
        maxMonopileDepth=maxMonopileDepth,
        maxJacketDepth=maxJacketDepth,
        year=techYear,
    )
    turbAndFoundInstallPlantCostDefault = _getSpecificTurbineInstallCost(_depth=waterDepth)
    connectionPlantCostDefault = getSpecificOffshoreConnectionCost(
        capacity=capacity,
        waterDepth=waterDepth,
        coastDistance=coastDistance,
        voltageType=voltageType,
        baseWFSize=baseWFSize,
        maxJacketDepthPlatform=maxJacketDepthPlatform,
        year=techYear,
    )[0]  # returns tuple for optimal voltage with min. cost as first and voltage type as second entry
    totalPlantCostDefault = (
        turbinePlantCostDefault
        + foundationPlantCostDefault
        + turbAndFoundInstallPlantCostDefault
        + connectionPlantCostDefault
    )
    # scale the plant specific default (Rogeau) cost to custom reference CAPEX level
    totalPlantCostCustom = totalPlantCostDefault * customScalingFactor

    return totalPlantCostCustom


# %%
# this function returns the complete connection cost, including cable and all required converters and platforms
def getSpecificOffshoreConnectionCost(
    capacity: int | float | np.ndarray,
    waterDepth: int | float | np.ndarray,
    coastDistance: int | float | np.ndarray,
    year: int,
    portDistance: int | float | np.ndarray = None,
    voltageType: str = "optimal",
    baseWFSize: int | float = 180000,
    maxJacketDepthPlatform: int = 150,
):
    """
    Get offshore and - if applicable - onshore platform/converter cost plus
    cable cost and return the total cost (including installation) for the given
    connection and capacity.

    capacity : int|float|np.ndarray
        The electrical capacity of the cable connection in [kW].
    waterDepth : int|float|np.ndarray
        The water depth (positive values) at the location of the offshore
        converter, in [m].
    coastDistance : int|float|np.ndarray
        The distance to coast in [km] (>0).
    year: int
        The year for which the reference cost shall be returned.
    portDistance : int, optional
        If given in [km], it will be used to calculate the installation cost,
        else the coast distance [km] will be assumed as proxy.
    voltageType : str, optional
        Either "dc" or "ac" to return the cost for the respective connection
        type, or "optimal" to return the cheaper option. By default "optimal".
        NOTE: Returns a tuple then with the optimal voltageType as second entry
        like (cost:float, optimalVoltageType:str).
    baseWFSize : int, optional
        The average (base) wind farm size in [kW], by default 106858 (based on
        global average extracted from processed theWindPower.net database v2025/07).
    maxJacketDepthPlatform : int, optional
        The maximum depth up to which jacket foundations can be installed,
        by default 150 [m].

    Returns
    -------
    tuple
        A tuple of (min. specific cost, respective ac/dc connection type).
        Will contain scalar entries if all inputs were scalar, else arrays.

    References
    ----------
    [1] Rogeau, Antoine; Vieubled, Julien; Coatpont, Matthieu de; Affonso
    Nobrega, Pedro; Erbs, Guillaume; Girard, Robin (2023): Techno-economic
    evaluation and resource assessment of hydrogen production through
    offshore wind farms: A European perspective. In Renewable and
    Sustainable Energy Reviews 187, p. 113699. DOI: 10.1016/j.rser.2023.113699.
    [2] Ea Energy Analyses A / S, Energynautics, van Uden, J., Ebersbach, N.,
    Reijntjes, J., Ayivor, P., & Campagne, A. (2024). Pathway 2.0
    Techno-economic data (11.0) Pathway Databook_v11 - Public Version.
    Zenodo. https://doi.org/10.5281/zenodo.13382786
    """
    # check scalar inputs
    assert isinstance(baseWFSize, (int, float)) and baseWFSize > 0, "baseWFSize must be an integer or float > 0"
    assert isinstance(maxJacketDepthPlatform, int) and maxJacketDepthPlatform > 0, "maxJacketDepthPlatform must be an integer > 0"
    assert voltageType in ["ac", "dc", "optimal"], f"Unknown voltageType: {voltageType}"
    # check and preprocess vectorized inputs
    isScalar = all([np.isscalar(_arg) for _arg in [portDistance, waterDepth, capacity, coastDistance]])
    if not portDistance is None:
        portDistance = np.atleast_1d(portDistance)
        if any(portDistance < 0):
            raise ValueError("If not None, portDistance must be integers or floats >= 0 for all locations.")
    waterDepth = np.atleast_1d(waterDepth)
    if any(waterDepth < 0):
        raise ValueError(f"waterDepth must be >= 0 for all offshore locations.")
    capacity = np.atleast_1d(capacity)
    if any(capacity < 0):
        raise ValueError(f"capacity must be >= 0 for all locations.")
    coastDistance = np.atleast_1d(coastDistance)
    if any(coastDistance < 0):
        raise ValueError(f"coastDistance must be >= 0 for all locations.")
    if portDistance is None:
        portDistance = coastDistance

    def _getTotalSpecificConnectionCost(_voltageType):
        """Connection cost consists of 3 elements: onshore and offshore converter + cable"""
        assert _voltageType in ["ac", "dc"]
        # get specific onshore converter cost first
        convertercost_onshore_baseWFSize = getSpecificConverterStationCost(
            capacity=baseWFSize,
            waterDepth=None,
            voltageType=_voltageType,
            portDistance=0,  # onshore
            maxJacketDepthPlatform=maxJacketDepthPlatform,
        )
        convertercost_onshore = (
            convertercost_onshore_baseWFSize * capacity / baseWFSize
        )  # scale linearly to the actual size

        # then specific offshore converter cost
        convertercost_offshore_baseWFSize = getSpecificConverterStationCost(
            capacity=baseWFSize,
            waterDepth=waterDepth,
            voltageType=_voltageType,
            portDistance=portDistance,
            maxJacketDepthPlatform=maxJacketDepthPlatform,
        )
        convertercost_offshore = convertercost_offshore_baseWFSize * capacity / baseWFSize  # scale to actual size again

        # last specific cable cost
        cableCost = getSpecificOffshoreCableCost(
            distance=coastDistance,
            capacity=capacity,
            voltageType=_voltageType,
            fixedCost=0,
            variableCostFactor=None,
            year=year,
        )
        return cableCost + convertercost_offshore + convertercost_onshore

    if voltageType == "optimal":
        # check both AC and DC and return the cheaper option
        voltageTypes = ["ac", "dc"]
    else:
        # calculate only for the given voltageType
        voltageTypes = [voltageType]
    # calculate the cost for all eligible voltageTypes...
    _totalCostDict = {}
    for _voltageType in voltageTypes:
        _totalCostDict[_voltageType] = _getTotalSpecificConnectionCost(_voltageType=_voltageType)
    _totalCostArray = np.vstack([_totalCostDict[k] for k in _totalCostDict.keys()])  # stack all options
    # ... and return the lowest cost option (the ONLY option if one specific voltageType was given)
    minCost = np.min(_totalCostArray, axis=0)
    minType = (
        list(_totalCostDict.keys())[0]
        if _totalCostArray.shape[0] == 1
        else np.array(list(_totalCostDict.keys()))[np.argmin(_totalCostArray, axis=0)]
    )
    # make scalar if all inputs were scalar
    if isScalar:
        minCost = minCost[0]
        minType = minType[0]
    # return a tuple: (minCost:float|np.array, optimalVoltageType:str|np.array)
    return (minCost, minType)


# %%
def getOffshoreTurbineFoundationCost(
    depth: int | float | np.ndarray,
    maxMonopileDepth: int | float = 25,
    maxJacketDepth: int | float = 55,
    year: int = 2030,
    returnType: bool = False,
):
    """
    Estimates the rated cost (and optionally type) of offshore wind turbine
    foundations based on water depth and year for one or multiple locations.
    Does not include the cost for the turbine itself or cable connection cost.

    Note: Excludes installation cost.

    Parameters
    ----------
    depth : int | float | np.ndarray
        Water depth at the installation site in [m], can be provided as an
        array per location.
    maxMonopileDepth : int | float, optional
        Threshold depth for monopile foundations in [m], by default 25.
    maxJacketDepth : int | float, optional
        Threshold depth for jacket foundations in [m], by default 55.
    year : int, optional
        Determines the scaling factors acc. to Rogeau et al., will interpolate
        if year is not provided by Rogeau et al., by default 2030.
    returnType : bool, optional
        If True, a tuple will be returned of (cost, foundationType), else only
        cost. By default False.

    Returns
    -------
    float | np.ndarray | tuple
        Rated cost in in [EUR_2022/kW], or (Rated cost in [€_2022/kW], foundation type)
        if returnType is True. Both rated cost and foundation type will be
        scalar or numpy arrays depending on the input data type of "depth".

    References
    ----------
    [1] Rogeau et al. (2023), Renewable and Sustainable Energy Reviews.
    [2] Jonathan Bosch, Iain Staffell, Adam D. Hawkes, Global levelised cost of
        electricity from offshore wind, Energy, Volume 189, 2019, 116357,
        ISSN 0360-5442, https://doi.org/10.1016/j.energy.2019.116357.
    """
    # check and preprocess inputs
    isScalar = np.isscalar(depth)  # will be returned as a scalar as well then
    depth = np.atleast_1d(depth)
    if any(depth < 0):
        raise ValueError("depth must be >= 0 for all locations.")
    assert isinstance(maxMonopileDepth, (int, float)) and maxMonopileDepth > 0, (
        "maxMonopileDepth must be integer or float and > 0"
    )
    assert isinstance(maxJacketDepth, (int, float)) and maxJacketDepth > maxMonopileDepth, (
        "maxJacketDepth must be integer or float and > maxMonopileDepth"
    )
    assert isinstance(year, int), "year must be integer"
    assert isinstance(returnType, bool), "returnType must be boolean"

    # set monopile respectively jacket as foundation type whereever it is under the threshold, else set floating
    foundtypes = np.where(
        depth <= maxMonopileDepth, "monopile", np.where(depth <= maxJacketDepth, "jacket", "floating")
    )
    # map also to integer index for later extraction of parameters (order must match below params order)
    foundmapper = {"monopile": 0, "jacket": 1, "floating": 2}
    foundints = np.vectorize(foundmapper.__getitem__, otypes=[int])(foundtypes)
    # define cost function coefficients per year and foundation type, sorted by above integer index
    coeffsDict = {
        2020: np.array(
            [
                (201, 613, 812),  # monopile
                (114, -2270, 932),  # jacket
                (0, 774, 1481),  # floating
            ],
            dtype=float,
        ),
        2030: np.array(
            [
                (181, 552, 370),
                (103, -2043, 478),
                (0, 697, 1223),
            ],
            dtype=float,
        ),
        2050: np.array(
            [
                (171, 521, 170),
                (97, -1930, 272),
                (0, 658, 844),
            ],
            dtype=float,
        ),
    }

    # now get the nearest data years to the given reference year
    years = sorted(coeffsDict.keys())
    assert min(years) <= year <= max(years), (
        f"year {year} is outside of the cost range defined by Rogeau et al.: 2020-2050"
    )
    yearBefore = max((y for y in years if y <= year))
    yearAfter = min((y for y in years if y >= year))
    # get coefficients for both bracketing years and interpolate them (all linear)
    coeffsBefore = coeffsDict[yearBefore]
    coeffsAfter = coeffsDict[yearAfter]
    weighing = 0 if yearBefore == yearAfter else (year - yearBefore) / (yearAfter - yearBefore)
    coeffsInterp = (1.0 - weighing) * coeffsBefore + weighing * coeffsAfter
    # now extract the actual coefficients per location via foundation code and calculate cost
    a, b, c = coeffsInterp[foundints, 0], coeffsInterp[foundints, 1], coeffsInterp[foundints, 2]
    costs = (
        a * depth**2 + b * depth + c * 1000.0
    )  # Rogeau et al. do not indicate unit, but secondary source Bosch et al. [2] uses cost per [MW]
    costs = costs / 1000  # convert to [EUR_2022/kW]

    # prepare outputs and return
    if isScalar:
        # get the scalar entry
        costs = np.asarray(costs).item()
        foundtypes = np.asarray(foundtypes).item()
    if returnType:
        return costs, foundtypes  # tuple
    else:
        return costs


# %%
def getSpecificOffshoreCableCost(
    distance: int | float | np.ndarray,
    capacity: int | float | np.ndarray,
    voltageType: str | np.ndarray,
    variableCostFactor: int | float | np.ndarray = None,
    fixedCost: int | float | np.ndarray = 0,
    year: int = None,
    detourFactor: int | float | np.ndarray = 1.2,
):
    """
    Calculates the default cost for the cable connecting an offshore wind power
    plant to the coastline, including installation cost. Does not include cost
    for converter stations and its offshore platform.

    Parameters
    ----------
    distance : int | float | np.ndarray
        Distance to coastline in [km].
    capacity : int | float | np.ndarray
        Power plant's capacity in [kW].
    voltageType : str | np.ndarray
        'ac' or 'dc', takes no effect when variableCostFactor is provided.
    variableCostFactor : int | float | np.ndarray, optional
        Cost multiplier in [EUR_2022/kW/km], by default None.
    fixedCost : float, optional
        Fixed absolute connection cost, must be in [EUR_2022]. Defaults to 0.
    year : int, optional
        The year for which the reference cost shall be returned in case of
        voltageType == 'ac' (year is then mandatory, else it has no effect).
        By default None.
    detourFactor: int | float | np.ndarray
        The detour factor to be applied onto the coast distance of the 
        windfarms, set to 1.0 for no effect. By default 1.2 in accordance 
        with assumptions in [1].

    Returns
    -------
    np.ndarray
        Total cable connection cost in [EUR_2022].

    References
    ----------
    [1] Rogeau et al. (2023), "Review and modeling of offshore wind CAPEX",
    Renewable and Sustainable Energy Reviews, DOI: 10.1016/j.rser.2023.113699
    [2] Ea Energy Analyses A / S, Energynautics, van Uden, J., Ebersbach, N.,
    Reijntjes, J., Ayivor, P., & Campagne, A. (2024). Pathway 2.0
    Techno-economic data (11.0) Pathway Databook_v11 - Public Version.
    Zenodo. https://doi.org/10.5281/zenodo.13382786
    """
    # check if we have only scalar inputs and save as flag
    isScalar = all([np.isscalar(x) for x in [distance, capacity, fixedCost, voltageType, variableCostFactor]])
    # convert all inputs to arrays of the same shape
    distance = np.asarray(distance, dtype=float)
    capacity = np.asarray(capacity, dtype=float)
    fixedCost = np.asarray(fixedCost, dtype=float)
    detourFactor = np.asarray(detourFactor)
    voltageType = np.asarray(voltageType)
    variableCostFactor = None if variableCostFactor is None else np.asarray(variableCostFactor, dtype=float)
    if variableCostFactor is None:
        distance, capacity, fixedCost, voltageType, detourFactor = np.broadcast_arrays(distance, capacity, fixedCost, voltageType, detourFactor)
    else:
        distance, capacity, fixedCost, voltageType, variableCostFactor, detourFactor = np.broadcast_arrays(
            distance, capacity, fixedCost, voltageType, variableCostFactor, detourFactor
        )

    # check inputs
    assert (distance >= 0).all(), "All distances must be larger or equal to 0"
    assert (capacity >= 0).all() > 0, "All turbine capacities must be larger than 0"
    assert (fixedCost >= 0).all(), "All fixed cost must be positive or 0."
    assert (detourFactor > 0).all(), "All detour factors must be positive."
    assert ((voltageType == "ac") | (voltageType == "dc")).all(), "All voltageType must be 'ac' or 'dc'"
    assert variableCostFactor is None or (variableCostFactor > 0).all(), (
        "All variableCostFactor must be larger than 0 if not None"
    )
    assert isinstance(year, int) and year > 0, f"Year must be positive integer"

    if variableCostFactor is not None:
        # use the one that is provided
        costPerKm = variableCostFactor
    else:
        # create a container for the cost per km and fill depending on voltage type
        costPerKm = np.empty(distance.shape, dtype=float)
        # for dc, use time-independent DC cable cost per kW and km as in Rogeau et al. (Note unit error, is indicated per W*km but actually kW*km)
        # cost already include "delivery and installation" as per Rogeau et al.
        dc_mask = voltageType == "dc"
        costPerKm[dc_mask] = 1.35 * 1.5  # + 50% for installation acc. to Rogeau et al. [1]
        # for ac, use AC cost per km and kW from Pathway 2.0 (Ea Energy Analyses A / S et al. [2]) instead of Rogeau (Rogeau is very confusing here)
        # values include installation and have been corrected to EUR_2022 to align with cost data from Rogeau et al. [1]
        ac_mask = voltageType == "ac"
        if ac_mask.sum() > 0:
            # we DO have ac cases, check year and get bracketing data years first
            assert year is not None, "year is required for voltageType 'ac' if no variableCostFactor is provided."
            acCostPerKmDict = {
                2020: 7.70,
                2030: 7.47,
                2050: 7.05,
            }  # includes installation acc. to [2], converted to EUR_2022
            years = np.array(sorted(acCostPerKmDict.keys()), dtype=int)
            if not (years.min() <= year <= years.max()):
                raise ValueError(f"year {year} is outside range {years.min()}-{years.max()}")
            yearBefore = max((y for y in acCostPerKmDict.keys() if y <= year))
            yearAfter = min((y for y in acCostPerKmDict.keys() if y >= year))
            acCostPerKm = acCostPerKmDict[yearBefore]
            if yearAfter > yearBefore:
                acCostPerKm = (
                    acCostPerKm
                    * (acCostPerKmDict[yearAfter] - acCostPerKmDict[yearBefore])
                    * (year - yearBefore)
                    / (yearAfter - yearBefore)
                )
            costPerKm[ac_mask] = acCostPerKm

    # scale and add up the cost components, return as array or scalar
    totalSpecCost = (costPerKm * distance * detourFactor * capacity + fixedCost) / capacity
    if isScalar:
        totalSpecCost = np.asarray(totalSpecCost).item()
    return totalSpecCost


# %%


def getSpecificOffshorePlatformCost(
    applicationType: str | np.ndarray,
    capacity: int | float | np.ndarray,
    waterDepth: int | float | np.ndarray,
    portDistance: int | float | np.ndarray,
    foundationType: str | np.ndarray = None,
    maxJacketDepthPlatform: int | float = 150,
    baseWFSize: int | float = 180000,
    convention: str = "RogeauEtAl2023",
):
    """
    Returns the specific cost of an offshore foundation in one or multiple locations
    for offshore substations or electrolysis (but not wind turbines!) depending
    on application type, water depth, port distance and installed capacity.
    Includes installation cost.

    applicationType : str | np.ndarray
        The type of application that shall be installed on the platform,
        e.g. "ac" (substation), "dc" (substation) or (offshore) "electrolysis",
        or an array thereof with individual values per location.
    capacity: int | float | np.ndarray
        The installed electrical capacity in [kW] for the respective application.
    waterDepth: int | float | np.ndarray
        The location water depth in [m] per location.
    portDistance : int | float|np.ndarray
        The distance from the nearest installation base port in [km].
    foundationType: str | np.ndarray, optional
        The type of foundation per location, will be specified automatically
        based on maxJacketDepth if not provided.
    maxJacketDepthPlatform : int | float, optional
        The max. possible jacket foundation depth in [m]. By default 150 m
        following [1].
    baseWFSize : int | float, optional
        The average (base) wind farm size in [kW], by default 180000 (based on
        global average extracted from processed theWindPower.net database v2025/07 (15 plants and 12000 MW standardsize from Rougeau)).
    convention : str, optional
        The convention by which the foundation cost shall be determined,
        e.g. "RogeauEtAl2023" based on the equations in [1].

    [1] Rogeau, Antoine; Vieubled, Julien; Coatpont, Matthieu de; Affonso
    Nobrega, Pedro; Erbs, Guillaume; Girard, Robin (2023): Techno-economic
    evaluation and resource assessment of hydrogen production through
    offshore wind farms: A European perspective. In Renewable and
    Sustainable Energy Reviews 187, p. 113699. DOI: 10.1016/j.rser.2023.113699.
    """
    # set flag if results should be scalar
    isScalar = all([np.isscalar(x) for x in [waterDepth, capacity, portDistance, applicationType]])
    # preprocess vectorized inputs
    waterDepth = np.atleast_1d(waterDepth)
    capacity = np.atleast_1d(capacity)
    portDistance = np.atleast_1d(portDistance)
    applicationType = np.atleast_1d(applicationType)
    for name, x in [
        ("waterDepth", waterDepth),
        ("capacity", capacity),
        ("portDistance", portDistance),
        ("applicationType", applicationType),
    ]:
        if x.ndim > 1:
            raise ValueError(f"{name} must be scalar or 1D (one value per location). Got shape {x.shape}.")
    if foundationType is None:
        waterDepth, capacity, portDistance, applicationType = np.broadcast_arrays(
            waterDepth, capacity, portDistance, applicationType
        )
    else:
        waterDepth, capacity, portDistance, applicationType, foundationType = np.broadcast_arrays(
            waterDepth, capacity, portDistance, applicationType, foundationType
        )

    # check inputs
    if np.any(capacity <= 0):
        raise ValueError(f"capacity must be an int or float > 0 kW, or an array thereof")
    if np.any(waterDepth < 0):
        raise ValueError(f"waterDepth must be an int or float >= 0 m, or an array thereof")
    if np.any(portDistance < 0):
        raise ValueError(f"portDistance must be an int or float >= 0 km, or an array thereof")

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

        # get and check foundation type
        isFixed = waterDepth <= maxJacketDepthPlatform  # fixed foundation if under max jacket depth, else floating
        if foundationType is None:
            foundationType = np.where(isFixed, "jacket", "floating")
        else:
            # format as array if not done yet
            foundationType = np.asarray(foundationType)
            if not np.isin(foundationType, list(RCPF_factors.keys())).all():
                # unknown foundation type
                raise ValueError(f"foundationType must be in: {', '.join(RCPF_factors.keys())}")
            # warn if the foundation type is wrong acc. to given threshold
            if ((waterDepth > maxJacketDepthPlatform) & (foundationType == "jacket")).any():
                warnings.warn(
                    f"waterDepth ({waterDepth} m) exceeds maxJacketDepthPlatform ({maxJacketDepthPlatform} m) but 'jacket' is enforced as foundationType."
                )

        # get the coefficients for the Rogeau cost equation, depending on foundation type per location
        c2_R = np.where(foundationType == "jacket", RCPF_factors["jacket"]["c2"], RCPF_factors["floating"]["c2"])
        c3_R = np.where(foundationType == "jacket", RCPF_factors["jacket"]["c3"], RCPF_factors["floating"]["c3"])
        c2_U = np.where(foundationType == "jacket", UCPF_factors["jacket"]["c2"], UCPF_factors["floating"]["c2"])
        c3_U = np.where(foundationType == "jacket", UCPF_factors["jacket"]["c3"], UCPF_factors["floating"]["c3"])

        # get RCPF and UCPF as per Rogeau
        RCPF = c2_R * waterDepth + c3_R * 10**3
        UCPF = c2_U * waterDepth + c3_U * 10**3

        # relative theoretical power (vectorized) based on power densities/footprints, eq.(9) [1]
        powerDensity_factors = np.empty_like(capacity, dtype=float)
        # write all dc, ac and electrolysis power densities via mask
        if not np.isin(applicationType, ["dc", "ac", "electrolysis"]).all():
            raise ValueError(f"applicationType must be one of: {', '.join(['dc', 'ac', 'electrolysis'])}")
        mask_dc = applicationType == "dc"
        powerDensity_factors[mask_dc] = baseWFSize / 1000.0
        mask_ac = applicationType == "ac"
        powerDensity_factors[mask_ac] = 0.5 * baseWFSize / 1000.0
        mask_el = applicationType == "electrolysis"
        powerDensity_factors[mask_el] = 2.0 * baseWFSize / 1000.0

        # calculate total equipment platform cost as the sum of capacity-dependent and fixed cost as per eq. (8)
        ECPF = RCPF * powerDensity_factors + UCPF
        specECPF = ECPF / (baseWFSize)  # normalize to specific values per kW

        # now calculate the platform installation cost function based on Rogeau et al. section 3.2.2
        ICPF = np.zeros_like(waterDepth, dtype=float)  # initiate Installation Cost container
        # first deal with fixed foundation locations
        ICPF[isFixed] = ((1.0 / 1) * (2.0 * portDistance[isFixed] / 18.5 + 24.0) + 96*1) * (200.0 / 24.0)
        # now deal with floating foundations
        # floating has 2 terms, so define params as np.arrays of len 2
        A = np.array([[1], [3]])
        B = np.array([[22.5], [18.5]])
        C = np.array([[10], [30.0]])
        D = np.array([[0.0], [90.0]])
        E = np.array([[40], [40.0]])
        # then apply the function to each "column" of the params separately and then add up
        ICPF[~isFixed] = (((1.0 / A) * (2.0 * portDistance[~isFixed][None, :] / B + C) + D*1) * (E / 24.0)).sum(axis=0)
        # make specific - assume cost per ship stay constant but load-carrying capacity grows -> implicit (specific) transport cost (per kW) decrease with larger, future turbines
        specICPF = (ICPF) / (baseWFSize/1000)  # Rogeau is in k€ x units, here 1 unit, divide by cap. per unit
        # add up
        totalSpecCost = specECPF + specICPF
        print(ICPF)
        print(f"totalSpecCost Platfom Substation: {totalSpecCost}, specECPF: {specECPF}, specICPF: {specICPF}")

    else:
        raise NotImplementedError(f"convention '{convention}' is not implemented.")

    if isScalar:
        totalSpecCost = np.asarray(totalSpecCost).item()
    return totalSpecCost


# %%
# This function returns the cost for an on- or offshore converter station, includes platform cost if offshore
def getSpecificConverterStationCost(
    capacity: int | float | np.ndarray,
    waterDepth: int | float | np.ndarray | None,
    voltageType: str | np.ndarray,
    portDistance: int | float | np.ndarray,
    maxJacketDepthPlatform: int | float = 150,
    convention="RogeauEtAl2023",
):
    """
    Calculates the cost of an onshore or offshore converter station for AC or DC
    connections of wind farms, including platform cost in the case of offshore
    locations. Includes installation cost explicitly for offshore platforms,
    converter electrical installation cost is not treated separately in
    Rogeau et al. [1].

    capacity : int | float | np.ndarray
        Converter station power in [kW].
    waterDepth : int | float | np.ndarray
        The water depth in [m] in case of an offshore substation, does then
        include then platform cost of either jacket or floating type depending
        on depth. If None, an onshore substation without platform will be assumed.
    voltageType : str | np.ndarray
        If the substation is for "ac"or "dc" connections, can be defined per
        location and passed as array.
    portDistance : int | float|np.ndarray
        The distance from the nearest installation base port in [km].
    maxJacketDepthPlatform : int, optional
        The max. possible jacket foundation depth in [m]. By default 150 m
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

    # enforce "max 1d" for np arrays and normalize scalars to 1-element arrays
    def _as1d(x, name: str):
        if x is None:
            return None, True  # treat None as scalar-like for isScalar
        arr = np.asarray(x)
        if arr.ndim > 1:
            raise ValueError(f"{name} must be scalar or a max 1d np array, got ndim={arr.ndim}")
        is_scalar_like = arr.ndim == 0
        if is_scalar_like:
            arr = arr.reshape(1)
        return arr, is_scalar_like

    capacity, _cap_scalar = _as1d(capacity, "capacity")
    waterDepth, _wd_scalar = _as1d(waterDepth, "waterDepth")
    voltageType, _vt_scalar = _as1d(voltageType, "voltageType")
    portDistance, _pd_scalar = _as1d(portDistance, "portDistance")
    voltageType, _vt_scalar = _as1d(voltageType, "voltageType")
    # define scalar flag to return scalars if inputs were scalar as well
    isScalar = _cap_scalar and _wd_scalar and _vt_scalar and _pd_scalar and _vt_scalar  # required by user
    # broadcast all arrays to the same dimensions
    if waterDepth is None:
        capacity, portDistance, voltageType = np.broadcast_arrays(capacity, portDistance, voltageType)
    else:
        capacity, waterDepth, portDistance, voltageType = np.broadcast_arrays(
            capacity, waterDepth, portDistance, voltageType
        )

    if convention == "RogeauEtAl2023":
        # calculate electrical powerstation cost based on equation (10) and table 6
        RCPS = {"ac": 22.87, "dc": 102.93}  # EUR/kW
        UCPS = {"ac": 3.1750000, "dc": 7.060000}  # EUR
        assert sorted(RCPS.keys()) == sorted(UCPS.keys())  # make sure
        if not np.all(np.isin(voltageType, list(RCPS.keys()))):
            raise ValueError(f"unknown voltageType, select from: {', '.join(RCPS.keys())}")
        # get voltage-type specific coefficients per location
        RCPS_coeffs = np.where(voltageType == "ac", RCPS["ac"], RCPS["dc"])
        UCPS_coeffs = np.where(voltageType == "ac", UCPS["ac"], UCPS["dc"])
        # Add up powerstation cost. Note that Rogeau does list separate installation cost for the powerstation itself, only for the platform
        ECPS = RCPS_coeffs * capacity + UCPS_coeffs * 10**3
        specECPS = ECPS / capacity

        print(f"total Cost Substation: {ECPS}, specific Cost Substation: {specECPS}")

        if waterDepth is None:
            # an onshore station, no additional platform cost
            specECPF = np.zeros_like(capacity, dtype=float)
        else:
            # get platform cost (incl. installation cost) from separate function
            specECPF = getSpecificOffshorePlatformCost(
                capacity=capacity,
                applicationType=voltageType,
                waterDepth=waterDepth,
                portDistance=portDistance,
                foundationType=None,
                maxJacketDepthPlatform=maxJacketDepthPlatform,
                convention=convention,
            )
        # combine electrical and platform cost components
        
        print(f'specific Platform Cost Substation: {specECPF}')
        totalSpecCost = specECPS + specECPF
        
    else:
        raise NotImplementedError(f"Unknown convention: '{convention}'")

    if isScalar:
        totalSpecCost = np.asarray(totalSpecCost).item()

    return totalSpecCost
