import numpy as np


def onshore_turbine_capex(capacity, hub_height, rotor_diam, base_capex=1100 * 4200, base_capacity=4200, base_hub_height=120, base_rotor_diam=136, tcc_share=0.673, bos_share=0.229, **k):
    """
    A cost and scaling model (CSM) to calculate the total cost of a 3-bladed, direct drive onshore wind turbine according to Fingersh et al. [1] and Maples et al. [2].
    A CSM normalization is done such that a chosen baseline turbine, with a capacity of 4200 kW, hub height of 120 m, and rotor diameter of 136 m, corresponds to a expected typical specific cost of 1100 Eur/kW in a 2050 context according to the 2016 cost of wind energy review by Stehly [3]
    The turbine cost includes the turbine capital cost (TCC) and balance of system costs (BOS), amounting to 67.3% and 22.9% respectively [3], as well as finantial costs equivalent to the the complementary percentage.
    
    Parameters
    ----------
        capacity : float or array-like
            Turbine's nominal capacity in kW.

        hubHeight : float or array-like
            Turbine's hub height in m.

        rotordiam : float or array-like
            Turbine's hub height in m.

    Returns
    --------
        Onshore turbine total cost: float

    Notes
    -------
        The expected turbine cost estimation by Stehly et al [3]. is claimed to be derived from real cost data and valid until 10 MW capacity.
        The baseline (or standard) turbine characteristics correspond to the expected typical onshore turbine in 2050 [].

    Sources
    ---------
        [1] Fingersh, L., Hand, M., & Laxson, A. (2006). Wind Turbine Design Cost and Scaling Model. Nrel. https://www.nrel.gov/docs/fy07osti/40566.pdf

        [2] Maples, B., Hand, M., & Musial, W. (2010). Comparative Assessment of Direct Drive High Temperature Superconducting Generators in Multi-Megawatt Class Wind Turbines. Energy. https://doi.org/10.2172/991560

        [3] Stehly, T., Heimiller, D., & Scott, G. (2016). Cost of Wind Energy Review. Technical Report. https://www.nrel.gov/docs/fy18osti/70363.pdf

    """

    # PREPROCESS INPUTS
    rd = np.array(rotor_diam)
    hh = np.array(hub_height)
    cp = np.array(capacity)
    # rr = rd / 2

    # COMPUTE COSTS
    # normalizations chosen to make the default turbine (4200-cap, 120-hub, 136-rot) match both a total
    # cost of 1100 EUR/kW as well as matching the percentages given in [3]
    tcc_scaling = base_capex * tcc_share / onshore_tcc(cp=base_capacity, hh=base_hub_height, rd=base_rotor_diam)
    tcc = onshore_tcc(cp=cp, hh=hh, rd=rd) * tcc_scaling

    bos_scaling = base_capex * bos_share / onshore_bos(cp=base_capacity, hh=base_hub_height, rd=base_rotor_diam)
    bos = onshore_bos(cp=cp, hh=hh, rd=rd) * bos_scaling

    # print(tcc_scaling, bos_scaling)

    total_costs = (tcc + bos) / (tcc_share + bos_share)

    # other_costs = total_costs * (1-tcc_share-bos_share)

    return total_costs


def onshore_tcc(cp, hh, rd, gdp_escalator=1, blade_material_escalator=1, blades=3):
    """

    A function to determine the tubine capita cost (TCC) based on the capacity, hub height and rotor diamter values accoiding to Fingersh et al. [1].

    Parameters
    ----------
    cp : int
        Turbine's capacity in kW
    hh : int
        Turbine's hub height in m
    rd : int
        Turbine's rotor diamter in m
    gdpEscalator : int, optional
        Labor cost escalator, by default 1
    bladeMaterialEscalator : int, optional
        Blade material cost escalator, by default 1
    blades : int, optional
        Number of blades, by default 3

    Returns
    -------
    Turbine capital cost: float
        Turbine's capital cost in monetary units.
    
    Sources
    ---------
        [1] Fingersh, L., Hand, M., & Laxson, A. (2006). Wind Turbine Design Cost and Scaling Model. Nrel. https://www.nrel.gov/docs/fy07osti/40566.pdf

    """
    rr = rd / 2
    sa = np.pi * rr * rr

    # Blade Cost
    singleBladeMass = 0.4948 * np.power(rr, 2.53)
    singleBladeCost = ((0.4019 * np.power(rr, 3) - 21051) * blade_material_escalator + 2.7445 * np.power(rr, 2.5025) * gdp_escalator) * (1 - 0.28)

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
    turbineCapitalCost = singleBladeCost * blades + \
        hubCost + \
        pitchSystemCost + \
        noseConeCost + \
        lowSpeedShaftCost + \
        bearingCost + \
        breakCouplingCost + \
        generatorCost + \
        electronicsCost + \
        yawSystemCost + \
        mainframeCost + \
        platformAndRailingCost + \
        electricalConnectionCost + \
        hydraulicAndCoolingSystemCost + \
        nacelleCost + \
        towerCost

    return turbineCapitalCost


def onshore_bos(cp, hh, rd):
    """
    
    A function to determine the balance of the system cost (BOS) of an onshore turbine based on the capacity, hub height and rotor diamter values accoriding to Fingersh et al. [1].

    Parameters
    ----------
    cp : int
        Turbine's capacity in kW
    hh : int
        Turbine's hub height in m
    rd : int
        Turbine's rotor diamter in m
    Returns
    -------
    bosCosts: float
        Turbine's BOS in monetary units.
    
    Sources
    ---------
        [1] Fingersh, L., Hand, M., & Laxson, A. (2006). Wind Turbine Design Cost and Scaling Model. Nrel. https://www.nrel.gov/docs/fy07osti/40566.pdf

    """
    
    rr = rd / 2
    sa = np.pi * rr * rr

    # Foundation
    foundationCost = 303.24 * np.power((hh * sa), 0.4037)

    # Transportation
    transporationCostFactor = 1.581E-5 * np.power(cp, 2) - 0.0375 * cp + 54.7
    transporationCost = transporationCostFactor * cp

    # Roads and civil work
    roadsAndCivilWorkFactor = 2.17E-6 * np.power(cp, 2) - 0.0145 * cp + 69.54
    roadsAndCivilWorkCost = roadsAndCivilWorkFactor * cp

    # Assembly and installation
    assemblyAndInstallationCost = 1.965 * np.power((hh * rd), 1.1736)

    # Electrical Interface and connections
    electricalInterfaceAndConnectionFactor = (3.49E-6 * np.power(cp, 2)) - (0.0221 * cp) + 109.7
    electricalInterfaceAndConnectionCost = electricalInterfaceAndConnectionFactor * cp

    # Engineering and permit factor
    engineeringAndPermitCostFactor = 9.94E-4 * cp + 20.31
    engineeringAndPermitCost = engineeringAndPermitCostFactor * cp

    # Add up other costs
    bosCosts = foundationCost + \
        transporationCost + \
        roadsAndCivilWorkCost + \
        assemblyAndInstallationCost + \
        electricalInterfaceAndConnectionCost +\
        engineeringAndPermitCost

    return bosCosts
