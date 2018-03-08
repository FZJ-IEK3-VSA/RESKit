from ._util import *

class _BaselineOnshoreTurbine(dict):
    """
    The baseline onshore turbine is chosen to reflect future trends in wind turbine characteristics.
    """

baselineOnshoreTurbine = _BaselineOnshoreTurbine(capacity=4200, hubHeight=129, rotordiam=141)

def turbineCostCalculator(capacity, hubHeight, rotordiam, depth=None, shoreDistance=None, busDistance=None, offshore=False, ):
    """
    **NEEDS UPDATE**
    Onshore wind turbine cost and scaling model (csm) built following [1] and update following [2]. 
    Considers only the turbine capital cost estimations for a 3-bladed, direct drive turbine.
    Claimed to be derived from real cost data and valid (for costs at the time) up until 10 MW capacity.
    
    Base-line (default) turbine characteristics correspond to the expected typical onshore turbine in 2050.
    Output values are adjusted such that the the baseline onshore turbine matches 1100 Eur/kW including all costs.
    Only the turbine capital cost (tcc), amounting to 67.3% [3], is adjusted according to capacity, rotor diameter, and hub height.
    Balance of system costs and other financial costs are added as fixed percentages.

    Inputs:
        capacity : Turbine nameplate capacity in kW
            float - Single value
            np.ndarray - multidimensional values 

        hubHeight : Turbine hub height in meters
            float - Single value
            np.ndarray - multidimensional values

        rotordiam : Turbine rotor diameter in meters
            float - Single value
            np.ndarray - multidimensional values
    
    Sources:
    [1] L. Fingersh, M. Hand, and A. Laxson 
        "Wind Turbine Design Cost and Scaling Model"
        2006. NREL

    [2] B. Maples, M. Hand, and W. Musial
        "Comparative Assessment of Direct Drive High Temperature Superconducting Generators in Multi-Megawatt Class Wind Turbines"
        2010. NREL

    [3] Tyler Stehly, Donna Heimiller, and George Scott
        "2016 Cost of Wind Energy Review"
        2017. NREL

    [4] Lixuan Hong and Bernd Moeller.
        "Offshore wind energy potential in China: Under technical, spatial and economic constraints"
        2011. Energy

    """
    gdpEscalator=1#2.5 # Chosen to match example given in [1]
    bladeMaterialEscalator=1
    blades = 3
    
    rd = np.array(rotordiam)
    hh = np.array(hubHeight)
    cp = np.array(capacity)
    rr = rd/2
    sa = np.pi*rr*rr

    if offshore:
        depth = np.array(depth)
        busD = np.array(busDistance)
        shoreD = np.array(shoreDistance)
    
    turbineCapitalCostNormalization = 0.874035081173
    
    # Blade Cost
    singleBladeMass = 0.4948 * np.power(rr,2.53)
    singleBladeCost = ((0.4019*np.power(rr, 3)-21051)*bladeMaterialEscalator + 2.7445*np.power(rr, 2.5025)*gdpEscalator)*(1-0.28)

    # Hub 
    hubMass = 0.945*singleBladeMass+5680.3
    hubCost = hubMass*4.25

    # Pitch and bearings
    pitchBearingMass = 0.1295*(singleBladeMass*blades)+491.31
    pitchSystemMass = pitchBearingMass*1.328+555
    pitchSystemCost = 2.28*(0.2106*np.power(rd,2.6578))

    # Spinner and nosecone
    noseConeMass = 18.5*rd - 520.5
    noseConeCost = noseConeMass*5.57

    # Low Speed Shaft
    lowSpeedShaftMass = 0.0142 * np.power(rd, 2.888)
    lowSpeedShaftCost = 0.01 * np.power(rd,2.887)

    # Main bearings
    bearingMass = (rd*8/600 - 0.033) * 0.0092 * np.power(rd,2.5)
    bearingCost = 2*bearingMass * 17.6

    # Gearbox
    ## Gearbox not included for direct drive turbines

    # Break, coupling, and others
    breakCouplingCost = 1.9894 * cp - 0.1141
    breakCouplingMass = breakCouplingCost/10

    # Generator (Assuming direct drive)
    # generatorMass = 6661.25 * np.power(lowSpeedShaftTorque, 0.606) # wtf is the torque?
    generatorCost = cp*219.33

    # Electronics
    electronicsCost = cp * 79

    # Yaw drive and bearing
    yawSystemMass = 1.6*(0.0009*np.power(rd, 3.314))
    yawSystemCost = 2*(0.0339*np.power(rd,2.964))

    # Mainframe (Assume direct drive)
    mainframeMass = 1.228*np.power(rd, 1.953)
    mainframeCost = 627.28 * np.power(rd, 0.85)

    # Platform and railings
    platformAndRailingMass = 0.125 * mainframeMass
    platformAndRailingCost = platformAndRailingMass * 8.7

    # Electrical Connections
    electricalConnectionCost = cp*40

    # Hydraulic and Cooling systems
    hydraulicAndCoolingSystemMass = 0.08 * cp
    hydraulicAndCoolingSystemCost = cp * 12

    # Nacelle Cover
    nacelleCost = 11.537*cp + 3849.7
    nacelleMass = nacelleCost/10

    # Tower
    towerMass = 0.2694*sa*hh + 1779
    towerCost = towerMass*1.5
    
    # Add up the turbine capital cost
    turbineCapitalCost= singleBladeCost*blades + \
                        hubCost + \
                        pitchSystemCost + \
                        noseConeCost + \
                        lowSpeedShaftCost + \
                        bearingCost + \
                        breakCouplingCost + \
                        generatorCost + \
                        electronicsCost+ \
                        yawSystemCost + \
                        mainframeCost + \
                        platformAndRailingCost + \
                        electricalConnectionCost + \
                        hydraulicAndCoolingSystemCost + \
                        nacelleCost + \
                        towerCost

    turbineCapitalCost *= turbineCapitalCostNormalization

    if not offshore:
        # Foundation
        foundationCost = 303.24*np.power((hh*sa), 0.4037)

        # Transportation
        transporationCostFactor = 1.581E-5 * np.power(cp,2) - 0.0375 * cp + 54.7
        transporationCost = transporationCostFactor * cp

        # Roads and civil work
        roadsAndCivilWorkFactor = 2.17E-6 * np.power(cp,2) - 0.0145 * cp + 69.54
        roadsAndCivilWorkCost = roadsAndCivilWorkFactor * cp

        # Assembly and installation
        assemblyAndInstallationCost = 1.965 * np.power((hh*rd), 1.1736)

        # Electrical Interface and connections
        electricalInterfaceAndConnectionFactor = (3.49E-6 * np.power(cp,2)) - (0.0221 * cp) + 109.7
        electricalInterfaceAndConnectionCost = electricalInterfaceAndConnectionFactor * cp

        # Engineering and permit factor
        engineeringAndPermitCostFactor = 9.94E-4 * cp + 20.31
        engineeringAndPermitCost = engineeringAndPermitCostFactor * cp

        # Add up other costs 
        otherCosts= foundationCost + \
                    transporationCost + \
                    roadsAndCivilWorkCost + \
                    assemblyAndInstallationCost + \
                    electricalInterfaceAndConnectionCost + \
                    engineeringAndPermitCost 


        # Get total cost
        totalCost = turbineCapitalCost + otherCosts*turbineCapitalCostNormalization

    else: # Offshore, following [4]

        turbineCapitalCost *= 1.135 # Marinization of turbine [1]

        # Get Foundation Cost
        foundationCost = np.zeros(depth.shape)

        lt25 = depth < 25
        if lt25.any(): foundationCost[lt25] = 1.4*((499*np.power(depth[lt25],2))+(6219*depth[lt25])+311810)

        gt25 = depth >= 25
        if gt25.any(): foundationCost[gt25] = 1.4*((440*np.power(depth[gt25],2))+(19695*depth[gt25])+901691)

        foundationCost *= cp/1000 # Make into Eur

        # Get grid cost
        gridCost = (0.38*busD+0.4*shoreD+76.6)*1e6/600
        gridCost *= cp/1000 # Make into Eur

        # Other Costs
        otherCosts = (turbineCapitalCost + foundationCost + gridCost) * 0.10

        # Some all costs
        totalCost = turbineCapitalCost + foundationCost + gridCost + otherCosts

    return totalCost
