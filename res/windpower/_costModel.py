from ._util import *

def onshoreCSM(capacity=4200, hubHeight=129, rotorDiam=141, normalization=0.708365390407, tccShare=0.673):
    """
    Onshore wind turbine cost and scaling model (csm) built following [1] and update following [2]. 
    Considers only the turbine capital cost estimations for a 3-bladed, direct drive turbine.
    Claimed to be derived from real cost data and valid (for costs at the time) up until 10 MW capacity.
    
    Base-line (default) turbine characteristics correspond to the expected typical onshore turbine in 2050.
    The normalization value adjusts the output to match 1000 Eur/kW including all costs. 
    Only the turbine capital cost (tcc) is adjusted according to capacity, rotor diameter, and hub height.
    Balance of system costs and other financial costs are added as fixed percentages according to [3].

    Summary equation:
        cost = originalNrelCSM(capacity, hubheight, rotorDiam ) * normalization / tccShare

    Inputs:
        capacity : Turbine nameplate capacity in kW
            float - Single value
            np.ndarray - multidimensional values 

        hubHeight : Turbine hub height in meters
            float - Single value
            np.ndarray - multidimensional values

        rotorDiam : Turbine rotor diameter in meters
            float - Single value
            np.ndarray - multidimensional values

        normalization - float : A normalization value to adjust the output to a desired context

        tccShare - float : The share of turbine capital cost in the total cost
    
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

    """
    gdpEscalator=2.5 # Chosen to match example given in [1]
    bladeMaterialEscalator=1
    blades = 3    
    
    rd = np.array(rotorDiam)
    rr = np.array(rotorDiam/2)
    hh = np.array(hubHeight)
    cp = np.array(capacity)
    sa = np.array((np.pi*rr*rr))
    
    # Blade Cost
    singleBladeMass = 0.4948 * np.power(rr,2.53)
    singleBladeCost = ((0.4019*np.power(rr, 3)-21051)*bladeMaterialEscalator + 2.7445*np.power(rr, 2.5025)*gdpEscalator)*(1-0.28)

    # Hub 
    hubMass = 0.954*singleBladeMass+5680.3
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
    """
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

    # Levelized Replacement Cost
    ## Not included here
    """
    # Get total cost
    totalCost = singleBladeCost*blades + \
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
                ### Balance of system costs not included....
                #foundationCost + \
                #transporationCost + \
                #roadsAndCivilWorkCost + \
                #assemblyAndInstallationCost + \
                #electricalInterfaceAndConnectionCost + \
                #engineeringAndPermitCost 

    return totalCost*normalization/tccShare

