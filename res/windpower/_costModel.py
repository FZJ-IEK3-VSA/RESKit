from ._util import *

def nrelCostModel(capacity, hubHeight, rotorDiameter, gearBox="direct", gdpEscalator=1, bladeMaterialEscalator=1, blades=3):
    """Cost model built following the NREL report

    - capacity in MW
    """

    rr = rotorDiameter/2
    hh = hubHeight
    cp = capacity*1000

    totalCost = 0 
    
    # Blade costs
    singleBladeMass = 0.1452 * np.power(rr, 2.9158)
    singleBladeCost = ((0.4019*np.power(rr, 3)-955.24)*bladeMaterialEscalator + 2.7445*np.power(rr, 2.5025)*gdpEscalator)*(1-0.28)
    totalCost += singleBladeCost*blades

    # Hub cost
    hubMass = 0.954 * singleBladeMass + 5680.3
    hubCost = hubMass * 4.25
    totalCost += hubCost

    # Pitch system costs
    pitchBearingMass = 0.1295*singleBladeMass*blades + 491.3
    pitchSystemMass = pitchBearingMass * 1.328 + 555
    pitchSystemCost = 2.28 * (0.2106 * np.power(2*rr, 2.6578))
    totalCost += pitchSystemCost

    # Spinner/Nose cone
    noseConeMass = 18.5 * 2 * rr -520.5
    noseConeCost = noseConeMass * 5.57
    totalCost += noseConeCost

    # Low speed shaft
    lowSpeedShaftMass = 0.0142 * np.power(2 * rr, 2.888)
    lowSpeedShaftCost = 0.01 * np.power(2*rr, 2.887)
    totalCost += lowSpeedShaftCost

    # Main bearing
    bearingMass = (2*rr*8/600 - 0.033) * 0.0092 * np.power(2*rr, 2.5)
    bearingCost = 2*bearingMass * 17.6
    totalCost += bearingCost

    # Gearbox
    if gearBox=="direct":
        gearBoxCost = 0
    else:
        raise RuntimeError("implement the others!")

    totalCost += gearBoxCost

    # Brakes, Coupling, and others
    brakesAndCouplingCost = 1.9894 * cp - 0.1141
    brakesAndCouplingMass = brakesAndCouplingCost/10
    totalCost = brakesAndCouplingCost

    # Generator 
    if gearBox=="direct":
        generatorCost = cp * 219.33    
    else:
        raise RuntimeError("implement the others!")

    totalCost += generatorCost

    # Electronics
    electronicsCost = cp * 79
    totalCost += electronicsCost

    # Yaw drive and bearing
    ydabCost = 2*(0.0339*np.power(2*rr, 2.964))
    totalCost += ydabCost

    # Mainframe
    if gearBox=="direct":
        mainframeMass = 1.228 * np.power(2*rr, 1.953)
        mainframeCost = 627.28 * np.power(2*rr, 0.85)
    else:
        raise RuntimeError("implement the others!")

    totalCost += mainframeCost

    # Platform and railings
    pfrlMass = 0.125*mainframeMass
    pfrlCost = pfrlMass * 8.7
    totalCost += pfrlCost

    # Electrical Connections
    connectionCost = cp * 40
    totalCost += connectionCost

    # Hydraulic and cooling systems
    hyCoCost = cp * 12
    totalCost += hyCoCost

    # Nacelle Cover
    nacelleCost = 11.537*cp + 3849.7
    totalCost += nacelleCost

    # Tower
    towerMass = 0.3973 * np.pi * np.power(rr,2) * hh - 1414
    towerCost = towerMass * 1.5
    totalCost += towerCost

    # Foundation
    foundationCost = 303.24 * np.power( hh*np.pi * np.power(rr,2), 0.4037)
    totalCost += foundationCost

    # Assembly & Installation
    assInCost = 1.965 * np.power( hh * 2 * rr, 1.1736)
    totalCost += assInCost

    # Electrical interfaces 
    elecIntCost = cp*(3.49E-6 * cp**2 - 0.0221 * cp + 109.7)
    totalCost += elecIntCost

    # Done!
    return totalCost

    
def NormalizedCostModel(baseModel=nrelCostModel, normalizedCapacity=3.6, normalizedHubHeight=90, normalizedRotorDiameter=120, normalizedCost=3600000, constantCost=0, **kwargs):
    """Normalize a given cost model based on the expected cost of a particular set of turbine parameters

     * The default setup normalizes the following turbine parameters to the basic assumption of 1000 Euros/kW:
        - Capacity: 3.6 MW
        - Hub Height: 90 meters
        - Rotor Diameter: 120 meters
     * Returns a function which provides cost estimates based on the inputs of the base model
     * All executions of the returned function will be scaled around the given normalized cost
    """
    baseCost = baseModel(capacity=normalizedCapacity, hubHeight=normalizedHubHeight, rotorDiameter=normalizedRotorDiameter, **kwargs)
    scaling = normalizedCost / baseCost

    def outputFunc(capacity, hubHeight, rotorDiameter, **kwargs):
        """Normalized cost model. Returns cost in Euros

        inputs:
            capacity - float : The wind turbine capcity in MW
            hubHeight - int : The hub height in meters
            rotorDiameter - int : The rotor diameter in meters
        """
        return scaling*baseModel(capacity=capacity, hubHeight=hubHeight, rotorDiameter=rotorDiameter, **kwargs) + constantCost

    return outputFunc