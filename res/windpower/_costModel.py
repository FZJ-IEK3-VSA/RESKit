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
        if gt25.any(): foundationCost[gt25] = 1.4*((440*np.power(depth[gt25],2))+(19695*depth[gt25])+11785)

        foundationCost *= cp/1000 # Make into Eur

        # Get grid cost
        gridCost = (0.38*busD+0.4*shoreD+76.6)*1e6/600
        gridCost *= cp/1000 # Make into Eur

        # Other Costs
        otherCosts = (turbineCapitalCost + foundationCost + gridCost) * 0.10

        # Some all costs
        totalCost = turbineCapitalCost + foundationCost + gridCost + otherCosts

    return totalCost

def offshoreBOS(capacity, rotordiam, hubHeight, depth, distanceToShore, distanceToBus, foundation="monopile", mooringCount=3, anchor="DEA", turbineNumber=80, turbineSpacing=5, rowSpacing=9):
    # [1] https://www.nrel.gov/docs/fy17osti/66874.pdf
    # [2] Anders Mhyr, Catho Bjerkseter, Anders Agotnes and Tor A. Nygaard (2014) Levelised costs of energy for offshore floating wind turbines in a life cycle perspective
    # [3] Catho Bjerkseter and Anders Agotnes(2013) Levelised costs of energy for offshore floating wind turbine concenpts
    # [4] www.rpgcables.com/images/product/EHV-catalogue.pdf
    # [5] https://www.nrel.gov/docs/fy16osti/66262.pdf

    ## PREPROCESS INPUTS
    foundation = foundation.lower()
    if foundation=="monopile" or foundation=="jacket": fixedType = True
    elif foundation=="spar" or foundation=="semisubmersible": fixedType = False
    else: raise ValueError("Please choose one of the four foundation types: monopile, jacket, spar, or semisubmersible")

    cp = np.array(capacity/1000)
    rr = np.array(rotordiam/2)
    rd = np.array(rotordiam)
    hh = np.array(hubHeight)
    depth = np.array(depth)
    shoreD = np.array(distanceToShore)
    busD = np.array(distanceToBus)
    
    ## CONSTANTS AND ASSUMPTIONS (all from [1] except where noted)
    # Stucture are foundation
    embedmentDepth = 30 #meters
    monopileCostRate = 2250 #dollars/tonne
    monopileTPCostRate = 3230 #dollars/tonne
    sparSCCostRate = 3120 #dollars/tonne
    sparTCCostRate = 4222 #dollars/tonne
    sparBallCostRate = 100 #dollars/tonne
    jacketMLCostRate = 4680 #dollars/tonne
    jacketTPCostRate = 4500 #dollars/tonne
    jacketPileCostRate = 2250 #dollars/tonne
    semiSubmersibleSCCostRate = 3120 #dollars/tonne
    semiSubmersibleTCostRate = 6250 #dollars/tonne
    semiSubmersibleHPCostRate = 6250 #dollars/tonne
    mooringCostRate = 721 #dollars/tonne -- 0.12m diameter is chosen since it is the median in [1]
    outfittingSteelCost = 7250 #dollars/tonne

    #the values of anchor cost is calculated from Table8 in [2] by assuming a euro to dollar rate of 1.35
    DEA_anchorCost = 154 #dollars [2]
    SPA_anchorCost = 692 #dollars [2]

    # Electrical
    #current rating values are taken from source an approximate number is chosen from tables[4]
    cable1CurrentRating = 400 # [4]
    cable2CurrentRating = 600 # [4]
    exportCableCurrentRating = 1000 # [4]
    arrayVoltage = 33
    exportCableVoltage = 220
    powerFactor = 0.95
    buriedDepth = 1 #this value is chosen from [5] IF THIS CHANGES FROM ONE "singleStringPower1" needs to be updated
    catenaryLengthFactor = 0.04
    excessCableFactor = 0.1
    numberOfSubStations = 1 # From the example used in [5]
    arrayCableCost = 281000*1.35 # dollars/km (converted from EUR) [3]
    externalCableCost = 443000*1.35 # dollars/km (converted from EUR) [3]
    singleTurbineInterfaceCost = 0 # Could not find a number...
    substationInterfaceCost = 0 # Could not find a number...
    dynamicCableFactor = 2
    mainPowerTransformerCostRate = 12500 # dollers/MVA
    highVoltageSwitchgearCost = 950000 # dollars
    mediumVoltageSwitchgearCost = 500000 # dollars
    shuntReactorCostRate = 35000 # dollars/MVA
    dieselGeneratorBackupCost = 1000000 # dollars
    workspaceCost = 2000000 # dollars
    otherAncillaryCosts = 3000000 # dollars
    fabricationCostRate = 14500 # dollars/tonne
    topsideDesignCost = 4500000 # dollars
    assemblyFactor = 1 # could not find a number...
    offshoreSubstationSubstructureCostRate = 6250 # dollars/tonne
    substationSubstructurePileCostRate = 2250 # dollars/tonne
    interconnectVoltage = 345 # kV

    ## GENERAL (APEENDIX B in NREL BOS MODEL)
    hubDiam = cp/4 +2
    bladeLength = (rotordiam-hubDiam)/2

    nacelleWidth = hubDiam + 1.5
    nacelleLength = 2 * nacelleWidth

    # RNAMass is rotor nacelle assembly
    RNAMass = 2.082 * cp * cp + 44.59 * cp +22.48

    towerDiam = cp/2 + 4
    towerMass = (0.4 * np.pi * np.power(rr, 2) * hh -1500)/1000

    ## STRUCTURE AND FOUNDATION
    if foundation == 'monopile':
        monopileLength = depth + embedmentDepth + 5

        monopileMass = (np.power((cp*1000), 1.5) + (np.power(hh, 3.7)/10) + 2100 * np.power(depth, 2.25) + np.power((RNAMass*1000), 1.13))/10000
        monopileCost = monopileMass * monopileCostRate

        #monopile transition piece mass is called as monopileTPMass

        monopileTPMass =  np.exp(2.77 + 1.04*np.power(cp, 0.5) + 0.00127 * np.power(depth, 1.5))
        monopileTPCost = monopileTPMass * monopileTPCostRate

        foundationCost = monopileCost + monopileTPCost
        mooringAndAnchorCost = 0

    elif foundation == 'jacket':
        #jacket main lattice mass is called as jacketMLMass
        jacketMLMass = np.exp(3.71 + 0.00176 * np.power(cp , 2.5) + 0.645 * np.log(np.power(depth, 1.5)))
        jacketMLCost = jacketMLMass * jacketMLCostRate

        #jacket transition piece mass is called as jacketTPMass
        jacketTPMass = 1/(((-0.0131+0.0381)/np.log(cp))-0.00000000227*np.power(depth,3))
        jacketTPCost = jacketTPMass * jacketTPCostRate

        #jacket pile mass is called as jacketPileMass
        jacketPileMass = 8 * np.power(jacketMLMass, 0.5574)
        jacketPileCost = jacketPileMass * jacketPileCostRate

        foundationCost = jacketMLCost + jacketTPCost + jacketPileCost
        mooringAndAnchorCost=0

    elif foundation == 'spar':
        #spar stiffened column mass is called as sparSCMass
        sparSCMass = 535.93 + 17.664 * np.power(cp, 2) + 0.02328 * depth * np.log(depth)
        sparSCCost = sparSCMass * sparSCCostRate

        #spar tapered column mass is called as sparTCMass
        sparTCMass = 125.81 * np.log(cp) + 58.712
        sparTCCost = sparTCMass * sparTCCostRate

        #spar ballast mass is called as sparBallMass
        sparBallMass = -16.536 * np.power(cp,2) + 1261.8*cp - 1554.6
        sparBallCost = sparBallMass * sparBallCostRate

        foundationCost = sparSCCost + sparTCCost + sparBallCost

        if anchor == 'DEA': 
            anchorCost = DEA_anchorCost
            #the equation is derived from [3]
            mooringLength = 1.5 * depth + 350

        elif anchor == 'SPA':
            anchorCost = SPA_anchorCost
            #since it is assumed to have an angle of 45 degrees it is multiplied by 1.41 which is squareroot of 2 [3]
            mooringLength = 1.41 * depth

        else: raise ValueError("Please choose an anchor type!")

        mooringAndAnchorCost = mooringLength * mooringCostRate + anchorCost

    elif foundation == 'semisubmersible':
        #semiSubmersible stiffened column mass is called as semiSubmersibleSCMass
        semiSubmersibleSCMass = -0.9571 * np.power(cp , 2) + 40.89 * cp + 802.09
        semiSubmersibleSCCost = semiSubmersibleSCMass * semiSubmersibleSCCostRate

        #semiSubmersible truss mass is called as semiSubmersibleTMass
        semiSubmersibleTMass = 2.7894 * np.power(cp , 2) + 15.591 * cp + 266.03
        semiSubmersibleTCost = semiSubmersibleTMass * semiSubmersibleTCostRate

        #semiSubmersible heavy plate mass is called as semiSubmersibleHPMass
        semiSubmersibleHPMass = -0.4397 * np.power(cp , 2) + 21.145 * cp + 177.42
        semiSubmersibleHPCost = semiSubmersibleHPMass * semiSubmersibleHPCostRate

        foundationCost = semiSubmersibleSCCost + semiSubmersibleTCost + semiSubmersibleHPCost

        if anchor == 'DEA': 
            anchorCost = DEA_anchorCost
            #the equation is derived from [3]
            mooringLength = 1.5 * depth + 350

        elif anchor == 'SPA':
            anchorCost = SPA_anchorCost
            #since it is assumed to have an angle of 45 degrees it is multiplied by 1.41 which is squareroot of 2 [3]
            mooringLength = 1.41 * depth

        else: raise ValueError("Please choose an anchor type!")

        mooringAndAnchorCost = mooringLength * mooringCostRate + anchorCost

    if fixedType:
        secondarySteelSubstructureMass = np.zeros(depth.shape)
        gt4 = cp>4
        if gt4.any(): secondarySteelSubstructureMass[gt4] = 40 + (0.8 * (18 + depth[gt4]))
        lte4 = cp<=4
        if lte4.any(): secondarySteelSubstructureMass[lte4] = 40 + (0.8 * (18 + depth[lte4]))

    elif foundation == 'spar':
        secondarySteelSubstructureMass = np.exp(3.58+0.196*np.power(cp, 0.5)*np.log(cp) + 0.00001*depth*np.log(depth))

    elif foundation == 'semiSubmersible':
        secondarySteelSubstructureMass = -0.153 * np.power(cp,2) + 6.54 * cp + 128.34
    
    secondarySteelSubstructureCost = secondarySteelSubstructureMass * outfittingSteelCost

    totalStructureAndFoundationCosts = foundationCost +\
                                       mooringAndAnchorCost*mooringCount +\
                                       secondarySteelSubstructureCost

    ##ELECTRICAL INFRASTRUCTURE
    singleStringPower1 = np.sqrt(3)*cable1CurrentRating*arrayVoltage*powerFactor/1000
    singleStringPower2 = np.sqrt(3)*cable2CurrentRating*arrayVoltage*powerFactor/1000

    numberofStrings = np.floor_divide(turbineNumber*cp , singleStringPower2)

    numberofTurbinesperPartialString = np.round(np.remainder((turbineNumber*cp) , singleStringPower2))

    numberofTurbinesperArrayCable1 = np.floor_divide(singleStringPower1 , cp)

    numberofTurbinesperArrayCable2 = np.floor_divide(singleStringPower2 , cp)

    if numberofTurbinesperPartialString == 0:
        numberofTurbineInterfacesPerArrayCable1 = numberofTurbinesperArrayCable1 * numberofStrings * 2

        max1_Cable1 = np.maximum(numberofTurbinesperArrayCable1-numberofTurbinesperArrayCable2, 0)
        max2_Cable1 = 0
        numberofTurbineInterfacesPerArrayCable2 = (max1_Cable1 * numberofStrings + max2_Cable1) * 2

    else:
        numberofTurbineInterfacesPerArrayCable1 = (numberofTurbinesperArrayCable1 * numberofStrings + np.minimum((numberofTurbinesperPartialString-1),numberofTurbinesperArrayCable1)) * 2

        max1_Cable1 = np.maximum(numberofTurbinesperArrayCable1-numberofTurbinesperArrayCable2, 0)
        max2_Cable1 = 0 #due to the no partial string assumption
        numberofTurbineInterfacesPerArrayCable2 = (max1_Cable1 * numberofStrings + max2_Cable1) * 2 + 1


    numberofArrayCableSubstationInterfaces = numberofStrings
    
    if fixedType:
        arrayCable1Length = (turbineSpacing*rd+depth*2)*(numberofTurbineInterfacesPerArrayCable1/2)*(1+excessCableFactor)
        arrayCable1Length /= 1000 # convert to km

    else:
        systemAngle = -0.0047 * depth + 18.743

        freeHangingCableLength = (depth/np.cos(systemAngle*np.pi/180)*(catenaryLengthFactor+1))+ 190
        freeHangingCableLength /= 1000 # convert to km
        print("freeHangingCableLength:", freeHangingCableLength)

        fixedCableLength =(turbineSpacing * rd) - (2*np.tan(systemAngle*np.pi/180)*depth)-70
        fixedCableLength /= 1000 # convert to km
        print("fixedCableLength:", fixedCableLength)

        arrayCable1Length = (2 * freeHangingCableLength) * (numberofTurbineInterfacesPerArrayCable1/2)*(1+excessCableFactor)
        arrayCable1Length /= 1000 # convert to km
        print("arrayCable1Length:", arrayCable1Length)

    max1_Cable2 = np.maximum(numberofTurbinesperArrayCable2-1, 0)
    max2_Cable2 = np.maximum( numberofTurbinesperPartialString - numberofTurbinesperArrayCable2 -1, 0 )

    if numberofTurbinesperPartialString == 0:
        strFac = numberofStrings / numberOfSubStations
    else:
        strFac = numberofStrings / numberOfSubStations + 1

    if fixedType:
        arrayCable2Length = (turbineSpacing*rd+2*depth)*(max1_Cable2*numberofStrings+max2_Cable2) +\
                            numberOfSubStations*(strFac*(rd*rowSpacing)+\
                            (np.sqrt(np.power((rd*turbineSpacing*(strFac-1)),2)+np.power((rd*rowSpacing), 2))/2) +\
                            strFac*depth)*(excessCableFactor+1)
        arrayCable2Length /= 1000 # convert to km
        
        arrayCable1AndAncillaryCost = arrayCable1Length*arrayCableCost + singleTurbineInterfaceCost *\
            (numberofTurbineInterfacesPerArrayCable1+numberofTurbineInterfacesPerArrayCable2)

        arrayCable2AndAncillaryCost = arrayCable2Length*arrayCableCost +\
            singleTurbineInterfaceCost * (numberofTurbineInterfacesPerArrayCable1+numberofTurbineInterfacesPerArrayCable2) +\
            substationInterfaceCost * numberofArrayCableSubstationInterfaces

    else: 
        arrayCable2Length = (fixedCableLength +2*freeHangingCableLength)*(max1_Cable2*numberofStrings+max2_Cable2) +\
                            numberOfSubStations*(strFac*(rd*rowSpacing)+\
                            np.sqrt(np.power(( (2*freeHangingCableLength)*(strFac-1)+(rd*rowSpacing)-(2*np.tan(systemAngle*np.pi/180)*depth)-70), 2) +\
                                    np.power(fixedCableLength+2*freeHangingCableLength, 2))/2)*(excessCableFactor+1)
        arrayCable2Length /= 1000 # convert to km

        arrayCable1AndAncillaryCost = dynamicCableFactor *(arrayCable1Length*arrayCableCost +\
            singleTurbineInterfaceCost*(numberofTurbineInterfacesPerArrayCable1+numberofTurbineInterfacesPerArrayCable2))


        arrayCable2AndAncillaryCost = dynamicCableFactor *(arrayCable2Length*arrayCableCost +\
            singleTurbineInterfaceCost * (numberofTurbineInterfacesPerArrayCable1+numberofTurbineInterfacesPerArrayCable2) +\
            substationInterfaceCost * numberofArrayCableSubstationInterfaces)

    singleExportCablePower =  np.sqrt(3)*cable2CurrentRating*arrayVoltage*powerFactor/1000
    numberOfExportCables = np.floor_divide( cp*turbineNumber, singleExportCablePower)+1

    if fixedType:
        exportCableLength = (shoreD*1000+depth)*numberOfExportCables*1.1
        exportCableLength /= 1000 # convert to km

        exportCableandAncillaryCost = exportCableLength*externalCableCost + numberOfExportCables*substationInterfaceCost
    else:
        exportCableLength = (shoreD*1000+freeHangingCableLength+500)*numberOfExportCables*1.1
        exportCableLength /= 1000 # convert to km

        exportCableandAncillaryCost = externalCableCost +\
            ((exportCableLength - freeHangingCableLength -500)+dynamicCableFactor*(500+freeHangingCableLength)) +\
            numberOfExportCables*substationInterfaceCost

    numberOfSubStations = numberOfSubStations

    numberOfMainPowerTransformers = np.floor_divide(turbineNumber*cp,250)

    singleMptRating = np.round(turbineNumber*cp*1.15/numberOfMainPowerTransformers, -1)

    mainPowerTransformerCost = numberOfMainPowerTransformers*singleMptRating*mainPowerTransformerCostRate

    switchgearCost = numberOfMainPowerTransformers*(highVoltageSwitchgearCost+mediumVoltageSwitchgearCost)

    shuntReactorCost = singleMptRating * numberOfMainPowerTransformers * shuntReactorCostRate * 0.5

    ancillarySystemsCost = dieselGeneratorBackupCost + workspaceCost + otherAncillaryCosts

    offshoreSubstationTopsideMass = 3.85 * (singleMptRating*numberOfMainPowerTransformers) + 285
    offshoreSubstationTopsideCost = offshoreSubstationTopsideMass * fabricationCostRate + topsideDesignCost
    assemblyFactor = 1 # could not find a number...

    offshoreSubstationTopsideLandAssemblyCost = (switchgearCost+shuntReactorCost+mainPowerTransformerCost)*assemblyFactor

    if fixedType:
        offshoreSubstationSubstructureMass = 0.4*offshoreSubstationTopsideMass

        substationSubstructurePileMass = 8*np.power(offshoreSubstationSubstructureMass, 0.5574)

        offshoreSubstationSubstructureCost = offshoreSubstationSubstructureMass * offshoreSubstationSubstructureCostRate +\
                                             substationSubstructurePileMass * substationSubstructurePileCostRate
    else:

        # copied from above in case of spar
        if foundation == 'spar':
            semiSubmersibleSCMass = -0.9571 * np.power(cp , 2) + 40.89 * cp + 802.09
            semiSubmersibleSCCost = semiSubmersibleSCMass * semiSubmersibleSCCostRate

            #semiSubmersible truss mass is called as semiSubmersibleTMass
            semiSubmersibleTMass = 2.7894 * np.power(cp , 2) + 15.591 * cp + 266.03
            semiSubmersibleTCost = semiSubmersibleTMass * semiSubmersibleTCostRate

            #semiSubmersible heavy plate mass is called as semiSubmersibleHPMass
            semiSubmersibleHPMass = -0.4397 * np.power(cp , 2) + 21.145 * cp + 177.42
            semiSubmersibleHPCost = semiSubmersibleHPMass * semiSubmersibleHPCostRate

        semisubmersibleMass = semiSubmersibleSCMass + semiSubmersibleTMass + semiSubmersibleHPMass

        offshoreSubstationSubstructureMass = 2*(semiSubmersibleMass + secondarySteelSubstructureMass)

        substationSubstructurePileMass = 0
        
        semiSubmersibleCost = semiSubmersibleSCCost + semiSubmersibleTCost + semiSubmersibleHPCost
        offshoreSubstationSubstructureCost = 2*(semiSubmersibleTCostRate + mooringAndAnchorCost)

    onshoreSubstationCost = 11652 * (interconnectVoltage + cp*turbineNumber) + 1200000

    onshoreSubstationMiscCost = 11795 * np.power(cp*turbineNumber, 0.3549) + 350000

    overheadTransmissionLineCost = (1176*interconnectVoltage+218257) * np.power(busD,-0.1063)*busD

    switchyardCost = 18115*interconnectVoltage+165944

    totalElectricalInfrastructureCosts = arrayCable1AndAncillaryCost +\
                                         arrayCable2AndAncillaryCost +\
                                         exportCableandAncillaryCost +\
                                         mainPowerTransformerCost +\
                                         switchgearCost +\
                                         shuntReactorCost +\
                                         ancillarySystemsCost +\
                                         offshoreSubstationTopsideCost +\
                                         offshoreSubstationTopsideLandAssemblyCost +\
                                         offshoreSubstationSubstructureCost +\
                                         onshoreSubstationCost +\
                                         onshoreSubstationMiscCost +\
                                         overheadTransmissionLineCost +\
                                         switchyardCost
    totalElectricalInfrastructureCosts /= turbineNumber
    
    ## ASSEMBLY AND INSTALLATION


    ## TOTAL COST
    totalCost = totalStructureAndFoundationCosts + totalElectricalInfrastructureCosts
    return totalCost