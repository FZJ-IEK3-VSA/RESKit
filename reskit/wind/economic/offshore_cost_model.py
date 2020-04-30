import numpy as np
from .onshore_cost_model import onshore_tcc


def offshore_turbine_capex(capacity, hub_height, rotor_diam, depth, distance_to_shore, distance_to_bus=3, foundation="monopile", mooring_count=3, anchor="DEA", turbine_count=80, turbine_spacing=5, turbine_row_spacing=9):
    """
    TODO: Generalize this function further (like with the onshore cost model)
    Offshore turbine capital cost (TCC) is calculated by using the turbine capital cost 
    Fingersh et al.(2006) and Maples et al. (2010). Balance of system cost (BOS) is 
    calculated by using the equations of Maness et al.(2017). Afterwards, TCC and BOS 
    are scaled by cost breakdown suggested by Stehly et al.(2017).

    Stardard turbine for scaling the results is chosen from optimal turbine analysis.
    Capacity: 9.4 MW, Hub height: 135 m and rotor diameter: 210 m, monopile foundation
    Reference water depth is chosen as 40 m and distance to shore is chosen as 60 km.
    Inputs:
        capacity : Turbine nameplate capacity in kW
            float - Single value
            np.ndarray - multidimensional values 

        hub_height : Turbine hub height in meters
            float - Single value
            np.ndarray - multidimensional values

        rotor_diam : Turbine rotor diameter in meters
            float - Single value
            np.ndarray - multidimensional values

        depth : Water depth in meters (absolute value)
            float - Single value
            np.ndarray - multidimensional values

        distance_to_shore : Distance from shore in kilometers
            float - Single value
            np.ndarray - multidimensional values

        distance_to_bus : Distance from bus in kilometers
            float - Single value
            np.ndarray - multidimensional values

        foundation : Foundation type (monopile, jacket, semisubmersible, spar)
            str - Single value
            np.ndarray - multidimensional values

        mooring_count : Mooring count
            float - Single value
            np.ndarray - multidimensional values

        anchor : Anchor type
            str - Single value
            np.ndarray - multidimensional values

        turbine_count : Number of turbines in the windpark
            float - Single value
            np.ndarray - multidimensional values

        turbine_spacing : Spacing of turbines (5 * rotor diameter)
            float - Single value
            np.ndarray - multidimensional values

        turbine_row_spacing : Spacing of rows (9 * rotor diameter)
            float - Single value
            np.ndarray - multidimensional values

    # Sources: (Defaults from [1] or [5])
    # [1] https://www.nrel.gov/docs/fy17osti/66874.pdf
    # [2] Anders Mhyr, Catho Bjerkseter, Anders Agotnes and Tor A. Nygaard (2014) Levelised costs of energy for offshore floating wind turbines in a life cycle perspective
    # [3] Catho Bjerkseter and Anders Agotnes(2013) Levelised costs of energy for offshore floating wind turbine concenpts
    # [4] www.rpgcables.com/images/product/EHV-catalogue.pdf
    # [5] https://www.nrel.gov/docs/fy16osti/66262.pdf
    # [6] L. Fingersh, M. Hand, and A. Laxson. "Wind Turbine Design Cost and Scaling Model". 2006. NREL
    # [7] Tyler Stehly, Donna Heimiller, and George Scott. "2016 Cost of Wind Energy Review". 2017. NREL. https://www.nrel.gov/docs/fy18osti/70363.pdf

    """
    # PREPROCESS INPUTS
    cp = np.array(capacity / 1000)
    # rr = np.array(rotor_diam / 2)
    rd = np.array(rotor_diam)
    hh = np.array(hub_height)
    depth = np.abs(np.array(depth))
    distance_to_shore = np.array(distance_to_shore)
    distance_to_bus = np.array(distance_to_bus)

    # COMPUTE COSTS
    tcc = onshore_tcc(cp=cp * 1000, hh=hh, rd=rd)
    tcc *= 0.7719832742256006

    bos = offshore_bos(cp=cp, rd=rd, hh=hh, depth=depth, distance_to_shore=distance_to_shore, distance_to_bus=distance_to_bus, foundation=foundation,
                       mooring_count=mooring_count, anchor=anchor, turbine_count=turbine_count,
                       turbine_spacing=turbine_spacing, turbine_row_spacing=turbine_row_spacing, )

    bos *= 0.3669156255898912

    if foundation == 'monopile' or foundation == 'jacket':
        fin = (tcc + bos) * 20.9 / (32.9 + 46.2)  # Scaled according to tcc [7]
    else:
        fin = (tcc + bos) * 15.6 / (60.8 + 23.6)  # Scaled according to tcc [7]
    return tcc + bos + fin
    # return np.array([tcc,bos,fin])


def offshore_bos(cp, rd, hh, depth, distance_to_shore, distance_to_bus, foundation, mooring_count, anchor, turbine_count, turbine_spacing, turbine_row_spacing):
    # rr = rd / 2

    # prevent problems with negative depth values
    depth = np.abs(depth)

    foundation = foundation.lower()
    anchor = anchor.lower()
    if foundation == "monopile" or foundation == "jacket":
        fixedType = True
    elif foundation == "spar" or foundation == "semisubmersible":
        fixedType = False
    else:
        raise ValueError("Please choose one of the four foundation types: monopile, jacket, spar, or semisubmersible")

    # CONSTANTS AND ASSUMPTIONS (all from [1] except where noted)
    # Stucture are foundation
    # embedmentDepth = 30  # meters
    monopileCostRate = 2250  # dollars/tonne
    monopileTPCostRate = 3230  # dollars/tonne
    sparSCCostRate = 3120  # dollars/tonne
    sparTCCostRate = 4222  # dollars/tonne
    sparBallCostRate = 100  # dollars/tonne
    jacketMLCostRate = 4680  # dollars/tonne
    jacketTPCostRate = 4500  # dollars/tonne
    jacketPileCostRate = 2250  # dollars/tonne
    semiSubmersibleSCCostRate = 3120  # dollars/tonne
    semiSubmersibleTCostRate = 6250  # dollars/tonne
    semiSubmersibleHPCostRate = 6250  # dollars/tonne
    mooringCostRate = 721  # dollars/tonne -- 0.12m diameter is chosen since it is the median in [1]
    outfittingSteelCost = 7250  # dollars/tonne

    # the values of anchor cost is calculated from Table8 in [2] by assuming a euro to dollar rate of 1.35
    DEA_anchorCost = 154  # dollars [2]
    SPA_anchorCost = 692  # dollars [2]

    # Electrical
    # current rating values are taken from source an approximate number is chosen from tables[4]
    cable1CurrentRating = 400  # [4]
    cable2CurrentRating = 600  # [4]
    # exportCableCurrentRating = 1000  # [4]
    arrayVoltage = 33
    # exportCableVoltage = 220
    powerFactor = 0.95
    # buriedDepth = 1  # this value is chosen from [5] IF THIS CHANGES FROM ONE "singleStringPower1" needs to be updated
    catenaryLengthFactor = 0.04
    excessCableFactor = 0.1
    numberOfSubStations = 1  # From the example used in [5]
    arrayCableCost = 281000 * 1.35  # dollars/km (converted from EUR) [3]
    externalCableCost = 443000 * 1.35  # dollars/km (converted from EUR) [3]
    singleTurbineInterfaceCost = 0  # Could not find a number...
    substationInterfaceCost = 0  # Could not find a number...
    dynamicCableFactor = 2
    mainPowerTransformerCostRate = 12500  # dollers/MVA
    highVoltageSwitchgearCost = 950000  # dollars
    mediumVoltageSwitchgearCost = 500000  # dollars
    shuntReactorCostRate = 35000  # dollars/MVA
    dieselGeneratorBackupCost = 1000000  # dollars
    workspaceCost = 2000000  # dollars
    otherAncillaryCosts = 3000000  # dollars
    fabricationCostRate = 14500  # dollars/tonne
    topsideDesignCost = 4500000  # dollars
    assemblyFactor = 1  # could not find a number...
    offshoreSubstationSubstructureCostRate = 6250  # dollars/tonne
    substationSubstructurePileCostRate = 2250  # dollars/tonne
    interconnectVoltage = 345  # kV

    # GENERAL (APEENDIX B in NREL BOS MODEL)
    # hubDiam = cp / 4 + 2
    # bladeLength = (rd - hubDiam) / 2

    # nacelleWidth = hubDiam + 1.5
    # nacelleLength = 2 * nacelleWidth

    # RNAMass is rotor nacelle assembly
    RNAMass = 2.082 * cp * cp + 44.59 * cp + 22.48

    # towerDiam = cp / 2 + 4
    # towerMass = (0.4 * np.pi * np.power(rr, 2) * hh - 1500) / 1000

    # STRUCTURE AND FOUNDATION
    if foundation == 'monopile':
        # monopileLength = depth + embedmentDepth + 5

        monopileMass = (np.power((cp * 1000), 1.5) + (np.power(hh, 3.7) / 10) + 2100 * np.power(depth, 2.25) + np.power((RNAMass * 1000), 1.13)) / 10000
        monopileCost = monopileMass * monopileCostRate

        # monopile transition piece mass is called as monopileTPMass

        monopileTPMass = np.exp(2.77 + 1.04 * np.power(cp, 0.5) + 0.00127 * np.power(depth, 1.5))
        monopileTPCost = monopileTPMass * monopileTPCostRate

        foundationCost = monopileCost + monopileTPCost
        mooringAndAnchorCost = 0

    elif foundation == 'jacket':
        # jacket main lattice mass is called as jacketMLMass
        jacketMLMass = np.exp(3.71 + 0.00176 * np.power(cp, 2.5) + 0.645 * np.log(np.power(depth, 1.5)))
        jacketMLCost = jacketMLMass * jacketMLCostRate

        # jacket transition piece mass is called as jacketTPMass
        jacketTPMass = 1 / (((-0.0131 + 0.0381) / np.log(cp)) - 0.00000000227 * np.power(depth, 3))
        jacketTPCost = jacketTPMass * jacketTPCostRate

        # jacket pile mass is called as jacketPileMass
        jacketPileMass = 8 * np.power(jacketMLMass, 0.5574)
        jacketPileCost = jacketPileMass * jacketPileCostRate

        foundationCost = jacketMLCost + jacketTPCost + jacketPileCost
        mooringAndAnchorCost = 0

    elif foundation == 'spar':
        # spar stiffened column mass is called as sparSCMass
        sparSCMass = 535.93 + 17.664 * np.power(cp, 2) + 0.02328 * depth * np.log(depth)
        sparSCCost = sparSCMass * sparSCCostRate

        # spar tapered column mass is called as sparTCMass
        sparTCMass = 125.81 * np.log(cp) + 58.712
        sparTCCost = sparTCMass * sparTCCostRate

        # spar ballast mass is called as sparBallMass
        sparBallMass = -16.536 * np.power(cp, 2) + 1261.8 * cp - 1554.6
        sparBallCost = sparBallMass * sparBallCostRate

        foundationCost = sparSCCost + sparTCCost + sparBallCost

        if anchor == 'dea':
            anchorCost = DEA_anchorCost
            # the equation is derived from [3]
            mooringLength = 1.5 * depth + 350

        elif anchor == 'spa':
            anchorCost = SPA_anchorCost
            # since it is assumed to have an angle of 45 degrees it is multiplied by 1.41 which is squareroot of 2 [3]
            mooringLength = 1.41 * depth

        else:
            raise ValueError("Please choose an anchor type!")

        mooringAndAnchorCost = mooringLength * mooringCostRate + anchorCost

    elif foundation == 'semisubmersible':
        # semiSubmersible stiffened column mass is called as semiSubmersibleSCMass
        semiSubmersibleSCMass = -0.9571 * np.power(cp, 2) + 40.89 * cp + 802.09
        semiSubmersibleSCCost = semiSubmersibleSCMass * semiSubmersibleSCCostRate

        # semiSubmersible truss mass is called as semiSubmersibleTMass
        semiSubmersibleTMass = 2.7894 * np.power(cp, 2) + 15.591 * cp + 266.03
        semiSubmersibleTCost = semiSubmersibleTMass * semiSubmersibleTCostRate

        # semiSubmersible heavy plate mass is called as semiSubmersibleHPMass
        semiSubmersibleHPMass = -0.4397 * np.power(cp, 2) + 21.145 * cp + 177.42
        semiSubmersibleHPCost = semiSubmersibleHPMass * semiSubmersibleHPCostRate

        foundationCost = semiSubmersibleSCCost + semiSubmersibleTCost + semiSubmersibleHPCost

        if anchor == 'dea':
            anchorCost = DEA_anchorCost
            # the equation is derived from [3]
            mooringLength = 1.5 * depth + 350

        elif anchor == 'spa':
            anchorCost = SPA_anchorCost
            # since it is assumed to have an angle of 45 degrees it is multiplied by 1.41 which is squareroot of 2 [3]
            mooringLength = 1.41 * depth

        else:
            raise ValueError("Please choose an anchor type!")

        mooringAndAnchorCost = mooringLength * mooringCostRate + anchorCost

    if fixedType:
        if cp > 4:
            secondarySteelSubstructureMass = 40 + (0.8 * (18 + depth))
        else:
            secondarySteelSubstructureMass = 35 + (0.8 * (18 + depth))

    elif foundation == 'spar':
        secondarySteelSubstructureMass = np.exp(3.58 + 0.196 * np.power(cp, 0.5) * np.log(cp) + 0.00001 * depth * np.log(depth))

    elif foundation == 'semisubmersible':
        secondarySteelSubstructureMass = -0.153 * np.power(cp, 2) + 6.54 * cp + 128.34

    secondarySteelSubstructureCost = secondarySteelSubstructureMass * outfittingSteelCost

    totalStructureAndFoundationCosts = foundationCost +\
        mooringAndAnchorCost * mooring_count +\
        secondarySteelSubstructureCost

    # ELECTRICAL INFRASTRUCTURE
    # in the calculation of singleStringPower1 and 2, bur depth is assumed to be 1. Because of that the equation is simplified.
    singleStringPower1 = np.sqrt(3) * cable1CurrentRating * arrayVoltage * powerFactor / 1000
    singleStringPower2 = np.sqrt(3) * cable2CurrentRating * arrayVoltage * powerFactor / 1000

    numberofStrings = np.floor_divide(turbine_count * cp, singleStringPower2)

    # Only no partial string will be implemented
    numberofTurbinesperPartialString = 0  # np.round(np.remainder((turbine_count*cp) , singleStringPower2))

    numberofTurbinesperArrayCable1 = np.floor_divide(singleStringPower1, cp)

    numberofTurbinesperArrayCable2 = np.floor_divide(singleStringPower2, cp)

    numberofTurbineInterfacesPerArrayCable1 = numberofTurbinesperArrayCable1 * numberofStrings * 2

    max1_Cable1 = np.maximum(numberofTurbinesperArrayCable1 - numberofTurbinesperArrayCable2, 0)
    max2_Cable1 = 0
    numberofTurbineInterfacesPerArrayCable2 = (max1_Cable1 * numberofStrings + max2_Cable1) * 2

    numberofArrayCableSubstationInterfaces = numberofStrings

    if fixedType:
        arrayCable1Length = (turbine_spacing * rd + depth * 2) * (numberofTurbineInterfacesPerArrayCable1 / 2) * (1 + excessCableFactor)
        arrayCable1Length /= 1000  # convert to km
        #print("arrayCable1Length:", arrayCable1Length)
    else:
        systemAngle = -0.0047 * depth + 18.743

        freeHangingCableLength = (depth / np.cos(systemAngle * np.pi / 180) * (catenaryLengthFactor + 1)) + 190

        fixedCableLength = (turbine_spacing * rd) - (2 * np.tan(systemAngle * np.pi / 180) * depth) - 70

        arrayCable1Length = (2 * freeHangingCableLength) * (numberofTurbineInterfacesPerArrayCable1 / 2) * (1 + excessCableFactor)
        arrayCable1Length /= 1000  # convert to km

    max1_Cable2 = np.maximum(numberofTurbinesperArrayCable2 - 1, 0)
    max2_Cable2 = np.maximum(numberofTurbinesperPartialString - numberofTurbinesperArrayCable2 - 1, 0)

    strFac = numberofStrings / numberOfSubStations

    if fixedType:
        arrayCable2Length = (turbine_spacing * rd + 2 * depth) * (max1_Cable2 * numberofStrings + max2_Cable2) +\
            numberOfSubStations * (strFac * (rd * turbine_row_spacing) +
                                   (np.sqrt(np.power((rd * turbine_spacing * (strFac - 1)), 2) + np.power((rd * turbine_row_spacing), 2)) / 2) +
                                   strFac * depth) * (excessCableFactor + 1)
        arrayCable2Length /= 1000  # convert to km

        arrayCable1AndAncillaryCost = arrayCable1Length * arrayCableCost + singleTurbineInterfaceCost *\
            (numberofTurbineInterfacesPerArrayCable1 + numberofTurbineInterfacesPerArrayCable2)

        arrayCable2AndAncillaryCost = arrayCable2Length * arrayCableCost +\
            singleTurbineInterfaceCost * (numberofTurbineInterfacesPerArrayCable1 + numberofTurbineInterfacesPerArrayCable2) +\
            substationInterfaceCost * numberofArrayCableSubstationInterfaces

    else:
        arrayCable2Length = (fixedCableLength + 2 * freeHangingCableLength) * (max1_Cable2 * numberofStrings + max2_Cable2) +\
            numberOfSubStations * (strFac * (rd * turbine_row_spacing) +
                                   np.sqrt(np.power(((2 * freeHangingCableLength) * (strFac - 1) + (rd * turbine_row_spacing) - (2 * np.tan(systemAngle * np.pi / 180) * depth) - 70), 2) +
                                           np.power(fixedCableLength + 2 * freeHangingCableLength, 2)) / 2) * (excessCableFactor + 1)
        arrayCable2Length /= 1000  # convert to km

        arrayCable1AndAncillaryCost = dynamicCableFactor * (arrayCable1Length * arrayCableCost +
                                                            singleTurbineInterfaceCost * (numberofTurbineInterfacesPerArrayCable1 + numberofTurbineInterfacesPerArrayCable2))

        arrayCable2AndAncillaryCost = dynamicCableFactor * (arrayCable2Length * arrayCableCost +
                                                            singleTurbineInterfaceCost * (numberofTurbineInterfacesPerArrayCable1 + numberofTurbineInterfacesPerArrayCable2) +
                                                            substationInterfaceCost * numberofArrayCableSubstationInterfaces)

    singleExportCablePower = np.sqrt(3) * cable2CurrentRating * arrayVoltage * powerFactor / 1000
    numberOfExportCables = np.floor_divide(cp * turbine_count, singleExportCablePower) + 1

    if fixedType:
        exportCableLength = (distance_to_shore * 1000 + depth) * numberOfExportCables * 1.1
        exportCableLength /= 1000  # convert to km

        exportCableandAncillaryCost = exportCableLength * externalCableCost + numberOfExportCables * substationInterfaceCost
    else:
        exportCableLength = (distance_to_shore * 1000 + freeHangingCableLength + 500) * numberOfExportCables * 1.1
        exportCableLength /= 1000  # convert to km

        exportCableandAncillaryCost = exportCableLength * externalCableCost +\
            ((exportCableLength - freeHangingCableLength - 500) + dynamicCableFactor * (500 + freeHangingCableLength)) +\
            numberOfExportCables * substationInterfaceCost

    numberOfSubStations = numberOfSubStations

    numberOfMainPowerTransformers = np.floor_divide(turbine_count * cp, 250) + 1

    # equation 72 in [1] is simplified
    singleMPTRating = np.round(turbine_count * cp * 1.15 / numberOfMainPowerTransformers, -1)

    mainPowerTransformerCost = numberOfMainPowerTransformers * singleMPTRating * mainPowerTransformerCostRate

    switchgearCost = numberOfMainPowerTransformers * (highVoltageSwitchgearCost + mediumVoltageSwitchgearCost)

    shuntReactorCost = singleMPTRating * numberOfMainPowerTransformers * shuntReactorCostRate * 0.5

    ancillarySystemsCost = dieselGeneratorBackupCost + workspaceCost + otherAncillaryCosts

    offshoreSubstationTopsideMass = 3.85 * (singleMPTRating * numberOfMainPowerTransformers) + 285
    offshoreSubstationTopsideCost = offshoreSubstationTopsideMass * fabricationCostRate + topsideDesignCost
    assemblyFactor = 1  # could not find a number...

    offshoreSubstationTopsideLandAssemblyCost = (switchgearCost + shuntReactorCost + mainPowerTransformerCost) * assemblyFactor

    if fixedType:
        offshoreSubstationSubstructureMass = 0.4 * offshoreSubstationTopsideMass

        substationSubstructurePileMass = 8 * np.power(offshoreSubstationSubstructureMass, 0.5574)

        offshoreSubstationSubstructureCost = offshoreSubstationSubstructureMass * offshoreSubstationSubstructureCostRate +\
            substationSubstructurePileMass * substationSubstructurePileCostRate
    else:

        # copied from above in case of spar
        if foundation == 'spar':  # WHY WAS IT SPAR BEFORE? WE ARE DOING THINGS WITH SEMISUBMERSIBLE
            # if foundation == 'semisubmersible':
            semiSubmersibleSCMass = -0.9571 * np.power(cp, 2) + 40.89 * cp + 802.09
            semiSubmersibleSCCost = semiSubmersibleSCMass * semiSubmersibleSCCostRate

            # semiSubmersible truss mass is called as semiSubmersibleTMass
            semiSubmersibleTMass = 2.7894 * np.power(cp, 2) + 15.591 * cp + 266.03
            semiSubmersibleTCost = semiSubmersibleTMass * semiSubmersibleTCostRate

            # semiSubmersible heavy plate mass is called as semiSubmersibleHPMass
            semiSubmersibleHPMass = -0.4397 * np.power(cp, 2) + 21.145 * cp + 177.42
            semiSubmersibleHPCost = semiSubmersibleHPMass * semiSubmersibleHPCostRate

        semiSubmersibleMass = semiSubmersibleSCMass + semiSubmersibleTMass + semiSubmersibleHPMass

        offshoreSubstationSubstructureMass = 2 * (semiSubmersibleMass + secondarySteelSubstructureMass)

        substationSubstructurePileMass = 0

        # semiSubmersibleCost = semiSubmersibleSCCost + semiSubmersibleTCost + semiSubmersibleHPCost
        offshoreSubstationSubstructureCost = 2 * (semiSubmersibleTCostRate + mooringAndAnchorCost)

    onshoreSubstationCost = 11652 * (interconnectVoltage + cp * turbine_count) + 1200000

    onshoreSubstationMiscCost = 11795 * np.power(cp * turbine_count, 0.3549) + 350000

    overheadTransmissionLineCost = (1176 * interconnectVoltage + 218257) * np.power(distance_to_bus, -0.1063) * distance_to_bus

    switchyardCost = 18115 * interconnectVoltage + 165944

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
    totalElectricalInfrastructureCosts /= turbine_count

    # ASSEMBLY AND INSTALLATION
    """
    assembly and installation could not be implemented due to the excessive number of unspecified 
    constants in [1]. Therefore empirical equations were derived which fit the sensitivities to
    the baseline plants shown in [1]. These ended up being linear equations in turbine capacity and 
    sea depth (only for floating turbines).
    """
    assemblyAndInstallationCost = np.ones(totalElectricalInfrastructureCosts.shape)

    if fixedType:
        assemblyAndInstallationCost *= 4200000
    else:
        assemblyAndInstallationCost *= 5500000

    # depth depedance
    if fixedType:
        pass
    else:
        # Normalized to 1 at 250m depth
        assemblyAndInstallationCost *= 0.00041757917648320338 * depth + 0.89560520587919934

    # Capacity dependance
    # Normalized to 1 at 6 MW
    assemblyAndInstallationCost *= 0.05947387 * cp + 0.64371944

    # OTHER THINGS
    # Again, many constants were used in [1] but not defined. Also, many of the costs were given in the
    # context of the USA. Therefore the other groups were are simply treated as percentages which
    # fit the examples shown in [1] or [7]

    #########################################
    # The below corresponds to other costs in [1]
    #tot = (assemblyAndInstallationCost + totalElectricalInfrastructureCosts + totalStructureAndFoundationCosts)/(1-0.06)

    #commissioning = tot*0.015
    #portAndStaging = tot*0.005
    #engineeringManagement = tot*0.02
    #development = tot*0.02

    #########################################
    # The below cooresponds to cost percentages in [7]
    if fixedType:
        tot = (assemblyAndInstallationCost * 19.0 +
               totalElectricalInfrastructureCosts * 9.00 +
               totalStructureAndFoundationCosts * 13.9) / 46.2

        commissioning = tot * (0.8 / 46.2)
        portAndStaging = tot * (0.5 / 46.2)
        engineeringManagement = tot * (1.6 / 46.2)
        development = tot * (1.4 / 46.2)

    else:
        tot = (assemblyAndInstallationCost * 11.3 +
               totalElectricalInfrastructureCosts * 10.9 +
               totalStructureAndFoundationCosts * 34.1) / 60.8

        commissioning = tot * (0.8 / 60.8)
        portAndStaging = tot * (0.6 / 60.8)
        engineeringManagement = tot * (2.2 / 60.8)
        development = tot * (1 / 60.8)

    # TOTAL COST
    totalCost = commissioning +\
        assemblyAndInstallationCost +\
        totalElectricalInfrastructureCosts +\
        totalStructureAndFoundationCosts +\
        portAndStaging +\
        engineeringManagement +\
        development

    return totalCost
