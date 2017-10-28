import numpy as np
import pandas as pd
from scipy.interpolate import splrep, splev
from scipy.stats import norm
from collections import namedtuple, OrderedDict
from glob import glob
from os.path import join, dirname

from res.util import *
from res.weather import windutil, NCSource

##################################################
## Make a turbine model library
TurbineInfo = namedtuple('TurbineInfo','profile meta')
def parse_turbine(path):
    meta = OrderedDict()
    with open(path) as fin:
        # Meta extraction mode
        while True:
            line = fin.readline()[:-1]
            if line == "" or line[0]=="#": continue # skip blank lines and comment lines
            if 'power curve' in line.lower(): break
            sLine = line.split(',')
            try:
                meta[sLine[0].title()] = float(sLine[1])
            except:
                meta[sLine[0].title()] = sLine[1]
        
        # Extract power profile
        tmp = pd.read_csv(fin)
        power = [(ws,output) for i,ws,output in tmp.iloc[:,:2].itertuples()]
    
    return TurbineInfo(power, meta)     

turbineFiles = glob(join(dirname(__file__),"..","..","..","data","turbines","*.csv"))


tmp = []
for f in turbineFiles:
    try:
        tmp.append(parse_turbine(f))
    except:
        print("failed to parse:", f)


TurbineLibrary = pd.DataFrame([i.meta for i in tmp])
TurbineLibrary.set_index('Model', inplace=True)
TurbineLibrary['Performance'] = [x.profile for x in tmp]

####################################################
## Simulation for a single turbine
TurbinePerformance = namedtuple("TurbinPerformance", "production capacityFactor")
def simulateTurbine( windspeed, performance='E115 3.0MW', measuredHeight=None, roughness=None, alpha=None, hubHeight=None, loss=0.08, capacity=None):
    """
    Perform simple windpower simulation for a single turbine. Can also project to a hubheight before
    simulating.

    Notes:
        * In order to project to a hub height, the measuredHeight, hubHeight and either roughness or 
          alpha must be provided
            - weather.windutil.roughnessFromCLC, .roughnessFromGWA, and .alphaFromGWA can help 
              provide these factors
        * If no projection factors are given, windspeeds are assumed to already be at teh desired 
          hub height
    Inputs:
        windspeed - np-array, list of np-arrays, pd-Series, or pd-DataFrame
            * Time series of measured wind speeds

        performance 
            [ (float, float), ... ]
                * An array of "wind speed" to "power output" pairs, as two-member tuples, maping the 
                  power profile of the turbine to be simulated
                * The performance pairs must contain the boundary benhavior:
                    - The first (after sorting by wind speed) pair will be used as the 
                      "cut in"
                    - The last (after sorting) pair will be used as the "cut out" 
                    - The maximal pair will be used as the rated speed
            str
                * An identifier from the TurbineLibrary dictionary

        measuredHeight - float, or list of floats matching the number of wind speed time series
            * The height (in meters) where the wind speeds were measured 

        roughness - float, or list of floats matching the number of wind speed time series
            * The roughness length of the area associated with the measured wind speeds
            ! Providing this input instructs the res.weather.windutil.projectByLogLaw function
            ! Cannot be used in conjunction with 'alpha'
    
        alpha - float, or list of floats matching the number of wind speed time series
            * The alpha value of the area associated with the measured wind speeds
            ! Providing this input instructs the res.weather.windutil.projectByPowerLaw function
    
        hubHeight - float, or list of floats matching the number of wind speed time series
            * The hub height (in meters) of the wind turbine to simulate

        loss - float
            * A constant loss rate to apply to the simulated turbine(s)

        capacity - floar
            * The maximal capacity of the turbine being simulated
            * If 'None' is given, then the following occurs:
                - When performance is the name of a turbine, the capacity is read from the TurbineLibrary
                - When the performance is a list of windspeed-power_output pairs, then the capacity is 
                  taken as the maximum power output
    
    Returns: ( performance, hub-wind-speeds )
        performance - A numpy array of performance values
        hub-wind-speeds - The projected wind speeds at the turbine's hub height
    """
    ############################################
    # make sure we have numpy types or pandas types
    if isinstance(windspeed, pd.DataFrame): 
        isNumpy = False
        isSeries = False
        N = windspeed.shape[1]
    elif isinstance(windspeed, pd.Series): 
        isNumpy = False
        isSeries = True
        N = 1
    elif isinstance(windspeed, np.ndarray): 
        isNumpy = True
        if len(windspeed.shape)>1:
            N = windspeed.shape[1]
        else:
            N = 1
    elif isinstance(windspeed, list):
        try: # Test if we have a list of lists
            windspeed[0][0]
            windspeed = np.column_stack(windspeed)
            N = windspeed.shape[1]
        except TypeError: # We only have a single time series as a list
            windspeed = np.array(windspeed)
            N = 1
        isNumpy = True
    else:
        raise ResError("Could not understand Input")

    ############################################
    # Set performance
    if isinstance(performance,str): 
        if capacity is None: capacity = TurbineLibrary.ix[performance].Capacity
        performance = np.array(TurbineLibrary.ix[performance].Performance)
    elif isinstance(performance, list):
        performance = np.array(performance)
        if capacity is None: capacity = performance[:,1].max()

    ############################################
    # Convert to wind speeds at hub height
    #  * Follows the "log-wind profile" assumption
    if not (measuredHeight is None and hubHeight is None and roughness is None and alpha is None):
        # check inputs
        if measuredHeight is None or hubHeight is None:
            raise ResError("When projecting, both a measuredHeight and hubHeight must be provided")

        # make sure all types are float, pandas series, or numpy array
        def fixVal(val, name):
            if isinstance(val, float) or isinstance(val, int):
                val = np.float(val)
            elif isinstance(val, list):
                if len(val)==N: val = np.array(val)
                else: raise ResError(name + " does not have an appropriate length")
            elif isinstance(val, np.ndarray):
                if val.shape == (N,): val = val
                elif val.shape == (N,1): val = val[:,0]
                else: raise ResError(name + " does not have an appropriate shape")
            elif isinstance(val, pd.Series) or isinstance(val, pd.DataFrame):
                if val.shape == (N,): val = val
                elif val.shape == (N,1): val = val.iloc[:,0]
                else: raise ResError(name + " does not have an appropriate shape")

                if isNumpy: val = val.values
                else:
                    if not val.index.equals(windspeed.columns):
                        raise ResError("%s indexes do not match windspeed columns"%name)
            elif val is None: val = None
            else: raise ResError(name+" is not appropriate. (must be a numeric type, or a one-dimensionsal set of numeric types (one for each windspeed time series)")

            return val


        measuredHeight = fixVal(measuredHeight,"measuredHeight")
        hubHeight      = fixVal(hubHeight,"hubHeight")
        roughness      = fixVal(roughness,"roughness")
        alpha          = fixVal(alpha,"alpha")

        # do projection
        if not roughness is None:
            windspeed = windutil.projectByLogLaw(windspeed, measuredHeight=measuredHeight,
                                        targetHeight=hubHeight, roughness=roughness)
        elif not alpha is None:
            windspeed = windutil.projectByPowerLaw(windspeed, measuredHeight=measuredHeight,
                                        targetHeight=hubHeight, alpha=alpha)
        else:
            raise ResError("When projecting, either roughness or alpha must be given")

    ############################################
    # map wind speeds to power curve using a spline
    powerCurve = splrep(performance[:,0], performance[:,1])
    if isNumpy:
        powerGen = splev(windspeed, powerCurve)*(1-loss)
    else:
        powerGen = splev(windspeed.values, powerCurve)*(1-loss)

    # Do some "just in case" clean-up
    maxPower = performance[:,1].max() # use the max power as as ceiling
    cutin = performance[:,0].min() # use the first defined windspeed as the cut in
    cutout = performance[:,0].max() # use the last defined windspeed as the cut out 

    powerGen[powerGen<0]=0 # floor to zero
    powerGen[powerGen>maxPower]=maxPower # ceiling at max
    if isNumpy:
        powerGen[windspeed<cutin]=0 # Drop power to zero before cutin
        powerGen[windspeed>cutout]=0 # Drop power to zero after cutout
    else:
        powerGen[windspeed.values<cutin]=0 # Drop power to zero before cutin
        powerGen[windspeed.values>cutout]=0 # Drop power to zero after cutout
    
    ############################################
    # make outputs
    if not isNumpy:
        if isSeries:
            powerGen = pd.Series(powerGen, index=windspeed.index, name='production')
        else:
            powerGen = pd.DataFrame(powerGen, columns=windspeed.columns, index=windspeed.index)
    capFactor = powerGen.mean(axis=0)/capacity

    # Done!
    return TurbinePerformance(powerGen, capFactor)

def singleTurbine(**kwargs):
    print( "Forwarding to 'simulateTurbine'" )
    return simulateTurbine(**kwargs)

####################################################
## Simulation for a single turbine
def simulateArea( source, area, performance='E115_3200', measuredHeight=None, hubHeight=None, loss=0.08, leSource=None, gwaSource=None, **kwargs):
    """
    Perform wind power simulation for an area. A statistical distribution of expected wind speeds will be generated
    which will then be used to convolve the wind turbine's power-curve. A projection from a measure height to 
    a hub height can also be incorporated.

    Notes:
        * In order to project to a hub height, the measuredHeight, hubHeight and either roughness or 
          alpha must be provided
            - weather.windutil.roughnessFromCLC, .roughnessFromGWA, and .alphaFromGWA can help 
              provide these factors
        * If no projection factors are given, wind speeds are assumed to already be at the desired 
          hub height
    """
    ############################################

    ## Ensure source is okay
    if isinstance(source, NCSource):
        if not "windspeed" in source.data:
            source.loadWindspeed()
    else:
        raise ResError("Source is not an NCSource type")

    ## Ensure the area input is a geokit RegionMask
    rmKwargs = {}
    rmKwargs["srs"] = kwargs.pop("srs","europe_m")
    rmKwargs["pixelSize"] = kwargs.pop("pixelSize",100)

    area = rm.RegionMask.load(area)

    ## Evaluate eligibility if given (otherwise just use the area's mask)
    if not leSource is None:
        le = area.warp(leSource) > 0.5 # greater than 0.5 ensure we have a boolean mask
    else:
        le = area.mask

    ## Break the surviving areas into contexts from the wind speed source
    # Find the range of indexes which the region falls into
    lonMin, latMin, lonMax, latMax = area.extent.castTo(gk.srs.EPSG4326).xyXY

    extentIdx = source

###########################################################
## Convolute Power Curve
def convolutePowerCurveByGuassian(stdScaling=0.2, stdBase=0.6, performance='E115 3.0MW', minSpeed=0.01, maxSpeed=40, steps=4000, outputResolution=0.1):
    # Set performance
    if isinstance(performance,str): 
        performance = np.array(TurbineLibrary.ix[performance].Performance)
    elif isinstance(performance, list):
        performance = np.array(performance)

    # Initialize windspeed axis
    ws = np.linspace(minSpeed, maxSpeed, steps)
    dws = ws[1]-ws[0]

    # check if we have enough resolution
    tmp = (stdScaling*5+stdBase)/dws
    if  tmp < 1.0: # manually checked threshold
        if tmp < 0.25: # manually checked threshold
            raise ResError("Insufficient number of 'steps'")
        else:
            print("WARNING: 'steps' may not be high enough to properly compute the convoluted power curve. Check results or use a higher number of steps")
    
    # Initialize vanilla power curve
    powerCurve = splrep(performance[:,0], performance[:,1])

    perf = np.zeros(steps)
    perf[ws<performance[:,0].max()] = splev(ws[ws<performance[:,0].max()], powerCurve)

    perf[ws<performance[:,0].min()] = 0 # set all windspeed less than cut-in speed to 0
    perf[ws>performance[:,0].max()] = 0 # set all windspeed greater than cut-out speed to 0 (just in case)
    perf[perf<0] = 0 # force a floor of 0
    perf[perf>performance[:,1].max()] = performance[:,1].max() # force a ceiling of the max capacity
    
    # Begin convolution
    cPerf = np.zeros(steps)
    for i,ws_ in enumerate(ws):
        cPerf[i] = (norm.pdf(ws, loc=ws_, scale=stdScaling*ws_+stdBase)*perf).sum()*dws
        
    # Done!
    return np.column_stack([ws,cPerf])


class TerrainComplexityConvoluter(object):
    def __init__(s, terrainComplexityFile, turbine, tcStep=5, tcMax=400, preEvaluate=True):
        s.terrainComplexityFile = terrainComplexityFile
        s.turbine = turbine
        s._tcStep = int(tcStep)
        s._tcMax = int(tcMax)

        s.ws = np.linspace(0, 40, 4001)[1:]
        s._dws = s.ws[1]-s.ws[0]
        s.Capacity = TurbineLibrary.ix[turbine].Capacity

        s.evaluationTCs = np.arange(0, tcMax+0.01, tcStep)
        s._convolutedPowerCurves = pd.Series([None]*s.evaluationTCs.size, index=s.evaluationTCs)

        # make unconvoluted perfomrance
        performance = np.array(TurbineLibrary.ix[turbine].Performance)
        powerCurve = splrep(performance[:,0], performance[:,1])

        perf = np.zeros(s.ws.size)
        perf[s.ws<performance[:,0].max()] = splev(s.ws[s.ws<performance[:,0].max()], powerCurve)

        perf[s.ws<performance[:,0].min()] = 0 # set all windspeed less than cut-in speed to 0
        perf[s.ws>performance[:,0].max()] = 0 # set all windspeed greater than cut-out speed to 0 (just in case)
        perf[perf<0] = 0 # force a floor of 0
        perf[perf>performance[:,1].max()] = performance[:,1].max() # force a ceiling of the max capacity

        s.unconvolutedPowerCurve = perf

        s.storage = []

        # Setup all, maybe
        if preEvaluate:
            s._preEvaluateAll()

    @staticmethod
    def _A(tc): return 0.25*(1-np.exp(-tc/10))+0.4

    @staticmethod
    def _B(tc): return tc*0+0.525

    @staticmethod
    def _C(tc): return 0.9 - tc/300*0.4

    @staticmethod
    def _D(tc): return 0.085*(1-np.exp(-tc/150))+0.009

    def relativeSig(s, tc): 
        return s._A(tc)*np.exp( -np.power(s.ws,s._B(tc))*s._C(tc)) + s._D(tc)

    def _preEvaluateAll(s):
        for tc in s.evaluationTCs: s._evaluateComplexity(tc)

    def _evaluateComplexity(s,tc):
        convolutedPowerCurve = np.zeros(s.ws.size)

        for i, ws, rsig in zip(range(s.ws.size), s.ws, s.relativeSig(tc)):
            convolutedPowerCurve[i] = (s.unconvolutedPowerCurve*norm.pdf(s.ws, scale=ws*rsig, loc=ws)).sum()*s._dws

        s._convolutedPowerCurves.set_value(tc, convolutedPowerCurve)

    def performance(s,tc):
        return np.column_stack( [s.ws,s[tc]] )

    def __getitem__(s,tc):
        tc = s._tcStep*(int(tc)//s._tcStep)
        if tc > s._tcMax: tc = s._tcMax
        if s._convolutedPowerCurves.ix[tc] is None:
            s._evaluateComplexity(tc)

        return s._convolutedPowerCurves.ix[tc]

    def convolutedPowerCurveAtLocation(s, loc):
        loc = Location.ensureLocation(loc, forceAsArray=True)
        tcVals = gk.raster.extractValues(s.terrainComplexityFile, loc, noDataOkay=False).data.values

        if len(loc)==1:
            return s[tcVals[0]]
        else:
            for v in tcVals: yield s[v]

    def getTerrainComplexityAtLocation(s,loc):
        return gk.raster.extractValues(s.terrainComplexityFile, loc, noDataOkay=False).data.values


def costModelNrelBaseline(capacity, hubHeight, rotorDiameter, gearBox="direct", gdpEscalator=1, bladeMaterialEscalator=1, blades=3):
    """Cost model built following the NREL report

    - capacity in MW
    """

    rr = rotorDiameter/2
    hh = hubHeight
    cp = capacity/1000

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

    # Assembly & Insstallation
    assInCost = 1.965 * np.power( hh * 2 * rr, 1.1736)
    totalCost += assInCost

    # Electrical interfaces 
    elecIntCost = cp*(3.49E-6 * cp**2 - 0.0221 * cp + 109.7)
    totalCost += elecIntCost

    # Done!
    return totalCost

    
