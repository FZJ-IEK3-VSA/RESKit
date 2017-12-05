from ._util import *

###########################################################
## Convolute Power Curve
def convolutePowerCurveByGuassian(powerCurve, stdScaling=0.2, stdBase=0.6, minSpeed=0.01, maxSpeed=40, steps=4000, outputResolution=0.1):
    # Set performance
    if isinstance(powerCurve,str): 
        powerCurve = np.array(TurbineLibrary.ix[powerCurve].PowerCurve)
    elif isinstance(powerCurve, list):
        powerCurve = np.array(powerCurve)

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
    powerCurveInterp = splrep(powerCurve.ws, powerCurve.cf)

    cf = np.zeros(steps)
    sel = ws<powerCurve.ws.max()
    cf[sel] = splev(ws[sel], powerCurveInterp)

    cf[ws<powerCurve.ws.min()] = 0 # set all windspeed less than cut-in speed to 0
    cf[ws>powerCurve.ws.max()] = 0 # set all windspeed greater than cut-out speed to 0 (just in case)
    cf[cf<0] = 0 # force a floor of 0
    #cf[cf>powerCurve[:,1].max()] = powerCurve[:,1].max() # force a ceiling of the max capacity
    
    # Begin convolution
    convolutedCF = np.zeros(steps)
    for i,ws_ in enumerate(ws):
        convolutedCF[i] = (norm.pdf(ws, loc=ws_, scale=stdScaling*ws_+stdBase)*cf).sum()*dws
        
    # Done!
    ws = ws[::40]
    convolutedCF = convolutedCF[::40]
    return PowerCurve(ws,convolutedCF)

class TerrainComplexityConvoluter(object):
    def __init__(s, terrainComplexityFile, turbine, tcStep=5, tcMax=800, preEvaluate=False, mode="MERRA_20kTC"):
        print("WARNING: This has not been updated to the new power-curve definition, it will probably fail...")
        s.terrainComplexityFile = terrainComplexityFile
        s.turbine = turbine
        s._tcStep = int(tcStep)
        s._tcMax = int(tcMax)
        s.mode=mode

        s.ws = np.linspace(0, 40, 4001)[1:]
        s._dws = s.ws[1]-s.ws[0]
        s.Capacity = TurbineLibrary.ix[turbine].Capacity

        s.evaluationTCs = np.arange(tcStep, tcMax+0.01, tcStep)
        s._convolutedPowerCurves = OrderedDict()

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
    def _params_REA6_5kTC(tc):
        a = 0.25*(1-np.exp(-tc/10))+0.4
        b = tc*0+0.525
        c = 0.9 - tc/300*0.4
        d = 0.085*(1-np.exp(-tc/150))+0.009
        return a,b,c,d

    @staticmethod
    def _params_MERRA_20kTC(tc):
        a = 0.25*(1-np.exp(-tc/100))+0.49
        b = 0.55*np.exp(-tc/150)+0.5
        c = 0.25 + tc/685*0.20
        d = 0.06+0.25/650 * tc
        return a,b,c,d

    def relativeSig(s, tc): 
        
        tc = np.array(tc)
        tc[tc<5] = 5

        if s.mode=="MERRA_20kTC":
            a,b,c,d = s._params_MERRA_20kTC(tc)
        elif s.mode=="REA6_5kTC":
            a,b,c,d = s._params_REA6_5kTC(tc)
        else:
            raise ResError("mode not recognized")

        return a*np.exp( -np.power(s.ws,b)*c) + d

    def _preEvaluateAll(s):
        for tc in s.evaluationTCs: s._evaluateComplexity(tc)

    def _evaluateComplexity(s, tc):
        convolutedPowerCurve = np.zeros(s.ws.size)

        for i, ws, rsig in zip(range(s.ws.size), s.ws, s.relativeSig(tc)):
            convolutedPowerCurve[i] = (s.unconvolutedPowerCurve*norm.pdf(s.ws, scale=ws*rsig, loc=ws)).sum()*s._dws

        s._convolutedPowerCurves[tc] = convolutedPowerCurve

    def performance(s,tc):
        return np.column_stack( [s.ws,s[tc]] )

    def __getitem__(s, tc):
        tc = s._tcStep*(int(tc)//s._tcStep)
        if tc > s._tcMax: tc = s._tcMax
        if not tc in s._convolutedPowerCurves:
            s._evaluateComplexity(tc)

        return s._convolutedPowerCurves[tc]

    def convolutedPowerCurveAtLocation(s, loc):
        loc = Location.ensureLocation(loc, forceAsArray=True)
        tcVals = gk.raster.extractValues(s.terrainComplexityFile, loc, noDataOkay=False).data.values

        if len(loc)==1:
            return s[tcVals[0]]
        else:
            for v in tcVals: yield s[v]

    def getTerrainComplexityAtLocation(s,loc):
        a = gk.raster.extractValues(s.terrainComplexityFile, loc, noDataOkay=True).data.values
        return a