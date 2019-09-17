from ._util import *

###########################################################
## Convolute Power Curve
def convolutePowerCurveByGuassian(powerCurve, stdScaling=0.06, stdBase=0.1, minSpeed=0.01, maxSpeed=40, steps=4000, outputResolution=0.1, extendBeyondCutoff=True):
    """
    Convolutes a turbine power curve from a normal distribution function with wind-speed-dependent standard deviation.
    """
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
    powerCurveInterp = splrep(ws, np.interp(ws, powerCurve.ws, powerCurve.cf))

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

    # Correct cutoff, maybe
    if not extendBeyondCutoff: convolutedCF[ws>powerCurve.ws[-1]] = 0
        
    # Done!
    ws = ws[::40]
    convolutedCF = convolutedCF[::40]
    return PowerCurve(ws,convolutedCF)
