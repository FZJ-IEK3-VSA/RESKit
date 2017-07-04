import numpy as np
import pandas as pd

from scipy.interpolate import splrep, splev
from collections import namedtuple

##################################################
## Make a turbine model library
class Turbine(object):
	def __init__(s, name, performance, hubHeight, meta=None):
		s.name = name
		s.performance = performance
		s.hubHeight = hubHeight
		s.meta = meta

	def __str__(s): return "Turbine: "+s.name

TurbineLibrary = {}
TurbineLibrary["Enercon_E115"] = Turbine(
	name="Enercon_E115",
	hubHeight=100,
	performance=np.array([
    (0.0, 0.0),
    (1.0, 0.0),
    (2.0, 0.0),
    (3.0, 45),
    (4.0, 122),
    (5.0, 326),
    (6.0, 632),
    (7.0, 1053),
    (8.0, 1524),
    (9.0, 2123),
    (10.0, 2569),
    (11.0, 2938),
    (12.0, 3000.0),
    (13.0, 3000.0),
    (24.0, 3000.0),
    (25.0, 3000.0),]))

TurbinePerformance = namedtuple("TurbinPerformance", "production hubSpeeds fullLoadHours")
def singleTurbine( measuredWindSpeeds, measuredHeight=50, roughness=1, turbine=TurbineLibrary["Enercon_E115"], hubHeight=None, performance=None):
    """
    Perform simple windpower simulation for a single turbine
    Inputs:
        measuredWindSpeeds - array-like
            * A numerical array of measured wind speeds
        measuredHeight - float (default 50)
            * The height (in meters) where the wind speeds were measured 
        roughness - float (default 1)
            * The roughness length of the area associated with the measured wind speeds
        hubHeight - float (default 115)
            * The hub height (in meters) of the wind turbine to simulate
            * 115m corresponds to the Enercon E115 turbine
        performance - [ (float, float),...] (default E115_performance)
            * An array of "wind speed" to "power output" pairs, as two-member tuples, which maps the power profile of the turbine to be simulated
            * The performance pairs must contain the boundary benhavior
                - The first (after sorting by wind speed) pair will be used as the "cut in"
                - The last (after sorting) pair will be used as the "cut out" 
    Returns: ( performance, hub-wind-speeds )
        performance - A numpy array of performance values
        hub-wind-speeds - The projected wind speeds at the turbine's hub height
    """
    ############################################
    # make sure we have numpy types
    if not isinstance(measuredWindSpeeds, np.ndarray):
    	try: # maybe we have a pandas type
    		measuredWindSpeeds = measuredWindSpeeds.values
    	except AttributeError: # just try a simple cast
    		measuredWindSpeeds = np.array(measuredWindSpeeds)

    # Check for bad values
    nanSel = np.isnan(measuredWindSpeeds)
    if( nanSel.sum() > 0):
        print("WARNING: {0:d} nans found in measured windspeeds".format(nanSel.sum()))
        measuredWindSpeeds = measuredWindSpeeds[~nanSel]

    infSel = np.isinf(measuredWindSpeeds)
    if( infSel.sum() > 0):
        print("WARNING: {0:d} infs found in measure windspeeds".format(infSel.sum()))
        measuredWindSpeeds = measuredWindSpeeds[~infSel]

    ############################################
    # Set huheight and performance
    if performance is None: performance = turbine.performance
    if hubHeight is None: hubHeight = turbine.hubHeight

    ############################################
    # Convert to wind speeds at hub height
    #  * Follows the "log-wind profile" assumption
    if (hubHeight != measuredHeight):
        hubWindSpeeds = measuredWindSpeeds * (np.log(hubHeight/roughness)/np.log(measuredHeight/roughness))
    else:
        hubWindSpeeds = measuredWindSpeeds

    ############################################
    # map wind speeds to power curve using a spline
    powerCurve = splrep(performance[:,0], performance[:,1])
    powerGen = splev(hubWindSpeeds, powerCurve)

    # Do some "just in case" clean-up
    maxPower = performance[:,1].max() # use the max power as as ceiling
    cutin = performance[:,0].min() # use the first defined windspeed as the cut in
    cutout = performance[:,0].max() # use the last defined windspeed as the cut out 

    powerGen[powerGen<0]=0 # floor to zero
    powerGen[powerGen>maxPower]=maxPower # ceiling at max
    powerGen[hubWindSpeeds<cutin]=0 # Drop power to zero before cutin
    powerGen[hubWindSpeeds>cutout]=0 # Drop power to zero after cutout
    
    # Return
    return TurbinePerformance(powerGen, hubWindSpeeds, np.sum(powerGen, 0)/max(performance[:,1]))
