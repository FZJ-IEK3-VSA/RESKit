from ._util import *
from ._costModel import *

from scipy.optimize import differential_evolution
from scipy.stats import exponweib


class OptimalTurbine(namedtuple("OptimalTurbine","capacity rotordiam hubHeight opt")):
    def __str__(s):
        out = ""
        out += "Capacity:   %d\n"%int(s.capacity)
        out += "Rotor Diam: %d\n"%int(s.rotordiam)
        out += "Hub Height: %d\n"%int(s.hubHeight)
        out += "LCOE Value: %.5f"%s.opt.fun
        return out
    def __repr__(s): return str(s)

def determineBestTurbine(weibK=2, weibL=7, capacity=(3000,9000), rotordiam=(90,180), hubHeight=(80,200), roughness=0.02, costModel=nrelCostModel, measuredHeight=50, minSpecificCapacity=200, tol=1e-5, **kwargs):
    """
    Determine the best turbine characteristics (capacity, rotor diameter, and hub height) for a location defined by a 
    weibul distribution of windspeeds and a roughness length

    * A genetic algorithm is used to find the "optimal" solution
    * A synthetic turbine power curve is always generated according to the given capacity and rotor diameter
    * All characteristic inputs (capacity, rotordiam, hubHeight) can be given as a tuple, indicating an allowable 
      range, or a descrete value

    Inputs:
        weibK : float - Weibull k parameter describing the location's wind speed distribution

        weibL : float - Weibull lambda parameter describing the location's wind speed distribution

        Capacity : The allowable capacity value(s) in kW
            ( float, float ) - minimal and maximal value
            float - The explicit value to use

        rotordiam : The allowable rotor diameter value(s) in m
            ( float, float ) - minimal and maximal value
            float - The explicit value to use

        hubHeight : The allowable hub height value(s) in m
            ( float, float ) - minimal and maximal value
            float - The explicit value to use

        roughness : float - The location roughness length

        costModel : func - The cost model to use to estimate the turbine's cost
            * Must be a function which accepts keyword arguments of capacity, rotordiam, and hubHeight and returns a
              single float value

        measuredHeight : float - The implied height of the given windspeed distribution

        minSpecificCapacity : float - The minimal specific-capacity value to allow during the optimization
            * Can be 'None', implying no minimum

        tol : float - The tolerance to use during the optimization
            * See scipy.optimize.differential_evolution for more information

        **kwargs
            * All other kwargs are passed on to scipy.optimize.differential_evolution
    """
    ws = np.linspace(0,40,4000)
    dws = ws[1]-ws[0]
    _s = np.log(measuredHeight/roughness)

    # Determine unpacking and boundary structure
    if isinstance(capacity,tuple) and isinstance(rotordiam,tuple) and isinstance(hubHeight,tuple):
        unpack = lambda x: (x[0], x[1], x[2])
        bounds = [capacity, rotordiam, hubHeight, ]
        finalize = lambda x: OptimalTurbine(x.x[0], x.x[1], x.x[2], x)
    elif not isinstance(capacity,tuple) and isinstance(rotordiam,tuple) and isinstance(hubHeight,tuple):
        unpack = lambda x: (capacity, x[0], x[1])
        bounds = [rotordiam, hubHeight, ]
        finalize = lambda x: OptimalTurbine(capacity, x.x[0], x.x[1], x)
    elif isinstance(capacity,tuple) and not isinstance(rotordiam,tuple) and isinstance(hubHeight,tuple):
        unpack = lambda x: (x[0], rotordiam, x[1])
        bounds = [capacity, hubHeight, ]
        finalize = lambda x: OptimalTurbine(x.x[0], rotordiam, x.x[1], x)
    elif isinstance(capacity,tuple) and isinstance(rotordiam,tuple) and not isinstance(hubHeight,tuple):
        unpack = lambda x: (x[0], x[1], hubHeight)
        bounds = [capacity, rotordiam, ]
        finalize = lambda x: OptimalTurbine(x.x[0], x.x[1], hubHeight, x)
    elif not isinstance(capacity,tuple) and not isinstance(rotordiam,tuple) and isinstance(hubHeight,tuple):
        unpack = lambda x: (capacity, rotordiam, x[0])
        bounds = [hubHeight, ]
        finalize = lambda x: OptimalTurbine(capacity, rotordiam, x.x[0], x)
    elif not isinstance(capacity,tuple) and isinstance(rotordiam,tuple) and not isinstance(hubHeight,tuple):
        unpack = lambda x: (capacity, x[0], hubHeight)
        bounds = [rotordiam, ]
        finalize = lambda x: OptimalTurbine(capacity, x.x[0], hubHeight, x)
    elif isinstance(capacity,tuple) and not isinstance(rotordiam,tuple) and not isinstance(hubHeight,tuple):
        unpack = lambda x: (x[0], rotordiam, hubHeight)
        bounds = [capacity, ]
        finalize = lambda x: OptimalTurbine(x.x[0], rotordiam, hubHeight, x)
    else:
        raise RuntimeError("Something is wrong...")

    # Define scoring function
    def score(x):
        c,r,h = unpack(x)
        s = np.log(h/roughness)/_s
        pdf = exponweib.pdf(ws, a=1, c=weibK, loc=0, scale=weibL*s)
        
        pc = SyntheticPowerCurve(capacity=c, rotordiam=r)
        cf = np.interp(ws, pc.ws, pc.cf)
        
        expectedCapFac = (cf*pdf).sum()*dws
        capex = costModel(capacity=c, hubHeight=h, rotordiam=r)
        lcoe = simpleLCOE(capex, expectedCapFac*8760*c)
        
        # Dissuade against too-low specific capacity values
        if not minSpecificCapacity is None:
            specificCapacity = 1000*c/(np.pi*r*r/4)
            if specificCapacity<minSpecificCapacity:
                lcoe += np.power(minSpecificCapacity-specificCapacity,3)

        # Dissuade against too-low hub height compared to the rotor diameter
        tmp = h-(25+r/2)
        if tmp<0: lcoe += np.power(tmp,3)

        # Done!
        return lcoe

    res = differential_evolution(score, bounds=bounds, tol=tol, **kwargs)
    return finalize(res)