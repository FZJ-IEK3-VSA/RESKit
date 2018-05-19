from respower.util.util_ import *

def simpleLCOE(capex, meanProduction, opexPerCapex=0.02, lifetime=20, discountRate=0.08):
    r = discountRate
    N = lifetime
    return capex * ( r /(1-np.power(1+r,-N)) + opexPerCapex ) / (meanProduction)

def lcoe( expenditures, productions, discountRate=0.08 ):
    """Provides a raw computation of LCOE. Requires input time-series for annual expenditures and annual productions"""
    # Initialize variables
    exp = np.array(expenditures)
    pro = np.array(productions)
    if not exp.size==pro.size: raise ResError("expenditures length does not match productions length")

    yr = np.arange(exp.size)
    if isinstance(r,float):
        r = np.zeros(exp.size)+discountRate
    else:
        r = np.array(r)

    # Do summation and return
    lcoe = (exp/np.power(1+r, yr)).sum() / (pro/np.power(1+r, yr)).sum()

    return lcoe