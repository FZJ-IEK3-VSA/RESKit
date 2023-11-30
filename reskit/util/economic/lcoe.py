import numpy as np
from .. import ResError


def levelized_cost_of_electricity_simplified(
    capex, mean_production, opex_per_capex=0.02, lifetime=20, discount_rate=0.08
):
    """Compute the LCOE of a producer using the simple method

    Uses the equation:
    .. math::
        \\mathrm{LCOE} = C * \\frac{ (r/(1-(1+r)^{-N})) + O_c }{P_{mean}}

    where:
      * $C$ -> CAPEX [euros]
      * $O_c$ -> Fixed OPEX as a factor of CAPEX 
      * $P_{mean}$ -> The average production in each year [kWh]
      * $r$ -> The discount rate
      * $N$ -> The economic lifetime [years]


    Parameters
    ----------
    capex : numeric or array_like
        The capital expenditures

    mean_production : numeric or array_like
        The average annual production

    opex_per_capex : numeric
        The operational expenditures given as a factor or the capex

    lifetime : numeric
        The economic lifetime in years

    discount_rate : numeric
        The discount rate

    Returns
    -------
        numeric or array_like

    """
    r = discount_rate
    N = lifetime
    return capex * (r / (1 - np.power(1 + r, -N)) + opex_per_capex) / (mean_production)


def levelized_cost_of_electricity(expenditures, productions, discount_rate=0.08):
    """Compute the LCOE of a producer using explicitly given production and 
    expeditures for each year in the economic lifetime

    Uses the equation:
    .. math::
        \\mathrm{LCOE} = \\sum{\\frac{exp_y}{(1+r)^y}} / \\sum{\\frac{prod_y}{(1+r)^y}}

    where:
      * $exp_y$ -> The expenditures in year $y$ [Euro]
      * $prod_y$ -> The production in year $y$ [kWh]
      * $r$ -> The discount rate

    Parameters
    ----------
    expenditures : array_like
        All expenditures for each year in the lifetime

    productions : array_like
        Annual production for each year in the lifetime

    discount_rate : numeric or array_like
        The discount rate
          * If a numeric is given, the discount rate is applied to all years
          * If an array is given, a discount rate for each year must by provided

    Returns
    -------
        numeric or array_like

    """
    # Initialize variables
    exp = np.array(expenditures)
    pro = np.array(productions)
    if not exp.shape == pro.shape:
        raise ResError("expenditures length does not match productions length")

    yr = np.arange(exp.shape[0])
    if isinstance(discount_rate, float):
        r = np.zeros(exp.shape[0]) + discount_rate
    else:
        r = np.array(r)

    # Do summation and return
    lcoe = (exp / np.power(1 + r, yr)).sum() / (pro / np.power(1 + r, yr)).sum()

    return lcoe
