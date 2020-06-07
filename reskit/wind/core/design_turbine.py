# from ._util import *
# from ._costModel import *

# from scipy.optimize import differential_evolution
# from scipy.stats import exponweib
import numpy as np
import pandas as pd
from .power_curve import compute_specific_power


def onshore_turbine_from_avg_wind_speed(wind_speed, constant_rotor_diam=True, base_capacity=4200, base_hub_height=120, base_rotor_diam=136, reference_wind_speed=6.7, min_tip_height=20, min_specific_power=180):
    """
    TODO: NEEDS UPDATE!!!
    
    Suggest onshore turbine turbine desing characteristics based on an average wind speed such that a baseline turbine with 4200 kW capacity, 120m hub height, and 136 m rotor diameter is chosen for a wind speed equal to 6.7 m/s as per Ryberg et al. [1]
    
    

    Parameters:
    ----------
    wind_speed : float or array_like
        Local average wind speed close to or at the hub height.

    constant_rotor_diam : bool, optional
        Whether the rotor diameter is mantained constant or not, by default True
    base_capacity : int, optional
        Baseline turbine capacity in kW, by default 4200.
    base_hub_height : int, optional
        [description], by default 120.
    base_rotor_diam : int, optional
        [description], by default 136.
    reference_wind_speed : float, optional
        [description], by default 6.7.
    min_tip_height : int, optional.
        [description], by default 20.
    min_specific_power : int, optional
        [description], by default 180.





    rotordiam : float or array_like, optional
        Rotor diamter in meters. Default value is 136.
    
    Returns
    -------
    Onshore turbine suggested characteristcs: pandas data frame
        A pandas data frame with columns hub height in m, specific power in W/m2, and capacity in kW.

    Notes
    -------
    Suggestions are given such that with an average wind speed value of 6.7 m/s, a turbine with 4200 kW capacity, 120m hub height, and 136m rotor diameter is chosen
    The specific power (capacity/area of the rotor) is not permited to go less than 180 W/m2 (becase...)
    A minimum hub height to keep 20 m sepatarion distnce beteen the tip of the blade and the floor is maintaied.
    
    References
    -------
    [1] David S. Ryberg, Dilara C. Caglayana, Sabrina Schmitt, Jochen Linßena, Detlef Stolten, Martin Robinius - The Future of European Onshore Wind Energy Potential: 
    Detailed Distributionand Simulation of Advanced Turbine Designs, Energy, 2019, available at https://www.sciencedirect.com/science/article/abs/pii/S0360544219311818



    Suggest turbine characteristics based off an average wind speed and in relation to the 'baseline' onshore turbine.
    relationships are derived from turbine data between 2013 and 2017

    Parameters
    ----------
    wind_speed : float or array-like
        The wind speed in m/s at <(or close to)> the surface/hub height?.
    

    Returns
    -------
    [type]
        [description]
    """    """
    
    

    * Suggested specific power will not go less than 180 W/m2
    * Normalizations chosen for the context of 2050
        - Such that at 6.7 m/s, a turbine with 4200 kW capacity, 120m hub height, and 136m rotor diameter is chosen
    """
    wind_speed = np.array(wind_speed)
    multi = wind_speed.size > 1

    # Design Specific Power
    scaling = compute_specific_power(base_capacity, base_rotor_diam) / (np.exp(0.53769024 * np.log(reference_wind_speed) + 4.74917728))
    specific_power = scaling * np.exp(0.53769024 * np.log(wind_speed) + 4.74917728)
    if multi:
        lt180 = specific_power < min_specific_power
        if lt180.any():
            specific_power[lt180] = min_specific_power
    else:
        if specific_power < min_specific_power:
            specific_power = min_specific_power

    if constant_rotor_diam:
        rotor_diam = base_rotor_diam
        capacity = specific_power * np.pi * np.power((rotor_diam / 2), 2) / 1000
    else:
        capacity = base_capacity
        rotor_diam = 2 * np.sqrt(capacity * 1000 / specific_power / np.pi)

    # Design Hub Height
    scaling = base_hub_height / (np.exp(-0.84976623 * np.log(reference_wind_speed) + 6.1879937))
    hub_height = scaling * np.exp(-0.84976623 * np.log(wind_speed) + 6.1879937)
    if multi:
        lt20 = hub_height < (rotor_diam / 2 + min_tip_height)
        if lt20.any():
            hub_height[lt20] = rotor_diam[lt20] / 2 + min_tip_height
    else:
        if hub_height < (rotor_diam / 2 + min_tip_height):
            hub_height = rotor_diam / 2 + min_tip_height

    output = dict(capacity=capacity, hub_height=hub_height, rotor_diam=rotor_diam, specific_power=specific_power)
    if multi:
        return pd.DataFrame(output)
    else:
        return output
