# from ._util import *
# from ._costModel import *

# from scipy.optimize import differential_evolution
# from scipy.stats import exponweib
import numpy as np
import pandas as pd
from . import compute_specific_power


def onshore_turbine_from_avg_wind_speed(wind_speed, constant_rotor_diam=True, base_capacity=4200, base_hub_height=120, base_rotor_diam=136, reference_wind_speed=6.7, min_tip_height=20, min_specific_power=180):
    """
    TODO: NEEDS UPDATE!!!
    Suggest turbine characteristics based off an average wind speed and in relation to the 'baseline' onshore turbine.
    relationships are derived from turbine data between 2013 and 2017

    * Suggested specific power will not go less than 180 W/m2
    * Normalizations chosen for the context of 2050
        - Such that at 6.7 m/s, a turbine with 4200 kW capacity, 120m hub height, and 136m rotor diameter is chosen
    """
    wind_speed = np.array(wind_speed)
    if wind_speed.size > 1:
        multi = True
        rotor_diam = np.array([rotor_diam] * wind_speed.size)
    else:
        multi = False

    scaling = base_hub_height / (np.exp(-0.84976623 * np.log(reference_wind_speed) + 6.1879937))
    hub_height = scaling * np.exp(-0.84976623 * np.log(wind_speed) + 6.1879937)
    if multi:
        lt20 = hub_height < (rotor_diam / 2 + min_tip_height)
        if lt20.any():
            hub_height[lt20] = rotor_diam[lt20] / 2 + min_tip_height
    else:
        if hub_height < (rotor_diam / 2 + min_tip_height):
            hub_height = rotor_diam / 2 + min_tip_height
        # if hub_height>200: hub_height = 200

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

    output = dict(capacity=capacity, hub_height=hub_height, rotor_diam=rotor_diam, specific_power=specific_power)
    if multi:
        return pd.DataFrame(output)
    else:
        return output
