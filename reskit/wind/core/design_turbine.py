# from ._util import *
# from ._costModel import *

# from scipy.optimize import differential_evolution
# from scipy.stats import exponweib
import numpy as np
import pandas as pd
from .power_curve import compute_specific_power
from reskit.parameters.parameters import OnshoreParameters


def onshore_turbine_from_avg_wind_speed(
    wind_speed,
    constant_rotor_diam=None,
    base_capacity=None,
    base_hub_height=None,
    base_rotor_diam=None,
    reference_wind_speed=None,
    min_tip_height=None,
    min_specific_power=None,
    max_hub_height=None,
    tech_year=2050,
    baseline_turbine_fp=None,
):
    """
    Suggest onshore turbine design characteristics (capacity, hub height, rotor diameter, specific power) for a 2050 European context based on an average wind speed value.
    The default values and the function's normalization correspond to the baseline turbine design considered by Ryberg et al. [1] for a wind speed equal to 6.7 m/s. See notes.

    Parameters
    ----------
    wind_speed : numeric or array_like
        Local average wind speed close to or at the hub height.

    constant_rotor_diam : bool, optional
        Whether the rotor diameter is mantained constant or not, by default True

    base_capacity : numeric or array_like, optional
        Baseline turbine capacity in kW, by default 4200.

    base_hub_height : numeric or array_like, optional
        Baseline turbine hub height in m, by default 120.

    base_rotor_diam : numeric or array_like, optional
        Baseline turbine rotor diameter in m, by default 136.

    reference_wind_speed : numeric, optional
        Average wind speed corresponding to the baseline turbine design, by default 6.7.

    min_tip_height : numeric, optional.
        Minimum distance in m between the lower tip of the blades and the ground, by default 20.

    min_specific_power : numeric, optional
        Minimum specific power allowed in kw/m2, by default 180.

    max_hub_height : numeric, optional
        Maximum allowed hub height, any higher optimal hub height will be reduced to this
        value, by default 200.

    tech_year : int, optional
        The year definining the baseline turbine design that shall be used.

    baseline_turbine_fp : str, optional
        A json or csv file that contains baseline turbine parameters. Will
        replace the default data.

    Returns
    -------
    dict or pandas DataFrame
        Returns a the suggested values of hub height in m, specific power in W/m2, and capacity in kW as dictionary when numeric values are input or as a pandas DataFrame when array-like objects are input.

    Notes
    -------
    The default baseline onshore turbine has 4200 kW capacity, 120m hub height, and 136m rotor diameter [1]

    References
    -------
    [1] David S. Ryberg, Dilara C. Caglayan, Sabrina Schmitt, Jochen Linssen, Detlef Stolten, Martin Robinius - The Future of European Onshore Wind Energy Potential:
    Detailed Distributionand Simulation of Advanced Turbine Designs, Energy, 2019, available at https://www.sciencedirect.com/science/article/abs/pii/S0360544219311818
    """
    OnshoreParams = OnshoreParameters(fp=baseline_turbine_fp, year=tech_year)

    # define a dict to hold the parameter values
    baseline_params = dict()

    # iterate over arguments and retrieve defaults from OnshoreParams if not given explicitly
    for arg, val in locals().items():
        if arg in [
            "wind_speed",
            "baseline_turbine_fp",
            "OnshoreParams",
            "baseline_params",
        ]:
            continue
        print(arg, val)
        if val is None:
            val = getattr(OnshoreParams, arg)
        baseline_params[arg] = val

    wind_speed = np.array(wind_speed)
    multi = wind_speed.size > 1

    # Design Specific Power
    scaling = compute_specific_power(
        baseline_params["base_capacity"], baseline_params["base_rotor_diam"]
    ) / (
        np.exp(
            0.53769024 * np.log(baseline_params["reference_wind_speed"]) + 4.74917728
        )
    )
    specific_power = scaling * np.exp(0.53769024 * np.log(wind_speed) + 4.74917728)
    if multi:
        lt180 = specific_power < baseline_params["min_specific_power"]
        if lt180.any():
            specific_power[lt180] = baseline_params["min_specific_power"]
    else:
        if specific_power < baseline_params["min_specific_power"]:
            specific_power = baseline_params["min_specific_power"]

    if baseline_params["constant_rotor_diam"]:
        rotor_diam = baseline_params["base_rotor_diam"]
        capacity = specific_power * np.pi * np.power((rotor_diam / 2), 2) / 1000
    else:
        capacity = baseline_params["base_capacity"]
        rotor_diam = 2 * np.sqrt(capacity * 1000 / specific_power / np.pi)

    # Design Hub Height
    scaling = baseline_params["base_hub_height"] / (
        np.exp(
            -0.84976623 * np.log(baseline_params["reference_wind_speed"]) + 6.1879937
        )
    )
    hub_height = scaling * np.exp(-0.84976623 * np.log(wind_speed) + 6.1879937)
    if multi:
        lowerlt = hub_height < (rotor_diam / 2 + baseline_params["min_tip_height"])
        if lowerlt.any():
            if baseline_params["constant_rotor_diam"]:
                hub_height[lowerlt] = rotor_diam / 2 + baseline_params["min_tip_height"]
            else:
                hub_height[lowerlt] = (
                    rotor_diam[lowerlt] / 2 + baseline_params["min_tip_height"]
                )

        upperlt = hub_height > baseline_params["max_hub_height"]
        if upperlt.any():
            hub_height[upperlt] = baseline_params["max_hub_height"]

    else:
        if hub_height < (rotor_diam / 2 + baseline_params["min_tip_height"]):
            hub_height = rotor_diam / 2 + baseline_params["min_tip_height"]
        elif hub_height > baseline_params["max_hub_height"]:
            hub_height = baseline_params["max_hub_height"]

    output = dict(
        capacity=capacity,
        hub_height=hub_height,
        rotor_diam=rotor_diam,
        specific_power=specific_power,
    )
    if multi:
        return pd.DataFrame(output)
    else:
        return output
