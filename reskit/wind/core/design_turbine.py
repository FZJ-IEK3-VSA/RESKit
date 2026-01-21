# from ._util import *
# from ._costModel import *

# from scipy.optimize import differential_evolution
# from scipy.stats import exponweib
import numpy as np
import pandas as pd
from .power_curve import compute_specific_power
from reskit.parameters.parameters import OnshoreParameters, OffshoreParameters
import warnings
from copy import copy

def onshore_turbine_from_avg_wind_speed(wind_speed, **kwargs):
    """
    Convenience function for backward compatibility, will be removed soon.
    All kwargs are passed to turbine_design_from_avg_wind_speed() with
    technology='onshore'. 
    
    NOTE: 'reference_wind_speed' is allowed as kwarg here for backward 
    compatibility only and will be set for both 'reference_wind_speed_hubheight' 
    and 'reference_wind_speed_specpow'. This allows to use the former workflow 
    reference_wind_speed param with the new turbine_design_from_avg_wind_speed() 
    function. It is recommended to use specific values for each parameter though.

    wind_speed : numeric or array_like
        Local average wind speed close to or at the hub height.
    """
    # deprecation warning
    warnings.warn(
        "onshore_turbine_from_avg_wind_speed() will be retired soon, please use turbine_design_from_avg_wind_speed() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # check or set technology arg as onshore
    assert not "technology" in kwargs or kwargs["technology"] == "onshore", (
        f"'technology' argument not required here, but must be 'onshore' if provided."
    )
    kwargs["wind_speed"] = wind_speed
    kwargs["technology"] = "onshore"
    # check if reference_wind_speed if in kwargs, if so, replace by 
    # reference_wind_speed_specpow and reference_wind_speed_hubheight
    if "reference_wind_speed" in kwargs:
        assert not any([k in kwargs for k in ["reference_wind_speed_specpow", "reference_wind_speed_hubheight"]]),\
            "legacy argument 'reference_wind_speed' cannot be given together with current arguments 'reference_wind_speed_specpow' or 'reference_wind_speed_hubheight'"
        refws = kwargs.pop("reference_wind_speed", "None")
        kwargs["reference_wind_speed_specpow"] = refws
        kwargs["reference_wind_speed_hubheight"] = refws
    # return results of turbine_design_from_avg_wind_speed
    return turbine_design_from_avg_wind_speed(**kwargs)


def turbine_design_from_avg_wind_speed(
    wind_speed,
    technology,
    constant_rotor_diam=None,
    base_capacity=None,
    base_hub_height=None,
    base_rotor_diam=None,
    reference_wind_speed_hubheight=None,
    reference_wind_speed_specpow=None,
    min_tip_height=None,
    min_specific_power=None,
    max_hub_height=None,
    tech_year=2050,
    baseline_turbine_fp=None,
    convention="RybergEtAl2019",
):
    """
    Suggest onshore turbine design characteristics (capacity, hub height, rotor diameter, specific power) for a 2050 European context based on an average wind speed value.
    The default values and the function's normalization correspond to the baseline turbine design considered by Ryberg et al. [1] for a wind speed equal to 6.7 m/s. See notes.

    Parameters #TODO remove all default values, depends on the default baseline turbine file!
    ----------
    wind_speed : numeric or array_like
        Local average wind speed close to or at the hub height.

    technology : str
        Either "onshore" or "offshore" to define the turbine scaling functions.

    constant_rotor_diam : bool, optional
        Whether the rotor diameter is maintained constant or not, by default True

    base_capacity : numeric or array_like, optional
        Baseline turbine capacity in kW, by default 4200.

    base_hub_height : numeric or array_like, optional
        Baseline turbine hub height in m, by default 120.

    base_rotor_diam : numeric or array_like, optional
        Baseline turbine rotor diameter in m, by default 136.

    reference_wind_speed_hubheight : numeric, optional
        Average wind speed corresponding to the baseline hub height value, 
        takes effect only for conventions which differentiate between parameter-
        specific windspeeds (e.g. WinklerEtAl2026).

    reference_wind_speed_specpow : numeric, optional
        Average wind speed corresponding to the baseline specific power value, 
        takes effect only for conventions which differentiate between parameter-
        specific windspeeds (e.g. WinklerEtAl2026).

    min_tip_height : numeric, optional.
        Minimum distance in m between the lower tip of the blades and the ground, by default 20.

    min_specific_power : numeric, optional
        Minimum specific power allowed in kw/m2, by default 180.

    max_hub_height : numeric, optional
        Maximum allowed hub height, any higher optimal hub height will be reduced to this
        value, by default 200.

    tech_year : int, optional
        The year defining the baseline turbine design that shall be used.

    baseline_turbine_fp : str, optional
        A json or csv file that contains baseline turbine parameters. Will
        replace the default data.

    convention : str, optional
        Author and year of the publication that contains the exact scaling
        approach of hub height and specific power over wind speed. Available
        conventions (depending on the technology) might be:
        - RybergEtAl2019 : Approach from [1]
        - WinklerEtAl2026 : unpublished, coming soon

    Returns
    -------
    dict or pandas DataFrame
        Returns a the suggested values of hub height in m, specific power in W/m2, and capacity in kW as dictionary when numeric values are input or as a pandas DataFrame when array-like objects are input.

    Notes
    -----
    The default baseline onshore turbine has 4200 kW capacity, 120m hub height, and 136m rotor diameter [1]

    References
    ----------
    [1] David S. Ryberg, Dilara C. Caglayan, Sabrina Schmitt, Jochen Linssen, Detlef Stolten, Martin Robinius - The Future of European Onshore Wind Energy Potential:
    Detailed Distributionand Simulation of Advanced Turbine Designs, Energy, 2019, available at https://www.sciencedirect.com/science/article/abs/pii/S0360544219311818
    """
    # define scaling functions
    func_mapper = {
        "onshore": {
            "RybergEtAl2019": {
                "specific_power": lambda ws, base_sp, ref_ws: base_sp / (np.exp(
                    0.53769024 * np.log(ref_ws) + 4.74917728
                )) * (np.exp(
                    0.53769024 * np.log(ws) + 4.74917728
                )),
                "hub_height": lambda ws, base_hh, ref_ws: base_hh / (np.exp(-0.84976623 * np.log(ref_ws) + 6.1879937)) * (np.exp(-0.84976623 * np.log(ws) + 6.1879937)),
            },

            "WinklerEtAl2026": {
                "specific_power": lambda ws, base_sp, ref_ws: 187.993 * (base_sp/295) * np.log(ws/ref_ws) + base_sp,
                "hub_height": lambda ws, base_hh, ref_ws: -94.126 * (base_hh/118) * np.log(ws/ref_ws) + base_hh,
            },
        },
        "offshore": {
            "WinklerEtAl2026": {
                "specific_power": lambda ws, base_sp, ref_ws: 287.772 * (base_sp/320) * np.log(ws/ref_ws) + base_sp,
                "hub_height": lambda ws, base_hh, ref_ws: 0.742 * (base_hh/96) * np.log(ws/ref_ws) + base_hh,
            },
        },
    }
    # first extract the technology subdict or raise error for unknown techs
    if not isinstance(technology, str):
        raise TypeError(f"'technology' must be str type, here: {technology}")
    try:
        # extract the sub dicts for the available conventions for the given tech
        conv_mapper = func_mapper[technology.lower()]
    except:
        raise KeyError(f"'technology' (case insensitive) must be in: {', '.join(func_mapper.keys())}")
    # then extract the correct scaling functions for the given convention or flag error
    try:
        # extract the scaling functions for the given convention as dict of functions
        scaling_funcs = conv_mapper[convention]
    except:
        # no matching convention found
        raise ValueError(
            f"convention for technology '{technology}' must be in: {', '.join(conv_mapper.keys())}"
        )

    # define a dict to hold the parameter values
    baseline_params = dict()
    Params = None  # initialize with None, overwrite with singleton later if needed

    # iterate over arguments and retrieve defaults from Params if not given explicitly
    _locals = copy(locals())
    for arg, val in _locals.items():
        if arg in [
            "wind_speed",
            "baseline_turbine_fp",
            "Params",
            "baseline_params",
            "technology",
            "convention",
            "func_mapper",
            "Params",
        ]:
            continue
        _val = copy(val)
        if _val is None:
            if Params is None:
                # get the correct params singleton when it is needed for the 1st time
                if technology.lower() == "onshore":
                    Params = OnshoreParameters(fp=baseline_turbine_fp, year=tech_year)
                elif technology.lower() == "offshore":
                    Params = OffshoreParameters(fp=baseline_turbine_fp, year=tech_year)
                else:
                    raise ValueError(
                        f"Parameters singleton cannot be initialized for technology '{technology}'."
                    )
            # set value from Params
            _val = getattr(Params, arg)
            print(f"Parameter '{arg}' taken from Params as: {_val}", flush=True)
        baseline_params[arg] = _val

    wind_speed = np.array(wind_speed)
    multi = wind_speed.size > 1

    # Design Specific Power
    #TODO delete when confirmed via comparison with elder branch
    # scaling = compute_specific_power( 
    #     baseline_params["base_capacity"], baseline_params["base_rotor_diam"]
    # ) / scaling_funcs["specific_power"](ws=baseline_params["reference_wind_speed"])
    # specific_power = scaling * scaling_funcs["specific_power"](ws=wind_speed)

    # get reference wind speed, can be general (e.g. RybergEtAl2019) or parameter-specific (e.g. WinklerEtAl2026)
    reference_wind_speed_specpow = baseline_params["reference_wind_speed_specpow"]
    assert not pd.isnull(reference_wind_speed_specpow), "reference_wind_speed_specpow must be given."
    # apply the respective scaling function
    specific_power = scaling_funcs["specific_power"](
        ws=wind_speed, 
        base_sp=compute_specific_power(baseline_params["base_capacity"], baseline_params["base_rotor_diam"]),
        ref_ws=reference_wind_speed_specpow)
    # limit to min. specific power
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
    #TODO delete when confirmed 
    # scaling = baseline_params["base_hub_height"] / (
    #     scaling_funcs["hub_height"](ws=baseline_params["reference_wind_speed"])
    # )
    # hub_height = scaling * scaling_funcs["hub_height"](ws=wind_speed)
    # get reference wind speed, can be general (e.g. RybergEtAl2019) or parameter-specific (e.g. WinklerEtAl2026)
    reference_wind_speed_hubheight = baseline_params["reference_wind_speed_hubheight"]
    assert not pd.isnull(reference_wind_speed_hubheight), "reference_wind_speed_hubheight must be given."
    # apply the respective scaling function
    hub_height = scaling_funcs["hub_height"](
        ws=wind_speed, 
        base_hh=baseline_params["base_hub_height"],
        ref_ws=reference_wind_speed_hubheight)
    # limit to min. tip height and maximum hub height
    if multi:
        lowerlt = hub_height < (rotor_diam / 2 + baseline_params["min_tip_height"])
        if lowerlt.any():
            if baseline_params["constant_rotor_diam"]:
                hub_height[lowerlt] = rotor_diam / 2 + baseline_params["min_tip_height"]
            else:
                hub_height[lowerlt] = rotor_diam[lowerlt] / 2 + baseline_params["min_tip_height"]

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
