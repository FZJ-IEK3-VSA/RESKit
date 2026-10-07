import functools
import warnings
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike
from wisdem.nrelcsm.nrel_csm_mass_2015 import nrel_csm_2015
import openmdao.api as om
from openmdao.utils.units import unit_conversion

from reskit.parameters.parameters import OnshoreParameters


def onshore_turbine_capex(
    capacity,
    hub_height,
    rotor_diam,
    base_capex=None,
    base_capacity=None,
    base_hub_height=None,
    base_rotor_diam=None,
    tcc_share=None,
    bos_share=None,
):
    """
    A cost and scaling model (CSM) to calculate the total cost of a 3-bladed, direct drive onshore wind turbine according to Fingersh et al. [1] and Maples et al. [2].
    A CSM normalization is done such that a chosen baseline turbine, with a capacity of 4200 kW, hub height of 120 m, and rotor diameter of 136 m, corresponds to a expected typical specific cost of 1100 Eur/kW in a 2050 European context according to Ryberg et al. [4]
    The turbine cost includes the turbine capital cost (TCC) and balance of system costs (BOS), amounting to 67.3% and 22.9% respectively [3], as well as finantial costs equivalent to the the complementary percentage.


    Parameters
    ----------
    capacity : numeric or array-like
        Turbine's nominal capacity in kW.

    hub_height : numeric or array-like
        Turbine's hub height in m.

    rotor_diam : numeric or array-like
        Turbine's hub height in m.

    base_capex : numeric, optional
        The baseline turbine's capital costs in €, by default 1100*4200 [€/kW * kW]

    base_capacity : int, optional
        The baseline turbine's capacity in kW, by default 4200

    base_hub_height : int, optional
        The baseline turbine's hub height in m, by default 120

    base_rotor_diam : int, optional
        The baseline turbine's rotor diameter in m, by default 136

    tcc_share : float, optional
        The baseline turbine's turbine capital cost (TCC) percentage contribution in the total cost, by default 0.673

    bos_share : float, optional
        The baseline turbine's balance of system costs (BOS) percentage contribution in the total cost, by default 0.229

    Returns
    -------
    numeric or array-like
        Onshore turbine total cost


    Notes
    -----
        Pass many designs as arrays in one call: they are evaluated together (~10 ms for 1000
        designs), while every call also loads the baseline parameters and costs the baseline turbine.

        The expected turbine cost shares by Stehly et al. [3] are claimed to be derived from real cost data and valid until 10 MW capacity.

    Sources
    -------
    [1] Fingersh, L., Hand, M., & Laxson, A. (2006). Wind Turbine Design Cost and Scaling Model. NREL. https://www.nrel.gov/docs/fy07osti/40566.pdf
    [2] Maples, B., Hand, M., & Musial, W. (2010). Comparative Assessment of Direct Drive High Temperature Superconducting Generators in Multi-Megawatt Class Wind Turbines. Energy. https://doi.org/10.2172/991560
    [3] Stehly, T., Heimiller, D., & Scott, G. (2016). Cost of Wind Energy Review. Technical Report. https://www.nrel.gov/docs/fy18osti/70363.pdf
    [4] Ryberg, D. S., Caglayan, D. G., Schmitt, S., Linßen, J., Stolten, D., & Robinius, M. (2019). The future of European onshore wind energy potential: Detailed distribution and simulation of advanced turbine designs. Energy. https://doi.org/10.1016/j.energy.2019.06.052
    """
    # initialize OnshoreParameters class and feed with custom param values
    OnshoreParams = OnshoreParameters(
        **{k: v for k, v in locals().items() if not k in ["capacity", "hub_height", "rotor_diam"]}
    )

    # PREPROCESS INPUTS
    rd = np.array(rotor_diam)
    hh = np.array(hub_height)
    cp = np.array(capacity)
    # rr = rd / 2

    # COMPUTE COSTS
    # normalizations chosen to make the default turbine (4200-cap, 120-hub, 136-rot) match both a total
    # cost of 1100 EUR/kW as well as matching the percentages given in [3]
    tcc_scaling = (
        OnshoreParams.base_capex
        * OnshoreParams.tcc_share
        / onshore_tcc(
            cp=OnshoreParams.base_capacity,
            hh=OnshoreParams.base_hub_height,
            rd=OnshoreParams.base_rotor_diam,
        )
    )
    tcc = onshore_tcc(cp=cp, hh=hh, rd=rd) * tcc_scaling

    bos_scaling = (
        OnshoreParams.base_capex
        * OnshoreParams.bos_share
        / onshore_bos(
            cp=OnshoreParams.base_capacity,
            hh=OnshoreParams.base_hub_height,
            rd=OnshoreParams.base_rotor_diam,
        )
    )
    bos = onshore_bos(cp=cp, hh=hh, rd=rd) * bos_scaling

    # print(tcc_scaling, bos_scaling)

    total_costs = (tcc + bos) / (OnshoreParams.tcc_share + OnshoreParams.bos_share)

    # other_costs = total_costs * (1 - OnshoreParams.tcc_share - OnshoreParams.bos_share)

    return total_costs


def onshore_tcc(cp, hh, rd, gdp_escalator=None, blade_material_escalator=None, blades=None, **kwargs):
    """
    A function to determine the turbine capital cost (TCC) of a 3 blade standard onshore wind turbine based capacity, hub height and rotor diameter values according to the cost model by Fingersh et al. [1].

    Parameters
    ----------
    cp : numeric or array-like
        Turbine's capacity in kW
    hh : numeric or array-like
        Turbine's hub height in m
    rd : numeric or array-like
        Turbine's rotor diameter in m
    gdp_escalator : int, optional
        Labor cost escalator, by default 1
        DEPRECATED: ``gdp_escalator`` == 1 mandatory.
        This argument will be removed in a coming release.
    blade_material_escalator : int, optional
        Blade material cost escalator, by default 1
        DEPRECATED: ``blade_material_escalator`` == 1 mandatory.
        This argument will be removed in a coming release.
    blades : int, optional
        Number of blades, by default 3
        DEPRECATED: Use ``blade_number`` instead.
        This argument will be removed in a coming release.
    **kwargs
        Inputs of WISDEM's nrel_csm_2015() model, scalars or arrays broadcastable to the
        designs. See _onshore_tcc_scalar() for details.

    Returns
    -------
    numeric or array-like
        Turbine's turbine capital cost (TCC) in USD_2015.

    Notes
    -----
        All designs are evaluated in one pass of the model's components, so passing many designs
        as arrays is much faster than calling this function once per design.

    References
    ----------
    [1] Fingersh, L., Hand, M., & Laxson, A. (2006). Wind Turbine Design Cost and Scaling Model. NREL. https://www.nrel.gov/docs/fy07osti/40566.pdf

    """
    # deal with deprecated arguments
    if blades is not None:
        warnings.warn(
            "blades argument has been deprecated and replaced by optional blades_number key in kwargs, 'blades' arg will be removed soon.",
            DeprecationWarning,
            stacklevel=2,
        )
        if "blade_number" in kwargs:
            assert kwargs["blade_number"] == blades, "blades value cannot differ from 'blade_number' in kwargs"
        else:
            kwargs["blade_number"] = blades  # write into kwargs as blade_number
    if gdp_escalator is not None:
        warnings.warn(
            "gdp_escalator has been deprecated and will be removed soon.",
            DeprecationWarning,
            stacklevel=2,
        )
        assert gdp_escalator == 1  # make sure it has no unexpected non-impact
    if blade_material_escalator is not None:
        warnings.warn(
            "blade_material_escalator has been deprecated and will be removed soon.",
            DeprecationWarning,
            stacklevel=2,
        )
        assert blade_material_escalator == 1  # make sure it has no unexpected non-impact

    cp, hh, rd = np.broadcast_arrays(cp, hh, rd)

    spinner_mass_coeff = kwargs.get("spinner_mass_coeff", 15.5)
    spinner_mass_intercept = kwargs.get("spinner_mass_intercept", -980.0)
    spinner_mass = spinner_mass_coeff * rd + spinner_mass_intercept
    if np.any(spinner_mass < 0.0):
        warnings.warn(
            "At least one rotor diameter gives a negative spinner mass in WISDEM's NREL CSM model. "
            "The default 2015 spinner relation is spinner_mass = 15.5 * rotor_diameter - 980, "
            "which becomes positive only above about 63.2 m rotor diameter. Negative spinner mass "
            "will also produce negative spinner cost. Override spinner_mass_coeff/spinner_mass_intercept "
            "or avoid this regression for small turbines.",
            UserWarning,
            stacklevel=2,
        )

    design_shape = cp.shape
    design_count = cp.size
    if design_count == 0:
        return np.empty(design_shape, dtype=float)

    # all designs at once: the same inputs as _onshore_tcc_scalar() sets, one value per design
    # (_run_nrel_csm_2015() converts the continuous inputs to float)
    model_inputs = {
        "machine_rating": cp.ravel(),
        "rotor_diameter": rd.ravel(),
        "tower_length": hh.ravel(),
        "turbine_class": 2,
        "main_bearing_number": 2,
        "blade_number": 3,
        "max_tip_speed": 80,
        "max_efficiency": 0.90,
    }
    for input_name, input_value in kwargs.items():
        if np.ndim(input_value) == 0:  # the same value for all designs
            model_inputs[input_name] = input_value
        else:
            value_per_design = np.broadcast_to(input_value, design_shape)
            model_inputs[input_name] = value_per_design.ravel()

    model_values = _run_nrel_csm_2015(model_inputs, design_count=design_count)
    specific_turbine_cost = model_values["turbine_cost_kW"]  # in USD_2015/kW
    specific_turbine_cost = np.broadcast_to(specific_turbine_cost, design_count)
    specific_turbine_cost = specific_turbine_cost.reshape(design_shape)
    # previous functions expect absolute cost
    turbineCapitalCost = specific_turbine_cost * cp

    if design_shape == ():
        return turbineCapitalCost.item()
    return turbineCapitalCost


class _ModelStep(NamedTuple):
    """One component of WISDEM's NREL CSM 2015 model, as _run_nrel_csm_2015() computes it."""

    component: om.ExplicitComponent
    # (name in compute(), promoted name, unit conversion factor, unit conversion offset)
    continuous_inputs: list[tuple[str, str, float, float]]
    # (name in compute(), promoted name)
    discrete_inputs: list[tuple[str, str]]
    # (name in compute(), promoted name)
    outputs: list[tuple[str, str]]


@functools.lru_cache(maxsize=None)
def _nrel_csm_2015_model() -> tuple[list[_ModelStep], dict[str, ArrayLike]]:
    """
    Sets up WISDEM's NREL CSM 2015 model once and returns what _run_nrel_csm_2015() needs to
    evaluate its components without OpenMDAO: setting up an OpenMDAO problem takes tens of
    milliseconds, its components compute in microseconds.

    Returns
    -------
    steps : list of _ModelStep
        One per component, in execution order.
    defaults : dict
        Default value of every input not computed by a component, by promoted name, in the
        units of the inputs it feeds (as OpenMDAO takes values set with prob[name] = value):
        float arrays for continuous inputs, plain values (e.g. int or bool) for discrete inputs.
    """
    problem = om.Problem(reports=False)
    problem.model = nrel_csm_2015()
    problem.setup()
    problem.final_setup()

    def _variables(component, io_type):
        # (name in the component's compute(), metadata) of each of its inputs or outputs
        metadata_by_name = component.get_io_metadata(iotypes=io_type, metadata_keys=["units"], get_remote=False)
        return metadata_by_name.items()

    components = []
    for component in problem.model.system_iter(recurse=True, typ=om.ExplicitComponent):
        if isinstance(component, om.IndepVarComp):
            continue  # the automatic one holding the inputs
        components.append(component)

    output_units = {}  # by promoted name
    for component in components:
        for _, metadata in _variables(component, "output"):
            output_units[metadata["prom_name"]] = metadata["units"]

    steps = []
    defaults = {}
    input_units = {}  # of the inputs not computed by a component, by promoted name
    for component in components:
        continuous_inputs = []
        discrete_inputs = []
        for name, metadata in _variables(component, "input"):
            promoted_name = metadata["prom_name"]
            is_computed = promoted_name in output_units
            if metadata["discrete"]:
                discrete_inputs.append((name, promoted_name))
                if not is_computed:
                    defaults[promoted_name] = problem.get_val(promoted_name)
                continue
            if is_computed:
                # converted from the output's units to this input's units
                unit_factor, unit_offset = unit_conversion(output_units[promoted_name], metadata["units"])
            else:
                unit_factor = 1.0
                unit_offset = 0.0
                first_units = input_units.setdefault(promoted_name, metadata["units"])
                if first_units != metadata["units"]:
                    raise NotImplementedError(f"NREL CSM input '{promoted_name}' has inputs in different units")
                default_value = problem.get_val(promoted_name)
                defaults[promoted_name] = np.array(default_value, dtype=float)
            continuous_inputs.append((name, promoted_name, unit_factor, unit_offset))

        outputs = []
        for name, metadata in _variables(component, "output"):
            if metadata["discrete"]:
                raise NotImplementedError(f"NREL CSM component '{component.pathname}' has discrete outputs")
            outputs.append((name, metadata["prom_name"]))

        steps.append(_ModelStep(component, continuous_inputs, discrete_inputs, outputs))
    return steps, defaults


def _run_nrel_csm_2015(model_inputs: dict[str, ArrayLike], design_count: int) -> dict[str, ArrayLike]:
    """
    Evaluates WISDEM's NREL CSM 2015 model (nrel_csm_2015) for `design_count` designs at once by
    calling its components' compute() on arrays, in the order and with the unit conversions
    OpenMDAO uses. Components that branch on a value differing between the designs (e.g.
    TowerCost2015 on `outputs["tower_parts_cost"] == 0.0`) are computed one design at a time.

    Parameters
    ----------
    model_inputs : dict
        Model inputs by promoted name, as for prob[name] = value; scalars or arrays of length
        `design_count`.
    design_count : int
        Number of designs.

    Returns
    -------
    dict
        All model inputs and outputs by promoted name: float arrays of length 1 or
        `design_count`, except for discrete inputs, which are as given or their default.
    """
    steps, defaults = _nrel_csm_2015_model()

    computed_names = set()
    for step in steps:
        for _, promoted_name in step.outputs:
            computed_names.add(promoted_name)
    unknown_names = [name for name in model_inputs if name not in defaults and name not in computed_names]
    if unknown_names:
        raise KeyError(f"Not inputs of WISDEM's NREL CSM 2015 model: {sorted(unknown_names)}")

    model_values = dict(defaults)  # by promoted name
    for name, value in model_inputs.items():
        # continuous inputs have array defaults, discrete inputs scalar ones
        is_continuous_input = name in defaults and np.ndim(defaults[name]) > 0
        if is_continuous_input:
            float_value = np.asarray(value, dtype=float)
            model_values[name] = np.atleast_1d(float_value)
        else:
            model_values[name] = value

    for component, continuous_inputs, discrete_inputs, outputs in steps:
        component_inputs = {}
        for name, promoted_name, unit_factor, unit_offset in continuous_inputs:
            value = model_values[promoted_name]
            component_inputs[name] = (value + unit_offset) * unit_factor

        component_discrete_inputs = {}
        for name, promoted_name in discrete_inputs:
            component_discrete_inputs[name] = model_values[promoted_name]

        try:
            component_outputs = _compute(component, component_inputs, component_discrete_inputs)
        except ValueError:  # the truth value of an array is ambiguous
            component_outputs = _compute_one_design_at_a_time(
                component, component_inputs, component_discrete_inputs, design_count
            )

        for name, promoted_name in outputs:
            model_values[promoted_name] = component_outputs[name]
    return model_values


def _compute_one_design_at_a_time(component, inputs, discrete_inputs, design_count):
    """
    Calls _compute() once per design, for components that branch on a value differing between
    the designs; returns the outputs of all designs, concatenated.
    """
    input_per_design = {}
    for name, value in inputs.items():
        input_per_design[name] = np.broadcast_to(value, design_count)

    outputs_per_design = []
    for design in range(design_count):
        design_inputs = {}
        for name, value_per_design in input_per_design.items():
            design_inputs[name] = value_per_design[design : design + 1]

        design_discrete_inputs = {}
        for name, value in discrete_inputs.items():
            if np.ndim(value) == 0:  # the same value for all designs
                design_discrete_inputs[name] = value
            else:
                design_discrete_inputs[name] = value[design].item()

        design_outputs = _compute(component, design_inputs, design_discrete_inputs)
        outputs_per_design.append(design_outputs)

    outputs = {}
    for name in outputs_per_design[0]:
        values_per_design = [design_outputs[name] for design_outputs in outputs_per_design]
        outputs[name] = np.concatenate(values_per_design)
    return outputs


def _compute(component, inputs, discrete_inputs):
    """
    Calls component.compute() as OpenMDAO does, with plain dicts; returns the outputs as float
    arrays of at least one dimension.
    """
    outputs = {}
    if discrete_inputs:
        discrete_outputs = {}
        component.compute(inputs, outputs, discrete_inputs, discrete_outputs)
    else:
        component.compute(inputs, outputs)

    float_outputs = {}
    for name, value in outputs.items():
        float_value = np.asarray(value, dtype=float)
        float_outputs[name] = np.atleast_1d(float_value)
    return float_outputs


def _onshore_tcc_scalar(cp, hh, rd, **kwargs):
    """
    Calculates the absolute turbine capital cost in USD according to
    https://wisdem.readthedocs.io/en/master/examples/01_nrelcsm/tutorial.html

    This is the reference implementation, setting up and running an OpenMDAO problem for a
    single design; onshore_tcc() evaluates the same model for many designs at once.

    Parameters
    ----------
    cp : int | float
        Capacity in kW.
    hh : int | float
        Hub height in meters.
    rd : int | float
        Rotor diamater in meters.
    **kwargs
        Will be set as attributes of nrel_csm_2015() model.
        Default values in addition to nrel_csm_2015() are:
        "machine_rating": cp
        "rotor_diameter": rd
        "tower_length": hh
        "turbine_class": 2
        "main_bearing_number": 2
        "blade_number": 3
        "max_tip_speed": 80
        "max_efficiency": 0.90

    Returns
    -------
    float
        Absolute CAPEX in USD_2015.
    """
    prob = om.Problem(reports=False)
    prob.model = nrel_csm_2015()
    prob.setup()
    prob.model.turbine_costs.options["verbosity"] = False
    # ensure or set all mandatory args
    # defaults are taken from https://wisdem.readthedocs.io/en/master/examples/01_nrelcsm/tutorial.html
    params = {
        "machine_rating": cp,
        "rotor_diameter": rd,
        "tower_length": hh,
        "turbine_class": 2,
        "main_bearing_number": 2,
        "blade_number": 3,
        "max_tip_speed": 80,
        "max_efficiency": 0.90,
    }
    params.update(kwargs)  # update default params with kwargs where parameters are missing
    # set all kwarg + default parameters
    for k, v in params.items():
        prob[k] = v

    # run and evaluate the model
    prob.run_model()
    return (
        prob.get_val("turbine_costs.turbine_c.turbine_cost_kW").item() * cp
    )  # previous functions expect absolute cost


def onshore_bos(cp, hh, rd):
    """

    A function to determine the balance of the system cost (BOS) of an onshore turbine based on the capacity, hub height and rotor diameter values according to Fingersh et al. [1].

    Parameters
    ----------
    cp : numeric or array-like
        Turbine's capacity in kW
    hh : numeric or array-like
        Turbine's hub height in m
    rd : numeric or array-like
        Turbine's rotor diameter in m

    Returns
    -------
    numeric or array-like
        Turbine's balance of system costs (BOS) in monetary units.

    References
    ----------
    [1] Fingersh, L., Hand, M., & Laxson, A. (2006). Wind Turbine Design Cost and Scaling Model. NREL. https://www.nrel.gov/docs/fy07osti/40566.pdf

    """
    rr = rd / 2
    sa = np.pi * rr * rr

    # Foundation
    foundationCost = 303.24 * np.power((hh * sa), 0.4037)

    # Transportation
    transporationCostFactor = 1.581e-5 * np.power(cp, 2) - 0.0375 * cp + 54.7
    transporationCost = transporationCostFactor * cp

    # Roads and civil work
    roadsAndCivilWorkFactor = 2.17e-6 * np.power(cp, 2) - 0.0145 * cp + 69.54
    roadsAndCivilWorkCost = roadsAndCivilWorkFactor * cp

    # Assembly and installation
    assemblyAndInstallationCost = 1.965 * np.power((hh * rd), 1.1736)

    # Electrical Interface and connections
    electricalInterfaceAndConnectionFactor = (3.49e-6 * np.power(cp, 2)) - (0.0221 * cp) + 109.7
    electricalInterfaceAndConnectionCost = electricalInterfaceAndConnectionFactor * cp

    # Engineering and permit factor
    engineeringAndPermitCostFactor = 9.94e-4 * cp + 20.31
    engineeringAndPermitCost = engineeringAndPermitCostFactor * cp

    # Add up other costs
    bosCosts = (
        foundationCost
        + transporationCost
        + roadsAndCivilWorkCost
        + assemblyAndInstallationCost
        + electricalInterfaceAndConnectionCost
        + engineeringAndPermitCost
    )

    return bosCosts
