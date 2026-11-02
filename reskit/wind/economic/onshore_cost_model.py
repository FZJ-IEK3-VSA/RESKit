import functools
import warnings
from collections.abc import ItemsView
from typing import Any, Literal, NamedTuple

import numpy as np
from numpy.typing import ArrayLike
from wisdem.nrelcsm.nrel_csm_mass_2015 import nrel_csm_2015
import openmdao.api as om
from openmdao.utils.units import unit_conversion

from reskit.parameters.parameters import OnshoreParameters


def onshore_turbine_capex(
    capacity: ArrayLike,
    hub_height: ArrayLike,
    rotor_diam: ArrayLike,
    base_capex: float | None = None,
    base_capacity: float | None = None,
    base_hub_height: float | None = None,
    base_rotor_diam: float | None = None,
    tcc_share: float | None = None,
    bos_share: float | None = None,
) -> float | np.ndarray:
    """
    A cost and scaling model (CSM) to calculate the total cost of a 3-bladed, direct drive onshore wind turbine according to Fingersh et al. [1] and Maples et al. [2].
    A CSM normalization is done such that a chosen baseline turbine, with a capacity of 4200 kW, hub height of 120 m, and rotor diameter of 136 m, corresponds to a expected typical specific cost of 1100 Eur/kW in a 2050 European context according to Ryberg et al. [4]
    The turbine cost includes the turbine capital cost (TCC) and balance of system costs (BOS), amounting to 67.3% and 22.9% respectively [3], as well as finantial costs equivalent to the the complementary percentage.


    Parameters
    ----------
    capacity : ArrayLike
        Turbine's nominal capacity in kW.

    hub_height : ArrayLike
        Turbine's hub height in m.

    rotor_diam : ArrayLike
        Turbine's rotor diameter in m.

    base_capex : float, optional
        The baseline turbine's capital costs in €, by default 1100*4200 [€/kW * kW]

    base_capacity : float, optional
        The baseline turbine's capacity in kW, by default 4200

    base_hub_height : float, optional
        The baseline turbine's hub height in m, by default 120

    base_rotor_diam : float, optional
        The baseline turbine's rotor diameter in m, by default 136

    tcc_share : float, optional
        The baseline turbine's turbine capital cost (TCC) percentage contribution in the total cost, by default 0.673

    bos_share : float, optional
        The baseline turbine's balance of system costs (BOS) percentage contribution in the total cost, by default 0.229

    Returns
    -------
    float | np.ndarray
        Onshore turbine total cost; a float for scalar inputs, else an array of their broadcast shape.


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

    total_costs = (tcc + bos) / (OnshoreParams.tcc_share + OnshoreParams.bos_share)

    return total_costs


def onshore_tcc(
    cp: ArrayLike,
    hh: ArrayLike,
    rd: ArrayLike,
    gdp_escalator: int | None = None,
    blade_material_escalator: int | None = None,
    blades: int | None = None,
    **kwargs: ArrayLike,
) -> float | np.ndarray:
    """
    A function to determine the turbine capital cost (TCC) of a 3 blade standard onshore wind turbine based capacity, hub height and rotor diameter values according to the cost model by Fingersh et al. [1].

    Parameters
    ----------
    cp : ArrayLike
        Turbine's capacity in kW
    hh : ArrayLike
        Turbine's hub height in m
    rd : ArrayLike
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
    **kwargs : ArrayLike
        Inputs of WISDEM's nrel_csm_2015() model, scalars or arrays broadcastable to the
        designs, see https://wisdem.readthedocs.io/en/master/examples/01_nrelcsm/tutorial.html
        Default values in addition to nrel_csm_2015() are:
        "turbine_class": 2
        "main_bearing_number": 2
        "blade_number": 3
        "max_tip_speed": 80
        "max_efficiency": 0.90

    Returns
    -------
    float | np.ndarray
        Turbine's turbine capital cost (TCC) in USD_2015; a float for scalar cp, hh and rd, else
        an array of their broadcast shape.

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

    # the model takes one flat array per input, the result gets the designs' shape back
    design_shape = cp.shape
    design_count = cp.size
    if design_count == 0:
        # nothing to compute, and computing one design at a time needs at least one
        return np.empty(design_shape, dtype=float)

    # all designs at once, one value per design; defaults taken from
    # https://wisdem.readthedocs.io/en/master/examples/01_nrelcsm/tutorial.html
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
        if np.ndim(input_value) == 0:
            # the same value for all designs, kept a scalar: components branching on a discrete
            # input (e.g. turbine_class) can then still compute all designs at once
            model_inputs[input_name] = input_value
        else:
            # flattened like cp, hh and rd, so that the values line up with the designs
            value_per_design = np.broadcast_to(input_value, design_shape)
            model_inputs[input_name] = value_per_design.ravel()

    model_values = _run_nrel_csm_2015(model_inputs, design_count=design_count)
    specific_turbine_cost = model_values["turbine_cost_kW"]  # in USD_2015/kW
    # model values have length 1 where they depend on no input differing between the designs
    specific_turbine_cost = np.broadcast_to(specific_turbine_cost, design_count)
    specific_turbine_cost = specific_turbine_cost.reshape(design_shape)
    # previous functions expect absolute cost
    turbineCapitalCost = specific_turbine_cost * cp

    if design_shape == ():
        return turbineCapitalCost.item()  # a float for scalar designs, as before
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
    Sets up WISDEM's NREL CSM 2015 model once manually and returns what _run_nrel_csm_2015() needs to
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
    # The model's structure (components, how they connect, units, defaults) is only available from
    # a set-up OpenMDAO problem. Nothing below changes the model: the loops only read this fixed
    # structure, so that _run_nrel_csm_2015() can do what OpenMDAO does when running it.
    problem = om.Problem(reports=False)
    problem.model = nrel_csm_2015()
    problem.setup()
    problem.final_setup()

    def _variables(
        component: om.ExplicitComponent, io_type: Literal["input", "output"]
    ) -> ItemsView[str, dict[str, Any]]:
        # (name in the component's compute(), metadata) of each of its inputs or outputs
        metadata_by_name = component.get_io_metadata(iotypes=io_type, metadata_keys=["units"], get_remote=False)
        return metadata_by_name.items()

    # the components doing the computation, in OpenMDAO's execution order, so that each one runs
    # after the components computing its inputs
    components = []
    for component in problem.model.system_iter(recurse=True, typ=om.ExplicitComponent):
        if isinstance(component, om.IndepVarComp):
            continue  # the automatic one holding the inputs
        components.append(component)

    # all outputs first, as an input connected to an output is fed with it, converted to the
    # input's units, instead of being a model input with a default
    output_units = {}  # by promoted name
    for component in components:
        for _, metadata in _variables(component, "output"):
            output_units[metadata["prom_name"]] = metadata["units"]

    # per component: where each input comes from and where each output goes, i.e. the wiring
    # OpenMDAO does between the components
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
                # passed on as they are: discrete variables have no units
                discrete_inputs.append((name, promoted_name))
                if not is_computed:
                    defaults[promoted_name] = problem.get_val(promoted_name)
                continue
            if is_computed:
                # converted from the output's units to this input's units
                unit_factor, unit_offset = unit_conversion(output_units[promoted_name], metadata["units"])
            else:
                # a model input: given by the caller or its default, both taken in this input's
                # units, which is only right if every input it feeds has the same units
                unit_factor = 1.0
                unit_offset = 0.0
                first_units = input_units.setdefault(promoted_name, metadata["units"])
                if first_units != metadata["units"]:
                    raise NotImplementedError(f"NREL CSM input '{promoted_name}' has inputs in different units")
                default_value = problem.get_val(promoted_name)
                defaults[promoted_name] = np.array(default_value, dtype=float)
            continuous_inputs.append((name, promoted_name, unit_factor, unit_offset))

        # stored under their promoted names, where the inputs of later components look them up
        outputs = []
        for name, metadata in _variables(component, "output"):
            if metadata["discrete"]:
                # none in this model; they would need to be passed on like discrete inputs
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

    # Checks the keys of model_inputs and rejects those that are neither inputs nor outputs of the OpenMDAO model,
    # otherwise a misspelled keyword argument of onshore_tcc() would silently get its default
    computed_names = set()
    for step in steps:
        for _, promoted_name in step.outputs:
            computed_names.add(promoted_name)
    unknown_names = [name for name in model_inputs if name not in defaults and name not in computed_names]
    if unknown_names:
        raise KeyError(f"Not inputs of WISDEM's NREL CSM 2015 model: {sorted(unknown_names)}")

    # Replaces the defaults with the inputs from model_inputs in the OPENMdaro components.
    model_values = dict(defaults)  # by promoted name
    for name, value in model_inputs.items():
        # continuous inputs have array defaults, discrete inputs scalar ones
        is_continuous_input = name in defaults and np.ndim(defaults[name]) > 0
        if is_continuous_input:
            # OpenMDAO stores continuous values as float arrays, so the components compute with
            # floats also for int inputs such as max_tip_speed=80
            float_value = np.asarray(value, dtype=float)
            model_values[name] = np.atleast_1d(float_value)
        else:
            model_values[name] = value

    # in execution order, so the values every component needs were set or computed before
    for component, continuous_inputs, discrete_inputs, outputs in steps:
        # each input's value from the model input or the output it is connected to, in the units
        # the component expects, as OpenMDAO converts them
        component_inputs = {}
        for name, promoted_name, unit_factor, unit_offset in continuous_inputs:
            value = model_values[promoted_name]
            component_inputs[name] = (value + unit_offset) * unit_factor

        component_discrete_inputs = {}
        for name, promoted_name in discrete_inputs:
            component_discrete_inputs[name] = model_values[promoted_name]

        # Computes all designs at once where possible. A component with an `if` on a value that is an array of
        # several designs raises a ValueError, and is computed one design at a time instead.
        try:
            component_outputs = _compute(component, component_inputs, component_discrete_inputs)
        except ValueError as error:
            # re-raises any other ValueError, which computing one design at a time would not fix
            is_array_in_if = "truth value of an array with more than one element is ambiguous" in str(error)
            if not is_array_in_if:
                raise
            component_outputs = _compute_one_design_at_a_time(
                component, component_inputs, component_discrete_inputs, design_count
            )

        # by promoted name, for the inputs of later components and the caller
        for name, promoted_name in outputs:
            model_values[promoted_name] = component_outputs[name]
    return model_values


def _compute_one_design_at_a_time(
    component: om.ExplicitComponent,
    inputs: dict[str, np.ndarray],
    discrete_inputs: dict[str, ArrayLike],
    design_count: int,
) -> dict[str, np.ndarray]:
    """
    Calls _compute() once per design, for components that branch on a value differing between
    the designs.

    Parameters
    ----------
    component : om.ExplicitComponent
        The component to compute.
    inputs : dict
        Continuous inputs by name in compute(): float arrays of length 1 or `design_count`.
    discrete_inputs : dict
        Discrete inputs by name in compute(): plain values (e.g. int or bool) or arrays of
        length `design_count`.
    design_count : int
        Number of designs, at least 1.

    Returns
    -------
    dict
        Outputs by name in compute(): float arrays of length `design_count`, the outputs of all
        designs concatenated.
    """
    # inputs not differing between the designs have length 1, so they can be indexed per design
    input_per_design = {}
    for name, value in inputs.items():
        input_per_design[name] = np.broadcast_to(value, design_count)

    outputs_per_design = []
    for design in range(design_count):
        # slices of length 1, the shape OpenMDAO passes for a single design
        design_inputs = {}
        for name, value_per_design in input_per_design.items():
            design_inputs[name] = value_per_design[design : design + 1]

        # the plain value of the design (e.g. int or bool), as prob[name] = value sets it
        design_discrete_inputs = {}
        for name, value in discrete_inputs.items():
            if np.ndim(value) == 0:  # the same value for all designs
                design_discrete_inputs[name] = value
            else:
                design_discrete_inputs[name] = value[design].item()

        design_outputs = _compute(component, design_inputs, design_discrete_inputs)
        outputs_per_design.append(design_outputs)

    # back to one array per output, of length design_count like the vectorised outputs
    outputs = {}
    for name in outputs_per_design[0]:
        values_per_design = [design_outputs[name] for design_outputs in outputs_per_design]
        outputs[name] = np.concatenate(values_per_design)
    return outputs


def _compute(
    component: om.ExplicitComponent,
    inputs: dict[str, np.ndarray],
    discrete_inputs: dict[str, ArrayLike],
) -> dict[str, np.ndarray]:
    """
    Calls component.compute() as OpenMDAO does, with plain dicts.

    Parameters
    ----------
    component : om.ExplicitComponent
        The component to compute.
    inputs : dict
        Continuous inputs by name in compute(): float arrays.
    discrete_inputs : dict
        Discrete inputs by name in compute(); empty if the component has none.

    Returns
    -------
    dict
        Outputs by name in compute(): float arrays of at least one dimension.
    """
    outputs = {}
    if discrete_inputs:
        # components with discrete variables define compute() with the discrete arguments too
        discrete_outputs = {}
        component.compute(inputs, outputs, discrete_inputs, discrete_outputs)
    else:
        component.compute(inputs, outputs)

    # components may write Python scalars; OpenMDAO stores them in float arrays, which the
    # unit conversion and np.concatenate() in _compute_one_design_at_a_time() rely on
    float_outputs = {}
    for name, value in outputs.items():
        float_value = np.asarray(value, dtype=float)
        float_outputs[name] = np.atleast_1d(float_value)
    return float_outputs


def onshore_bos(
    cp: float | np.ndarray,
    hh: float | np.ndarray,
    rd: float | np.ndarray,
) -> float | np.ndarray:
    """

    A function to determine the balance of the system cost (BOS) of an onshore turbine based on the capacity, hub height and rotor diameter values according to Fingersh et al. [1].

    Parameters
    ----------
    cp : float | np.ndarray
        Turbine's capacity in kW
    hh : float | np.ndarray
        Turbine's hub height in m
    rd : float | np.ndarray
        Turbine's rotor diameter in m

    Returns
    -------
    float | np.ndarray
        Turbine's balance of system costs (BOS) in monetary units, of the broadcast shape of
        cp, hh and rd.

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
