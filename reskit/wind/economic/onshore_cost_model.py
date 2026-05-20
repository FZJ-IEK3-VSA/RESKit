import warnings

import numpy as np
from wisdem.nrelcsm.nrel_csm_mass_2015 import nrel_csm_2015
import openmdao.api as om

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

    See Also
    --------
        offshore_turbine_capex(capacity, hub_height, rotor_diam, depth, distance_to_shore, distance_to_bus, foundation, mooring_count, anchor, turbine_count, turbine_spacing, turbine_row_spacing)

    Notes
    -----
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
    blade_material_escalator : int, optional
        Blade material cost escalator, by default 1
    blades : int, optional
        Number of blades, by default 3

    #TODO: kwargs

    Returns
    -------
    numeric or array-like
        Turbine's turbine capital cost (TCC) in monetary units.

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

    if cp.shape == ():
        turbineCapitalCost = _onshore_tcc_scalar(
            cp=cp.item(),
            hh=hh.item(),
            rd=rd.item(),
            **kwargs,
        )
    else:
        turbineCapitalCost = np.empty(cp.shape, dtype=float)
        for idx in np.ndindex(cp.shape):
            turbineCapitalCost[idx] = _onshore_tcc_scalar(
                cp=cp[idx],
                hh=hh[idx],
                rd=rd[idx],
                **_select_scalar_kwargs(kwargs, idx),
            )

    return turbineCapitalCost


def _select_scalar_kwargs(kwargs, idx):
    scalar_kwargs = {}
    for k, v in kwargs.items():
        value = np.asarray(v)
        if value.shape == ():
            scalar_kwargs[k] = value.item()
        else:
            scalar_kwargs[k] = value[idx]
    return scalar_kwargs


def _onshore_tcc_scalar(cp, hh, rd, **kwargs):
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
    return prob.get_val("turbine_costs.turbine_c.turbine_cost_kW").item() * cp # previous functions expect absolute cost


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
