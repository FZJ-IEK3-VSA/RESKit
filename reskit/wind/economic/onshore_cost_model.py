import numpy as np

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
    onshore_params = OnshoreParameters(
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
        onshore_params.base_capex
        * onshore_params.tcc_share
        / onshore_tcc(
            cp=onshore_params.base_capacity,
            hh=onshore_params.base_hub_height,
            rd=onshore_params.base_rotor_diam,
        )
    )
    tcc = onshore_tcc(cp=cp, hh=hh, rd=rd) * tcc_scaling

    bos_scaling = (
        onshore_params.base_capex
        * onshore_params.bos_share
        / onshore_bos(
            cp=onshore_params.base_capacity,
            hh=onshore_params.base_hub_height,
            rd=onshore_params.base_rotor_diam,
        )
    )
    bos = onshore_bos(cp=cp, hh=hh, rd=rd) * bos_scaling

    # print(tcc_scaling, bos_scaling)

    total_costs = (tcc + bos) / (onshore_params.tcc_share + onshore_params.bos_share)

    # other_costs = total_costs * (1 - OnshoreParams.tcc_share - OnshoreParams.bos_share)

    return total_costs


def onshore_tcc(cp, hh, rd, gdp_escalator=None, blade_material_escalator=None, blades=None):
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

    Returns
    -------
    numeric or array-like
        Turbine's turbine capital cost (TCC) in monetary units.

    References
    ----------
    [1] Fingersh, L., Hand, M., & Laxson, A. (2006). Wind Turbine Design Cost and Scaling Model. NREL. https://www.nrel.gov/docs/fy07osti/40566.pdf

    """
    # initialize OnshoreParameters class and feed with custom param values
    onshore_params = OnshoreParameters(
        gdp_escalator=gdp_escalator,
        blade_material_escalator=blade_material_escalator,
        blades=blades,
    )

    rr = rd / 2
    sa = np.pi * rr * rr

    # Blade Cost
    single_blade_mass = 0.4948 * np.power(rr, 2.53)
    single_blade_cost = (
        (0.4019 * np.power(rr, 3) - 21051) * onshore_params.blade_material_escalator
        + 2.7445 * np.power(rr, 2.5025) * onshore_params.gdp_escalator
    ) * (1 - 0.28)

    # Hub
    hub_mass = 0.945 * single_blade_mass + 5680.3
    hub_cost = hub_mass * 4.25

    # Pitch and bearings
    # pitchBearingMass = 0.1295 * (singleBladeMass * blades) + 491.31
    # pitchSystemMass = pitchBearingMass*1.328+555
    pitch_system_cost = 2.28 * (0.2106 * np.power(rd, 2.6578))

    # Spinner and nosecone
    nose_cone_mass = 18.5 * rd - 520.5
    nose_cone_cost = nose_cone_mass * 5.57

    # Low Speed Shaft
    # lowSpeedShaftMass = 0.0142 * np.power(rd, 2.888)
    low_speed_shaft_cost = 0.01 * np.power(rd, 2.887)

    # Main bearings
    bearing_mass = (rd * 8 / 600 - 0.033) * 0.0092 * np.power(rd, 2.5)
    bearing_cost = 2 * bearing_mass * 17.6

    # Gearbox
    # Gearbox not included for direct drive turbines

    # Break, coupling, and others
    break_coupling_cost = 1.9894 * cp - 0.1141
    # breakCouplingMass = breakCouplingCost/10

    # Generator (Assuming direct drive)
    # generatorMass = 6661.25 * np.power(lowSpeedShaftTorque, 0.606) # wtf is the torque?
    generator_cost = cp * 219.33

    # Electronics
    electronics_cost = cp * 79

    # Yaw drive and bearing
    # yawSystemMass = 1.6*(0.0009*np.power(rd, 3.314))
    yaw_system_cost = 2 * (0.0339 * np.power(rd, 2.964))

    # Mainframe (Assume direct drive)
    mainframe_mass = 1.228 * np.power(rd, 1.953)
    mainframe_cost = 627.28 * np.power(rd, 0.85)

    # Platform and railings
    platform_and_railing_mass = 0.125 * mainframe_mass
    platform_and_railing_cost = platform_and_railing_mass * 8.7

    # Electrical Connections
    electrical_connection_cost = cp * 40

    # Hydraulic and Cooling systems
    # hydraulicAndCoolingSystemMass = 0.08 * cp
    hydraulic_and_cooling_system_cost = cp * 12

    # Nacelle Cover
    nacelle_cost = 11.537 * cp + 3849.7
    # nacelleMass = nacelleCost/10

    # Tower
    tower_mass = 0.2694 * sa * hh + 1779
    tower_cost = tower_mass * 1.5

    # Add up the turbine capital cost
    turbine_capital_cost = (
        single_blade_cost * onshore_params.blades
        + hub_cost
        + pitch_system_cost
        + nose_cone_cost
        + low_speed_shaft_cost
        + bearing_cost
        + break_coupling_cost
        + generator_cost
        + electronics_cost
        + yaw_system_cost
        + mainframe_cost
        + platform_and_railing_cost
        + electrical_connection_cost
        + hydraulic_and_cooling_system_cost
        + nacelle_cost
        + tower_cost
    )

    return turbine_capital_cost


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
    foundation_cost = 303.24 * np.power((hh * sa), 0.4037)

    # Transportation
    transportation_cost_factor = 1.581e-5 * np.power(cp, 2) - 0.0375 * cp + 54.7
    transportation_cost = transportation_cost_factor * cp

    # Roads and civil work
    roads_and_civil_work_factor = 2.17e-6 * np.power(cp, 2) - 0.0145 * cp + 69.54
    roads_and_civil_work_cost = roads_and_civil_work_factor * cp

    # Assembly and installation
    assembly_and_installation_cost = 1.965 * np.power((hh * rd), 1.1736)

    # Electrical Interface and connections
    electrical_interface_and_connection_factor = (3.49e-6 * np.power(cp, 2)) - (0.0221 * cp) + 109.7
    electrical_interface_and_connection_cost = electrical_interface_and_connection_factor * cp

    # Engineering and permit factor
    engineering_and_permit_cost_factor = 9.94e-4 * cp + 20.31
    engineering_and_permit_cost = engineering_and_permit_cost_factor * cp

    # Add up other costs
    bos_costs = (
        foundation_cost
        + transportation_cost
        + roads_and_civil_work_cost
        + assembly_and_installation_cost
        + electrical_interface_and_connection_cost
        + engineering_and_permit_cost
    )

    return bos_costs
