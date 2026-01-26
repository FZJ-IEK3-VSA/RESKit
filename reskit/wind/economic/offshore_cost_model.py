import numpy as np

from reskit.parameters.parameters import OffshoreParameters
from reskit.wind.economic.onshore_cost_model import onshore_tcc


def offshore_turbine_capex(
    capacity,
    hub_height,
    rotor_diam,
    depth,
    distance_to_shore,
    distance_to_bus=None,
    foundation=None,
    mooring_count=None,
    anchor=None,
    turbine_count=None,
    turbine_spacing=None,
    turbine_row_spacing=None,
):
    """
    A cost and scaling model (CSM) to calculate the total cost of a 3-bladed, direct drive offshore wind turbine according to the cost model proposed by Fingersh et al. [1] and Maples et al. [2].
    The CSM distinguishes between seaflor-fixed foundation types; "monopile" and "jacket" and floating foundation types; "semisubmersible" and "spar".
    The total turbine cost includes the contributions of the turbine capital cost (TCC), amounting 32.9% for fixed or 23.9% for floating structures, the balance of system costs (BOS) contribution, amounting 46.2% and 60.8% respectively, as well as the finantial costs as the complementary percentage contribution (15.9% and 20.9%) in the same manner [3].
    A CSM normalization is done such that a chosen baseline offshore turbine taken by Caglayan et al. [4] (see notes for details) corresponds to an expected specific cost of 2300 €/kW in a 2050 European context as suggested by the 2016 cost of wind energy review by Stehly [3].

    Parameters
    ----------
    capacity : numeric or array-like
        Turbine's nominal capacity in kW.

    hub_height : numeric or array-like
        Turbine's hub height in m.

    rotor_diam : numeric or array-like
        Turbine's rotor diameter in m.

    depth : numeric or array-like
        Water depth in m (absolute value) at the turbine's location.

    distance_to_shore : numeric or array-like
        Distance from the turbine's location to the nearest shore in km.

    distance_to_bus : numeric or array-like, optional
        Distance from the wind farm's bus in km from the turbine's location.

    foundation : str or array-like of strings, optional
        Turbine's foundation type. Accepted  types are: "monopile", "jacket", "semisubmersible" or "spar", by default "monopile"

    mooring_count : numeric, optional
        Refers to the number of mooring lines are there attaching a turbine only applicable for floating foundation types. By default 3 assuming a triangular attachment to the seafloor.

    anchor : str, optional
        Turbine's anchor type only applicable for floating foundation types, by default as recommended by [1].
        Arguments accepted are "dea" (drag embedment anchor) or "spa" (suction pile anchor).

    turbine_count : numeric, optional
        Number of turbines in the offshore windpark. CSM valid for the range [3-200], by default 80

    turbine_spacing : numeric, optional
        Spacing distance in a row of turbines (turbines that share the electrical connection) to the bus. The value must be a multiplier of rotor diameter. CSM valid for the range [4-9], by default 5

    turbine_row_spacing : numeric, optional
        Spacing distance between rows of turbines. The value must be a multiplier of rotor diameter. CSM valid for the range [4-10], by default 9

    Returns
    -------
    numeric or array-like
        Offshore turbine total cost


    See Also
    --------
        onshore_turbine_capex(capacity, hub_height, rotor_diam, base_capex, base_capacity, base_hub_height, base_rotor_diam, tcc_share, bos_share)

    Notes
    -----
        The baseline offshore turbine correspongs to the optimal design for Europe according to Caglayan et al. [4]: capacity = 9400 kW, hub height = 135 m, rotor diameter = 210 m, "monopile" foundation, reference water depth = 40 m, and reference distance to shore = 60 km.

    Sources
    -------
    [1] Fingersh, L., Hand, M., & Laxson, A. (2006). Wind Turbine Design Cost and Scaling Model. Nrel. https://www.nrel.gov/docs/fy07osti/40566.pdf
    [2] Maples, B., Hand, M., & Musial, W. (2010). Comparative Assessment of Direct Drive High Temperature Superconducting Generators in Multi-Megawatt Class Wind Turbines. Energy. https://doi.org/10.2172/991560
    [3] Stehly, T., Heimiller, D., & Scott, G. (2016). Cost of Wind Energy Review. Technical Report. https://www.nrel.gov/docs/fy18osti/70363.pdf
    [4] Caglayan, D. G., Ryberg, D. S., Heinrichs, H., Linssen, J., Stolten, D., & Robinius, M. (2019). The techno-economic potential of offshore wind energy with optimized future turbine designs in Europe. Applied Energy. https://doi.org/10.1016/j.apenergy.2019.113794
    [5] Maness, M., Maples, B., & Smith, A. (2017). NREL Offshore Balance-of- System Model NREL Offshore Balance-of- System Model. https://www.nrel.gov/docs/fy17osti/66874.pdf
    [6] Myhr, A., Bjerkseter, C., Ågotnes, A., & Nygaard, T. A. (2014). Levelised cost of energy for offshore floating wind turbines in a life cycle perspective. Renewable Energy, 66, 714–728. https://doi.org/10.1016/j.renene.2014.01.017
    [7] Bjerkseter, C., & Ågotnes, A. (2013). Levelised Costs Of Energy For Offshore Floating Wind Turbine Concepts [Norwegian University of Life Sciences]. https://nmbu.brage.unit.no/nmbu-xmlui/bitstream/handle/11250/189073/Bjerkseter%2C C. %26 Ågotnes%2C A. %282013%29 - Levelised Costs of Energy for Offshore Floating Wind Turbine Concepts.pdf?sequence=1&isAllowed=y
    [8] Smart, G., Smith, A., Warner, E., Sperstad, I. B., Prinsen, B., & Lacal-Arantegui, R. (2016). IEA Wind Task 26: Offshore Wind Farm Baseline Documentation. https://doi.org/10.2172/1259255
    [9] RPG CABLES, & KEC International limited. (n.d.). EXTRA HIGH VOLTAGE cables. RPG CABLES. www.rpgcables.com/images/product/EHV-catalogue.pdf

    """
    # TODO: Generalize this function further(like with the onshore cost model)

    # initialize OffshoreParameters class and feed with custom param values
    offshore_params = OffshoreParameters(
        **{
            k: v
            for k, v in locals().items()
            if not k in ["capacity", "hub_height", "rotor_diam", "depth", "distance_to_shore"]
        }
    )

    # PREPROCESS INPUTS
    cp = np.array(capacity / 1000)  # in MW
    # rr = np.array(rotor_diam / 2)
    rd = np.array(rotor_diam)
    hh = np.array(hub_height)
    depth = np.abs(np.array(depth))  # positive values

    # COMPUTE COSTS
    tcc = onshore_tcc(cp=cp * 1000, hh=hh, rd=rd)
    tcc *= 0.7719832742256006

    bos = offshore_bos(
        cp=cp,
        rd=rd,
        hh=hh,
        depth=depth,
        distance_to_shore=np.array(distance_to_shore),
        distance_to_bus=np.array(offshore_params.distance_to_bus),
        foundation=offshore_params.foundation,
        mooring_count=offshore_params.mooring_count,
        anchor=offshore_params.anchor,
        turbine_count=offshore_params.turbine_count,
        turbine_spacing=offshore_params.turbine_spacing,
        turbine_row_spacing=offshore_params.turbine_row_spacing,
    )

    bos *= 0.3669156255898912

    if offshore_params.foundation in ["monopile", "jacket"]:
        fin = (tcc + bos) * 20.9 / (32.9 + 46.2)  # Scaled according to tcc [7]
    else:
        fin = (tcc + bos) * 15.6 / (60.8 + 23.6)  # Scaled according to tcc [7]
    return tcc + bos + fin
    # return np.array([tcc,bos,fin])


def offshore_bos(
    cp,
    rd,
    hh,
    depth,
    distance_to_shore,
    distance_to_bus,
    foundation,
    mooring_count,
    anchor,
    turbine_count,
    turbine_spacing,
    turbine_row_spacing,
):
    """
    A function to determine the balance of the system cost (BOS) of an offshore turbine based on the capacity, hub height and rotor diameter values according to Fingersh et al. [1].

    Parameters
    ----------
    cp : numeric or array-like
        Turbine's nominal capacity in kW

    rd : numeric or array-like
        Turbine's rotor diameter in m

    hh : numeric or array-like
        Turbine's hub height in m

    depth : numeric or array-like
        Water depth in m (absolute value) at the turbine's location.

    distance_to_shore : numeric or array-like
            Distance from the turbine's location to the nearest shore in km.

    distance_to_bus : numeric or array-like, optional
        Distance from the wind farm's bus in km from the turbine's location.

    foundation : str or array-like of strings, optional
        Turbine's foundation type. Accepted  types are: "monopile", "jacket", "semisubmersible" or "spar", by default "monopile"

    mooring_count : numeric, optional
        Refers to the number of mooring lines are there attaching a turbine only applicable for floating foundation types. By default 3 assuming a triangular attachment to the seafloor.

    anchor : str, optional
        Turbine's anchor type only applicable for floating foundation types, by default as recommended by [1].
        Arguments accepted are "dea" (drag embedment anchor) or "spa" (suction pile anchor).

    turbine_count : numeric, optional
        Number of turbines in the offshore windpark. CSM valid for the range [3-200], by default 80

    turbine_spacing : numeric, optional
        Spacing distance in a row of turbines (turbines that share the electrical connection) to the bus. The value must be a multiplier of rotor diameter. CSM valid for the range [4-9], by default 5

    turbine_row_spacing : numeric, optional
        Spacing distance between rows of turbines. The value must be a multiplier of rotor diameter. CSM valid for the range [4-10], by default 9

    Returns
    -------
    numeric
        Offshore turbine's balance of the system cost (BOS) in monetary units.

    Notes
    -----
    Assembly and installation costs could not be implemented due to the excessive number of unspecified constants considered by Smart et al. [8]. Therefore empirical equations were derived which fit the sensitivities to the baseline plants shown in [8]. These ended up being linear equations in turbine capacity and sea depth (only for floating turbines).

    Sources
    -------

    [1] Fingersh, L., Hand, M., & Laxson, A. (2006). Wind Turbine Design Cost and Scaling Model. Nrel. https://www.nrel.gov/docs/fy07osti/40566.pdf
    [2] Maples, B., Hand, M., & Musial, W. (2010). Comparative Assessment of Direct Drive High Temperature Superconducting Generators in Multi-Megawatt Class Wind Turbines. Energy. https://doi.org/10.2172/991560
    [3] Stehly, T., Heimiller, D., & Scott, G. (2016). Cost of Wind Energy Review. Technical Report. https://www.nrel.gov/docs/fy18osti/70363.pdf
    [4] Caglayan, D. G., Ryberg, D. S., Heinrichs, H., Linssen, J., Stolten, D., & Robinius, M. (2019). The techno-economic potential of offshore wind energy with optimized future turbine designs in Europe. Applied Energy. https://doi.org/10.1016/j.apenergy.2019.113794
    [5] Maness, M., Maples, B., & Smith, A. (2017). NREL Offshore Balance-of- System Model NREL Offshore Balance-of- System Model. https://www.nrel.gov/docs/fy17osti/66874.pdf
    [6] Myhr, A., Bjerkseter, C., Ågotnes, A., & Nygaard, T. A. (2014). Levelised cost of energy for offshore floating wind turbines in a life cycle perspective. Renewable Energy, 66, 714–728. https://doi.org/10.1016/j.renene.2014.01.017
    [7] Bjerkseter, C., & Ågotnes, A. (2013). Levelised Costs Of Energy For Offshore Floating Wind Turbine Concepts [Norwegian University of Life Sciences]
    [8] Smart, G., Smith, A., Warner, E., Sperstad, I. B., Prinsen, B., & Lacal-Arantegui, R. (2016). IEA Wind Task 26: Offshore Wind Farm Baseline Documentation. https://doi.org/10.2172/1259255
    [9] RPG CABLES, & KEC International limited. (n.d.). EXTRA HIGH VOLTAGE cables. RPG CABLES. www.rpgcables.com/images/product/EHV-catalogue.pdf

    """
    # rr = rd / 2

    # prevent problems with negative depth values
    depth = np.abs(depth)

    foundation = foundation.lower()
    anchor = anchor.lower()
    if foundation == "monopile" or foundation == "jacket":
        fixed_type = True
    elif foundation == "spar" or foundation == "semisubmersible":
        fixed_type = False
    else:
        raise ValueError(
            f"Please choose one of the four foundation types: monopile, jacket, spar, or semisubmersible. Here: {foundation}"
        )

    # CONSTANTS AND ASSUMPTIONS (all from [1] except where noted)
    # Structure are foundation
    # embedmentDepth = 30  # meters
    monopile_cost_rate = 2250  # dollars/tonne
    monopile_tp_cost_rate = 3230  # dollars/tonne
    spar_sc_cost_rate = 3120  # dollars/tonne
    spar_tc_cost_rate = 4222  # dollars/tonne
    spar_ball_cost_rate = 100  # dollars/tonne
    jacket_ml_cost_rate = 4680  # dollars/tonne
    jacket_tp_cost_rate = 4500  # dollars/tonne
    jacket_pile_cost_rate = 2250  # dollars/tonne
    semi_submersible_sc_cost_rate = 3120  # dollars/tonne
    semi_submersible_t_cost_rate = 6250  # dollars/tonne
    semi_submersible_hp_cost_rate = 6250  # dollars/tonne
    # dollars/tonne -- 0.12m diameter is chosen since it is the median in [1]
    mooring_cost_rate = 721
    outfitting_steel_cost = 7250  # dollars/tonne

    # the values of anchor cost is calculated from Table8 in [2] by assuming a euro to dollar rate of 1.35
    dea_anchor_cost = 154  # dollars [2]
    spa_anchor_cost = 692  # dollars [2]

    # Electrical
    # current rating values are taken from source an approximate number is chosen from tables[4]
    cable_1_current_rating = 400  # [4]
    cable_2_current_rating = 600  # [4]
    # exportCableCurrentRating = 1000  # [4]
    array_voltage = 33
    # exportCableVoltage = 220
    power_factor = 0.95
    # buriedDepth = 1  # this value is chosen from [5] IF THIS CHANGES FROM ONE "singleStringPower1" needs to be updated
    catenary_length_factor = 0.04
    excess_cable_factor = 0.1
    number_of_sub_stations = 1  # From the example used in [5]
    array_cable_cost = 281000 * 1.35  # dollars/km (converted from EUR) [3]
    external_cable_cost = 443000 * 1.35  # dollars/km (converted from EUR) [3]
    single_turbine_interface_cost = 0  # Could not find a number...
    substation_interface_cost = 0  # Could not find a number...
    dynamic_cable_factor = 2
    main_power_transformer_cost_rate = 12500  # dollars/MVA
    high_voltage_switch_gear_cost = 950000  # dollars
    medium_voltage_switchgear_cost = 500000  # dollars
    shunt_reactor_cost_rate = 35000  # dollars/MVA
    diesel_generator_backup_cost = 1000000  # dollars
    workspace_cost = 2000000  # dollars
    other_ancillary_costs = 3000000  # dollars
    fabrication_cost_rate = 14500  # dollars/tonne
    topside_design_cost = 4500000  # dollars
    assembly_factor = 1  # could not find a number...
    offshore_substation_substructure_cost_rate = 6250  # dollars/tonne
    substation_substructure_pile_cost_rate = 2250  # dollars/tonne
    interconnect_voltage = 345  # kV

    # GENERAL (APPENDIX B in NREL BOS MODEL)
    # hubDiam = cp / 4 + 2
    # bladeLength = (rd - hubDiam) / 2

    # nacelleWidth = hubDiam + 1.5
    # nacelleLength = 2 * nacelleWidth

    # RNAMass is rotor nacelle assembly
    rna_mass = 2.082 * cp * cp + 44.59 * cp + 22.48

    # towerDiam = cp / 2 + 4
    # towerMass = (0.4 * np.pi * np.power(rr, 2) * hh - 1500) / 1000

    # STRUCTURE AND FOUNDATION
    if foundation == "monopile":
        # monopileLength = depth + embedmentDepth + 5

        monopile_mass = (
            np.power((cp * 1000), 1.5)
            + (np.power(hh, 3.7) / 10)
            + 2100 * np.power(depth, 2.25)
            + np.power((rna_mass * 1000), 1.13)
        ) / 10000
        monopile_cost = monopile_mass * monopile_cost_rate

        # monopile transition piece mass is called as monopileTPMass

        monopile_tp_mass = np.exp(2.77 + 1.04 * np.power(cp, 0.5) + 0.00127 * np.power(depth, 1.5))
        monopile_tp_cost = monopile_tp_mass * monopile_tp_cost_rate

        foundation_cost = monopile_cost + monopile_tp_cost
        mooring_and_anchor_cost = 0

    elif foundation == "jacket":
        # jacket main lattice mass is called as jacketMLMass
        jacket_ml_mass = np.exp(3.71 + 0.00176 * np.power(cp, 2.5) + 0.645 * np.log(np.power(depth, 1.5)))
        jacket_ml_cost = jacket_ml_mass * jacket_ml_cost_rate

        # jacket transition piece mass is called as jacketTPMass
        jacket_tp_mass = 1 / (((-0.0131 + 0.0381) / np.log(cp)) - 0.00000000227 * np.power(depth, 3))
        jacket_tp_cost = jacket_tp_mass * jacket_tp_cost_rate

        # jacket pile mass is called as jacketPileMass
        jacket_pile_mass = 8 * np.power(jacket_ml_mass, 0.5574)
        jacket_pile_cost = jacket_pile_mass * jacket_pile_cost_rate

        foundation_cost = jacket_ml_cost + jacket_tp_cost + jacket_pile_cost
        mooring_and_anchor_cost = 0

    elif foundation == "spar":
        # spar stiffened column mass is called as sparSCMass
        spar_sc_mass = 535.93 + 17.664 * np.power(cp, 2) + 0.02328 * depth * np.log(depth)
        spar_sc_cost = spar_sc_mass * spar_sc_cost_rate

        # spar tapered column mass is called as sparTCMass
        spar_tc_mass = 125.81 * np.log(cp) + 58.712
        spar_tc_cost = spar_tc_mass * spar_tc_cost_rate

        # spar ballast mass is called as sparBallMass
        spar_ball_mass = -16.536 * np.power(cp, 2) + 1261.8 * cp - 1554.6
        spar_ball_cost = spar_ball_mass * spar_ball_cost_rate

        foundation_cost = spar_sc_cost + spar_tc_cost + spar_ball_cost

        if anchor == "dea":
            anchor_cost = dea_anchor_cost
            # the equation is derived from [3]
            mooring_length = 1.5 * depth + 350

        elif anchor == "spa":
            anchor_cost = spa_anchor_cost
            # since it is assumed to have an angle of 45 degrees it is multiplied by 1.41 which is squareroot of 2 [3]
            mooring_length = 1.41 * depth

        else:
            raise ValueError("Please choose an anchor type!")

        mooring_and_anchor_cost = mooring_length * mooring_cost_rate + anchor_cost

    elif foundation == "semisubmersible":
        # semiSubmersible stiffened column mass is called as semiSubmersibleSCMass
        semi_submersible_sc_mass = -0.9571 * np.power(cp, 2) + 40.89 * cp + 802.09
        semi_submersible_sc_cost = semi_submersible_sc_mass * semi_submersible_sc_cost_rate

        # semiSubmersible truss mass is called as semiSubmersibleTMass
        semi_submersible_t_mass = 2.7894 * np.power(cp, 2) + 15.591 * cp + 266.03
        semi_submersible_t_cost = semi_submersible_t_mass * semi_submersible_t_cost_rate

        # semiSubmersible heavy plate mass is called as semiSubmersibleHPMass
        semi_submersible_hp_mass = -0.4397 * np.power(cp, 2) + 21.145 * cp + 177.42
        semi_submersible_hp_cost = semi_submersible_hp_mass * semi_submersible_hp_cost_rate

        foundation_cost = semi_submersible_sc_cost + semi_submersible_t_cost + semi_submersible_hp_cost

        if anchor == "dea":
            anchor_cost = dea_anchor_cost
            # the equation is derived from [3]
            mooring_length = 1.5 * depth + 350

        elif anchor == "spa":
            anchor_cost = spa_anchor_cost
            # since it is assumed to have an angle of 45 degrees it is multiplied by 1.41 which is squareroot of 2 [3]
            mooring_length = 1.41 * depth

        else:
            raise ValueError("Please choose an anchor type!")

        mooring_and_anchor_cost = mooring_length * mooring_cost_rate + anchor_cost

    if fixed_type:
        if cp > 4:
            secondary_steel_substructure_mass = 40 + (0.8 * (18 + depth))
        else:
            secondary_steel_substructure_mass = 35 + (0.8 * (18 + depth))

    elif foundation == "spar":
        secondary_steel_substructure_mass = np.exp(
            3.58 + 0.196 * np.power(cp, 0.5) * np.log(cp) + 0.00001 * depth * np.log(depth)
        )

    elif foundation == "semisubmersible":
        secondary_steel_substructure_mass = -0.153 * np.power(cp, 2) + 6.54 * cp + 128.34

    secondary_steel_substructure_cost = secondary_steel_substructure_mass * outfitting_steel_cost

    total_structure_and_foundation_costs = (
        foundation_cost + mooring_and_anchor_cost * mooring_count + secondary_steel_substructure_cost
    )

    # ELECTRICAL INFRASTRUCTURE
    # in the calculation of singleStringPower1 and 2, bur depth is assumed to be 1. Because of that the equation is simplified.
    single_string_power1 = np.sqrt(3) * cable_1_current_rating * array_voltage * power_factor / 1000
    single_string_power2 = np.sqrt(3) * cable_2_current_rating * array_voltage * power_factor / 1000

    number_of_strings = np.floor_divide(turbine_count * cp, single_string_power2)

    # Only no partial string will be implemented
    # np.round(np.remainder((turbine_count*cp) , singleStringPower2))
    number_of_turbines_per_partial_string = 0

    number_of_turbines_per_array_cable1 = np.floor_divide(single_string_power1, cp)

    number_of_turbines_per_array_cable2 = np.floor_divide(single_string_power2, cp)

    number_of_turbine_tnterfaces_per_array_cable_1 = number_of_turbines_per_array_cable1 * number_of_strings * 2

    max1_cable1 = np.maximum(number_of_turbines_per_array_cable1 - number_of_turbines_per_array_cable2, 0)
    max2_cable1 = 0
    number_of_turbine_interfaces_per_array_cable_2 = (max1_cable1 * number_of_strings + max2_cable1) * 2

    number_of_array_cable_substation_interfaces = number_of_strings

    if fixed_type:
        array_cable_1_length = (
            (turbine_spacing * rd + depth * 2)
            * (number_of_turbine_tnterfaces_per_array_cable_1 / 2)
            * (1 + excess_cable_factor)
        )
        array_cable_1_length /= 1000  # convert to km
        # print("arrayCable1Length:", arrayCable1Length)
    else:
        system_angle = -0.0047 * depth + 18.743

        free_hanging_cable_length = (depth / np.cos(system_angle * np.pi / 180) * (catenary_length_factor + 1)) + 190

        fixed_cable_length = (turbine_spacing * rd) - (2 * np.tan(system_angle * np.pi / 180) * depth) - 70

        array_cable_1_length = (
            (2 * free_hanging_cable_length)
            * (number_of_turbine_tnterfaces_per_array_cable_1 / 2)
            * (1 + excess_cable_factor)
        )
        array_cable_1_length /= 1000  # convert to km

    max1_cable2 = np.maximum(number_of_turbines_per_array_cable2 - 1, 0)
    max2_cable2 = np.maximum(number_of_turbines_per_partial_string - number_of_turbines_per_array_cable2 - 1, 0)

    str_fac = number_of_strings / number_of_sub_stations

    if fixed_type:
        array_cable_2_length = (turbine_spacing * rd + 2 * depth) * (
            max1_cable2 * number_of_strings + max2_cable2
        ) + number_of_sub_stations * (
            str_fac * (rd * turbine_row_spacing)
            + (
                np.sqrt(np.power((rd * turbine_spacing * (str_fac - 1)), 2) + np.power((rd * turbine_row_spacing), 2))
                / 2
            )
            + str_fac * depth
        ) * (excess_cable_factor + 1)
        array_cable_2_length /= 1000  # convert to km

        array_cable_1_and_ancillary_cost = array_cable_1_length * array_cable_cost + single_turbine_interface_cost * (
            number_of_turbine_tnterfaces_per_array_cable_1 + number_of_turbine_interfaces_per_array_cable_2
        )

        array_cable_2_and_ancillary_cost = (
            array_cable_2_length * array_cable_cost
            + single_turbine_interface_cost
            * (number_of_turbine_tnterfaces_per_array_cable_1 + number_of_turbine_interfaces_per_array_cable_2)
            + substation_interface_cost * number_of_array_cable_substation_interfaces
        )

    else:
        array_cable_2_length = (fixed_cable_length + 2 * free_hanging_cable_length) * (
            max1_cable2 * number_of_strings + max2_cable2
        ) + number_of_sub_stations * (
            str_fac * (rd * turbine_row_spacing)
            + np.sqrt(
                np.power(
                    (
                        (2 * free_hanging_cable_length) * (str_fac - 1)
                        + (rd * turbine_row_spacing)
                        - (2 * np.tan(system_angle * np.pi / 180) * depth)
                        - 70
                    ),
                    2,
                )
                + np.power(fixed_cable_length + 2 * free_hanging_cable_length, 2)
            )
            / 2
        ) * (excess_cable_factor + 1)
        array_cable_2_length /= 1000  # convert to km

        array_cable_1_and_ancillary_cost = dynamic_cable_factor * (
            array_cable_1_length * array_cable_cost
            + single_turbine_interface_cost
            * (number_of_turbine_tnterfaces_per_array_cable_1 + number_of_turbine_interfaces_per_array_cable_2)
        )

        array_cable_2_and_ancillary_cost = dynamic_cable_factor * (
            array_cable_2_length * array_cable_cost
            + single_turbine_interface_cost
            * (number_of_turbine_tnterfaces_per_array_cable_1 + number_of_turbine_interfaces_per_array_cable_2)
            + substation_interface_cost * number_of_array_cable_substation_interfaces
        )

    single_export_cable_power = np.sqrt(3) * cable_2_current_rating * array_voltage * power_factor / 1000
    number_of_export_cables = np.floor_divide(cp * turbine_count, single_export_cable_power) + 1

    if fixed_type:
        export_cable_length = (distance_to_shore * 1000 + depth) * number_of_export_cables * 1.1
        export_cable_length /= 1000  # convert to km

        export_cable_and_ancillary_cost = (
            export_cable_length * external_cable_cost + number_of_export_cables * substation_interface_cost
        )
    else:
        export_cable_length = (
            (distance_to_shore * 1000 + free_hanging_cable_length + 500) * number_of_export_cables * 1.1
        )
        export_cable_length /= 1000  # convert to km

        export_cable_and_ancillary_cost = (
            export_cable_length * external_cable_cost
            + (
                (export_cable_length - free_hanging_cable_length - 500)
                + dynamic_cable_factor * (500 + free_hanging_cable_length)
            )
            + number_of_export_cables * substation_interface_cost
        )

    number_of_sub_stations = number_of_sub_stations

    number_of_main_power_transformers = np.floor_divide(turbine_count * cp, 250) + 1

    # equation 72 in [1] is simplified
    single_mpt_rating = np.round(turbine_count * cp * 1.15 / number_of_main_power_transformers, -1)

    main_power_transformer_cost = (
        number_of_main_power_transformers * single_mpt_rating * main_power_transformer_cost_rate
    )

    switchgear_cost = number_of_main_power_transformers * (
        high_voltage_switch_gear_cost + medium_voltage_switchgear_cost
    )

    shunt_reactor_cost = single_mpt_rating * number_of_main_power_transformers * shunt_reactor_cost_rate * 0.5

    ancillary_systems_cost = diesel_generator_backup_cost + workspace_cost + other_ancillary_costs

    offshore_substation_topside_mass = 3.85 * (single_mpt_rating * number_of_main_power_transformers) + 285
    offshore_substation_topside_cost = offshore_substation_topside_mass * fabrication_cost_rate + topside_design_cost
    assembly_factor = 1  # could not find a number...

    offshore_substation_topside_land_assembly_cost = (
        switchgear_cost + shunt_reactor_cost + main_power_transformer_cost
    ) * assembly_factor

    if fixed_type:
        offshore_substation_substructure_mass = 0.4 * offshore_substation_topside_mass

        substation_substructure_pile_mass = 8 * np.power(offshore_substation_substructure_mass, 0.5574)

        offshore_substation_substructure_cost = (
            offshore_substation_substructure_mass * offshore_substation_substructure_cost_rate
            + substation_substructure_pile_mass * substation_substructure_pile_cost_rate
        )
    else:
        # copied from above in case of spar
        if foundation == "spar":  # WHY WAS IT SPAR BEFORE? WE ARE DOING THINGS WITH SEMISUBMERSIBLE
            # if foundation == 'semisubmersible':
            semi_submersible_sc_mass = -0.9571 * np.power(cp, 2) + 40.89 * cp + 802.09
            semi_submersible_sc_cost = semi_submersible_sc_mass * semi_submersible_sc_cost_rate

            # semiSubmersible truss mass is called as semiSubmersibleTMass
            semi_submersible_t_mass = 2.7894 * np.power(cp, 2) + 15.591 * cp + 266.03
            semi_submersible_t_cost = semi_submersible_t_mass * semi_submersible_t_cost_rate

            # semiSubmersible heavy plate mass is called as semiSubmersibleHPMass
            semi_submersible_hp_mass = -0.4397 * np.power(cp, 2) + 21.145 * cp + 177.42
            semi_submersible_hp_cost = semi_submersible_hp_mass * semi_submersible_hp_cost_rate

        semi_submersible_mass = semi_submersible_sc_mass + semi_submersible_t_mass + semi_submersible_hp_mass

        offshore_substation_substructure_mass = 2 * (semi_submersible_mass + secondary_steel_substructure_mass)

        substation_substructure_pile_mass = 0

        # semiSubmersibleCost = semiSubmersibleSCCost + semiSubmersibleTCost + semiSubmersibleHPCost
        offshore_substation_substructure_cost = 2 * (semi_submersible_t_cost_rate + mooring_and_anchor_cost)

    onshore_substation_cost = 11652 * (interconnect_voltage + cp * turbine_count) + 1200000

    onshore_substation_misc_cost = 11795 * np.power(cp * turbine_count, 0.3549) + 350000

    overhead_transmission_line_cost = (
        (1176 * interconnect_voltage + 218257) * np.power(distance_to_bus, -0.1063) * distance_to_bus
    )

    switchyard_cost = 18115 * interconnect_voltage + 165944

    total_electrical_infrastructure_costs = (
        array_cable_1_and_ancillary_cost
        + array_cable_2_and_ancillary_cost
        + export_cable_and_ancillary_cost
        + main_power_transformer_cost
        + switchgear_cost
        + shunt_reactor_cost
        + ancillary_systems_cost
        + offshore_substation_topside_cost
        + offshore_substation_topside_land_assembly_cost
        + offshore_substation_substructure_cost
        + onshore_substation_cost
        + onshore_substation_misc_cost
        + overhead_transmission_line_cost
        + switchyard_cost
    )
    total_electrical_infrastructure_costs /= turbine_count

    # ASSEMBLY AND INSTALLATION

    assembly_and_installation_cost = np.ones(total_electrical_infrastructure_costs.shape)

    if fixed_type:
        assembly_and_installation_cost *= 4200000
    else:
        assembly_and_installation_cost *= 5500000

    # depth depedance
    if fixed_type:
        pass
    else:
        # Normalized to 1 at 250m depth
        assembly_and_installation_cost *= 0.00041757917648320338 * depth + 0.89560520587919934

    # Capacity dependence
    # Normalized to 1 at 6 MW
    assembly_and_installation_cost *= 0.05947387 * cp + 0.64371944

    # OTHER THINGS
    # Again, many constants were used in [1] but not defined. Also, many of the costs were given in the
    # context of the USA. Therefore the other groups were are simply treated as percentages which
    # fit the examples shown in [1] or [7]

    #########################################
    # The below corresponds to other costs in [1]
    # tot = (assemblyAndInstallationCost + totalElectricalInfrastructureCosts + totalStructureAndFoundationCosts)/(1-0.06)

    # commissioning = tot*0.015
    # portAndStaging = tot*0.005
    # engineeringManagement = tot*0.02
    # development = tot*0.02

    #########################################
    # The below corresponds to cost percentages in [7]
    if fixed_type:
        tot = (
            assembly_and_installation_cost * 19.0
            + total_electrical_infrastructure_costs * 9.00
            + total_structure_and_foundation_costs * 13.9
        ) / 46.2

        commissioning = tot * (0.8 / 46.2)
        port_and_staging = tot * (0.5 / 46.2)
        engineering_management = tot * (1.6 / 46.2)
        development = tot * (1.4 / 46.2)

    else:
        tot = (
            assembly_and_installation_cost * 11.3
            + total_electrical_infrastructure_costs * 10.9
            + total_structure_and_foundation_costs * 34.1
        ) / 60.8

        commissioning = tot * (0.8 / 60.8)
        port_and_staging = tot * (0.6 / 60.8)
        engineering_management = tot * (2.2 / 60.8)
        development = tot * (1 / 60.8)

    # TOTAL COST
    total_cost = (
        commissioning
        + assembly_and_installation_cost
        + total_electrical_infrastructure_costs
        + total_structure_and_foundation_costs
        + port_and_staging
        + engineering_management
        + development
    )

    return total_cost
