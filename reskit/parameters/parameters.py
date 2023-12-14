class Parameters:
    """
    This class holds the base techno-economic parameter assumptions on which
    the individual functions rely. The base parameter set can be updated by
    loader/setter functions.
    """

    def __init__(self):
        """
        This class is initialized without any arguments.
        """
        pass

    def load_custom_params_set(self):
        """
        This function loads a dictionary of parameters in json format and
        overwrites the preset static attributes.
        """
        pass


class OnshoreParameters(Parameters):
    """
    This class holds all onshore-wind specific techno-economic base parameter
    assumptions as static attributes as well as specific methods to manipulate
    onshore parameters.
    """

    # static baseline turbine attributes
    constant_rotor_diam = True
    base_capacity = 4200  # [kW]
    base_hub_height = 120  # [m]
    base_rotor_diam = 136  # [m]
    reference_wind_speed = 6.7  # [m/s]
    min_tip_height = 20
    min_specific_power = 180
    # static economic attributes
    base_capex_per_capacity = 1100  # [EUR/kW]
    base_capex = base_capex_per_capacity * base_capacity  # [EUR]
    tcc_share = 0.673  # [-]
    bos_share = 0.229  # [-]
    # static turbine design attributes
    gdp_escalator = 1
    blade_material_escalator = 1
    blades = 3

    def __init__(self):
        pass

    def load_individual_params(
        self,
        constant_rotor_diam=None,
        base_capacity=None,
        base_hub_height=None,
        base_rotor_diam=None,
        reference_wind_speed=None,
        min_tip_height=None,
        min_specific_power=None,
        base_capex_per_capacity=None,
        tcc_share=None,
        bos_share=None,
        gdp_escalator=None,
        blade_material_escalator=None,
        blades=None,
    ):
        """
        Function allows to load individual parameters into base parameter set
        that is applied in several functions throughout reskit.

        Parameters
        ----------
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

        base_capex : numeric, optional
            The baseline turbine's capital costs in €, by default 1100*4200 [€/kW * kW] #TODO change to

        tcc_share : float, optional
            The baseline turbine's TCC percentage contribution in the total cost, by default 0.673

        bos_share : float, optional
            The baseline turbine's BOS percentage contribution in the total cost, by default 0.229
        """
        pass


class OffshoreParameters(Parameters):
    """
    This class holds all offshore-wind specific techno-economic base parameter
    assumptions as static attributes as well as specific methods to manipulate
    offshore parameters.
    """

    distance_to_bus = 3
    foundation = "monopile"
    mooring_count = 3
    anchor = "DEA"
    turbine_count = 80
    turbine_spacing = 5
    turbine_row_spacing = 9

    def __init__(self):
        """
        This class is initiated without passing arguments.
        """
        pass

    def load_individual_params(
        self,
        distance_to_bus=None,
        foundation=None,
        mooring_count=None,
        anchor=None,
        turbine_count=None,
        turbine_spacing=None,
        turbine_row_spacing=None,
    ):
        """
        [Summary]

        Parameters
        ----------
        distance_to_bus : numeric or array-like, optional
            Distance from the wind farm's bus in km from the turbine's location.

        foundation : str or array-like of strings, optional
            Turbine's foundation type. Accepted  types are: "monopile", "jacket", "semisubmersible" or "spar", by default "monopile"

        mooring_count : numeric, optional
            Refers to the number of mooring lines are there attaching a turbine only applicable for floating foundation types. By default 3 assuming a triangular attachment to the seafloor.

        anchor : str, optional
            Turbine's anchor type only applicable for floating foundation types, by default as reccomended by [1].
            Arguments accepted are "dea" (drag embedment anchor) or "spa" (suction pile anchor).

        turbine_count : numeric, optional
            Number of turbines in the offshore windpark. CSM valid for the range [3-200], by default 80

        turbine_spacing : numeric, optional
            Spacing distance in a row of turbines (turbines that share the electrical connection) to the bus. The value must be a multiplyer of rotor diameter. CSM valid for the range [4-9], by default 5

        turbine_row_spacing : numeric, optional
            Spacing distance between rows of turbines. The value must be a multiplyer of rotor diameter. CSM valid for the range [4-10], by default 9
        """
        pass
