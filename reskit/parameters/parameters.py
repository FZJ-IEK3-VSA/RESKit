class Parameters:
    """
    This class holds the base techno-economic parameter assumptions on which 
    the individual functions rely. The base parameter set can be updated by 
    loader/setter functions.
    """
    # static deafult parameters as attributes for external quick access    
    onshore={
        'constant_rotor_diam': True, 
        'base_capacity': 4200, 
        'base_hub_height': 120, 
        'base_rotor_diam': 136, 
        'reference_wind_speed': 6.7, 
        'min_tip_height': 20, 
        'min_specific_power': 180,
        
        'base_capex': 1100 * 4200, #1100 EUR/kW at base capacity 4200kW #TODO change attribute to base_capex_per_kW to be independent of base_capacity
        'base_capacity': 4200, #[kW]
        'base_hub_height': 120, #[m]
        'base_rotor_diam': 136, #[m]
        'tcc_share': 0.673, #[-]
        'bos_share': 0.229, #[-]
        
        'gdp_escalator': 1, 
        'blade_material_escalator': 1, 
        'blades': 3,
    }
    offshore={
        'distance_to_bus': 3, 
        'foundation': "monopile", 
        'mooring_count': 3, 
        'anchor': "DEA", 
        'turbine_count': 80, 
        'turbine_spacing': 5, 
        'turbine_row_spacing': 9
    }
    solar={
    }

    def __init__(self):
        #TODO possibly already add loader here so initialization means new parameter set (if optional! json path is given)
        pass

    def load_custom_params_set(self):
        """
        This function loads a dictionary of parameters in json format and
        overwrites the preset static attributes.
        """
        pass

    def load_individual_params(self,         
        constant_rotor_diam=None,
        base_capacity=None,
        base_hub_height=None,
        base_rotor_diam=None,
        reference_wind_speed=None,
        min_tip_height=None,
        min_specific_power=None,
        base_capex=None,
        tcc_share=None,
        bos_share=None,
        gdp_escalator=None,
        blade_material_escalator=None,
        blades=None,
        distance_to_bus =None,
        foundation=None,
        mooring_count =None,
        anchor=None,
        turbine_count=None,
        turbine_spacing =None,
        turbine_row_spacing=None,):
        """
        Function allows to load individual parameters into base parameter set 
        that is applied in several functions throughout reskit.

        Parameters
        ----------
        constant_rotor_diam : [type], optional
            [description], by default None
        base_capacity : [type], optional
            [description], by default None
        base_hub_height : [type], optional
            [description], by default None
        base_rotor_diam : [type], optional
            [description], by default None
        reference_wind_speed : [type], optional
            [description], by default None
        min_tip_height : [type], optional
            [description], by default None
        min_specific_power : [type], optional
            [description], by default None
        base_capex : [type], optional
            [description], by default None
        tcc_share : [type], optional
            [description], by default None
        bos_share : [type], optional
            [description], by default None
        gdp_escalator : [type], optional
            [description], by default None
        blade_material_escalator : [type], optional
            [description], by default None
        blades : [type], optional
            [description], by default None
        distance_to_bus : [type], optional
            [description], by default None
        foundation : [type], optional
            [description], by default None
        mooring_count : [type], optional
            [description], by default None
        anchor : [type], optional
            [description], by default None
        turbine_count : [type], optional
            [description], by default None
        turbine_spacing : [type], optional
            [description], by default None
        turbine_row_spacing : [type], optional
            [description], by default None
        """
        pass
        # reset all self.attributes that are not None
        # for k in locals().keys():
        #     if not k is None:
        #         self.wind[k]==k



        # Parameters
        # ----------

        # constant_rotor_diam : bool, optional
        #     Whether the rotor diameter is mantained constant or not, by default True

        # base_capacity : numeric or array_like, optional
        #     Baseline turbine capacity in kW, by default 4200.

        # base_hub_height : numeric or array_like, optional
        #     Baseline turbine hub height in m, by default 120.

        # base_rotor_diam : numeric or array_like, optional
        #     Baseline turbine rotor diameter in m, by default 136.

        # reference_wind_speed : numeric, optional
        #     Average wind speed corresponding to the baseline turbine design, by default 6.7.

        # min_tip_height : numeric, optional.
        #     Minimum distance in m between the lower tip of the blades and the ground, by default 20.

        # min_specific_power : numeric, optional
        #     Minimum specific power allowed in kw/m2, by default 180.



        # base_capex : numeric, optional
        #     The baseline turbine's capital costs in €, by default 1100*4200 [€/kW * kW]

        # base_capacity : int, optional
        #     The baseline turbine's capacity in kW, by default 4200

        # base_hub_height : int, optional
        #     The baseline turbine's hub height in m, by default 120

        # base_rotor_diam : int, optional
        #     The baseline turbine's rotor diamter in m, by default 136

        # tcc_share : float, optional
        #     The baseline turbine's TCC percentage contribution in the total cost, by default 0.673

        # bos_share : float, optional
        #     The baseline turbine's BOS percentage contribution in the total cost, by default 0.229 



        # distance_to_bus : numeric or array-like, optional
        #     Distance from the wind farm's bus in km from the turbine's location.

        # foundation : str or array-like of strings, optional
        #     Turbine's foundation type. Accepted  types are: "monopile", "jacket", "semisubmersible" or "spar", by default "monopile"

        # mooring_count : numeric, optional
        #     Refers to the number of mooring lines are there attaching a turbine only applicable for floating foundation types. By default 3 assuming a triangular attachment to the seafloor.

        # anchor : str, optional
        #     Turbine's anchor type only applicable for floating foundation types, by default as reccomended by [1].
        #     Arguments accepted are "dea" (drag embedment anchor) or "spa" (suction pile anchor).

        # turbine_count : numeric, optional
        #     Number of turbines in the offshore windpark. CSM valid for the range [3-200], by default 80

        # turbine_spacing : numeric, optional
        #     Spacing distance in a row of turbines (turbines that share the electrical connection) to the bus. The value must be a multiplyer of rotor diameter. CSM valid for the range [4-9], by default 5

        # turbine_row_spacing : numeric, optional
        #     Spacing distance between rows of turbines. The value must be a multiplyer of rotor diameter. CSM valid for the range [4-10], by default 9