import json
import numpy as np
import os
import pandas as pd

class Parameters:
    """
    This class holds the base techno-economic parameter assumptions on which
    the individual functions rely. The base parameter set can be updated by
    loader/setter functions.
    """

    # if param is a key in the following dict, parameter values will be rounded to the given digits
    rounding = {
        "base_capacity": 0,
        "base_hub_height": 0,
        "base_rotor_diam": 0,
        "min_tip_height": 0,
        "min_specific_power": 0,
    }

    def __init__(self):
        """
        This class is initialized without any arguments.
        """
        pass

    def load_and_set_custom_params(self, fp, year, subclass, verbose=False):
        """
        This function loads a dictionary of parameters in json format and writes the
        parameter values into class attributes.

        Parameters
        ----------
        fp : str
            The filepath of a json file that contains the parameter names
            and values as key/value pairs. Values can be dicts with
            integer years as sub keys and the actual parameters as values
            per year.

        year : integer, optional
            The year for which the parameter shall be returned. Can be
            interpreted as a technical year or a cost year depending
            on the parameter, by default 2050.

        subclass : sub class instance
            The sub class to which the attribute shall be added.

        Returns:
        --------
            None
        """
        # check the input file
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"Parameter filepath does not exist: {fp}")
        
        def _interpolate_vals(data, year):
            """Interpolates values between neighboring years, or returns exact value when available."""
            if isinstance(data, dict):
                data=pd.Series(dict)
            else:
                assert isinstance(data, pd.Series), f"data must be of dict or pd.Series type. Here: {type(data)}: {data}"
            # avoid extrapolation
            assert (
                year >= data.index.min() and year <= data.index.max()
            ), f"'year' {year} must be between the min. ({data.index.min()}) and max. ({data.index.max()}) given data years to avoid interpolation."
            # get the nearest year below and above the passed 'year' (if not 'year' available)
            _lower_year = data.index[data.index >= year].min()
            _higher_year = data.index[data.index <= year].max()
            # get the actual value for that year
            if _higher_year == _lower_year:
                # simply choose any year since both the same, here lower
                _val = data[_lower_year]
            else:
                # interpolate between the nearest years and return result
                _val = data[_lower_year] + (
                    data[_higher_year] - data[_lower_year]
                ) * (year - _lower_year) / (_higher_year - _lower_year)
            return _val
        
        def _round_val(_val):
            if _param in self.rounding.keys():
                if self.rounding[_param] == 0:
                    _val = int(round(_val, 0))
                else:
                    _val = round(_val, self.rounding[_param])
            return _val
        
        # handle json files
        if os.path.splitext(os.path.basename(fp))[-1] == ".json":
            # load data from json
            with open(fp, "r") as fp:
                json_params = json.load(fp)
            # extract data from dict
            for _param, _val in json_params.items():
                # convert sub year keys to int where needed
                if isinstance(_val, dict) and all([k.isnumeric() for k in _val.keys()]):
                    # assume we have a param with yar-dependent values, convert to ints and write back to json params
                    _val = {int(k): v for k, v in _val.items()}
                    _val = _interpolate_vals(data=_val, year=year)

                # round the parameters where needed
                _val =_round_val(_val)
                # set the parameter value as class rttribute
                setattr(subclass, _param, _val)
                if verbose:
                    print(f"Baseline plant parameter '{_param}' set to: {_val}", flush=True)
        
        # handle csv baseline files
        elif os.path.splitext(os.path.basename(fp))[-1] == ".csv":
            
            # load data from csv
            params_df = pd.read_csv(fp)

            # reset anc check index years
            if all([c in params_df.columns for c in ["Unnamed: 0", "Index"]]) \
                or not any([c in params_df.columns for c in ["Unnamed: 0", "Index"]]):
                raise AttributeError(f"When loading baseline plant data from csv, a column 'Index' or 'Unnamed: 0' is expected (but not both) that contain the index years.")
            try:
                params_df.set_index('Index', drop=True, inplace=True)
            except:
                params_df.set_index('Unnamed: 0', drop=True, inplace=True)
            if not all([isinstance(i, int) for i in params_df.index]):
                raise TypeError(f"Index/Unnamed: 0 column in csv data must contain only integer year values.")

            # check the csv columns, must all be baseline plant attrs
            for _param in params_df.columns:
                try:
                    getattr(subclass, _param)
                except:
                    AttributeError(f"Baseline plant csv column '{_param}' is not an attribute of {subclass}.")
            # make sure the year is avalailable in csv data
            if not year in params_df.index:
                raise AttributeError(f"Year '{year}' is not available in the input data. Select from: {', '.join([str(i) for i in params_df.index])}")

            # now get and set the respective values
            for _param in params_df.columns:
                # get and interpolate where needed
                _val = _interpolate_vals(data=params_df[_param], year=year)
                # round if needed
                _val = _round_val(_val)
                # set as attr
                setattr(subclass, _param, _val)
                if verbose:
                    print(f"Baseline plant parameter '{_param}' set to: {_val}", flush=True)

        # other extensions cannot be processed
        else:
            raise TypeError(f"Baseline plant data file is expected to be a .json or .csv file.")

class OnshoreParameters(Parameters):
    """
    This class holds all onshore-wind specific techno-economic base parameter
    assumptions as static attributes as well as specific methods to manipulate
    onshore parameters.

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

    # static baseline turbine attributes
    constant_rotor_diam = True
    base_capacity = 4200  # [kW]
    base_hub_height = 120  # [m]
    base_rotor_diam = 136  # [m]
    reference_wind_speed = 6.7  # [m/s]
    min_tip_height = 20
    min_specific_power = 180
    # max. projection value from expert survey in Wiser et al. (2021)
    max_hub_height = 200
    # static economic attributes
    base_capex_per_capacity = 1100  # [EUR/kW]
    base_capex = base_capex_per_capacity * base_capacity  # [EUR]
    tcc_share = 0.673  # [-]
    bos_share = 0.229  # [-]
    # static turbine design attributes
    gdp_escalator = 1
    blade_material_escalator = 1
    blades = 3

    def __init__(self, fp=None, year=2050):
        if not fp is None:
            # extract json params from file
            self.load_and_set_custom_params(fp=fp, year=year, subclass=self)
        else:
            pass


class OffshoreParameters(Parameters):
    """
    This class holds all offshore-wind specific techno-economic base parameter
    assumptions as static attributes as well as specific methods to manipulate
    offshore parameters.

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

    distance_to_bus = 3
    foundation = "monopile"
    mooring_count = 3
    anchor = "DEA"
    turbine_count = 80
    turbine_spacing = 5
    turbine_row_spacing = 9

    def __init__(self, fp=None, year=2050):
        if not fp is None:
            # extract json params from file
            self.load_and_set_custom_params(fp=fp, year=year, subclass=self)
        else:
            pass
