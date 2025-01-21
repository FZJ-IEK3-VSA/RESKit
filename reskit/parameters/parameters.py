# primary packages
import json
import numpy as np
import os
import pandas as pd

# other modules
from reskit.wind.core.data import DATAFOLDER
from reskit.default_paths import DEFAULT_PATHS

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

    def load_and_set_custom_params(self, fp, year, subclass, verbose=False, **kwargs):
        """
        This function loads a parameter table in csv format and writes the
        parameter values into class attributes.

        Parameters
        ----------
        fp : str
            The filepath of a csv file that contains the parameter values
            in a tabular format with the parameter names/units as column
            names and the years as row indices.

        year : integer, optional
            The year for which the parameter shall be returned. Can be
            interpreted as a technical year or a cost year depending
            on the parameter.

        subclass : sub class instance
            The sub class to which the attribute shall be added.

        Returns:
        --------
            None
        """
        # check the input file
        if not isinstance(fp, str) and os.path.splitext(fp)[-1] == ".csv":
            raise TypeError(
                f"Parameter filepath must be a str-formatted '.csv' file: {fp}"
            )
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"Parameter filepath does not exist: {fp}")

        def _round_val(_val):
            if _param in self.rounding.keys():
                if self.rounding[_param] == 0:
                    _val = int(round(_val, 0))
                else:
                    _val = round(_val, self.rounding[_param])
            return _val

        def _get_value(data, year):
            """Interpolates values between neighboring years, or returns
            exact value when available."""
            assert isinstance(
                data, pd.Series
            ), f"data must be of pd.Series type. Here: {type(data)}: {data}"
            # avoid extrapolation
            assert (
                year >= data.index.min() and year <= data.index.max()
            ), f"'year' {year} must be between the min. and max. ({data.index.min()}-{data.index.max()}) given data years to avoid extrapolation."
            # get the nearest year below and above the passed 'year' (if not 'year' available)
            _lower_year = data.index[data.index >= year].min()
            _higher_year = data.index[data.index <= year].max()
            # get the actual value for that year
            if _higher_year == _lower_year:
                # simply choose any year since both the same, here lower
                _val = data[_lower_year]
            else:
                # interpolate between the nearest years and return result
                _val = data[_lower_year] + (data[_higher_year] - data[_lower_year]) * (
                    year - _lower_year
                ) / (_higher_year - _lower_year)

            return _val

        # handle csv files
        if os.path.splitext(os.path.basename(fp))[-1] == ".csv":

            # load data from csv
            params_df = pd.read_csv(fp)

            # make sure year is in columns and set as index
            if not "year" in params_df.columns:
                raise AttributeError(
                    f"'year' is a mandatory column in parameter dataframe: {fp}"
                )
            if not all([isinstance(x, int) and x >= 0 for x in params_df.year]):
                raise ValueError(
                    f"All 'year' entries in parameter dataframe must be integers > 0. Currently: {','.join([str(x) for x in params_df.year])}"
                )
            params_df.set_index("year", inplace=True)

            # check the csv columns, must all be baseline plant attrs
            def _return_colum_type(_param):
                try:
                    # check if we have a plant parameter
                    assert (_param in getattr(subclass, "mand_args")) or (
                        _param in getattr(subclass, "opt_args")
                    )
                    return "parameter"
                except:
                    try:
                        # check if we have a plant parameter unit
                        assert (
                            _param.strip("_unit") in getattr(subclass, "mand_args")
                        ) or (_param.strip("_unit") in getattr(subclass, "opt_args"))
                        return "unit"
                    except:
                        return "other"

            # check and fail if not a param or unit
            for _param in params_df.columns:
                if _param == "remarks":
                    # skip remarks column
                    continue
                elif not _return_colum_type(_param) in ["param", "unit"]:
                    AttributeError(
                        f"Baseline plant parameter csv column '{_param}' is not an attribute of '{subclass.__class__.__name__}'."
                    )

            # make sure all mandatory parameters are provided
            for _param in getattr(subclass, "mand_args"):
                if not _param in params_df.columns:
                    raise AttributeError(
                        f"Mandatory parameter '{_param}' must be an attribute of the parameter dataframe loaded from csv: {fp}"
                    )

            # now get and set the respective values
            for _param in params_df.columns:
                if not _return_colum_type(_param) == "parameter":
                    # skip remarks and units
                    continue
                # get and interpolate where needed
                _val = _get_value(data=params_df[_param], year=year)
                # round if needed
                _val = _round_val(_val)
                # set as attr
                setattr(subclass, _param, _val)
                if verbose and not _param in kwargs.keys():
                    print(
                        f"Baseline plant parameter '{_param}' set to: {_val}",
                        flush=True,
                    )

            # now add optional parameter values that have not been provided in csv
            for _param, _value in getattr(subclass, "opt_args").items():
                if not _param in params_df.columns:
                    # this has not been provided, set default
                    setattr(subclass, _param, _value)

        # other extensions cannot be processed
        else:
            raise TypeError(f"Baseline plant data file is expected to be a .csv file.")

    def update_custom_parameters(self, subclass, **kwargs):
        """
        Iterates over custom parameter names and values, checks if they
        are actually class attributes and overwrites them

        subclass : Parameters() sub class instance
            The sub class in which the custom parameters shall be updated

        **kwargs : optional
            parameter_name = value of custom plant baseline parameters,
            must be attributes of the respective Parameters() class.
        """
        # now iterate over kwargs and overwrite default data where needed
        for _param, _value in kwargs.items():
            if hasattr(subclass, _param):
                # we have an actual attribute
                if _value is not None:
                    # we have an actual custom value, overwrite
                    setattr(subclass, _param, _value)
                    print(
                        f"Baseline plant parameter '{_param}' overwritten by custom value: {_value}",
                        flush=True,
                    )
            else:
                raise AttributeError(
                    f"kwarg '{_param}' is not an attribute of '{subclass.__class__.__name__}'"
                )


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

    # the mandatory arguments that are always required in the file for scaling
    mand_args = [
        "base_capacity",
        "base_hub_height",
        "base_rotor_diam",
        "reference_wind_speed",
        "base_capex_per_capacity",
        "tcc_share",
        "bos_share",
        "gdp_escalator",
        "blade_material_escalator",
        "blades",
    ]
    # optional additional arguments with fallback values which mean the parameter has no effect
    opt_args = {
        "min_tip_height": 0,
        "min_specific_power": 0,
        "max_hub_height": np.inf,
    }

    def __init__(self, fp=None, year=2050, constant_rotor_diam=True, **kwargs):
        """Initializes an instance of the OnshoreParameters class."""
        # we need meaningful definition if rotor or capacity shall be scaled
        if not isinstance(constant_rotor_diam, bool):
            raise TypeError(f"constant_rotor_diam must be a boolean.")
        self.constant_rotor_diam = constant_rotor_diam

        # determine the parameter data file
        if fp is None:
            # use the default file
            if DEFAULT_PATHS["baseline_onshore_turbine_definition_path"] is None:
                fp = os.path.join(DATAFOLDER, "baseline_turbine_onshore_RybergEtAl2019.csv")
            else:
                fp = DEFAULT_PATHS["baseline_onshore_turbine_definition_path"]

        # extract baseline params from file
        self.load_and_set_custom_params(fp=fp, year=year, subclass=self, **kwargs)
        print(
            f"Baseline plant parameters have been loaded for year {year} from: {fp}",
            flush=True,
        )

        # generate dependent attributes
        self.base_capex = self.base_capex_per_capacity * self.base_capacity

        # update custom parameters
        self.update_custom_parameters(subclass=self, **kwargs)


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

    # the mandatory arguments that are always required in the file for scaling size
    mand_args = [
        "base_capacity",
        "base_hub_height",
        "base_rotor_diam",
        "reference_wind_speed",
        "distance_to_bus",
        "foundation",
        "mooring_count",
        "anchor",
        "turbine_count",
        "turbine_spacing",
        "turbine_row_spacing",
    ]
    # optional additional arguments with fallback values which mean the parameter has no effect
    opt_args = {
        "min_tip_height": 0,
        "min_specific_power": 0,
        "max_hub_height": np.inf,
    }

    def __init__(self, fp=None, year=2050, constant_rotor_diam=True, **kwargs):
        """Initializes an instance of the OffshoreParameters class."""
        # we need meaningful definition if rotor or capacity shall be scaled
        if not isinstance(constant_rotor_diam, bool):
            raise TypeError(f"constant_rotor_diam must be a boolean.")
        self.constant_rotor_diam = constant_rotor_diam

        if fp is None:
            # use the default file
            if DEFAULT_PATHS["baseline_offshore_turbine_definition_path"] is None:
                fp = os.path.join(
                    DATAFOLDER, "baseline_turbine_offshore_CaglayanEtAl2019.csv"
                )   
            else:
                fp = DEFAULT_PATHS["baseline_offshore_turbine_definition_path"]

        # extract json params from file
        self.load_and_set_custom_params(fp=fp, year=year, subclass=self, **kwargs)
        print(f"Baseline plant parameters have been loaded from: {fp}", flush=True)

        # update custom parameters
        self.update_custom_parameters(subclass=self, **kwargs)
