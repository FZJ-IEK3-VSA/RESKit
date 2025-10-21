import geokit as gk
import pandas as pd
import numpy as np
import yaml

from os.path import isfile
import warnings
from scipy.interpolate import RectBivariateSpline
import json
import numbers
from collections.abc import Iterable, Callable

# from reskit import solarpower

from reskit.solar import core as rk_solar_core
from reskit.workflow_manager import WorkflowManager

from reskit.solar import DATA

# Lazily import PVLib
import importlib

"""

Importing required packages.

"""


class LazyLoader:
    def __init__(self, lib_name):
        """
        LazyLoader is a utility class which postpones the "real" importing of the desired module until the time when it is actually needed
        """

        self.lib_name = lib_name
        self._mod = None

    def __getattr__(self, name):
        if self._mod is None:
            self._mod = importlib.import_module(self.lib_name)
        return getattr(self._mod, name)


pvlib = LazyLoader("pvlib")


class SolarWorkflowManager(WorkflowManager):
    def __init__(self, placements):
        """

        __init_(self, placements)

        Initialization of an instance of the generic SolarWorkflowManager class.

        Parameters
        ----------
        placements : pandas Dataframe
                     The locations that the simulation should be run for.
                     Columns must include "lon", "lat"

        Returns
        -------
        SolarWorkflorManager

        """

        # Do basic workflow construction
        super().__init__(placements)
        self._time_sel_ = None
        self._time_index_ = None
        self.module = None

    ####################################
    # PREPROCESS LOCATIONAL ATTRIBUTES #
    ####################################

    def estimate_missing_params(
        self,
        elev,
        ground_albedo,
        fixed_module_tilt_convention=None,
        fixed_module_azimuth_convention=None,
        singleaxis_tilt_convention=None,
        singleaxis_azimuth_convention=None,
        crossaxis_tilt_convention=None,
    ):
        # check placements columns for possible other/wrong column names
        def _check_existing_cols(substr):
            _allcols = [
                "modtilt",
                "modazimuth",
                "axazimuth",
                "axtilt",
                "caxtilt",
                "elev",
                "grdalbedo",
            ]
            _othercols = [
                col
                for col in self.placements
                if substr.lower() in col.lower() and col not in _allcols
            ]
            _possiblecols = [col for col in _allcols if substr.lower() in col.lower()]
            assert len(_possiblecols) > 0  # make sure substr makes sense
            if len(_othercols) > 0:
                warnings.warn(
                    f"The following placement column names contain '{substr}' but are none of the recognized arguments related to {substr} and will be ignored: '"+"', '".join(_othercols)+f"'. Missing columns will be added. Regognized column names for '{substr}' are: '"+"', '".join(_possiblecols)+"'"
                )

        for param in ["tilt", "azimuth", "albedo", "elev"]:
            _check_existing_cols(substr=param)

        # set elevation and albedo - always required
        self.assign_elevation(
            elev=elev, fallback_elev=840
        )  # mean landmass elevation as fallback
        self.assign_ground_albedo(
            ground_albedo=ground_albedo
        )

        # set required tilts and azimuths depending on tracking type
        if self.tracking in ["fixed"]:
            assert fixed_module_tilt_convention is not None, "fixed_module_tilt_convention must not be None when tracking='fixed'"
            assert fixed_module_azimuth_convention is not None, "fixed_azimuth_tilt_convention must not be None when tracking='fixed'"
            self.estimate_module_tilt_from_latitude(
                convention=fixed_module_tilt_convention
            )
            self.estimate_module_azimuth_from_latitude(
                convention=fixed_module_azimuth_convention
            )
        elif self.tracking in ["single-axis"]:
            assert singleaxis_tilt_convention is not None, "singleaxis_tilt_convention must not be None when tracking='single-axis'"
            assert singleaxis_azimuth_convention is not None, "fixed_azimuth_tilt_convention must not be None when tracking='single-axis'"
            assert crossaxis_tilt_convention is not None, "crossaxis_tilt_convention must not be None when tracking='single-axis'"
            self.estimate_tracker_axis_tilt_from_latitude(
                convention=singleaxis_tilt_convention
            )
            self.estimate_tracker_axis_azimuth_from_latitude(
                convention=singleaxis_azimuth_convention
            )
            self.estimate_cross_axis_tilt_from_latitude(
                convention=crossaxis_tilt_convention
            )
            
        return self


    def _assign_attribute(
        self,
        attr: str,
        attr_default: object,
        attr_col: str,
        func: Callable,
        verbose: bool = False,
        allow_nan: bool = False,
        **funcargs,
    ):
        """
        Auxiliary function that checks if an attribute exists in self.placements
        already and renames column to default attribute name if necessary, else
        it sets the attribute either based on an attribute-specific function or
        based on a scalar constant default value.

        attr : str
            The attribute name used in the wfm
        attr_default : object
            The default value to be set if no attibute column in placements
            dataframe and no function given. Must be None if attr_col exists in
            placements dataframe or an extraction func is given.
        attr_col : str
            Name of the input column in placements dataframe containing values.
        func : Callable
            Function that will be applied to extract location-specific values,
            must return an iterable of the same length as placements df.
        verbose : bool, optional
            Prints additional information if True, by default False.
        allow_nan : bool, optional
            Will fail if Nans remain after trying to fill the gaps. By default False.
        **funcargs
            Arguments will be passed on to func.

        Returns
        -------
        obj
            reference to the invoking SolarWorkflowManager object
        """
        # if a different attr col is given, the attr name itself cannot be a column name as well
        if attr_col != attr and attr in self.placements:
            raise AttributeError(
                f"'{attr}' cannot be an existing column in placements when {attr} column != '{attr}'"
            )

        # if the attr column exists already in placements, use it
        if attr_col in self.placements:
            if attr_default is not None:
                warnings.warn(
                    f"Function argument for {attr} is not None (here: {attr_default}) but will be ignored since '{attr_col}' column self.placements exists and will be used."
                )
            if verbose and attr_col == attr:
                print(
                    f"'{attr}' column in placements dataframe exists and will not be overwritten."
                )
            elif verbose:
                print(
                    f"'{attr_col}' column in placements dataframe exists and will be renamed to '{attr}'."
                )
            self.placements.rename(columns={attr_col: attr}, inplace=True)
            assert not self.placements[attr].isna().any(), f"{attr} data provided as self.placements column '{attr_col}' must not contain NaN values."

        # else either apply the given function or default to assign missing values
        elif func is not None:
            # use function
            if attr_default is not None:
                raise TypeError(
                    f"'{attr}' default must be None if data function shall be used to assign '{attr}'."
                )
            self.placements[attr] = func(**funcargs)
        else:
            # use default
            if ~isinstance(attr_default, str) and hasattr(attr_default, "__iter__") and len(attr_default) != len(self.placements):
                raise TypeError(
                    f"{attr} default must be scalar or an iterable of length of placements dataframe ({len(self.placements)}) if not None: {e}"
                )
            self.placements[attr] = attr_default

        if not allow_nan:
            # make sure none remains
            assert not any(
                self.placements[attr].isna()
            ), f"Remaining NaN values for '{attr}'"

        return self


    def assign_elevation(self, elev: str | int | Iterable, fallback_elev: int = 0):
        """
        Ensures or adds an elevation ('elev') column to the placements data frame.

        Parameters
        ----------
        elev : str, int, Iterable
            If a string is given it must be a path to a rasterfile including the elevations.
            If an iterable is given it has to include the elevations at each location and be
            of equal length to self.placements dataframe.
            If an integer is given, it will be applied to all locations equally
        fallback_elev : int, optional
            The fallback value that will be used in case that elev is a raster
            path and the extraction of the elevation from raster fails (applied
            only to no-data locations), by default 0

        Returns
        -------
        obj
            reference to the invoking SolarWorkflowManager object
        """
        assert isinstance(
            fallback_elev, int
        ), "'fallback_elev' must be an integer elevantion in [m]."

        def _elev_func():
            if elev is None:
                # we don't have given elevation info, neither as elev arg nor in placements dataframe column
                # set all values to fallback
                return np.array([fallback_elev] * len(self.locs))
            elif isinstance(elev, str):
                # assume we have a str formatted elevation raster path
                clipped_elev = self.ext.pad(0.5).rasterMosaic(elev)
                if clipped_elev is None:
                    _elevs = np.array([np.nan] * len(self.locs))
                else:
                    _elevs = gk.raster.interpolateValues(clipped_elev, self.locs)
                    if np.isnan(_elevs).any():
                        # if getting values fails, it could be because of interpolation method
                        # replace by 'near' interpolation
                        _elevs_near = gk.raster.interpolateValues(
                            clipped_elev, self.locs, mode="near"
                        )
                        _elevs[np.isnan(_elevs)] = _elevs_near[np.isnan(_elevs)]
                if np.isnan(_elevs).any():
                    # if we still have nans, replace nans by fallback value
                    _elevs[np.isnan(_elevs)] = (
                        np.ones(shape=_elevs.shape) * fallback_elev
                    )[np.isnan(_elevs)]
                return _elevs

        _default = elev if isinstance(elev, int) else None  # integers are default elevs
        _func = None if isinstance(elev, int) else _elev_func

        self._assign_attribute(
            attr="elev", attr_default=_default, attr_col="elev", func=_func, funcargs={}
        )

        assert all([isinstance(x, numbers.Number) for x in self.placements["elev"]])

        return self


    def assign_ground_albedo(self, ground_albedo:float|Iterable):
        """
        Assigns a ground albedo value to every placement in self.placements in a
        new column, unless column exists already, if so existing values will be 
        checked and used.

        Parameters
        -------
        ground_albedo : float, tuplr, list
            * float : value will be set to all placements
            * tuple/list : Must then contain landcover dataset information and 
              be formatted like (dataset_name:str, dataset_filepath:str).

        Returns
        -------
        obj
            reference to the invoking SolarWorkflowManager object
        """
        if not all([isinstance(x, float) for x in np.atleast_1d(ground_albedo)]) or (hasattr(ground_albedo, "__iter__") and len(ground_albedo==2) and all([isinstance(x, str) for x in ground_albedo])):
            raise TypeError(f"Unknown ground_albedo argument of type '{type(ground_albedo)}': {ground_albedo}. Must be float, iterable of floats or tuple/list of 2 strings.")
    
        # define an aux function to extract ground albedos from landcover name and path
        def _get_ground_albedo_from_landcover(landcover_name:str, landcover_path:str):
            """
            Get the mean snow-free broadband white sky albedos for the respective
            landcover per placement based on category type in the landcover dataset.
            
            Parameters
            ----------
            landcover_name : str
                Name of dataset, must be a key in dataset_mapper dict in the 
                ground_cover_albedos.yaml file in solar/data.
            landcover_path : str
                The filepath to the dataset described by landcover_name.

            -------
            list
                list with local ground albedo values of length of placements 
                attribute of invoking SolarWorkflowManager object
            """
            # load the albedo data from yaml file
            with open(DATA.get('ground_cover_albedos.yaml'), "r") as stream:
                ground_cover_albedos = yaml.safe_load(stream)
            
            # make sure landcover input is legit
            if landcover_name not in ground_cover_albedos["dataset_classtype_mapper"]:
                raise KeyError(f"Unknown landcover_name '{landcover_name}'. Select from: {', '.join(ground_cover_albedos['dataset_classtype_mapper'].keys())}")
            if not isfile(landcover_path):
                raise FileNotFoundError(f"landcover_path file does not exist: {landcover_path}")
            try:
                gk.raster.loadRaster(landcover_path)
            except Exception as e:
                raise TypeError(f"landcover_path must point to a raster type file: {e}")
            
            # get the landcover type categories for all placements
            if "geom" not in self.placements:
                self.placements["geom"] = self.placements.apply(lambda x : gk.geom.point(x.lon, x.lat, srs=4326), axis=1)
            assert all([g.GetGeometryName()=="POINT" for g in self.placements.geom]) # make sure
            LCclasses = gk.raster.interpolateValues(
                source = landcover_path,
                points = self.placements.geom.to_list(), 
                pointSRS="latlon", 
                mode="near"
                )
            # map the LC class number to albedo via ground types 
            classtype = ground_cover_albedos["dataset_classtype_mapper"][landcover_name]
            LCclass_groundtype_mapper = ground_cover_albedos["LCclass_groundtype_mapper"][classtype]
            groundtype_albedo_mapper = ground_cover_albedos["groundtype_albedo_mapper"]
            ground_albedos = [groundtype_albedo_mapper[LCclass_groundtype_mapper[c]] for c in LCclasses]
            assert not any([np.isnan(a) for a in ground_albedos]), "NaN values found in extracted ground_albedos."
            return ground_albedos

        # prepare the assign attribute function arguments
        _default = ground_albedo if all([isinstance(x, float) for x in np.atleast_1d(ground_albedo)]) else None # integers are default elevs
        _func = None if all([isinstance(x, float) for x in np.atleast_1d(ground_albedo)]) else _get_ground_albedo_from_landcover
        _funcargs = {} if all([isinstance(x, float) for x in np.atleast_1d(ground_albedo)]) else {
                "landcover_name" : ground_albedo[0],
                "landcover_path" : ground_albedo[1]
            }
        
        # apply generic function with ground albedo args
        self._assign_attribute(
            attr="grdalbedo", 
            attr_default=_default, 
            attr_col="grdalbedo", 
            func=_func, 
            **_funcargs)
        
        # final sanity check
        if not self.placements.grdalbedo.apply(lambda x : 0<x<1).all():
            raise ValueError("ground albedo must be >0 and <1 for all locations.")
        
        return self


    def estimate_module_azimuth_from_latitude(self, convention: str):
        """
        Estimates the fixed module azimuth of the placements of the instance and
        writes them into the placements dataframe as 'modazimuth' column. Will use
        existing values in 'modazimuth' column of self.placements dataframe.

        Parameters
        ----------
        convention : str
            The calculation method used to suggest module azimuths For details
            see rk.solar.core.system_design.location_to_module_azimuth():
            * "NorthSouth"
            * Path to raster file.

        Returns
        -------
        obj
            reference to the invoking SolarWorkflowManager object
        """
        if not self.tracking == "fixed":
            warnings.warn(
                "estimate_module_azimuth_from_latitude() is called but tracking is not 'fixed'"
            )

        self._assign_attribute(
            attr="modazimuth",
            attr_default=None,  # no "standard azimuth"
            attr_col="modazimuth",
            func=rk_solar_core.system_design.location_to_module_azimuth,
            **{"locs": self.locs, "convention": convention},
        )

        return self


    def estimate_tracker_axis_azimuth_from_latitude(self, convention:str):
        """
        Estimates the (primary) tracker axis azimuths of the placements of the 
        instance and writes them into the placements dataframe as 'axazimuth' 
        column. Will use existing values in 'axazimuth' column of 
        self.placements dataframe.

        Parameters
        ----------
        convention : str
            The calculation method used to suggest axis azimuths For details 
            see rk.solar.core.system_design.location_to_tracker_axis_azimuth():
            * "North"
            * Path to raster file.

        Returns
        -------
        obj
            reference to the invoking SolarWorkflowManager object
        """
        if not self.tracking == "single-axis":
            warnings.warn("estimate_tracker_axis_azimuth_from_latitude() is called but tracking is not 'singe-axis'")
        
        self._assign_attribute(
            attr="axazimuth", 
            attr_default=None, # no "standard azimuth"
            attr_col="axazimuth", 
            func=rk_solar_core.system_design.location_to_tracker_axis_azimuth, 
            **{
                "locs" : self.locs,
                "convention" : convention
            })

        return self


    def estimate_module_tilt_from_latitude(self, convention: str):
        """
        Estimates the module surface tilt of the solar panels based on the
        latitude of the placements of the instance and writes them into the
        placements dataframe as 'modtilt' column. Will use existing values in
        'modtilt' column of self.placements dataframe.

        Parameters
        ----------
        convention : str
            The calculation method used to suggest module tilts. For details see
            rk.solar.core.system_design.location_to_module_tilt():
            * "Ryberg2020"
            * string consumable by 'eval'. This string can use the variable
              'latitude', for example "latitude*0.76".
            * path to a rasterfile.

        Returns
        -------
        obj
            reference to the invoking SolarWorkflowManager object
        """
        if not self.tracking == "fixed":
            warnings.warn(
                "estimate_module_tilt_from_latitude() is called but tracking is not 'fixed'"
            )

        self._assign_attribute(
            attr="modtilt",
            attr_default=None,  # no "standard module tilt"
            attr_col="modtilt",
            func=rk_solar_core.system_design.location_to_module_tilt,
            **{"locs": self.locs, "convention": convention},
        )

        return self


    def estimate_tracker_axis_tilt_from_latitude(self, convention: str):
        """
        Estimates the (primary) tracker axis tilt for a single-axis tracking 
        system based on the latitude of the placements of the instance and 
        writes them into the placements dataframe as 'axtilt' column. Will use 
        existing values in 'axtilt' column of self.placements dataframe.

        Parameters
        ----------
        convention : str
            The calculation method used to suggest axis tilts. For details see 
            rk.solar.core.system_design.location_to_tracker_axis_tilt():
            * "flat" assigns 0° axis tilt to all locations.
            * A path to a rasterfile with angles facing in axis azimuth direction 
              to extract axis tilts per location.

        Returns
        -------
        obj
            reference to the invoking SolarWorkflowManager object
        """
        if not self.tracking == "single-axis":
            warnings.warn(
                "estimate_tracker_axis_tilt_from_latitude() is called but tracking is not 'single-axis'"
            )
        
        self._assign_attribute(
            attr="axtilt", 
            attr_default=None, # no "standard axis tilt"
            attr_col="axtilt", 
            func=rk_solar_core.system_design.location_to_tracker_axis_tilt,
            **{
                "locs" : self.locs,
                "convention" : convention
            })

        return self


    def estimate_cross_axis_tilt_from_latitude(self, convention: str):
        """
        Estimates the cross axis azimuths at the placements of the 
        instance and writes them into the placements dataframe as 'caxazimuth' 
        column. Will use existing values in 'caxazimuth' column of 
        self.placements dataframe.

        Parameters
        ----------
        convention : str
            The calculation method used to suggest cross axis tilts. For details 
            see rk.solar.core.system_design.location_to_cross_axis_tilt():
            * "flat" assigns 0° cross-axis tilt to all locations.
            * A path to a rasterfile with angles facing in axis azimuth direction 
              to extract axis tilts per location.

        Returns
        -------
        obj
            reference to the invoking SolarWorkflowManager object
        """
        if not self.tracking == "single-axis":
            warnings.warn(
                "estimate_cross_axis_tilt_from_latitude() is called but tracking is not 'singe-axis'"
            )
        
        self._assign_attribute(
            attr="caxtilt", 
            attr_default=None, # no "standard azimuth"
            attr_col="caxtilt", 
            func=rk_solar_core.system_design.location_to_cross_axis_tilt, 
            **{
                "locs" : self.locs,
                "convention" : convention
            })

        return self

    ########################
    # GEOMETRIC OPERATIONS #
    ########################

    def determine_solar_position(
        self, lon_rounding=1, lat_rounding=1, elev_rounding=-2
    ):
        """
        Calculates azimuth and apparent zenith for each location using the pvlib
        fuction pvlib.solarposition.spa_python() [1]. Adds azimuth and apparent 
        zenit to the sim_data dictionary.

        Parameters
        ----------
        lon_rounding: int, optional
            Decimal places that longitude should be rounded to. Default is 1.
        lat_rounding: int, optional
            Decimal places that latitude should be rounded to. Default is 1.
        elev_rounding: int, optional
            Decimal places that elevation should be rounded to. Default is -2.

        Returns
        -------
        obj
            reference to the invoking SolarWorkflowManager object

        Notes
        -----
        Required columns in the placements dataframe to use this functions are 
        'lon', 'lat' and 'elev'. Required data in the sim_data dictionary are 
        'surface_pressure' and 'surface_air_temperature'.

        References
        ----------
        [1] https://pvlib-python.readthedocs.io/en/stable/generated/pvlib.solarposition.spa_python.html

        [2] I. Reda and A. Andreas, Solar position algorithm for solar
            radiation applications. Solar Energy, vol. 76, no. 5, pp. 577-589, 2004.

        [3] I. Reda and A. Andreas, Corrigendum to Solar position algorithm for
            solar radiation applications. Solar Energy, vol. 81, no. 6, p. 838,
            2007.

        [4] USNO delta T:
            http://www.usno.navy.mil/USNO/earth-orientation/eo-products/long-term

        """
        # check placements dataframe and sim_data attributes
        assert "lon" in self.placements.columns, \
            "'lon' is a mandatory column in placements dataframe"
        assert "lat" in self.placements.columns, \
            "'lat' is a mandatory column in placements dataframe"
        assert "elev" in self.placements.columns, \
            "'elev' is a mandatory column in placements dataframe"
        assert "surface_pressure" in self.sim_data,\
            "'surface_pressure' must be read in first via wfm.read()"
        assert "surface_air_temperature" in self.sim_data,\
            "'surface_air_temperature' must be read in first via wfm.read()"

        rounded_locs = pd.DataFrame()
        rounded_locs["lon"] = np.round(self.placements["lon"].values, lon_rounding)
        rounded_locs["lat"] = np.round(self.placements["lat"].values, lat_rounding)
        rounded_locs["elev"] = np.round(self.placements["elev"].values, elev_rounding)

        solar_position_library = dict()

        # pd.DataFrame(np.nan, index=self.time_index, columns=self.locs)
        self.sim_data["solar_azimuth"] = np.full_like(
            self.sim_data["surface_pressure"], np.nan
        )
        # pd.DataFrame(np.nan, index=self.time_index, columns=self.locs)
        self.sim_data["apparent_solar_zenith"] = np.full_like(
            self.sim_data["surface_pressure"], np.nan
        )
        # self.sim_data['apparent_solar_elevation'] = np.full_like(self.sim_data['surface_pressure'], np.nan)  # pd.DataFrame(np.nan, index=self.time_index, columns=self.locs)

        for loc, row in enumerate(rounded_locs.itertuples()):
            key = (row.lon, row.lat, row.elev)
            if key in solar_position_library:
                _solpos_ = solar_position_library[key]
            else:
                # make sure that no input is nan to avoid very hard-to-understand errors later on
                _req = [
                    self.time_index,
                    row.lat,
                    row.lon,
                    row.elev,
                    self.sim_data["surface_pressure"][:, loc],
                    self.sim_data["surface_air_temperature"][:, loc],
                ]
                assert not any(
                    [
                        np.isnan(x).any() if hasattr(x, "__iter__") else np.isnan(x)
                        for x in _req
                    ]
                ), "Arguments for pvlib.solarposition.spa_python() may not be NaN."
                _solpos_ = pvlib.solarposition.spa_python(
                    self.time_index,
                    latitude=row.lat,
                    longitude=row.lon,
                    altitude=row.elev,
                    pressure=self.sim_data["surface_pressure"][:, loc],
                    temperature=self.sim_data["surface_air_temperature"][:, loc],
                )
                solar_position_library[key] = _solpos_

            self.sim_data["solar_azimuth"][:, loc] = _solpos_["azimuth"]
            self.sim_data["apparent_solar_zenith"][:, loc] = _solpos_["apparent_zenith"]
            # self.sim_data['apparent_solar_elevation'][:, loc] = _solpos_["apparent_elevation"]

        assert not np.isnan(self.sim_data["solar_azimuth"]).any()
        assert not np.isnan(self.sim_data["apparent_solar_zenith"]).any()
        # assert not np.isnan(self.sim_data['apparent_solar_elevation']).any()

        return self


    def filter_positive_solar_elevation(self):
        """
        Filters positive solar elevations so that future operations are only 
        executed for time steps when the sun is above (or at least near-to) 
        the horizon

        Parameters
        ----------
        None

        Returns
        -------
        obj
            reference to the invoking SolarWorkflowManager object

        Notes
        -----
        Required data in the sim_data dictionary are 'apparent_solar_zenith'.
        """

        if self._time_sel_ is not None:
            warnings.warn("Filtering already applied, skipping...")
            return self
        assert "apparent_solar_zenith" in self.sim_data,\
            "'apparent_solar_zenith' is a mandatory self.sim_data argument. Calculate e.g. via self.determine_solar_position()"

        self._time_sel_ = (self.sim_data["apparent_solar_zenith"] < 95).any(axis=1)

        for key in self.sim_data.keys():
            self.sim_data[key] = self.sim_data[key][self._time_sel_, :]

        self._time_index_ = self.time_index[self._time_sel_]
        self._set_sim_shape()

        return self


    def determine_extra_terrestrial_irradiance(self, **kwargs):
        """
        Determines extra terrestrial irradiance using the 
        pvlib.irradiance.get_extra_radiation() function [1].

        Parameters
        ----------
        **kwargs
            Passed on to pvlib.irradiance.get_extra_radiation().

        Returns
        -------
        obj
            reference to the invoking SolarWorkflowManager object.

        References
        ----------
        [1] https://pvlib-python.readthedocs.io/en/stable/generated/pvlib.irradiance.get_extra_radiation.html

        [2]	M. Reno, C. Hansen, and J. Stein, “Global Horizontal Irradiance Clear Sky Models: Implementation and Analysis”, Sandia National Laboratories, SAND2012-2389, 2012.

        [3]	<http://solardat.uoregon.edu/SolarRadiationBasics.html>, Eqs. SR1 and SR2

        [4]	Partridge, G. W. and Platt, C. M. R. 1976. Radiative Processes in Meteorology and Climatology.

        [5]	Duffie, J. A. and Beckman, W. A. 1991. Solar Engineering of Thermal Processes, 2nd edn. J. Wiley and Sons, New York.

        [6]	ASCE, 2005. The ASCE Standardized Reference Evapotranspiration Equation, Environmental and Water Resources Institute of the American Civil Engineers, Ed. R. G. Allen et al.

        """

        dni_extra = pvlib.irradiance.get_extra_radiation(
            self._time_index_, **kwargs
        ).values

        shape = len(self._time_index_), self.locs.count
        self.sim_data["extra_terrestrial_irradiance"] = np.broadcast_to(
            dni_extra.reshape((shape[0], 1)), shape
        )

        return self


    def determine_air_mass(self, model:str="kastenyoung1989"):
        """
        Determines air mass using the pvlib function 
        pvlib.atmosphere.get_relative_airmass() [1].

        Parameters
        ----------
        model: str, optional
            The model used to compute airmass.
            * 'simple' - secant(apparent zenith angle) - Note that this gives -inf at zenith=90 [2]
            * 'kasten1966' - See reference [2] - requires apparent sun zenith [2]
            * 'youngirvine1967' - See reference [3] - requires true sun zenith [2]
            * 'kastenyoung1989' - See reference [4] - requires apparent sun zenith [2]
            * 'gueymard1993' - See reference [5] - requires apparent sun zenith [2]
            * 'young1994' - See reference [6] - requries true sun zenith [2]
            * 'pickering2002' - See reference [7] - requires apparent sun zenith [2]
            By default 'kastenyoung1989' [1]

        Notes
        -----
        Required data in the sim_data dictionary are 'apparent_solar_zenith'.

        References
        ----------
        [1] https://pvlib-python.readthedocs.io/en/stable/generated/pvlib.atmosphere.get_relative_airmass.html

        [2]	Fritz Kasten. “A New Table and Approximation Formula for the Relative Optical Air Mass”. Technical Report 136, Hanover, N.H.: U.S. Army Material Command, CRREL.

        [3]	A. T. Young and W. M. Irvine, “Multicolor Photoelectric Photometry of the Brighter Planets,” The Astronomical Journal, vol. 72, pp. 945-950, 1967.

        [4]	Fritz Kasten and Andrew Young. “Revised optical air mass tables and approximation formula”. Applied Optics 28:4735-4738

        [5]	C. Gueymard, “Critical analysis and performance assessment of clear sky solar irradiance models using theoretical and measured data,” Solar Energy, vol. 51, pp. 121-138, 1993.

        [6]	A. T. Young, “AIR-MASS AND REFRACTION,” Applied Optics, vol. 33, pp. 1108-1110, Feb 1994.

        [7]	Keith A. Pickering. “The Ancient Star Catalog”. DIO 12:1, 20,

        [8]	Matthew J. Reno, Clifford W. Hansen and Joshua S. Stein, “Global Horizontal Irradiance Clear Sky Models: Implementation and Analysis” Sandia Report, (2012).

        """

        assert "apparent_solar_zenith" in self.sim_data,\
            "'apparent_solar_zenith' is a mandatory self.sim_data argument. Calculate e.g. via self.determine_solar_position()"

        # 29 becasue that what the function seems to max out at as zenith approaches 90
        self.sim_data["air_mass"] = np.full_like(
            self.sim_data["apparent_solar_zenith"], 29
        )

        s = self.sim_data["apparent_solar_zenith"] < 90
        self.sim_data["air_mass"][s] = pvlib.atmosphere.get_relative_airmass(
            self.sim_data["apparent_solar_zenith"][s], model=model
        )


    def apply_DIRINT_model(self, use_pressure:bool=True, use_dew_temperature:bool=True):
        """
        Determines direct normal irradiance (DNI) using the 
        pvlib.irradiance.dirint() function [1].

        Parameters
        ----------
        use_pressure: boolian, optional
            Default: True

        use_dew_temperature: boolian, optional
            Default: True

        Returns
        -------
        obj
            a reference to the invoking SolarWorkflowManager object.

        Notes
        -----
        Required data in the sim_data dictionary are 'global_horizontal_irradiance', 'surface_pressure',
        'surface_dew_temperature', 'apparent_solar_zenith', 'air_mass' and 'extra_terrestrial_irradiance'.

        References
        ----------
        [1] https://pvlib-python.readthedocs.io/en/stable/generated/pvlib.irradiance.dirint.html

        [2]	Perez, R., P. Ineichen, E. Maxwell, R. Seals and A. Zelenka, (1992). “Dynamic Global-to-Direct Irradiance Conversion Models”. ASHRAE Transactions-Research Series, pp. 354-369

        [3]	Maxwell, E. L., “A Quasi-Physical Model for Converting Hourly Global Horizontal to Direct Normal Insolation”, Technical Report No. SERI/TR-215-3087, Golden, CO: Solar Energy Research Institute, 1987.

        """

        assert "global_horizontal_irradiance" in self.sim_data
        assert "surface_pressure" in self.sim_data
        assert "surface_dew_temperature" in self.sim_data
        assert "apparent_solar_zenith" in self.sim_data
        assert "air_mass" in self.sim_data
        assert "extra_terrestrial_irradiance" in self.sim_data

        g = self.sim_data["global_horizontal_irradiance"].flatten()
        z = self.sim_data["apparent_solar_zenith"].flatten()
        p = self.sim_data["surface_pressure"].flatten() if use_pressure else None
        td = (
            self.sim_data["surface_dew_temperature"].flatten()
            if use_dew_temperature
            else None
        )
        times = pd.DatetimeIndex(
            np.column_stack(
                [self._time_index_ for x in range(self._sim_shape_[1])]
            ).flatten()
        )

        self.sim_data["direct_normal_irradiance"] = (
            pvlib.irradiance.dirint(
                ghi=g, solar_zenith=z, times=times, pressure=p, temp_dew=td
            )
            .fillna(0)
            .values.reshape(self._sim_shape_)
        )

        return self


    def diffuse_horizontal_irradiance_from_trigonometry(self):
        """
        Calculates the diffuse horizontal irradiance from global horizontal 
        irradiance, direct normal irradiance and apparent zenith.

        Parameters
        ----------
        None

        Returns
        -------
        obj
            reference to the invoking SolarWorkflowManager object.

        Notes
        -----
        Required data in the sim_data dictionary are 'global_horizontal_irradiance', 'direct_normal_irradiance' and
        'apparent_solar_zenith'.
        """
        # check required sim_data attributes
        assert "global_horizontal_irradiance" in self.sim_data
        assert "direct_normal_irradiance" in self.sim_data
        assert "apparent_solar_zenith" in self.sim_data

        ghi = self.sim_data["global_horizontal_irradiance"]
        dni = self.sim_data["direct_normal_irradiance"]
        sol_zenith = np.radians(90 - self.sim_data["apparent_solar_zenith"])

        self.sim_data["diffuse_horizontal_irradiance"] = ghi - dni * np.sin(sol_zenith)
        self.sim_data["diffuse_horizontal_irradiance"][
            self.sim_data["diffuse_horizontal_irradiance"] < 0
        ] = 0

        return self


    def direct_normal_irradiance_from_trigonometry(self):
        """

        direct_normal_irradiance_from_trigonometry(self):


        Parameters
        ----------
        None

        Returns
        -------
        obj
            A reference to the invoking SolarWorkflowManager object.

        Notes
        -----
        Required columns in the placements dataframe to use this functions are 'lon', 'lat' and 'elev'.
        Required data in the sim_data dictionary are 'direct_horizontal_irradiance' and 'apparent_solar_zenith'.

        Calculates the direct normal irradiance from the following equation:
            .. math:: dir_nor_irr = dir_hor_irr / cos( solar_zenith )

            Where:
            dir_nor_irr  -> The direct irradiance on the normal plane
            dir_hor_irr  -> The direct irradiance on the horizontal plane
            solar_zenith -> The solar zenith angle in radians

        """

        # TODO: This can also cover the case when we know GHI & DiffHI
        assert "direct_horizontal_irradiance" in self.sim_data
        assert "apparent_solar_zenith" in self.sim_data

        dni_flat = self.sim_data["direct_horizontal_irradiance"]
        zen = np.radians(self.sim_data["apparent_solar_zenith"])

        self.sim_data["direct_normal_irradiance"] = dni_flat / np.maximum(
            np.cos(zen), 0.2
        )

        # catch outliners from zero devision
        index_out = (dni_flat < 25) & (np.cos(zen) < 0.05)
        self.sim_data["direct_normal_irradiance"][index_out] = 0

        index_out = (dni_flat < 25) & (np.cos(zen) < 0.05)
        self.sim_data["direct_normal_irradiance"][index_out] = 0

        sel = ~np.isfinite(self.sim_data["direct_normal_irradiance"])
        sel = np.logical_or(sel, self.sim_data["direct_normal_irradiance"] < 0)
        sel = np.logical_or(sel, self.sim_data["direct_normal_irradiance"] > 1600)

        self.sim_data["direct_normal_irradiance"][sel] = 0

        return self


    def permit_single_axis_tracking(self, max_angle=90, backtrack=True, gcr=2.0 / 7.0):
        """
        Permits single axis tracking in the simulation using the 
        pvlib.tracking.singleaxis() function [1].

        Parameters
        ----------
        max_angle: float, optional
            A value denoting the maximum rotation angle, in decimal degrees, of the one-axis tracker from its horizontal position
            (horizontal if axis_tilt = 0). A max_angle of 90 degrees allows the tracker to rotate to a vertical position to point the
            panel towards a horizon. max_angle of 180 degrees allows for full rotation [1]. By default 90.

        backtrack: bool, optional
            Controls whether the tracker has the capability to “backtrack” to avoid row-to-row shading.
            False denotes no backtrack capability. True denotes backtrack capability [1]. By default True.

        gcr: float, optional
            A value denoting the ground coverage ratio of a tracker system which utilizes backtracking; i.e. the ratio between the
            PV array surface area to total ground area. A tracker system with modules 2 meters wide, centered on the tracking axis,
            with 6 meters between the tracking axes has a gcr of 2/6=0.333. If gcr is not provided, a gcr of 2/7 is default. gcr 
            must be <=1 [1]. By default 2.0/7.0

        Returns
        -------
        obj
            a reference to the invoking SolarWorkflowManager object.

        Notes
        -----
        Required columns in the placements dataframe to use this functions are 'lon', 'lat', 'elev', 'axtilt' and 'axazimuth'.
        Required data in the sim_data dictionary are 'apparent_solar_zenith' and 'solar_azimuth'.

        References
        ----------
        [1] https://wholmgren-pvlib-python-new.readthedocs.io/en/doc-reorg2/generated/tracking/pvlib.tracking.singleaxis.html

        [2]	Lorenzo, E et al., 2011, “Tracking and back-tracking”, Prog. in Photovoltaics: Research and Applications, v. 19, pp. 747-753.

        """

        """See pvlib.tracking.singleaxis for parameter info"""
        assert self.tracking == "single-axis", \
            f"tracking flag must be 'single-axis' for permit_single_axis_tracking() but is instead: {self.tracking}"
        
        assert "apparent_solar_zenith" in self.sim_data
        assert "solar_azimuth" in self.sim_data
        assert "axtilt" in self.placements.columns
        assert "axazimuth" in self.placements.columns

        self.register_workflow_parameter("tracking_mode", "single_axis")
        self.register_workflow_parameter("tracking_max_angle", max_angle)
        self.register_workflow_parameter("tracking_backtrack", backtrack)
        self.register_workflow_parameter("tracking_gcr", gcr)

        system_modtilt = np.empty(self._sim_shape_)
        system_modazimuth = np.empty(self._sim_shape_)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for i in range(self.locs.count):
                placement = self.placements.iloc[i]

                tmp = pvlib.tracking.singleaxis(
                    apparent_zenith=pd.Series(
                        self.sim_data["apparent_solar_zenith"][:, i],
                        index=self._time_index_,
                    ),
                    apparent_azimuth=pd.Series(
                        self.sim_data["solar_azimuth"][:, i], index=self._time_index_
                    ),
                    axis_tilt=placement.axtilt,
                    axis_azimuth=placement.axazimuth,
                    max_angle=max_angle,
                    backtrack=backtrack,
                    gcr=gcr,
                )

                system_modtilt[:, i] = tmp["surface_tilt"].values
                system_modazimuth[:, i] = tmp["surface_azimuth"].values

                # fix nan values. Why are they there???
                s = np.isnan(system_modtilt[:, i])
                system_modtilt[s, i] = placement.axtilt

                s = np.isnan(system_modazimuth[:, i])
                system_modazimuth[s, i] = placement.axazimuth

        self.sim_data["system_modtilt"] = system_modtilt
        self.sim_data["system_modazimuth"] = system_modazimuth

        return self

    def determine_angle_of_incidence(self):
        """

        determine_angle_of_incidence(self)

        Determines the angle of incidence [TODO: credit the PVLib function as you've done in previous examples].

        Parameters
        ----------
        None

        Returns
        -------
        Returns a reference to the invoking SolarWorkflowManager object.

        Notes
        -----
        Required data in the sim_data dictionary are 'apparent_solar_zenith' and 'solar_azimuth'.

        """

        """tracking can be: 'fixed' or 'singleaxis'"""
        assert "apparent_solar_zenith" in self.sim_data
        assert "solar_azimuth" in self.sim_data

        # get either the time-variable tracking tilt/azimuths or fixed values for fixed tilt
        modazimuths = self.sim_data.get("system_modazimuth", self.placements["modazimuth"].values)
        modtilts = self.sim_data.get("system_modtilt", self.placements["modtilt"].values)

        self.sim_data["angle_of_incidence"] = np.nan_to_num(
            pvlib.irradiance.aoi(
                modtilts,
                modazimuths,
                self.sim_data["apparent_solar_zenith"],
                self.sim_data["solar_azimuth"],
            ),
            0,
        )

        return self

    def estimate_plane_of_array_irradiances(
        self, transposition_model="perez", **kwargs
    ):
        """
        Estimates the plane of array irradiance using the pvlib.irradiance.get_total_irradiance() function [1].


        Parameters
        ----------
        transportion_model: str, optional
                            default "perez"

        **kwargs

        Returns
        -------
        Returns a reference to the invoking SolarWorkflowManager object.

        Notes
        -----
        Required data in the sim_data dictionary are 'apparent_solar_zenith', 'solar_azimuth', 'direct_normal_irradiance',
        'global_horizontal_irradiance', 'diffuse_horizontal_irradiance', 'extra_terrestrial_irradiance' and 'air_mass'.

        References
        ----------
        [1] https://pvlib-python.readthedocs.io/en/stable/generated/pvlib.irradiance.get_total_irradiance.html

        """

        assert "apparent_solar_zenith" in self.sim_data
        assert "solar_azimuth" in self.sim_data
        assert "direct_normal_irradiance" in self.sim_data
        assert "global_horizontal_irradiance" in self.sim_data
        assert "diffuse_horizontal_irradiance" in self.sim_data
        assert "extra_terrestrial_irradiance" in self.sim_data
        assert "air_mass" in self.sim_data

        def _set_total_irradiance_per_side(front=True):
            """Calculates and sets to self.sim_data the POA global and its components for front or backside"""
            
            # get system ground albedos and tilts and azimuths for the module surfaces
            _grdalbedos = self.sim_data.get("system_grdalbedo", self.placements["grdalbedo"].values)
            _modtilts = self.sim_data.get("system_modtilt", self.placements["modtilt"].values)
            _modazimuths = self.sim_data.get("system_modazimuth", self.placements["modazimuth"].values)

            _poa = pvlib.irradiance.get_total_irradiance(
                surface_tilt=_modtilts if front else 180-_modtilts,
                surface_azimuth=_modazimuths,
                solar_zenith=self.sim_data["apparent_solar_zenith"],
                solar_azimuth=self.sim_data["solar_azimuth"],
                dni=self.sim_data["direct_normal_irradiance"],
                ghi=self.sim_data["global_horizontal_irradiance"],
                dhi=self.sim_data["diffuse_horizontal_irradiance"],
                dni_extra=self.sim_data["extra_terrestrial_irradiance"],
                airmass=self.sim_data["air_mass"],
                albedo=_grdalbedos,
                model=transposition_model,
                **kwargs,
            )
            # set results as sim_data attributes
            # at the same time sett unrealistically high POA values due to numerical sin effects at low sun angles to zero
            sel_bad_poa = _poa["poa_global"] >= 1600
            for key, tmp in _poa.items():
                # This should set: 'poa_global', 'poa_direct', 'poa_diffuse', 'poa_sky_diffuse', and 'poa_ground_diffuse' or the respective poa_backside value
                tmp[np.isnan(tmp)] = 0
                _key = key+"_raw" if front else key.replace("poa_", "poa_backside_")+"_raw"
                self.sim_data[_key] = np.where(
                        sel_bad_poa, 0, tmp
                    )
            return
        
        # first do frontside, always
        _set_total_irradiance_per_side(front=True)
        # then do backside only if needed
        if self.bifacial:
            _set_total_irradiance_per_side(front=False)

        return self


    def estimate_absorbed_plane_of_array_irradiances(self, **kwargs):
        """
        Estimates the incoming and absorbed plane of array irradiance using the 
        pvlib.bifacial.pvfactors.pvfactors_timeseries function [1] both for the 
        front and backside of the module.

        Parameters
        ----------
        **kwargs

        Returns
        -------
        obj
            a reference to the invoking SolarWorkflowManager object.

        Notes
        -----
        Required data sim_data attributes are 'diffuse_horizontal_irradiance'
        'apparent_solar_zenith', 'solar_azimuth' and 'direct_normal_irradiance'.

        Will save backside irradiances as well only when self.bifacial
        flag is True in the invoking class instance.

        Unless otherwise specified in kwargs, the following argument values will 
        be extracted from module data: 'gcr', 'pvrow_height', 'pv_row_width.

        References
        ----------
        [1] https://pvlib-python.readthedocs.io/en/stable/generated/pvlib.irradiance.get_total_irradiance.html
        [1] Winkler (2025), unpublished
        """
        # check required sim_data attributes first
        assert "apparent_solar_zenith" in self.sim_data
        assert "solar_azimuth" in self.sim_data
        assert "direct_normal_irradiance" in self.sim_data
        assert "diffuse_horizontal_irradiance" in self.sim_data

        # iteration is required through the locations since pvfactors_timeseries is not vectorized.
        
        # initialize result containers for the iteration over locs
        poa_frontside = np.empty(self._sim_shape_)
        poa_backside = np.empty(self._sim_shape_)
        poa_frontside_absorbed = np.empty(self._sim_shape_)
        poa_backside_absorbed = np.empty(self._sim_shape_)

        # iterate over all locs
        for iloc in range(self._sim_shape_[1]):

            # helper function to extract and preprocess shape for variables either from sim data or placements or defaults
            def _extract_var(var, sim_var=None, fallback=None):
                """First tries to get location- and time-variable sim_data, then location-variable placements column, else default."""
                if self.sim_data.get(sim_var) is not None:
                    # get only the timeseries for the representative location
                    return self.sim_data.get(sim_var)[:, iloc]
                elif var is not None and var in self.placements:
                    # value is not time-variable but in placements df, duplicate for T timesteps
                    return np.full(self._sim_shape_[0], self.placements.iloc[iloc][var])
                elif fallback is not None:
                    # set to variable fallback value
                    return np.full(self._sim_shape_[0], fallback)
                else:
                    # fall back to defaults
                    defaults = {"gcr" : 0.45, "axis_azimuth" : 0, "pvrow_height" : 1.12, "n_pvrows" : 1, "index_observed_pvrow" : 0} #TODO change n_pvrows and pvrow_width and index_observed_pvrow
                    if var not in defaults:
                        raise KeyError(f"Variable '{var}' is neither a sim_data system variable, nor a column in placements dataframe nor has a default value.")
                    return defaults[var]
            
            # define the base input args for this location
            # define a fallback for the axis azimuth in case of tracked systems where the value does not exist
            # the tracker axis is not used here but value is expected, orientation is always rectangular to module azimuth
            _axazimuth_fallback = _extract_var("modazimuth", "system_modazimuth") + 90 if self.tracking == "fixed" else None
            pvfts_args = {
                "solar_azimuth" : self.sim_data["solar_azimuth"][:, iloc], 
                "solar_zenith" : self.sim_data["apparent_solar_zenith"][:, iloc], 
                "surface_azimuth" : _extract_var("modazimuth", "system_modazimuth"), 
                "surface_tilt" : _extract_var("modtilt", "system_modtilt"), 
                "axis_azimuth" : _extract_var("axazimuth", "system_axazimuth", _axazimuth_fallback),
                "timestamps" : np.arange(self._sim_shape_[0]),
                "dhi" : self.sim_data["diffuse_horizontal_irradiance"][:, iloc], 
                "dni" : self.sim_data["direct_normal_irradiance"][:, iloc], 
                "gcr" : _extract_var("gcr"),
                "pvrow_height" : _extract_var("pvrow_height"),
                "albedo" : _extract_var("grdalbedo", "system_grdalbedo"),
                "n_pvrows" : _extract_var("n_pvrows"),
                "index_observed_pvrow" : _extract_var("index_observed_pvrow"),
            }

            # handle kwargs for this location
            kwargs_iloc = {}
            for k, v in kwargs.items():
                if isinstance(v, np.ndarray) and not v.shape==(self._sim_shape_[0],):
                    # we have a multi-dimensional numpy array, make sure it is of shape (t,n)
                    if not v.shape==self._sim_shape_:
                        raise ValueError(f"kwarg '{k}' was passed as {v.shape} numpy.ndarray, must either be 1d or of shape (Ntimesteps, Nlocations), here: {self._sim_shape_}.")
                    # set only the respective locational slice
                    kwargs_iloc[k] = v[:, iloc]
                elif not hasattr(v, "__iter__"):
                    # scalar value, set the same for all locations
                    kwargs_iloc[k] = v
                else:
                    raise TypeError("kwargs for pvlib.bifacial.pvfactors.pvfactors_timeseries() must be scalar or numpy.ndarray type.")
            pvfts_args.update(kwargs_iloc) # add/overwrite with location-specific kwargs if applicable
            # special case: PV row width if not given should match gcr, assume lateral module placement -> module row is 2m wide
            if "pvrow_width" not in pvfts_args:
                # calculate it based on the GCR and axis height
                _panel_width = 2 # m
                assert 0 < pvfts_args["gcr"] <= 1, "gcr must be a float value between 0 and 1 for all placements" # make sure
                pvfts_args["pvrow_width"] = _panel_width/pvfts_args["gcr"]
            
            # simulate and append locational output to total results
            _poa_frontside, _poa_backside, _poa_frontside_absorbed, _poa_backside_absorbed = pvlib.bifacial.pvfactors.pvfactors_timeseries(**pvfts_args)

            poa_frontside[:, iloc] = _poa_frontside.values
            poa_backside[:, iloc] = _poa_backside.values
            poa_frontside_absorbed[:, iloc] = _poa_frontside_absorbed.values
            poa_backside_absorbed[:, iloc] = _poa_backside_absorbed.values

        assert not (poa_frontside >= 1600).any() # sanity check

        # finally set the results as sim_data attributes
        self.sim_data['poa_global_raw'] = poa_frontside
        self.sim_data['poa_global'] = poa_frontside_absorbed
        if self.bifacial:
            # set POA values for backside only when bifacial flag is True
            self.sim_data['poa_backside_global_raw'] = poa_backside
            self.sim_data['poa_backside_global'] = poa_backside_absorbed

        return self


    def cell_temperature_from_sapm(self, mounting="glass_open_rack"):
        """
        cell_temperature_from_sapm(self, mounting="glass_open_rack")

        Calculates the cell temperature based on the pvlib.temperature.sapm_cell() function [1].


        Parameters
        ----------
        mounting: str
                  Options:
                  "glass_open_rack" [1]
                  "glass_close_roof" [1]
                  "polymer_open_rack" [1]
                  "polymer_insulated_back" [1]

        Returns
        -------
        Returns a reference to the invoking SolarWorkflowManager object.

        Notes
        -----
        Required data in the sim_data dictionary are 'surface_wind_speed', 'surface_air_temperature' and 'poa_global'.

        References
        ----------
        [1] https://pvlib-python.readthedocs.io/en/stable/generated/pvlib.temperature.sapm_cell.html


        """
        assert "surface_wind_speed" in self.sim_data
        assert "surface_air_temperature" in self.sim_data
        assert "poa_global" in self.sim_data

        if mounting == "glass_open_rack":
            a, b, dT = -3.47, -0.0594, 3
        elif mounting == "glass_close_roof":
            a, b, dT = -2.98, -0.0471, 1
        elif mounting == "polymer_open_rack":
            a, b, dT = -3.56, -0.075, 3
        elif mounting == "polymer_insulated_back":
            a, b, dT = -2.81, -0.0455, 0
        else:
            raise RuntimeError(
                "mounting not one of: 'glass_open_rack', 'glass_close_roof', 'polymer_open_rack', or 'polymer_insulated_back'"
            )

        self.sim_data["cell_temperature"] = pvlib.temperature.sapm_cell(
            self.sim_data["poa_global"],
            self.sim_data["surface_air_temperature"],
            self.sim_data["surface_wind_speed"],
            a=a,
            b=b,
            deltaT=dT,
            irrad_ref=1000,
        )

        return self
    
    
    def apply_angle_of_incidence_losses_to_poa(self):
        """
        apply_angle_of_incidence_losses_to_poa(self)


        Applies the angle of incidence losses to the plane-of-array irradiance using the pvlib.pvsystem.iam.physical() function [1].

        Parameters
        ----------
        None

        Returns
        -------
        Returns a reference to the invoking SolarWorkflowManager object.

        Notes
        -----
        Required data in the sim_data dictionary are 'poa_direct', 'poa_ground_diffuse' and 'poa_sky_diffuse'.

        References
        ----------
        [1] https://pvlib-python.readthedocs.io/en/stable/generated/pvlib.iam.physical.html


        """

        assert "poa_direct_raw" in self.sim_data
        assert "poa_ground_diffuse_raw" in self.sim_data
        assert "poa_sky_diffuse_raw" in self.sim_data

        modtilts = self.sim_data.get("system_modtilt", self.placements["modtilt"].values)

        self.sim_data["poa_direct"] = self.sim_data["poa_direct_raw"]*pvlib.pvsystem.iam.physical(
            aoi=self.sim_data["angle_of_incidence"],
            n=1.526,  # PVLIB v0.7.2 default
            K=4.0,  # PVLIB v0.7.2 default
            L=0.002,  # PVLIB v0.7.2 default
        )

        # Effective angle of incidence values from "Solar-Engineering-of-Thermal-Processes-4th-Edition"
        self.sim_data["poa_ground_diffuse"] = self.sim_data["poa_ground_diffuse_raw"]*pvlib.pvsystem.iam.physical(
            aoi=(90 - 0.5788 * modtilts + 0.002693 * np.power(modtilts, 2)),
            n=1.526,  # PVLIB v0.7.2 default
            K=4.0,  # PVLIB v0.7.2 default
            L=0.002,  # PVLIB v0.7.2 default
        )

        self.sim_data["poa_sky_diffuse"] = self.sim_data["poa_sky_diffuse_raw"]*pvlib.pvsystem.iam.physical(
            aoi=(59.7 - 0.1388 * modtilts + 0.001497 * np.power(modtilts, 2)),
            n=1.526,  # PVLIB v0.7.2 default
            K=4.0,  # PVLIB v0.7.2 default
            L=0.002,  # PVLIB v0.7.2 default
        )

        self.sim_data["poa_diffuse"] = (
            self.sim_data["poa_ground_diffuse"] + self.sim_data["poa_sky_diffuse"]
        )
        self.sim_data["poa_global"] = (
            self.sim_data["poa_direct"] + self.sim_data["poa_diffuse"]
        )

        # make sure poa is realistic, should be less than poa_global_raw which should have been checked before
        assert (self.sim_data["poa_global"] < 1600).all() 

        return self

    def configure_cec_module(
        self,
        module:str="WINAICO WSx-240P6",
        tracking:str="fixed",
        tech_year:int=2050,
        bifaciality_factor:float|None=None
    ):
        """
        configure_cec_module(self, module="WINAICO WSx-240P6")

        Configures CEC of a module based on the outputs of the pvlib.pvsystem.retrieve_sam() function [1].

        Parameters
        ----------
        module: str or dict
            The module name used for the simulation, must be one of:
            * A module found in the pvlib.pvsystem.retrieve_sam("CECMod") database
            * "WINAICO WSx-240P6" -> Good for open-field applications
            * "LG Electronics LG370Q1C-A5" -> Good for rooftop applications
            * A dict containing a set of module parameters, including:
              T_NOCT, A_c, N_s, I_sc_ref, V_oc_ref, I_mp_ref, V_mp_ref, alpha_sc,
              beta_oc, a_ref, I_L_ref, I_o_ref, R_s, R_sh_ref, Adjust, gamma_r, PTC, 
              Bifacial
        tracking : str, optional
            The tracking mechanism, can be 'fixed' or 'single-axis', by default 'fixed'.
        tech_year : int, optional
            If given in combination with the projected module str names "WINAICO WSx-240P6" or
            "LG Electronics LG370Q1C-A5", the effifiency will be scaled linearly to the given
            year. Must then be between year of market comparison in analysis (2019) and 2050.
            Will be ignored when non-projected existing module names or specific parameters
            are given, can then be None. By default 2050.
        bifaciality_factor : float, optional
            Float between 0-1 describing the backside yield reduction compared 
            to frontside at equal radiation. Will take effect only if the module
            has a Bifacial attribute either as True, 1, "1", "Y", "YES", or 
            "Yes". By default None, i.e. bifacial energy production will NOT be 
            considered.
        
        Returns
        -------
        obj
            Returns a reference to the invoking SolarWorkflowManager object

        References
        ----------
        [1] https://pvlib-python.readthedocs.io/en/stable/generated/pvlib.pvsystem.retrieve_sam.html


        """
        # check inputs
        if bifaciality_factor is not None and not 0<bifaciality_factor<=1:
            raise ValueError(f"bifaciality_factor must be a float >0 and <=1, here: {bifaciality_factor}")
        if tracking not in [
            "fixed",
            "single_axis",
        ]:
            raise ValueError("tracking must be either 'fixed' or 'single_axis'")

        # set tracking type as class attribute
        self.tracking = tracking

        def _interpolate_module_params(
            projected_module, original_module_name, tech_year, start_year
        ):
            if not isinstance(tech_year, int):
                raise TypeError(
                    "tech_year must be an integer when projected module is selected"
                )
            # avoid extrapolations
            if not start_year <= tech_year <= 2050:
                raise ValueError(
                    f"tech_year must be between {start_year} and 2050 (max. projection) for this module"
                )

            # get the original (unprojected) module parameters
            db = pvlib.pvsystem.retrieve_sam("CECMod")
            original_module = getattr(db, original_module_name)
            # scale module parameters to tech_year
            module = pd.Series(index=projected_module.index, dtype="float64")
            for param, val_proj in zip(projected_module.index, projected_module):
                if param == "Date":
                    module[param] = str(tech_year)
                elif param in ["Version"]:
                    # ignore, set dummy nan
                    module[param] = np.nan
                elif isinstance(val_proj, (int, float, np.integer)):
                    module[param] = original_module[param] + (
                        val_proj - original_module[param]
                    ) * (tech_year - start_year) / (2050 - start_year)
                else:
                    assert (
                        val_proj == original_module[param]
                    ), f"parameter '{param}' is not the same for original ({original_module[param]}) and projected ({val_proj}) modules"
                    module[param] = val_proj

            return module

        if isinstance(module, str):
            self.register_workflow_parameter("module_name", module)

            if module == "WINAICO WSx-240P6":
                # define projected module parameters
                module_2050 = pd.Series(
                    dict(
                        Bifacial=0,
                        BIPV="N",
                        Date="6/2/2014",
                        T_NOCT=43,
                        A_c=1.663,
                        N_s=60,
                        I_sc_ref=8.41,
                        V_oc_ref=37.12,
                        I_mp_ref=7.96,
                        V_mp_ref=30.2,
                        alpha_sc=0.001164,
                        beta_oc=-0.12357,
                        a_ref=1.6704,
                        I_L_ref=8.961,
                        I_o_ref=1.66e-11,
                        R_s=0.405,
                        R_sh_ref=326.74,
                        Adjust=4.747,
                        gamma_r=-0.383,
                        Version="NRELv1",
                        PTC=220.2,
                        Technology="Multi-c-Si",
                    )
                )

                # scale module parameters to tech_year
                module = _interpolate_module_params(
                    projected_module=module_2050,
                    original_module_name="WINAICO_WSx_240P6",
                    tech_year=tech_year,
                    start_year=2019,
                )

                module.name = "WINAICO WSx-240P6"

            elif module == "LG Electronics LG370Q1C-A5":
                # define projected module parameters
                module_2050 = pd.Series(
                    dict(
                        Bifacial=0,
                        BIPV="N",
                        Date="12/14/2016",
                        T_NOCT=45.7,
                        A_c=1.673,
                        N_s=60,
                        I_sc_ref=10.82,
                        V_oc_ref=42.8,
                        I_mp_ref=10.01,
                        V_mp_ref=37,
                        alpha_sc=0.003246,
                        beta_oc=-0.10272,
                        a_ref=1.5532,
                        I_L_ref=10.829,
                        I_o_ref=1.12e-11,
                        R_s=0.079,
                        R_sh_ref=92.96,
                        Adjust=14,
                        gamma_r=-0.32,
                        Version="NRELv1",
                        PTC=347.2,
                        Technology="Mono-c-Si",
                    )
                )

                # scale module parameters to tech_year
                module = _interpolate_module_params(
                    projected_module=module_2050,
                    original_module_name="LG_Electronics_Inc__LG370Q1C_A5",
                    tech_year=tech_year,
                    start_year=2019,
                )

                module.name = "LG Electronics LG370Q1C-A5"

            elif isinstance(module, str):
                if tech_year is not None:
                    warnings.warn(
                        "NOTE: The tech_year argument is ignored when a specific module is given. Set tech_year to None to silence this warning."
                    )
                # Extract module parameters
                db = pvlib.pvsystem.retrieve_sam("CECMod")
                try:
                    module = getattr(db, module)
                except Exception:
                    raise RuntimeError(
                        "The module '{}' is not in the CEC database".format(module)
                    )
        else:
            if tech_year is not None:
                print(
                    "NOTE: The tech_year argument is ignored when specific module parameters are given."
                )
            module = pd.Series(module)
            assert "T_NOCT" in module.index
            assert "A_c" in module.index
            assert "N_s" in module.index
            assert "I_sc_ref" in module.index
            assert "V_oc_ref" in module.index
            assert "I_mp_ref" in module.index
            assert "V_mp_ref" in module.index
            assert "alpha_sc" in module.index
            assert "beta_oc" in module.index
            assert "a_ref" in module.index
            assert "I_L_ref" in module.index
            assert "I_o_ref" in module.index
            assert "R_s" in module.index
            assert "R_sh_ref" in module.index
            assert "Adjust" in module.index
            assert "gamma_r" in module.index
            assert "PTC" in module.index
            assert "Bifacial" in module.index

            try:
                module_desc = json.dumps(module)
            except Exception:
                module_desc = "user-configured"
            self.register_workflow_parameter("module_desc", module_desc)

        # # Check if we need to add the Desoto parameters
        # # defaults for EgRef and dEgdT taken from the note in the docstring for
        # #  'pvlib.pvsystem.calcparams_desoto'
        # if not "EgRef" in module:
        #     module['EgRef'] = 1.121
        # if not "dEgdT" in module:
        #     module['dEgdT'] = -0.0002677

        self.module = module
        self.bifacial = module["Bifacial"] in [1, "1", "YES", "Yes", "Y", True]
        if bifaciality_factor is not None and not self.bifacial:
            print(f"NOTE: bifaciality_factor is not None but module '{module._name}' is not bifacial, bifaciality_factor will be ignored.")
        self.bifaciality_factor = bifaciality_factor

        return self


    def simulate_with_interpolated_single_diode_approximation(self):
        """
        Does the simulation with an interpolated single diode approximation 
        using the pvlib.pvsystem.calcparams_desoto() [1] function and the
        pvlib.pvsystem.singlediode() [2] function.

        Returns
        -------
        obj
            A reference to the invoking SolarWorkflowManager object.

        Notes
        -----
        Required data in the sim_data dictionary are 'poa_global' and 
        'cell_temperature'. 
        Requires wfm class attribute 'module' to be configured.

        References
        ----------
        [1] https://pvlib-python.readthedocs.io/en/stable/generated/pvlib.pvsystem.calcparams_desoto.html

        [2] https://pvlib-python.readthedocs.io/en/stable/generated/pvlib.pvsystem.singlediode.html

        [3]	(1, 2) W. De Soto et al., “Improvement and validation of a model for photovoltaic array performance”, Solar Energy, vol 80, pp. 78-88, 2006.

        [4]	System Advisor Model web page. https://sam.nrel.gov.

        [5]	A. Dobos, “An Improved Coefficient Calculator for the California Energy Commission 6 Parameter Photovoltaic Module Model”, Journal of Solar Energy Engineering, vol 134, 2012.

        [6]	O. Madelung, “Semiconductors: Data Handbook, 3rd ed.” ISBN 3-540-40488-0

        [7]	S.R. Wenham, M.A. Green, M.E. Watt, “Applied Photovoltaics” ISBN 0 86758 909 4

        [8]	A. Jain, A. Kapoor, “Exact analytical solutions of the parameters of real solar cells using Lambert W-function”, Solar Energy Materials and Solar Cells, 81 (2004) 269-277.

        [9]	D. King et al, “Sandia Photovoltaic Array Performance Model”, SAND2004-3535, Sandia National Laboratories, Albuquerque, NM

        [10]	“Computer simulation of the effects of electrical mismatches in photovoltaic cell interconnection circuits” JW Bishop, Solar Cell (1988) https://doi.org/10.1016/0379-6787(88)90059-2

        """
        assert "poa_global" in self.sim_data
        assert "cell_temperature" in self.sim_data

        assert self.module is not None, "Configure module te be simulated first via configure_cec_module()."

        sel = self.sim_data["poa_global"] > 0
        cell_temp = self.sim_data["cell_temperature"][sel]

        # Use RectBivariateSpline to speed up simulation, but at the cost of accuracy (should still be >99.996%)
        maxpoa = np.nanmax(poa)

        _poa = np.concatenate(
            [
                np.logspace(-1, np.log10(maxpoa / 10), 20, endpoint=False),
                np.linspace(maxpoa / 10, maxpoa, 80),
            ]
        )
        _temp = np.linspace(cell_temp.min(), cell_temp.max(), 100)
        poaM, tempM = np.meshgrid(_poa, _temp)

        sotoParams = pvlib.pvsystem.calcparams_desoto(
            effective_irradiance=poaM.flatten(),
            temp_cell=tempM.flatten(),
            alpha_sc=self.module.alpha_sc,
            a_ref=self.module.a_ref,
            I_L_ref=self.module.I_L_ref,
            I_o_ref=self.module.I_o_ref,
            R_sh_ref=self.module.R_sh_ref,
            R_s=self.module.R_s,
            EgRef=1.121,  # PVLIB v0.7.2 Default
            dEgdT=-0.0002677,  # PVLIB v0.7.2 Default
            irrad_ref=1000,  # PVLIB v0.7.2 Default
            temp_ref=25,  # PVLIB v0.7.2 Default
        )

        photoCur, satCur, resSeries, resShunt, nNsVth = sotoParams
        gen = pvlib.pvsystem.singlediode(
            photocurrent=photoCur,
            saturation_current=satCur,
            resistance_series=resSeries,
            resistance_shunt=resShunt,
            nNsVth=nNsVth,
            ivcurve_pnts=None,  # PVLIB v0.7.2 Default
            method="lambertw",  # PVLIB v0.7.2 Default
        )

        interpolator = RectBivariateSpline(
            _temp, _poa, gen["p_mp"].reshape(poaM.shape), kx=3, ky=3
        )
        self.sim_data["module_dc_power_at_mpp"] = np.zeros_like(
            self.sim_data["poa_global"]
        )
        self.sim_data["module_dc_power_at_mpp"][sel] = interpolator(
            cell_temp, poa, grid=False
        )

        interpolator = RectBivariateSpline(
            _temp, _poa, gen["v_mp"].reshape(poaM.shape), kx=3, ky=3
        )
        self.sim_data["module_dc_voltage_at_mpp"] = np.zeros_like(
            self.sim_data["poa_global"]
        )
        self.sim_data["module_dc_voltage_at_mpp"][sel] = interpolator(
            cell_temp, poa, grid=False
        )

        self.sim_data["capacity_factor"] = self.sim_data["module_dc_power_at_mpp"] / (
            self.module.I_mp_ref * self.module.V_mp_ref
        )

        # Estimate total system generation
        if "capacity" in self.placements.columns:
            self.sim_data["total_system_generation"] = self.sim_data[
                "capacity_factor"
            ] * np.broadcast_to(self.placements.capacity, self._sim_shape_)

        if (
            "modules_per_string" in self.placements.columns
            and "strings_per_inverter" in self.placements.columns
        ):
            total_modules = (
                self.placements.modules_per_string
                * self.placements.strings_per_inverter
                * getattr(self.placements, "number_of_inverters", 1)
            )

            self.sim_data["total_system_generation"] = self.sim_data[
                "module_dc_power_at_mpp"
            ] * np.broadcast_to(total_modules, self._sim_shape_)

        return self

    def apply_inverter_losses(
        self,
        inverter,
        method="sandia",
    ):
        """
         apply_inverter_losses(self, inverter, method="sandia", )

         Applies inverter losses using the pvlib.pvsystem.snlinverter() fuction [1], the pvlib.pvsystem.retrieve_sam() fuction [2] and the
         pvlib.pvsystem.adrinverter() fuction [3].


         Parameters
         ----------
         inverter: str
                   Describes the inverter.
                   [TODO: Add a more detailed desciption following the example of 'configure_cec_module']
         method: str
                 Options:
                 "scandia"
                 "driesse"
                 Describes the used method to apply the inverter losses.

         Returns
         -------
         Returns a reference to the invoking SolarWorkflowManager object.

         Notes
         -----
         Required data in the sim_data dictionary are 'module_dc_power_at_mpp' and 'module_dc_voltage_at_mpp'.
         Required data in the placements dataframe are 'modules_per_string' and 'strings_per_inverter'.
         Cannot simultaneously provide 'capacity' and inverter-string parameters.


         References
         ----------
        [1] https://pvlib-python.readthedocs.io/en/stable/generated/pvlib.pvsystem.snlinverter.html

        [2] https://pvlib-python.readthedocs.io/en/stable/generated/pvlib.pvsystem.retrieve_sam.html

        [3] https://pvlib-python.readthedocs.io/en/stable/generated/pvlib.pvsystem.adrinverter.html

        [4]	SAND2007-5036, “Performance Model for Grid-Connected Photovoltaic Inverters by D. King, S. Gonzalez, G. Galbraith, W. Boyson

        [5]	System Advisor Model web page. https://sam.nrel.gov.

        [6]	Beyond the Curves: Modeling the Electrical Efficiency of Photovoltaic Inverters, PVSC 2008, Anton Driesse et. al.

        """

        """method can be: 'sandia' or 'driesse'

        TODO: Make it work with multiplt inverter definitions
        """

        assert "module_dc_power_at_mpp" in self.sim_data
        assert "module_dc_voltage_at_mpp" in self.sim_data
        assert hasattr(self, "module")
        assert "modules_per_string" in self.placements.columns
        assert "strings_per_inverter" in self.placements.columns
        assert (
            "capacity" not in self.placements.columns
        ), "Cannot simultaneously provide 'capacity' and inverter-string parameters"

        if method == "sandia":
            if isinstance(inverter, str):
                db = pvlib.pvsystem.retrieve_sam("SandiaInverter")
                inverter = getattr(db, inverter)

            self.sim_data["inverter_ac_power_at_mpp"] = pvlib.inverter.sandia(
                v_dc=self.sim_data["module_dc_voltage_at_mpp"]
                * np.broadcast_to(self.placements.modules_per_string, self._sim_shape_),
                p_dc=self.sim_data["module_dc_power_at_mpp"]
                * np.broadcast_to(
                    self.placements.modules_per_string
                    * self.placements.strings_per_inverter,
                    self._sim_shape_,
                ),
                inverter=inverter,
            )

        elif method == "driesse":
            if isinstance(inverter, str):
                db = pvlib.pvsystem.retrieve_sam("CECInverter")
                inverter = getattr(db, inverter)

            self.sim_data["inverter_ac_power_at_mpp"] = pvlib.pvsystem.adrinverter(
                v_dc=self.sim_data["module_dc_voltage_at_mpp"]
                * np.broadcast_to(self.placements.modules_per_string, self._sim_shape_),
                p_dc=self.sim_data["module_dc_power_at_mpp"]
                * np.broadcast_to(
                    self.placements.modules_per_string
                    * self.placements.strings_per_inverter,
                    self._sim_shape_,
                ),
                inverter=inverter,
            )

        number_of_inverters = getattr(self.placements, "number_of_inverters", 1)
        self.sim_data["total_system_generation"] = self.sim_data[
            "inverter_ac_power_at_mpp"
        ] * np.broadcast_to(number_of_inverters, self._sim_shape_)

        total_capacity = (
            self.module.I_mp_ref
            * self.module.V_mp_ref
            * self.placements.modules_per_string
            * self.placements.strings_per_inverter
            * number_of_inverters
        )

        self.sim_data["capacity_factor"] = self.sim_data[
            "total_system_generation"
        ] / np.broadcast_to(total_capacity, self._sim_shape_)

        return self
