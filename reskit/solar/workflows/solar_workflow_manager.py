import geokit as gk
import pandas as pd
import numpy as np
import yaml
import rasterio

from os.path import isfile, splitext
from collections import OrderedDict
from types import NoneType
import warnings
from scipy.interpolate import RectBivariateSpline
import json
import re
from collections.abc import Iterable

from reskit.solar import core as rk_solar_core
from reskit.util.generic_helpers import _align_inputs
from reskit.workflow_manager import WorkflowManager

from reskit.solar import DATA #TODO move data into core analog to wind/core/data

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

    def preprocess_bifaciality_factor(
        self,
        bifaciality_factor = float | Iterable,
    ):
        """
        Preprocesses the bifaciality factor input into a 1d array with a single numeric 
        value per placement and saves it under plant_parameters_processed. Will check for
        realistic value ranges.

        Parameters
        ----------
        bifaciality_factor : float | Iterable
            Either a float value or a 1d iterable thereof with lenghth matching the 
            number of placements. #TODO allow also "from_module"

        Returns
        -------
        obj
            reference to the invoking SolarWorkflowManager object
        """
        # first save input to allow tracing the processing 
        self.plant_parameters_raw["bifaciality_factor"] = bifaciality_factor
        
        # then preprocess the inputs
        bifaciality_factor, _ = self._preprocess_variable(
            varname = "bifaciality_factor",
            value = bifaciality_factor,
            allow_none = False, # bifaciality factor MUST be provided consciously by user
            replace_none = None,
            assert_type = [float, int],
            force_cols = 1,
            force_dims = 1, # one scalar value per placement -> 1d
            as_dtype = float,
        )
        if any(isinstance(value, str) for value in _): 
            raise TypeError("bifaciality_factor must not contain str entries.")

        # TODO allow 'from_module' as a source of bifaciality_factor as soon as different modules are implemented as well, will then extract module["bifaciality_factor"] for the given location(s)
        # for now, only check if given bifaciality factor conflicts with potential module information, then inform user
        if self.module is not None and "Bifacial" in self.module.index:
            # we have information if the module is nifacial at all
            module_bifacials = self.module["Bifacial"] in [1, "1", "YES", "Yes", "Y", True] 
            if ((bifaciality_factor>0) != module_bifacials).any():
                print(f"NOTE: Module bifacial information and bifaciality according to bifaciality factors do not align for at least one plant", flush=True)
        if "bifaciality_factor" in self.module.index:
            # we have information on the quantitative bifaciality factor
            if not np.isclose(bifaciality_factor, self.module["bifaciality_factor"]).all():
                print(f"NOTE: Module has bifaciality_factor information but it does not match the provided bifaciality_factor argument for all locations.", flush=True)

        if not np.all((bifaciality_factor >= 0) & (bifaciality_factor <= 1.0)):
            raise ValueError(f"All bifaciality_factor values must be >=0 and <=1.0.")

        # finally save the processed data under plant_parameters_processed
        self.plant_parameters_processed["bifaciality_factor"] = bifaciality_factor

        return self


    def preprocess_capacity(
        self,
        capacity = float | int | Iterable,
    ):
        """
        Preprocesses the capacity [in kW] input into a 1d array with a single numeric 
        value per placement and saves it under plant_parameters_processed. 

        Parameters
        ----------
        capacity : float | Iterable
            Either a float value or a 1d iterable thereof with lenghth matching the 
            number of placements, in [kW]

        Returns
        -------
        obj
            reference to the invoking SolarWorkflowManager object
        """
        # first save input to allow tracing the processing 
        self.plant_parameters_raw["capacity"] = capacity

        if self._is_none(capacity).any() and not self._is_none(capacity).all():
            raise ValueError(f"'capacity' must be provided for all locations (no Nones) if provided for any.")
        
        # then preprocess the inputs
        capacity, _ = self._preprocess_variable(
            varname = "capacity",
            value = capacity,
            allow_none = True,
            replace_none = None,
            assert_type = [int, float],
            force_cols = 1,
            force_dims = 1, # one scalar value per placement -> 1d
        )
        if any(isinstance(value, str) for value in _): 
            raise TypeError("capacity must not contain str entries.")

        # finally save the processed data under plant_parameters_processed
        self.plant_parameters_processed["capacity"] = np.asarray(capacity, dtype=float)

        return self
    

    def preprocess_ground_coverage_ratio(
        self,
        gcr : float | str | Iterable,
        fallback: float | NoneType = None,
        min_gcr : float | NoneType = 0.169,
        min_interrow_distance : int | float | np.ndarray = 2.5,
    ):
        """
        Preprocesses the gcr input into a 1d array with a single numeric value per 
        placement and saves it under plant_parameters_processed. Will replace None, NaN, 
        pd.Na, null or strings thereof (including empty '' string) by fallback value and
        consider string entries as convention names based on which gcr per placement is 
        then assigned (see reskit.solar.core.system_design.location_to_gcr() for options).

        Parameters
        ----------
        gcr : str, float, Iterable
            If a string is given it must be a gcr convention name allowed in 
            reskit.solar.core.system_design.location_to_gcr(). 
            If a float is given, it will be applied to all locations equally.
            If an iterable is given it has to be of equal length to the number of 
            locations and contain one of the options above per placement, None etc.
            will be replaced by the fallback value.
        fallback : float | NoneType, optional
            The fallback value that can be used in case that a placement contains
            a None (etc.) value or gcr cannot be extracted via the given convention.
            By default None, will then trigger failure for any location for which 
            the primary value cannot be extracted.
        min_gcr : float | NoneType, optional
            If given as a float, GCR values will be limited to this minimum value.
            Has no effect if None, by default 0.169.
        min_interrow_distance : int | float | np.ndarray, optional
            The minimum width in m between the inner edges of 2 neighboring rows, 
            to allow e.g. acces for maintenance vehicles. By default 2.5 [m].

        Returns
        -------
        obj
            reference to the invoking SolarWorkflowManager object
        """
        assert fallback is None or isinstance(
            fallback, float
        ) and 0<fallback<=1, "gcr 'fallback' must be a float >0 and <=1."

        # preprocess min_interrow_distance to locs array shape
        _, min_interrow_distance, _ = _align_inputs(
            self.locs.lats,
            min_interrow_distance,
        )

        # first save input to allow tracing the processing 
        self.plant_parameters_raw["gcr"] = gcr
        
        # then preprocess the gcr inputs
        gcr, conventions = self._preprocess_variable(
            varname = "gcr",
            value = gcr,
            allow_none = fallback is not None,
            replace_none = fallback, # note that this replaces None etc. by fallback value already
            assert_type = float,
            force_cols = 1, # one per loc
            force_dims = 1
        )

        # now replace the entries with str conventions by the respective values
        fallbacks_applied = 0
        for conv in np.unique(conventions[~pd.isna(conventions)]):
            # get the gcrs only for the affected locations
            mask = (conventions == conv)
            # handle different conventions differently
            if isfile(conv):
                # assume it is a raster and extract value
                try:
                    iter_gcrs = gk.raster.interpolateValues(
                        source = conv,
                        points = self.locs[mask],
                    )
                except Exception as e:
                    raise RuntimeError(f"convention is an existing file but data could not be extracted via gk.raster.interpolateValues(): {e}")
            elif conv == "winter_solstice_rule":
                assert self.tracking == "fixed", f"'winter_solstice_rule' convention can only be applied to 'fixed', here '{self.tracking}' tracking."
                iter_row_pitches, iter_gcrs = rk_solar_core.system_design.location_to_gcr_and_row_pitch_winter_solstice_rule(
                    lats = self.locs.lats[mask], 
                    module_tilts = self.plant_parameters_processed["module_tilt"][mask], 
                    north_slopes = self.plant_parameters_processed["north_slope"][mask], 
                    solar_hour = 12, 
                    module_area_width = self.plant_parameters_processed["pvrow_width_sloped"][mask],
                    min_interrow_distance = min_interrow_distance[mask]
                )
            elif conv == "tonita_et_al_2023":
                iter_gcrs = rk_solar_core.system_design.location_to_gcr_tonita_et_al_2023(
                    lat = self.locs.lats[mask], 
                    bifaciality_factor = self.plant_parameters_processed["bifaciality_factor"][mask],
                    tracking = self.tracking,
                    shading_loss = 0.05, 
                )
            else:
                raise NotImplementedError(f"Location-to-GCR convention '{conv}' is not implemented yet. Choose from: 'winter_solstice_rule', 'tonita_et_al_2023' or a raster filepath with GCR information.")

            if np.isnan(iter_gcrs).any():
                # we have nans in the data that we just extracted/calculated
                if fallback is None:
                    # no fallback means we must fail if primary option does not work
                    raise ValueError("gcr values could not be extracted for all locations.")
                else:
                    # if allowed, replace nans by fallback value 
                    fallbacks_applied += np.isnan(iter_gcrs).sum()
                    iter_gcrs[np.isnan(iter_gcrs)] = (
                        np.ones(shape=iter_gcrs.shape) * fallback
                    )[np.isnan(iter_gcrs)]
            # last set the extracted elevations from this iteration in the main elevation array, duplicate iter_gcr row values per placement for every column
            gcr[mask] = iter_gcrs
        # warn of unavailable elevation data if required
        if fallbacks_applied > 0:
            warnings.warn(f"GCR for {fallbacks_applied} out of {self.locs.count} placements could not be extracted from desired convention and has been replaced with fallback value {fallback}.")

        # check and set min gcr if applicable (needed here explicitly because not all gcr values come from location_to_gcr)
        if min_gcr is not None:
            assert isinstance(min_gcr, float), "min_gcr must be a float if not None."
            # set all values below min gcr to min gcr
            gcr[gcr < min_gcr] = min_gcr

        # validity check
        if pd.isnull(gcr).any():
            raise ValueError("gcr contains NaN values.")
        if not np.all((gcr > 0) & (gcr <= 1.0)):
            raise ValueError("'gcr' contains values less or equal to zero or greater 1.0.")

        # finally save the processed gcr under plant_parameters_processed
        self.plant_parameters_processed["gcr"] = np.asarray(gcr, dtype=float)

        return self
    

    def preprocess_elevation(
        self,
        elevation: str | int | Iterable, 
        fallback: int = 840,
    ):
        """
        Preprocesses the elevation input into a 1d array with a single numeric value per 
        placement and saves it under plant_parameters_processed. Will replace None, NaN, 
        pd.Na, null or strings thereof (including empty '' string) by fallback value and
        consider string entries as DEM filepaths from which elevation for the respective
        placements will be extracted. Placements for which no data can be extracted will 
        be assigned the fallback value as well.

        Parameters
        ----------
        elevation : str, int, Iterable
            If a string is given it must be a path to a rasterfile including the elevations.
            If an iterable is given it has to include the elevations at each location and be
            of equal length to the number of locations.
            If an integer is given, it will be applied to all locations equally
        fallback : int, optional
            The fallback value in [m] that will be used in case that elev is a raster
            path and the extraction of the elevation from raster fails (applied
            only to no-data locations). Setting fallback to None means process will 
            fail if primary option fails, by default 840 m (average global elevation)

        Returns
        -------
        obj
            reference to the invoking SolarWorkflowManager object
        """
        assert isinstance(
            fallback, int
        ), "elevation 'fallback' must be an integer elevantion in [m]." #TODO check if we can somehow make use of the data that might have been extracted in the horizon profile already to save time

        # first save input to allow tracing the processing 
        self.plant_parameters_raw["elevation"] = elevation
        
        # then preprocess the elevation inputs
        elevation, filepaths = self._preprocess_variable(
            varname = "elevation",
            value = elevation,
            allow_none = fallback is not None,
            replace_none = fallback, # note that this replaces None etc. by fallback value already
            assert_type = [int, float, str],
            force_cols = 1,
            force_dims = 1,
        )

        # now deal with potential DEM filepath inputs, identified via string datatype
        # iterate over potentially different filepaths and extract elev for all locs that apply
        fallbacks_applied = 0
        for fp in np.unique(filepaths[~pd.isna(filepaths)]):
            if not isfile(fp):
                raise FileNotFoundError(f"elevation contains string entries which must be existing DEM rasters but cannot find file: {fp}")
            # generate a mask that applies also to the locations
            mask = (filepaths == fp)
            # extract a clipped sub raster covering the affected placements extent only to save time
            iter_ext = gk.Extent.fromLocationSet(gk.LocationSet(self.locs[mask])) # LocationSet can only be in EPSG:4326 as per geokit implementation
            clipped_elev = iter_ext.pad(0.5).rasterMosaic(fp)
            if clipped_elev is None:
                iter_vals = np.array([np.nan] * len(self.locs[mask]))
            else:
                iter_vals = gk.raster.interpolateValues(clipped_elev, self.locs[mask])
                if np.isnan(iter_vals).any():
                    # if getting values fails, it could be because of interpolation method
                    # replace only those that failed by 'near' interpolation
                    iter_vals_near = gk.raster.interpolateValues(
                        clipped_elev, self.locs[mask], mode="near"
                    )
                    iter_vals[np.isnan(iter_vals)] = iter_vals_near[np.isnan(iter_vals)]
            if np.isnan(iter_vals).any():
                # we still have nans
                if fallback is None:
                    # must fail then
                    raise ValueError("NaN values extracted for 'elevation' and fallback is None.")
                else:
                    # if allowed, replace nans by fallback value 
                    fallbacks_applied += np.isnan(iter_vals).sum()
                    iter_vals[np.isnan(iter_vals)] = (
                        np.ones(shape=iter_vals.shape) * fallback
                    )[np.isnan(iter_vals)]
            # last set the extracted elevations from this iteration in the main elevation array
            elevation[mask] = iter_vals
        # warn of unavailable elevation data if required
        if fallbacks_applied > 0:
            warnings.warn(f"Elevation for {fallbacks_applied} out of {self.locs.count} placements could not be extracted from provided DEM files and has been replaced with fallback value {fallback}m.")

        # finally save the processed elevation under plant_parameters_processed
        self.plant_parameters_processed["elevation"] = np.asarray(elevation, dtype=int)

        return self

    def preprocess_fixed_module_tilt(
        self,
        module_tilt : int | float | str | Iterable,
    ):
        """
        _summary_ #TODO mention degrees

        Parameters
        ----------
        module_tilt : int | float | str | Iterable
            _description_

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        # first save input to allow tracing the processing 
        self.plant_parameters_raw["module_tilt"] = module_tilt
        
        # then preprocess the inputs - do in a try-except to be able to explain the need for not-None values here (can and should be None when singleaxis tracking)
        try:
            module_tilt, conventions = self._preprocess_variable(
                varname = "module_tilt",
                value = module_tilt,
                allow_none = False,
                replace_none = None,
                assert_type = [int, float],
                force_cols = 1, # one per loc
                force_dims = 1,
                as_dtype = float,
            )
        except ValueError as e:
            if "must not have None values" in str(e):
                raise ValueError(
                    f"{e} This is required when tracking='fixed'."
                ) from e
            raise

        # now replace the entries with str conventions by the respective values
        for conv in np.unique(conventions[~pd.isna(conventions)]):
            # get the azimuths only for the affected locations
            mask = (conventions == conv)
            iter_vals = rk_solar_core.system_design.location_to_module_tilt(
                locs = self.locs[mask], 
                convention = conv, 
            )
            if np.isnan(iter_vals).any():
                # we have nans in the data that we just extracted
                raise ValueError(f"module_tilt values could not be extracted via required '{conv}' convention for all locations.")
            # last set the extracted elevations from this iteration in the main elevation array, duplicate iter_gcr row values per placement for every column
            module_tilt[mask, :] = iter_vals[:, None]

        # finally save the processed module tilt under plant_parameters_processed
        self.plant_parameters_processed["module_tilt"] = module_tilt

        return self


    def preprocess_fixed_module_azimuth(
        self,
        module_azimuth : int | float | str,
        ):

        # first save input to allow tracing the processing 
        self.plant_parameters_raw["module_azimuth"] = module_azimuth
        
        # then preprocess the inputs - do in a try-except to be able to explain the need for not-None values here (can and should be None when singleaxis tracking)
        try:
            module_azimuth, conventions = self._preprocess_variable(
                varname = "module_azimuth",
                value = module_azimuth,
                allow_none = False,
                replace_none = None,
                assert_type = [int, float],
                force_cols = 1, # one per loc
                force_dims = 1,
                as_dtype = float,
            )
        except ValueError as e:
            if "must not have None values" in str(e):
                raise ValueError(
                    f"{e} This is required when tracking='fixed'."
                ) from e
            raise

        # now replace the entries with str conventions by the respective values
        for conv in np.unique(conventions[~pd.isna(conventions)]):
            # get the azimuths only for the affected locations
            mask = (conventions == conv)
            iter_vals = rk_solar_core.system_design.location_to_module_azimuth(
                locs = self.locs[mask], 
                convention = conv, 
            )
            if np.isnan(iter_vals).any():
                # we have nans in the data that we just extracted
                raise ValueError(f"module_azimuth values could not be extracted via required '{conv}' convention for all locations.")
            # last set the extracted elevations from this iteration in the main elevation array, duplicate iter_gcr row values per placement for every column
            module_azimuth[mask, :] = iter_vals[:, None]

        # finally save the processed module azimuth under plant_parameters_processed
        self.plant_parameters_processed["module_azimuth"] = module_azimuth

        return self


    def preprocess_tracking_angle(
        self,
        max_tracking_angle : int | float | Iterable,
        ):
        """
        Preprocesses the input into a 1d array with a single numeric value per location 
        and saves it under plant_parameters_processed. Will not allow None, NaN, pd.Na, 
        null or strings thereof (including empty '' string) or any string entries.

        Parameters
        ----------
        max_tracking_angle : int | float | Iterable
            If an iterable is given it has to include the values for each location and 
            be of equal length to the number of locations. A scalar value will be 
            applied to all locations equally.  Must be in degrees.

        Returns
        -------
        obj
            reference to the invoking SolarWorkflowManager object
        """
        # first save input to allow tracing the processing 
        self.plant_parameters_raw["max_tracking_angle"] = max_tracking_angle
        
        # then preprocess the inputs
        max_tracking_angle, _ = self._preprocess_variable(
            varname = "max_tracking_angle",
            value = max_tracking_angle,
            allow_none = False,
            replace_none = None,
            assert_type = [float,int],
            force_cols = 1, # one per loc
            force_dims = 1,
            as_dtype = float,
        )

        # write to processed data
        self.plant_parameters_processed["max_tracking_angle"] = max_tracking_angle

        return self
    
    def preprocess_backtracking(
        self,
        backtracking : bool | Iterable,
        ):
        """
        Preprocesses the input into a 1d array with a single boolean value per location 
        and saves it under plant_parameters_processed. Will not allow None, NaN, pd.Na, 
        null or strings thereof (including empty '' string) or any string entries.

        Parameters
        ----------
        backtracking : int | float | Iterable
            If an iterable is given it has to include the values for each location and 
            be of equal length to the number of locations. A scalar value will be 
            applied to all locations equally.

        Returns
        -------
        obj
            reference to the invoking SolarWorkflowManager object
        """
        # first save input to allow tracing the processing 
        self.plant_parameters_raw["backtracking"] = backtracking
        
        # then preprocess the inputs
        backtracking, _ = self._preprocess_variable(
            varname = "backtracking",
            value = backtracking,
            allow_none = False,
            replace_none = None,
            assert_type = bool,
            force_cols = 1, # one per loc
            force_dims = 1,
            as_dtype = bool,
        )

        # write to processed data
        self.plant_parameters_processed["backtracking"] = backtracking

        return self


    def preprocess_pvrow_height(
        self,
        pvrow_height : float | Iterable | None = None,
        ):
        """
        _summary_

        Parameters
        ----------
        pvrow_height : float | Iterable | None, optional
            The height over ground in [m] of the row center axis. If None is given,
            the value will be calculated based on the array width (based on module
            dimensions and configuration) and (maximum) tilt angle plus minimum
            ground clearance. By default None.

        Returns
        -------
        obj
            reference to the invoking SolarWorkflowManager object
        """
        # assert preprocessed data exists
        if self.tracking == "singleaxis":
            if not "max_tracking_angle" in self.plant_parameters_processed:
                raise AttributeError(f"'max_tracking_angle' is a required attributes of self.plant_parameters_processed when tracking = 'singleaxis', run preprocess_tracking_angle() first.")
            if not "backtracking" in self.plant_parameters_processed:
                raise AttributeError(f"'backtracking' is a required attributes of self.plant_parameters_processed when tracking = 'singleaxis', run preprocess_backtracking() first.")
        elif self.tracking == "fixed" and not "module_tilt" in self.plant_parameters_processed:
            raise AttributeError(f"'module_tilt' is a required attributes of self.plant_parameters_processed when tracking = 'fixed', run preprocess_fixed_module_tilt() first.")
        if not "pvrow_width_sloped" in self.plant_parameters_processed:
            raise AttributeError(f"'pvrow_width_sloped' is a required attributes of self.plant_parameters_processed, run configure_cec_module() first.")

        # first save input to allow tracing the processing 
        self.plant_parameters_raw["pvrow_height"] = pvrow_height
        
        # then preprocess the inputs
        pvrow_height, _ = self._preprocess_variable(
            varname = "pvrow_height",
            value = pvrow_height,
            allow_none = True,
            replace_none = None,
            assert_type = [int, float, NoneType],
            force_cols = 1, # one per loc
            force_dims = 1,
        )
        # flag missing values
        noheightmask = self._is_none(pvrow_height)

        # if no pv row axis height (normal to sloped ground) is given, we must ensure that the modules will never tough ground in whatever angular position
        # so define the default pv row axis height as 0.5 x the width of the array (sloped), plus 1cm to ensure positive values despite numeric effects
        # this is particularly important as the module angles relative to the ground may change later in case of sloped ground
        # otherwise (sin(module tilt) x 0.5 x sloped array width) + 0.01 would have been sufficient
        min_pvrow_height = (1/2 * self.plant_parameters_processed["pvrow_width_sloped"] + 0.01)
        # set the minimum required pv row height only for locs without specifically given height values 
        pvrow_height[noheightmask] = min_pvrow_height[noheightmask]

        # calculate the maximum vertical center to lower row edge distance under greatest possible angle
        angles_deg = self.plant_parameters_processed["module_tilt"] if self.tracking == "fixed" else self.plant_parameters_processed["max_tracking_angle"]
        center_to_lower_edge_vertical = 1/2 * (self.plant_parameters_processed["pvrow_width_sloped"] * np.sin(np.radians(angles_deg)))

        # make sure the modules will never penetrate the ground, leads to physicalls unrealistic returns in pvlib
        if not np.all(pvrow_height - 1/2 * self.plant_parameters_processed["pvrow_width_sloped"] > 0):
            # the modules would penetrate the ground if vertical
            if np.all(pvrow_height - center_to_lower_edge_vertical > 0):
                # in the current tilt, the modules would NOT penetrate the ground, but they might when the angle becomes relatively steeper along a hill side
                raise ValueError(
                    f"Given pvrow_height values do not lead to a sufficient ground clearance for all locations. The values would be sufficient for current (maximum) module tilts, "
                    "but tilt values may increase relatively to sloped ground for hill slide plants; therefore vertical modules should be possible for all locations, too."
                    )
            raise ValueError(f"Given pvrow_height values do not lead to a sufficient ground clearance for all locations under the given module size and all potential module angles (incl. vertical).")

        # finally save the processed values under plant_parameters_processed
        self.plant_parameters_processed["pvrow_height"] = np.asarray(pvrow_height, dtype=float)

        return self


    def preprocess_ground_albedo(
            self,
            ground_albedo : float | str | Iterable, 
            consider_snow_albedo : bool = False,
            fallback : float = 0.25,
            ):
        """
        #TODO

        Parameters
        -------
        ground_albedo : float, tuple, list
            * float : value will be set to all placements
            * tuple/list : Must then contain landcover dataset information and 
              be formatted like (dataset_name:str, dataset_filepath:str, fallback: float, optional).

        consider_snow_albedo : bool, optional
            If True, will consider hourly snow cover in the ground albedo.
            Requires that snow cover data is available in sim_data attribute. #TODO which exact attribute?
            By default False.

        Returns
        -------
        obj
            reference to the invoking SolarWorkflowManager object
        """
        assert isinstance(fallback, float) and 0<fallback<1,\
            "fallback_albedo must be a float between 0.0 and 1.0"

        # write ground albedo raw input to storage
        self.plant_parameters_raw["ground_albedo"] = ground_albedo
    
        # first check if ground_albedo is possibly given as a tuple of landcover name and filepath
        landcover_name = None
        if isinstance(ground_albedo, (tuple, list)) and len(ground_albedo) == 2 and all([isinstance(x, str) for x in ground_albedo]): #TODO can we somehow allow the filepath per location?
            # we may have a (landcover_name, landcover_filepath) definition, load the albedo data from yaml file
            with open(DATA.get('ground_cover_albedos.yaml'), "r") as stream:
                ground_cover_albedos = yaml.safe_load(stream)
            if ground_albedo[0] in ground_cover_albedos["dataset_classtype_mapper"]:
                if isfile(ground_albedo[1]):
                    # we do have a known definition, adapt ground_albedo for further processing
                    landcover_name = ground_albedo[0]
                    ground_albedo = ground_albedo[1] 
                else:
                    raise ValueError(f"Tuple or list with len = 2 was passed as ground_cover with known landcover name '{ground_albedo[0]}' but filepath as second entry does not exist: '{ground_albedo[1]}'")
            elif isfile(ground_albedo[1]): 
                # the file exists but we have no known landcover name
                raise ValueError(f"Tuple or list with len = 2 was passed as ground_cover with existing filepath as second entry but unknown landcover name: '{ground_albedo[0]}'")
        # an Iterable of tuples has not been implemented yet
        elif (
            isinstance(ground_albedo, Iterable)
            and not isinstance(ground_albedo, (str, bytes))
            and any(isinstance(x, tuple) for x in ground_albedo)
        ):
            raise NotImplementedError("An iterable containing at least one tuple was passed as ground_albedo. Not implemented yet.")

        # then format ground albedo input
        albedo, filepaths = self._preprocess_variable(
                varname = "ground_albedo",
                value = ground_albedo,
                allow_none = True,
                replace_none = fallback,
                assert_type = None,
                force_cols = self._sim_shape_[0], # one per each timestep required
                force_dims = 2,
                transpose = True, # we need 8760 rows and Nloc columns here for sim shape
            )
        if any(isinstance(x, str) for x in filepaths) and landcover_name is None:
            raise TypeError("ground_albedo scalar or array values may not be/contain strings, landcover dataset paths must be passed as single tuple/list: (landcover_name, landcover_path)")

        # then get ground albedo from possibly linked land use datasets
        # define an aux function to extract ground albedos from landcover name and path
        def _get_ground_albedo_from_landcover(landcover_name:str, landcover_path:str, fallback_albedo:float):
            """
            Get the mean snow-free broadband white sky albedos for the respective
            landcover per placement based on category type in the landcover dataset.
            Extracts value for the centroid point in case of polygons.
            
            Parameters
            ----------
            landcover_name : str
                Name of dataset, must be a key in dataset_mapper dict in the 
                ground_cover_albedos.yaml file in solar/data.
            landcover_path : str
                The filepath to the dataset described by landcover_name.
            fallback_albedo : float
                The value to set for cells which have no data in the landcover 
                raster file. By default 0.2
            -------
            list
                list with local ground albedo values of length of placements 
                attribute of invoking SolarWorkflowManager object
            """
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
            LCclasses = np.atleast_1d(gk.raster.interpolateValues(
                source = landcover_path,
                points = self.locs, 
                pointSRS="latlon", 
                mode="near"
                ))
            # map the LC class number to albedo via ground types 
            classtype = ground_cover_albedos["dataset_classtype_mapper"][landcover_name]
            LCclass_groundtype_mapper = ground_cover_albedos["LCclass_groundtype_mapper"][classtype]
            groundtype_albedo_mapper = ground_cover_albedos["groundtype_albedo_mapper"]
            # deal with nans in landcover raster
            groundtype_albedo_mapper["nodata"] = fallback_albedo
            def val_to_lc(val):
                if isinstance(val, float) and np.isnan(val):
                    return "nodata"
                return LCclass_groundtype_mapper[val]
            if np.isnan(LCclasses).any():
                print(f"Landcover raster has missing cell values, will be filled with fallback_albedo: {fallback_albedo}")
            ground_albedos = [groundtype_albedo_mapper[val_to_lc(c)] for c in LCclasses]
            assert not any([np.isnan(a) for a in ground_albedos]), "NaN values found in extracted ground_albedos."
            return ground_albedos
        # iterate over all unique albedo filepaths
        for fp in np.unique(filepaths[~pd.isna(filepaths)]):
            # get the azimuths only for the affected locations
            mask = (filepaths == fp)
            iter_vals = np.asarray(_get_ground_albedo_from_landcover(
                landcover_name = landcover_name, 
                landcover_path = fp, 
                fallback_albedo = fallback
                ))
            # last set the extracted elevations from this iteration in the main elevation array, duplicate iter_gcr row values per placement for every column            
            albedo[:, mask] = iter_vals[mask][None, :]

        # # change scalar ground albedo per location to hourly system_grdalbedo array per location
        # self.sim_data["system_grdalbedo"] = np.tile(self.placements["grdalbedo"].to_numpy()[None, :], (self._sim_shape_[0], 1)) 
        assert albedo.shape == self._sim_shape_ #TODO delete this and commeted out lines hereabove when confirmed that above force_cols = self._sim_shape_[0] works

        # now write hourly no-snow albedo values to sim_data
        if not np.all((albedo > 0) & (albedo < 1)):
            raise ValueError("ground albedo must be >0 and <1 for all locations.")
        assert albedo.shape == self._sim_shape_ # make sure
        self.sim_data["system_grdalbedo_nosnow"] = np.asarray(albedo, dtype=float)

        # consider snow albedo if applicable
        consider_snow_albedo, _ = self._preprocess_variable(
            varname = "consider_snow_albedo",
            value = consider_snow_albedo,
            allow_none = False,
            replace_none = None,
            assert_type = bool,
            force_cols = 1, # one per loc
            force_dims = 1,
            as_dtype = bool,
        )

        if np.any(consider_snow_albedo):
            # make sure that we have loaded all required snow data variables
            if not all([var in self.sim_data for var in ["snow_albedo", "snow_depth_water_equivalent", "snow_density"]]):
                raise AttributeError("consider_snow_albedo=True but no 'snow_albedo', 'snow_depth_water_equivalent' and/or 'snow_density' data found in sim_data attribute.")
            # mask only the timesteps with actual snow (>= 1cm) on the ground
            snow_mask = self.sim_data["snow_depth_water_equivalent"] * (1000 / self.sim_data["snow_density"]) >= 0.01 # m/h actual snow height on the ground
            # apply snow albedo only to columns where consider_snow_albedo is True
            snow_mask &= consider_snow_albedo
            # replace all ground albedo values at snowy timesteps by snow albedo
            albedo = np.where(snow_mask, self.sim_data["snow_albedo"], albedo)

        # final sanity check
        if not np.all((albedo > 0) & (albedo < 1)):
            raise ValueError("ground albedo must be >0 and <1 for all locations.")
        
        # now write hourly albedo values to sim_data
        assert albedo.shape == self._sim_shape_ # make sure
        self.sim_data["system_grdalbedo"] = np.asarray(albedo, dtype=float)

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
        if "elevation" not in self.plant_parameters_processed:
            raise AttributeError(f"'elevation' attribute missing in self.plant_parameters_processed, run self.preprocess_elevation() first.")
        if "surface_pressure" not in self.sim_data:
            raise AttributeError("'surface_pressure' must be read in first via wfm.read()")
        if "surface_air_temperature" not in self.sim_data:
            raise AttributeError("'surface_air_temperature' must be read in first via wfm.read()")
        
        solar_position_library = dict()

        # pd.DataFrame(np.nan, index=self.time_index, columns=self.locs)
        self.sim_data["solar_azimuth"] = np.full_like(self.sim_data["surface_pressure"], np.nan)
        # pd.DataFrame(np.nan, index=self.time_index, columns=self.locs)
        self.sim_data["apparent_solar_zenith"] = np.full_like(self.sim_data["surface_pressure"], np.nan)
        # self.sim_data['apparent_solar_elevation'] = np.full_like(self.sim_data['surface_pressure'], np.nan)  # pd.DataFrame(np.nan, index=self.time_index, columns=self.locs) #TODO delete?

        # get and round values
        lats = np.round(self.locs.lats, lat_rounding)
        lons = np.round(self.locs.lons, lon_rounding)
        elevs = np.round(self.plant_parameters_processed["elevation"], elev_rounding)

        for loc, (lat, lon, elev) in enumerate(zip(lats, lons, elevs)):
            key = (lon, lat, elev)
            if key in solar_position_library:
                _solpos_ = solar_position_library[key] 
            else:
                # make sure that no input is nan to avoid very hard-to-understand errors later on
                _req = [
                    self.time_index,
                    lat,
                    lon,
                    elev,
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
                    latitude=lat,
                    longitude=lon,
                    altitude=elev,
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
        
        # variables which are connected between timesteps or describe non-solar parameters (e.g. for snow cover or slow-changing core temperatures) will be needed as a FULL timeseries
        # save them to an additional "sim_data_full" container before clipping the nighttime hours
        self.sim_data_full = OrderedDict()
        self._sim_shape_full_ = self._sim_shape_
        self._time_index_full_ = self.time_index
        for key in [
            "snow_depth_water_equivalent",
            "snow_density",
            "snowfall_water_equivalent",
            "surface_air_temperature", # needed for nighttime snow melt
        ]:
            if key in self.sim_data:
                val = self.sim_data[key]
                assert val.shape == self._sim_shape_full_ # make sure
                self.sim_data_full[key] = val 
        
        # then clip it only to "light hours", described by _time_sel_ mask
        self._time_sel_ = (self.sim_data["apparent_solar_zenith"] < 95).any(axis=1) # allow a few degrees below horizon to account for diffuse radiation after sunset and possibly slightly negative horizon angles
        for key in self.sim_data.keys():
            self.sim_data[key] = self.sim_data[key][self._time_sel_, :]
        # also update the time index and sim data shape
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
        dni_extra = pvlib.irradiance.get_extra_radiation(self._time_index_, **kwargs).values

        shape = len(self._time_index_), self.locs.count
        self.sim_data["extra_terrestrial_irradiance"] = np.broadcast_to(dni_extra.reshape((shape[0], 1)), shape)

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

        # 29 because that what the function seems to max out at as zenith approaches 90
        self.sim_data["air_mass"] = np.full_like(self.sim_data["apparent_solar_zenith"], 29)

        s = self.sim_data["apparent_solar_zenith"] < 90
        self.sim_data["air_mass"][s] = pvlib.atmosphere.get_relative_airmass(
            self.sim_data["apparent_solar_zenith"][s], model=model
        )

        return self


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
        td = self.sim_data["surface_dew_temperature"].flatten() if use_dew_temperature else None
        times = pd.DatetimeIndex(np.column_stack([self._time_index_ for x in range(self._sim_shape_[1])]).flatten())

        self.sim_data["direct_normal_irradiance"] = (
            pvlib.irradiance.dirint(ghi=g, solar_zenith=z, times=times, pressure=p, temp_dew=td)
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
        if "direct_horizontal_irradiance" not in self.sim_data:
            raise AttributeError("'direct_horizontal_irradiance' must be read in first via wfm.read()")
        if "apparent_solar_zenith" not in self.sim_data:
            raise AttributeError("'apparent_solar_zenith' must be processed first e.g. via wfm.determine_solar_position()")

        dni_flat = self.sim_data["direct_horizontal_irradiance"]
        zen = np.radians(self.sim_data["apparent_solar_zenith"])

        self.sim_data["direct_normal_irradiance"] = dni_flat / np.maximum(np.cos(zen), 0.2)

        # catch outliners from zero division
        index_out = (dni_flat < 25) & (np.cos(zen) < 0.05)
        self.sim_data["direct_normal_irradiance"][index_out] = 0

        sel = ~np.isfinite(self.sim_data["direct_normal_irradiance"])
        sel = np.logical_or(sel, self.sim_data["direct_normal_irradiance"] < 0)
        sel = np.logical_or(sel, self.sim_data["direct_normal_irradiance"] > 1600)

        self.sim_data["direct_normal_irradiance"][sel] = 0

        return self
    

    ########################
    ### HORIZON HANDLING ###
    ########################


    def preprocess_horizon_profile(
        self,
        distant_horizon_profile : str | Iterable | None,
        azimuthal_stepsize : float = 3.0,
        min_sampling_points : int = 12,
        **kwargs,
    ):
        """
        This convenience function calculates the horizon profiles based on the inputs for
        'distant_horizon_profile' as well as local horizons based on local 'hill_slopes' 
        and 'slope_azimuths'. It preprocessed the variables internally and combines them
        into the maximum relevant horizon angles per azimuthal view axis. The results are
        stored as 'horizon_profile' in self.plant_parameters_processed.

        Parameters
        ----------
        azimuthal_stepsize : float, optional
            The even angular spacing between azimuthal view axes, by default 3.0
            (i.e. 120 sampling points on a 360° round view)
        **kwargs
            Passed on to self.preprocess_distant_horizon_profile()

        Returns
        -------
        obj
            a reference to the invoking SolarWorkflowManager object.
        """
        # DISTANT HORIZON

        # save raw input
        self.plant_parameters_raw["distant_horizon_profile"] = distant_horizon_profile

        # first format the distant horizon profiles input
        distant_horizon_profile, filepaths = self._preprocess_variable(
            varname = "distant_horizon_profile",
            value = distant_horizon_profile,
            allow_none = True,
            replace_none = 0, # note that this replaces None etc. by fallback value already
            assert_type = [int, float, str],
            force_cols = None,
            force_dims = None,
        )
        # then process and get the values
        self.preprocess_distant_horizon_profile(
            distant_horizon_profile = distant_horizon_profile,
            azimuthal_stepsize = azimuthal_stepsize,
            min_sampling_points = min_sampling_points,
            **kwargs,
        )
        distant_horizon_profile = self.plant_parameters_processed["distant_horizon_profile"]

        # LOCAL HORIZON

        if "local_horizon_profile" not in self.plant_parameters_processed:
            # we need to first process it
            self.calculate_local_horizon_based_on_hillslope(
                azimuthal_stepsize = azimuthal_stepsize, 
                allow_negative = True)
        local_horizon_profile = self.plant_parameters_processed["local_horizon_profile"]

        # combine the two, always the maximum angle is the defining one blocking the sun
        assert distant_horizon_profile.shape == local_horizon_profile.shape, \
            f"Shape mismatch: distant_horizon_profile {distant_horizon_profile.shape} != local_horizon_profile {local_horizon_profile.shape}"
        horizon_profile = np.maximum(distant_horizon_profile, local_horizon_profile)

        # write to preprocessed params
        self.plant_parameters_processed["horizon_profile"] = np.asarray(horizon_profile, dtype=float)

        return self


    def preprocess_distant_horizon_profile(
        self,
        distant_horizon_profile : float | int | None | Iterable,
        azimuthal_stepsize = 3.0,
        min_sampling_points = 12,
        **kwargs,
    ):
        """
        Preprocesses the raw distant_horizon_profile inputs such that one distant horizon
        profile per placement with horizon angles over flat horizon is written into
        self.plant_parameters_processed. Checks numeric profiles, sets None to flat 
        profiles and calculates complex profiles from location and DEM filepaths where given.

        Parameters
        ----------
        distant_horizon_profile : str | None | Iterable
            The distant horizon profile as positive angles above flat horizon for evenly
            spaced azimuthal view angles, per every location. If a str per location is 
            given, a single filepath to a DEM is expected from which the distant horizon 
            profile will be calculated.
        azimuthal_stepsize : float, optional
            The even spacing between azimuthal view axes/sampling angles of the horizon 
            profile, by default 3.0 [°] (i.e. 120 sampling points for 360° view)
        min_sampling_points : int, optional
            The minn. number of azimuthal angles/sampling points in the horizon profile, 
            by default 12 (i.e. resolution not coarser than 30°)
        **kwargs
            Will be passed on to self._calculate_distant_horizon_profile()

        Returns
        -------
        obj
            a reference to the invoking SolarWorkflowManager object.
        """
        # first process the distant_horizon_profile as a copy of the input data
        distant_horizon_profile = np.asarray(distant_horizon_profile, dtype=object).copy()
        # must be either a single horizon profile array with e.g. None or one filepath per placement (1D) or one profile per placement (2D)
        if distant_horizon_profile.size == 1: # scalar
            if distant_horizon_profile.item() == 0:
                distant_horizon_profile = None
            distant_horizon_profile = np.full((self.locs.count, 1), distant_horizon_profile, dtype=object) # always use 2D view
        elif distant_horizon_profile.ndim == 1:
            # first check if this is given as value per location
            is_none = self._is_none(distant_horizon_profile)
            is_str = np.fromiter(
                (isinstance(value, str) for value in distant_horizon_profile),
                dtype=bool,
                count=distant_horizon_profile.size,
            )
            is_zero = (distant_horizon_profile == 0)
            if distant_horizon_profile.size == self.locs.count and np.all(is_none | is_str | is_zero):
                # first replace a single scalar zero per location (spacer for no profile) with a None to align later processing
                distant_horizon_profile[is_zero] = None
                # we have one value per location which is either str or None/zero (float/int makes no sense for a profile per location, should be None = zero or a DEM path)
                distant_horizon_profile = distant_horizon_profile[:, None] # always use 2D view
            else:
                # 1D input is interpreted as one angular horizon profile
                distant_horizon_profile = np.broadcast_to(
                    distant_horizon_profile[None, :],
                    (self.locs.count, distant_horizon_profile.size)
                ).copy()
        elif distant_horizon_profile.ndim == 2:
            if distant_horizon_profile.shape[0] != self.locs.count:
                raise ValueError(
                    f"The number of distant horizon profiles (first dimension of distant_horizon_profile: {distant_horizon_profile.shape[0]}) does not match the number of locations ({self.locs.count})."
                )
        else:
            raise ValueError(
                "'distant_horizon_profile' must be scalar or a 1D or 2D array."
            )
        
        # we have 3 different types of potential horizon info, assign them to 3 separate masks for processing:
        # None (no horizon = all zero), numeric iterable of actual horizon angles and a str formatted path to a digital elevation model raster
        n_rows = distant_horizon_profile.shape[0]
        num_mask = np.zeros(n_rows, dtype=bool)
        none_mask = np.zeros(n_rows, dtype=bool)
        str_mask = np.zeros(n_rows, dtype=bool)

        # store the single filepath/string associated with each string row
        str_values = np.empty(n_rows, dtype=object)
        str_values[:] = None

        for i, row in enumerate(distant_horizon_profile):
            # first check if entries are None
            row_none_mask = np.fromiter(
                (value is None for value in row),
                dtype=bool,
                count=row.size,
            )
            # set None mask to True for this row only if the whole row is None, then continue.
            if row_none_mask.all():
                none_mask[i] = True
                continue
            
            # else try strings
            row_str_mask = np.fromiter(
                (isinstance(value, str) for value in row),
                dtype=bool,
                count=row.size,
            )
            # same here: set str mask to True for this row only if the whole row is str, here also make sure the str are all the SAME per row! Then continue.
            if row_str_mask.all():
                if not np.all(row == row[0]):
                    raise ValueError(
                        f"distant_horizon_profile for placement #{i} contains multiple different strings."
                    )
                str_mask[i] = True
                str_values[i] = row[0] # additionally populate the array with single strings per row
                continue

            # then try numeric values
            row_num_mask = np.fromiter(
                (isinstance(value, (int, float, np.integer, np.floating))
                and not isinstance(value, (bool, np.bool_))
                for value in row),
                dtype=bool,
                count=row.size,
            )
            # set numeric mask to True for this row only if the whole row is numeric
            if row_num_mask.all():
                numeric_row = np.asarray(row, dtype=float)
                if not np.isfinite(numeric_row).all():
                    raise ValueError(
                        f"distant_horizon_profile for placement #{i} contains NaN or infinite values."
                    )
                num_mask[i] = True
                continue

            # must not contain rows that mix None or strings with other values
            if row_none_mask.any() or row_str_mask.any():
                raise ValueError(
                    f"distant_horizon_profile for placement #{i} mixes numeric values, None, and/or strings."
                )
            # cover possibly unforeseen value issues
            raise ValueError(
                f"distant_horizon_profile for placement #{i} contains unsupported non-numeric values: {row}."
            )     
        
        # deal with the azimuthal sampling angles
        if np.any(num_mask):
            # we do have actual exogenous distant horizon profiles given
            # assume 360° round view and overwite step size resulting from number of sampling points
            azimuthal_stepsize = float(360 / distant_horizon_profile.shape[1])
            if distant_horizon_profile.shape[1] < min_sampling_points:
                # there is no point in a far distant horizon "profile" with such coarse segments
                raise ValueError(
                    f"distant_horizon_profile is given as float iterable for at least one placement but has only {distant_horizon_profile.shape[1]} sampling points resulting in coarse {azimuthal_stepsize}° azimuthal resolution. Min. required sampling points are {min_sampling_points}.")
        # in any case generate an evenly spaced array from 0 to < 360 with the azimuthal stepsize
        azimuths = np.arange(0, 360, azimuthal_stepsize)

        # then make sure that distant_horizon_profile has one column per azimuthal sampling point
        if distant_horizon_profile.shape[1] == 1:
            distant_horizon_profile = np.repeat(
                distant_horizon_profile,
                len(azimuths),
                axis=1,
            )
        elif distant_horizon_profile.shape[1] != len(azimuths):
            raise ValueError(f"Length of distant horizon profile sampling points ({distant_horizon_profile.shape[1]}) does not match the number of azimuths ({len(azimuths)}).")
            
        # now deal with the 3 cases (numeric profiles, None, str) separately

        # 1) numeric profiles need to be checked only
        _num_array = np.asarray(distant_horizon_profile[num_mask], dtype=float)
        if np.any(_num_array < -3) or np.any(_num_array >= 90): # -3° is below even the most extreme angle that could occur on Mt. Everest
            raise ValueError("numeric distant_horizon_profile contains values < -3° or >= 90°.")

        # 2) None values stand for "no horizon considered" ergo set zero per every None row = all-zero profiles
        distant_horizon_profile[none_mask] = 0.0

        # 3) str values must be DEM filepaths, extract sampling points and calculate horizon profiles from them
        if np.sum(str_mask) > 0:
            # extract the profiles from at least one given DEM path
            args = { # default arguments
                "max_distance" : 10000,
                "distance_stepsize" : 30,
                "exp_spacing_factor" : 1.01,
                "out_of_bounds_tol" : 0,
            }
            args.update(**kwargs) # allow passing user kwargs
            distant_horizon_profile[str_mask] = self._calculate_distant_horizon_profile(
                digital_surface_model_paths = str_values[str_mask], # pass one dem filepath per location with strs
                lons = self.locs.lons[str_mask],
                lats = self.locs.lats[str_mask],
                angle_stepsize = azimuthal_stepsize,
                **args
            )

        self.plant_parameters_processed["distant_horizon_profile"] = np.asarray(distant_horizon_profile, dtype=float)

        return self
    

    def _calculate_distant_horizon_profile(
        self,
        digital_surface_model_paths: Iterable[str],
        lons: Iterable[float],
        lats: Iterable[float],
        angle_stepsize: float = 3.0,
        max_distance: int = 10000,
        distance_stepsize: int = 30,
        exp_spacing_factor: float = 1.01,
        out_of_bounds_tol : int = 0, #TODO 100 for the current run
    ):
        """
        Returns the distant horizon profile based on a digital elevation model raster 
        as an iterable of horizon angles for one or multiple locations. Azimuthal 
        sampling rate and reach/distance can be adapted. 

        Parameters
        ----------
        digital_surface_model_paths : Iterable[str]
            The paths to the digital elevation models that shall be used to
            extract the elevation of the horizon features, typically DSMs incl.
            tree, building etc. feature heights. Each path belongs to the location
            at the same position in lons and lats.
        lons : Iterable[float]
            Longitudes of the observer locations.
        lats : Iterable[float]
            Latitudes of the observer locations.
        angle_stepsize : float, optional
            The azimuthal angle steps for the view direction sampling points of 
            the horizont profile, by default every 3°.
        max_distance : int, optional
            The maximum distance in meters up to which horizon features will be 
            considered for the profile. By default 10 000 [m].
        distance_stepsize : int, optional
            The initial distance sampling step in every direction, i.e. every x 
            meters, an elevation value will be extracted. The resulting step 
            width beyond the first cell can be constant or non-linear, depending
            on exp_spacing_factor. By default 30 [m].
        exp_spacing_factor : float, optional
            Leads to a constant distance step size if 1.0, else the step size 
            depends on the distance to the location non-linearly as follows:
            s(n)= s_0**^(k**n), k=exp_spacing_factor and s_0=distance_stepsize
            By default k = 1.01.
        out_of_bounds_tol : int, optional
            The accepted number of pixels that a lat/lon can be out of bounds of
            the DEM raster so that still a NaN value will be returned for this 
            angle. Beyond this pixel tolerance it will fail.

        Returns
        -------
        np.ndarray
            A 2D-array with length of placements dataframe and width equal to the 
            number of angular sampling points, with horizon angles per view axis
            in degrees over flat horizon.
        """
        digital_surface_model_paths = list(digital_surface_model_paths)
        lons = list(lons)
        lats = list(lats)

        assert len(digital_surface_model_paths) == len(lons) == len(lats), \
            "digital_surface_model_paths ({len(digital_surface_model_paths)}), lons ({len(lons)}) and lats ({len(lats)}) must have identical lengths."
        assert all([isinstance(x, str) for x in digital_surface_model_paths]),\
            "digital_surface_model_paths iterable must only contain str formatted values"
        assert all([isfile(x) for x in digital_surface_model_paths]),\
            "All values of digital_surface_model_paths must be existing filepaths."

        # define a lines of view array adding up to 360° around the location
        azimuths = np.arange(0, 360, angle_stepsize)
        # iterate over all locations and generate horizon profiles for the azimuths
        horizons = [] # initialize collector for all locational profiles
        old_str = None # save the last DEM filepath to save the time for loading it again in case that the file remains the same
        for dsm_path, lat, lon in zip(digital_surface_model_paths, lats, lons):
            lats = np.atleast_1d(lat)
            lons = np.atleast_1d(lon)

            # load DEM only if necessary, i.e. only when new filepath
            if dsm_path != old_str:
                old_str = dsm_path
                with rasterio.open(dsm_path) as src: #TODO replace by osgeo approach to avoid rasterio import
                    dem = src.read(1)
                    transform = src.transform
                    crs = src.crs
                    if crs.to_epsg() != 4326:
                        raise ValueError("DEM must be in EPSG:4326")

            # preprocess dem data
            nrows, ncols = dem.shape
            res_lon = transform.a
            res_lat = -transform.e
            xmin = transform.c
            ymax = transform.f

            # get cell row/col ids for all locations
            def get_cell_id(lon_arr, lat_arr):
                cols = ((lon_arr - xmin) / res_lon).astype(int)
                rows = ((ymax - lat_arr) / res_lat).astype(int)
                return rows, cols
            r0, c0 = get_cell_id(lons, lats)

            # calculate if/by how many pixels the location exceeds the raster bounds
            exceeds = np.maximum.reduce([
                np.maximum(-r0, 0),              # top
                np.maximum(r0 - (nrows - 1), 0), # bottom
                np.maximum(-c0, 0),              # left
                np.maximum(c0 - (ncols - 1), 0)  # right
            ]).astype(int)

            # sometimes, the lat/lon is only SLIGHTLY out of bounds, in such cases allow returning NaNs - else fail based on out_of_bounds_tol
            if np.any(exceeds > out_of_bounds_tol):
                # this exceeds the tolerances, too far away - fail!
                raise IndexError(
                    f"Observer location is outside DEM by more than out_of_bounds_tol={out_of_bounds_tol} pixel, here {max(exceeds)} pixel/s."
                )
            elif np.any(exceeds > 0):
                # we are only < out_of_bounds_tol pixel away from the raster, move the location "inwards" to the outmost pixel
                print(f"NOTE: Location was {max(exceeds)} pixels out of bounds, but below tolerance = {out_of_bounds_tol} pixels. Location shifted inwards slightly.")
                r0 = np.clip(r0, 0, nrows-1)
                c0 = np.clip(c0, 0, ncols-1)

            # get the plant elevations (zero distance)
            elev0 = dem[r0, c0]
            
            # create a distance spacing array based on step and exponential growth factor
            distances = [distance_stepsize, distance_stepsize + distance_stepsize**exp_spacing_factor]
            while distances[-1] < max_distance:
                # growth linearly for exp_spacing_factor==1, else exponentially
                distances.append(distances[-1]+(distances[-1]-distances[-2])**exp_spacing_factor)
            distances = np.array(distances)

            # get the radians of the azimuths
            az_rad = np.radians(azimuths)[:, None]
            
            # dx/dy for all azimuths & distances
            d = distances[None, :]
            dx = d * np.sin(az_rad)
            dy = d * np.cos(az_rad)

            # convert meters to degrees (depends on observer lat!)
            meters_per_deg_lat = 111320
            meters_per_deg_lon = 111320 * np.cos(np.radians(lats))

            # reshape for broadcasting
            lats_r = lats[:, None, None]
            lons_r = lons[:, None, None]
            mpd_lat = meters_per_deg_lat
            mpd_lon = meters_per_deg_lon[:, None, None]
            dx = dx[None, :, :]
            dy = dy[None, :, :]

            # compute sampling points along the rays for all observers
            dlat = dy / mpd_lat # mpd_lat is scalar
            dlon = dx / mpd_lon # mpd_lon is an array, depends on lat
            lat_pts = lats_r + dlat
            lon_pts = lons_r + dlon

            # extract the elevation values for all of these points
            rows, cols = get_cell_id(lon_pts, lat_pts)

            # mask out-of-bounds
            in_bounds = (
                (rows >= 0) & (rows < nrows) &
                (cols >= 0) & (cols < ncols)
            )

            # flatten for indexing
            flat_rows = rows.ravel()
            flat_cols = cols.ravel()
            flat_mask = in_bounds.ravel()

            elev_flat = np.full(flat_rows.size, np.nan)
            valid_idx = np.where(flat_mask)[0]
            elev_flat[valid_idx] = dem[flat_rows[valid_idx], flat_cols[valid_idx]]

            elev_sampled = elev_flat.reshape(rows.shape)

            # calculate horizon angles
            elev_diff = elev_sampled - elev0[:, None, None]
            angles = np.degrees(np.arctan2(elev_diff, d))
            angles[~in_bounds] = -np.inf
            horizon = np.nanmax(angles, axis=2)

            # append profile to the overall list with all locations
            horizons.append(horizon)

        # recombine different locational profiles and set as attribute
        distant_horizon_profile = np.vstack(horizons, dtype=float)

        return distant_horizon_profile


    def preprocess_hill_slope_and_azimuth(
            self, 
            north_slope : int | float | str | Iterable | None,
            east_slope : int | float | str | Iterable | None,
            general_slope : int | float | str | Iterable | None = None,
            downhill_azimuth : int | float | str | Iterable | None = None,
            slopes_fallback : int | float | None = None,
            ):
        """
        Will process the "hill_slopes" and "slope_azimuths" inputs stored in 
        self.plant_parameters_raw. None values willbe aligned and set to zero,
        references to digital elevation model raster files will be opened and
        plant-specific data will be extracted. The parameters will be overwritten
        with all numeric float arrays.

        NOTE: North+East slope as well as overall slope + azimuth fully define a
        hill so only one of the pairs may be provided, the other will be calculated. 

        Parameters
        ----------
        north_slope : int | float | Iterable | None
            The slope angle in degrees between slope and horiztonal along an 
            axis from South to North, i.e. the downhill side of positive angle 
            north_slopes is facing North. None etc. values will be replaced by 
            slopes_fallback if the latter is not None. Slope per placement
            can be provided as an iterable of int or float.
        east_slope : int | float | Iterable | None
            The slope angle in degrees between slope and horiztonal along an 
            axis from West to East, i.e. the downhill side of positive angle 
            east_slopes is facing East. None etc. values will be replaced by 
            slopes_fallback if the latter is not None. Slope per placement
            can be provided as an iterable of int or float.
        general_slope : int | float | Iterable | None, optional
            The maximum slope angle in degrees between slope and horiztonal 
            along the downhill axis pointing in downhill_azimuth direction. 
            None etc. values will be replaced by slopes_fallback if the latter 
            is not None. Slope per placement can be provided as an iterable 
            of int or float. By default None.
        downhill_azimuth : int | float | Iterable | None, optional
            The direction of the downhill axis clockwise from North = 0°.
            By default None.
        slopes_fallback : int | float | None, optional
            Missing slope values (not azimuth) will be set to this fallback.
            Forces an error in case of missing values if slopes_fallback is
            None. Set to 0 to ignore slope effects in the given direction.
            By default None.

        Returns
        -------
        obj
            a reference to the invoking SolarWorkflowManager object.
        """
        # assert that the input combinations make sense
        assert sum([x is not None for x in [north_slope, east_slope, general_slope, downhill_azimuth]]) in [0,2],\
            "If not all inputs are None, either north_slope + east_slope OR general_slope + downhill_azimuzth must be given, the others must then be None."
        
        # first save the inputs to raw data
        self.plant_parameters_raw["north_slope"] = north_slope
        self.plant_parameters_raw["east_slope"] = east_slope
        self.plant_parameters_raw["general_slope"] = general_slope
        self.plant_parameters_raw["downhill_azimuth"] = downhill_azimuth

        # then format and check all params
        def _preprocess_data(varname, invals, fallback):
            # first general formatting
            vals, filepaths = self._preprocess_variable(
                varname = varname,
                value = invals,
                allow_none = fallback is not None, 
                replace_none = fallback,
                assert_type = [float, int],
                force_cols = 1,
                force_dims = 1, # one scalar value per placement -> 1d
            )
            # now deal with potential DEM filepath inputs, identified via string datatype
            # iterate over potentially different filepaths and extract vals for all locs that apply
            fallbacks_applied = 0
            for fp in np.unique(filepaths[~pd.isna(filepaths)]):
                if not isfile(fp):
                    raise FileNotFoundError(f"{varname} contains string entries which must be existing {varname} rasters but cannot find file: {fp}")
                # generate a mask that applies also to the locations
                mask = (filepaths == fp)
                # extract a clipped sub raster covering the affected placements extent only to save time
                iter_ext = gk.Extent.fromLocationSet(self.locs[mask])
                clipped_raster = iter_ext.pad(0.5).rasterMosaic(fp)
                if clipped_raster is None:
                    iter_vals = np.array([np.nan] * len(self.locs[mask]))
                else:
                    # use average mode over neighboring cells by default to void overestimating due to very local steep angles
                    iter_vals = gk.raster.interpolateValues(
                        clipped_raster, 
                        self.locs[mask],
                        mode = "average",
                        )
                    if np.isnan(iter_vals).any():
                        # if getting values fails, it could be because of interpolation method
                        # replace only those that failed by 'near' interpolation
                        iter_vals_near = gk.raster.interpolateValues(
                            clipped_raster, self.locs[mask], mode="near"
                        )
                        iter_vals[np.isnan(iter_vals)] = iter_vals_near[np.isnan(iter_vals)]
                if np.isnan(iter_vals).any():
                    # we still have nans
                    if fallback is None:
                        # must fail then
                        raise ValueError(f"NaN values extracted for '{varname}' and fallback is None.")
                    else:
                        # if allowed, replace nans by fallback value 
                        fallbacks_applied += np.isnan(iter_vals).sum()
                        iter_vals[np.isnan(iter_vals)] = (
                            np.ones(shape=iter_vals.shape) * fallback
                        )[np.isnan(iter_vals)]
                # last set the extracted vals from this iteration in the main vals array
                vals[mask] = iter_vals
            
            # warn of unavailable data if required
            if fallbacks_applied > 0:
                warnings.warn(f"{varname} for {fallbacks_applied} out of {len(self.locs)} placements could not be extracted from provided {varname} files and has been replaced with fallback value {fallback}.")
                
            return np.asarray(vals, dtype=float) # float is needed for later angle processing

        if north_slope is not None or east_slope is not None:
            if east_slope is None or north_slope is None:
                raise ValueError("When north_slope or east_slope is not None, east_slope and north_slope must both be provided.")
            # preprocess the north and east slopes and write to preprocessed params
            north_slope = _preprocess_data(varname="north_slope", invals=north_slope, fallback=slopes_fallback)
            east_slope = _preprocess_data(varname="east_slope", invals=east_slope, fallback=slopes_fallback)
            # general slope and azimuth are then None as asserted above - calculate them from north and east slopes via rad gradients!
            north_gradient = np.tan(np.radians(north_slope))
            east_gradient = np.tan(np.radians(east_slope))
            general_slope = np.degrees(
                np.arctan(np.hypot(north_gradient, east_gradient))
            )
        
        elif general_slope is not None or downhill_azimuth is not None:
            if general_slope is None or downhill_azimuth is None:
                raise ValueError("When general_slope or downhill_azimuth is not None, general_slope and downhill_azimuth must both be provided.")
            # first preprocess general slope and azimuth
            general_slope = _preprocess_data(varname="general_slope", invals=general_slope, fallback=slopes_fallback)
            downhill_azimuth = _preprocess_data(varname="downhill_azimuth", invals=downhill_azimuth, fallback=None)
            # now we only have the overall slope and orientation, but not the slope in North and East direction - invert above calculation
            general_gradient = np.tan(np.radians(general_slope))
            azimuth_radians = np.radians(downhill_azimuth)
            north_gradient = general_gradient * np.cos(azimuth_radians)
            east_gradient = general_gradient * np.sin(azimuth_radians)
            north_slope = np.degrees(np.arctan(north_gradient))
            east_slope = np.degrees(np.arctan(east_gradient))
        
        # postprocess azimuths to be in expected value range
        downhill_azimuth = np.degrees(np.arctan2(east_gradient, north_gradient)) % 360

        # write the resulting parameters to the preprocessed storage
        self.plant_parameters_processed["north_slope"] = north_slope
        self.plant_parameters_processed["east_slope"] = east_slope
        self.plant_parameters_processed["general_slope"] = general_slope
        self.plant_parameters_processed["downhill_azimuth"] = downhill_azimuth

        return self


    def calculate_local_horizon_based_on_hillslope(
            self,             
            azimuthal_stepsize : int | float = 3.0, 
            allow_negative : bool = True, 
            ):
        """
        Calculates an elliptical horizon based on the assumption of a plant 
        hillslope plane which is much larger than the plant height, leading to 
        a horizon angle equal to the hillslope angle in the direction of the 
        slope line and horizon angle = 0° crosswise from North = 0°. 
        The resulting profile will be stored "local_horizon_profile" under 
        self.plant_parameters_processed.

        Parameters
        ----------
        azimuthal_stepsize : int | float, optional
            The angular distance between sampling points for a 360° round view
            horizon profile starting from North 0° clockwise, e.g. 3° will lead 
            to 120 sampling points. Should be aligned with the distant horizon 
            sampling points for later combination. By default 3.0°. 
        allow_negative : bool, optional
            The local horizon may lead to significant negative horizon angles 
            towards the downhill direction. That is not realistic in the far 
            distance, so set allow_negative to False to enforce a minimum of 
            zero degrees unless you combine it with a far distant horizon later.
            By default True.

        Returns
        -------
        obj
            a reference to the invoking SolarWorkflowManager object.
        """
        assert isinstance(azimuthal_stepsize, (int, float)), "azimuthal_stepsize must be int or float"

        # get hill slopes and azimuths, preprocess if needed
        if not ("general_slope" in self.plant_parameters_processed and "downhill_azimuth" in self.plant_parameters_processed):
            raise AttributeError("'general_slope' and 'downhill_azimuth' are required, run self.preprocess_hill_slope_and_azimuth() first.", flush=True)
        hill_slopes = np.asarray(self.plant_parameters_processed["general_slope"], dtype=float)
        slope_azimuths = np.asarray(self.plant_parameters_processed["downhill_azimuth"], dtype=float)

        # invert the azimuth to look uphill
        uphill_slope_azimuth = (slope_azimuths  + 180.0) % 360.0
        uphill_slope_azimuth_rad = np.deg2rad(uphill_slope_azimuth)

        # define the sampling points for the azimuth angles
        if not np.isscalar(azimuthal_stepsize):
            raise TypeError("azimuthal_stepsize must be a scalar number.")
        azimuthal_stepsize = float(azimuthal_stepsize)
        if not np.isfinite(azimuthal_stepsize) or azimuthal_stepsize <= 0:
            raise ValueError(
                "azimuthal_stepsize must be finite and greater than zero."
            )
        number_of_steps = 360.0 / azimuthal_stepsize
        if not np.isclose(number_of_steps, round(number_of_steps)):
            raise ValueError(
                f"azimuthal_stepsize={azimuthal_stepsize} does not divide "
                "360 degrees evenly."
            )
        azimuths = np.arange(0, 360, azimuthal_stepsize)
        azimuths_rad = np.deg2rad(azimuths)
        # calculate the local horizon angles mathematically based on uphill slope and direction
        local_horizon_profile = np.rad2deg(
            np.arctan(
                np.tan(np.deg2rad(hill_slopes))[:, None]
                * np.cos(azimuths_rad[None, :] - uphill_slope_azimuth_rad[:, None])
            )
        )
        if not allow_negative:
            # set negative horizon angles to zero to account for the distant horizon unless it is applied explicitly later
            local_horizon_profile = np.maximum(local_horizon_profile, 0.0)
        
        # write as processed attribute
        self.plant_parameters_processed["local_horizon_profile"] = np.asarray(local_horizon_profile, dtype=float)

        return self


    def preprocess_singleaxis_and_crossaxis(
            self,
            singleaxis_azimuth : float | str | Iterable,
            singleaxis_tilt : float | str | Iterable | None = None,
            crossaxis_tilt : float | str | Iterable | None = None,
        ):
        """
        Preprocesses the input for singleaxis_azimuth and defines singleaxis and crossaxis 
        tilts either based on singleaxis_tilt and crossaxis_tilt inputs, or calculates it 
        from general (hill) slope and downhill azimuth, assuming constant distance of the 
        axis to the ground. If hill slope and orientation is not defined either, flat ground
        is assumed and singleaxis tilt and crossaxis tilt will be zero. 

        Parameters
        ----------
        singleaxis_azimuth : float | str | Iterable,
            The orientation of the single-axis PV tracker axis in degrees, clockwise from
            0° North. Can be provided as a scalar or an iterable of one value per location.
            Must be a known convention to reskit.core.system_design.location_to_tracker_axis_azimuth()
            if provided as a str or iterable of strings. 
        singleaxis_tilt : float | str | Iterable | None, optional
            The tilt of the single-axis PV tracker axis in degrees tilted towards singleaxis_azimuth. 
            Can be provided as a scalar or an iterable of one value per location. Must be a known 
            convention to reskit.core.system_design.location_to_tracker_axis_tilt() if provided as a 
            str or iterable of strings. If None is provided, the tilt will be derived from general 
            (hill) slope and downhill azimuth if given, else flat terrain will be assumed. By default 
            None.
        crossaxis_tilt : float | str | Iterable | None, optional
            The tilt of the plant surface perpendicular to the single-axis PV tracker axis in degrees 
            tilted upwards on the right-hand side looking in singleaxis_azimuth direction. 
            Can be provided as a scalar or an iterable of one value per location. Must be a known 
            convention to reskit.core.system_design.location_to_tracker_crossaxis_tilt() if provided 
            as a str or iterable of strings. If None is provided, the cross axis tilt will be derived 
            from general (hill) slope and downhill azimuth if given, else flat terrain will be assumed. 
            By default None.

        Returns
        -------
        obj
            a reference to the invoking SolarWorkflowManager object.
        """
        # first save the inputs to raw data
        self.plant_parameters_raw["singleaxis_azimuth"] = singleaxis_azimuth
        self.plant_parameters_raw["singleaxis_tilt"] = singleaxis_tilt
        self.plant_parameters_raw["crossaxis_tilt"] = crossaxis_tilt

        # singleaxis_azimuth is mandatory in any case
        # process in a try-except to be able to explain the need for not-None values here (can and should be None when fixed tilt)
        try:
            singleaxis_azimuth, saa_conventions = self._preprocess_variable(
                varname = "singleaxis_azimuth",
                value = singleaxis_azimuth,
                allow_none = False,
                replace_none = None,
                assert_type = [float, int],
                force_cols = 1,
                force_dims = 1, # one scalar value per placement -> 1d
                as_dtype = float,
            )
        except ValueError as e:
            if "must not have None values" in str(e):
                raise ValueError(
                    f"{e} This is required when tracking='singleaxis'."
                ) from e
            raise
        
        # fill potential str conventions with numbers
        for conv in np.unique(saa_conventions[~pd.isna(saa_conventions)]):
            # generate a mask that applies also to the locations
            mask = (saa_conventions == conv)
            # get numeric values based on convention
            iter_vals = rk_solar_core.system_design.location_to_tracker_axis_azimuth(
                locs=self.locs[mask], 
                convention=conv,
                )
            if np.isnan(iter_vals).any():
                # we still have nans and no fallback possible
                raise ValueError("NaN values extracted for 'singleaxis_azimuth' and no fallback possible.")
            # last set the extracted values from this iteration in the main value array
            singleaxis_azimuth[mask] = iter_vals

        # now singleaxis tilt and crossaxis tilt
        # both can either be given explicitly or calculated from hill slope and azimuth, else they will be set to zero
        if (not self._is_none(singleaxis_tilt).all()) and (not self._is_none(crossaxis_tilt).all()):
            # we do have at least some data for both, preprocess it analog to above, start with singleaxis_tilt
            singleaxis_tilt, sat_conventions = self._preprocess_variable(
                varname = "singleaxis_tilt",
                value = singleaxis_tilt,
                allow_none = False,
                replace_none = None,
                assert_type = [float, int],
                force_cols = 1,
                force_dims = 1, # one scalar value per placement -> 1d
                as_dtype = float,
            )
            # fill potential str conventions with numbers
            for conv in np.unique(sat_conventions[~pd.isna(sat_conventions)]):
                # generate a mask that applies also to the locations
                mask = (sat_conventions == conv)
                # get numeric values based on convention
                iter_vals = rk_solar_core.system_design.location_to_tracker_axis_tilt(
                    locs=self.locs[mask], 
                    convention=conv,
                    )
                if np.isnan(iter_vals).any():
                    # we still have nans and no fallback possible
                    raise ValueError("NaN values extracted for 'singleaxis_tilt' and no fallback possible.")
                # last set the extracted values from this iteration in the main value array
                singleaxis_tilt[mask] = iter_vals
            
            # the same for crossaxis_tilt
            crossaxis_tilt, cat_conventions = self._preprocess_variable(
                varname = "crossaxis_tilt",
                value = crossaxis_tilt,
                allow_none = False,
                replace_none = None,
                assert_type = [float, int],
                force_cols = 1,
                force_dims = 1, # one scalar value per placement -> 1d
                as_dtype = float,
            )
            # fill potential str conventions with numbers
            for conv in np.unique(cat_conventions[~pd.isna(cat_conventions)]):
                # generate a mask that applies also to the locations
                mask = (cat_conventions == conv)
                # get numeric values based on convention
                iter_vals = rk_solar_core.system_design.location_to_cross_axis_tilt(
                    locs=self.locs[mask], 
                    convention=conv,
                    )
                if np.isnan(iter_vals).any():
                    # we still have nans and no fallback possible
                    raise ValueError("NaN values extracted for 'crossaxis_tilt' and no fallback possible.")
                # last set the extracted values from this iteration in the main value array
                crossaxis_tilt[mask] = iter_vals
        else:
            # at least one of singleaxis_tilt or crossaxis_tilt is all None or equivalent, make sure both are then
            assert self._is_none(singleaxis_tilt).all() and self._is_none(crossaxis_tilt).all(), \
                "Either none or both of 'singleaxis_tilt' and 'crossaxis_tilt' inputs must be provided."
            # we have no explicit information and need to calculate it
            if self._is_none(self.plant_parameters_processed["general_slope"]).all() or self._is_none(self.plant_parameters_processed["downhill_azimuth"]).all():
                # we do not have any information sufficient to calculate the axis tilts, this means tilts are not being set explicitly -> assume horizontal
                print("NOTE: No singleaxis_tilt/crossaxis_tilt information nor hill orientation (general_slope, downhill_azimuth) provided, flat terrain/axes will be assumed.", flush=True)
                singleaxis_tilt = np.zeros_like(singleaxis_azimuth, dtype=np.float16)
                crossaxis_tilt = np.zeros_like(singleaxis_azimuth, dtype=np.float16)
            else:
                # we have the hill slope and orientation, assume constant axis distance to ground and CALCULATE the required axes tilts geometrically
                hill_slopes_rad = np.radians(self.plant_parameters_processed["general_slope"])
                slope_azimuths_rad = np.radians(self.plant_parameters_processed["downhill_azimuth"])
                singleaxis_azimuths_rad = np.radians(singleaxis_azimuth)
                # get then angular delta between downhill and axis orientation and derive resulting axis tilt across the hillface
                delta = singleaxis_azimuths_rad - slope_azimuths_rad
                singleaxis_tilt = np.degrees(
                    np.arctan(np.tan(hill_slopes_rad) * np.cos(delta))
                )
                # get the normal on the hillface
                tan_slope = np.tan(hill_slopes_rad)
                normal = np.stack([
                    tan_slope * np.sin(slope_azimuths_rad),
                    tan_slope * np.cos(slope_azimuths_rad),
                    np.ones_like(tan_slope),
                ], axis=-1)

                # Axis vector pointing toward singleaxis_azimuth
                # Its vertical component is negative when that direction is downhill
                axis = np.stack([
                    np.sin(singleaxis_azimuths_rad),
                    np.cos(singleaxis_azimuths_rad),
                    -tan_slope * np.cos(delta),
                ], axis=-1)

                # Points to the right when looking toward axis_azimuth, positive angles mean "going up towards the right-hand side"
                cross_axis = np.cross(normal, axis)

                # Positive means rising toward the right-hand side
                crossaxis_tilt = np.degrees(
                    np.arctan2(
                        cross_axis[..., 2],
                        np.hypot(cross_axis[..., 0], cross_axis[..., 1]),
                    )
                )
        
        # write all parameters into the storage container
        self.plant_parameters_processed["singleaxis_azimuth"] = singleaxis_azimuth
        self.plant_parameters_processed["singleaxis_tilt"] = singleaxis_tilt
        self.plant_parameters_processed["crossaxis_tilt"] = crossaxis_tilt

        return self


    def permit_single_axis_tracking(self):
        """
        Permits single axis tracking in the simulation using the 
        pvlib.tracking.singleaxis() function [1].

        NOTE: permit_single_axis_tracking() may update/invert the values in
        self.plant_parameters_processed['singleaxis_azimuth'] if negative.

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
        assert self.tracking == "singleaxis", \
            f"tracking flag must be 'singleaxis' for permit_single_axis_tracking() but is instead: {self.tracking}"
        
        assert "apparent_solar_zenith" in self.sim_data
        assert "solar_azimuth" in self.sim_data

        assert "singleaxis_azimuth" in self.plant_parameters_processed, \
            "singleaxis_azimuth not in self.plant_parameters_processed, run preprocess_singleaxis_and_crossaxis() first."
        assert "singleaxis_tilt" in self.plant_parameters_processed, \
            "singleaxis_tilt not in self.plant_parameters_processed, run preprocess_singleaxis_and_crossaxis() first."
        assert "crossaxis_tilt" in self.plant_parameters_processed, \
            "crossaxis_tilt not in self.plant_parameters_processed, run preprocess_singleaxis_and_crossaxis() first."
        assert "gcr" in self.plant_parameters_processed, \
            "gcr not in self.plant_parameters_processed, run preprocess_ground_coverage_ratio() first."
        assert "backtracking" in self.plant_parameters_processed, \
            "backtracking not in self.plant_parameters_processed, run preprocess_backtracking() first."
        assert "max_tracking_angle" in self.plant_parameters_processed, \
            "max_tracking_angle not in self.plant_parameters_processed, run preprocess_tracking_angle() first."
        
        # create a container for hourly module tilts and azimuths
        system_modtilt = np.empty(self._sim_shape_)
        system_modazimuth = np.empty(self._sim_shape_)
        system_axazimuth = np.empty(self._sim_shape_[1]) # expected as 1d by pvfactors_timeseries

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for i, (_axazimuth, _axtilt, _caxtilt, _gcr, _backtrack, _max_angle) in enumerate(zip(
                self.plant_parameters_processed["singleaxis_azimuth"], 
                self.plant_parameters_processed["singleaxis_tilt"], 
                self.plant_parameters_processed["crossaxis_tilt"],
                self.plant_parameters_processed["gcr"],
                self.plant_parameters_processed["backtracking"],
                self.plant_parameters_processed["max_tracking_angle"],
                )):
            # for i in range(self.locs.count):
                # placement = self.placements.iloc[i] #TODO remove

                # invert the axis azimuth (and hence axis and cross axis tilt) when axis tilt is negative, pvlib.tracking.singleaxis cannot deal with it
                _axazimuth = _axazimuth if _axtilt >= 0 else (_axazimuth + 180) % 360
                _caxtilt = _caxtilt if _axtilt >= 0 else -_caxtilt
                _axtilt = _axtilt if _axtilt >= 0 else -_axtilt
                # calculate the optimal tracking orientations
                tmp = pvlib.tracking.singleaxis(
                    # zenith is defined as angle from vertical and >90° is impossible, set maximum of 90° to avoid nans
                    apparent_zenith = pd.Series(
                        np.where(self.sim_data["apparent_solar_zenith"][:, i]<=90, self.sim_data["apparent_solar_zenith"][:, i], 90),
                        index=self._time_index_,
                    ),
                    apparent_azimuth=pd.Series(
                        self.sim_data["solar_azimuth"][:, i], index=self._time_index_
                    ),
                    axis_tilt=_axtilt,
                    axis_azimuth=_axazimuth,
                    max_angle=_max_angle,
                    backtrack=_backtrack,
                    gcr=_gcr,
                    cross_axis_tilt=_caxtilt,
                )

                # later simulation yields errors when fed with negative tilts, make sure that everything went as expected
                if (tmp["surface_tilt"] < -1e-9).any(): # with numeric tolerance
                    raise ValueError(
                        "pvlib.tracking.singleaxis returned negative surface_tilt."
                    )

                system_modtilt[:, i] = tmp["surface_tilt"].values
                system_modazimuth[:, i] = tmp["surface_azimuth"].values
                system_axazimuth[i] = _axazimuth # the axis azimuth has possibly been inverted above, store it as well

                assert not np.isnan(system_modtilt[:, i]).any()
                assert not np.isnan(system_modazimuth[:, i]).any()
                assert not np.isnan(system_axazimuth[i])

        # store the results in sim_data
        self.sim_data["system_modtilt"] = system_modtilt
        self.sim_data["system_modazimuth"] = system_modazimuth
        # also add the axis azimuth (time-invariant for single-axis tracking) formatted as timeseries over location shape, with possibly inverted values
        self.plant_parameters_processed["system_axazimuth"] = system_axazimuth
        
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
        modazimuths = self.sim_data.get("system_modazimuth", self.plant_parameters_processed["module_azimuth"])
        modtilts = self.sim_data.get("system_modtilt", self.plant_parameters_processed["module_tilt"])

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
        assert "system_grdalbedo" in self.sim_data, f"'system_grdalbedo' is expected in sim_data, run preprocess_ground_albedo() first."

        def _set_total_irradiance_per_side(front=True):
            """Calculates and sets to self.sim_data the POA global and its components for front or backside"""
            # get system ground albedos and tilts and azimuths for the module surfaces
            _grdalbedos = self.sim_data.get("system_grdalbedo")
            _modtilts = self.sim_data.get("system_modtilt", self.plant_parameters_processed["module_tilt"])
            _modazimuths = self.sim_data.get("system_modazimuth", self.plant_parameters_processed["module_azimuth"])
            
            # always consider all locations for the frontside but only those that have a bifaciality factor > 0 for the backside (save time calculating only what is really needed)
            bifaciality_mask = np.ones(self.locs.count, dtype=bool) if front else (self.plant_parameters_processed["bifaciality_factor" ] != 0)

            _poa = pvlib.irradiance.get_total_irradiance(
                surface_tilt=(_modtilts if front else 180-_modtilts)[..., bifaciality_mask],
                surface_azimuth=_modazimuths[..., bifaciality_mask] if front else ((_modazimuths + 180) % 360)[..., bifaciality_mask],
                solar_zenith=self.sim_data["apparent_solar_zenith"][..., bifaciality_mask],
                solar_azimuth=self.sim_data["solar_azimuth"][..., bifaciality_mask],
                dni=self.sim_data["direct_normal_irradiance"][..., bifaciality_mask],
                ghi=self.sim_data["global_horizontal_irradiance"][..., bifaciality_mask],
                dhi=self.sim_data["diffuse_horizontal_irradiance"][..., bifaciality_mask],
                dni_extra=self.sim_data["extra_terrestrial_irradiance"][..., bifaciality_mask],
                airmass=self.sim_data["air_mass"][..., bifaciality_mask],
                albedo=_grdalbedos[..., bifaciality_mask],
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
                # Initialize all locations with zero, then fill only processed locations
                self.sim_data[_key] = np.zeros(self._sim_shape_, dtype=np.asarray(tmp).dtype) 
                self.sim_data[_key][..., bifaciality_mask] = np.where(sel_bad_poa, 0, tmp)
            return
        
        # first do frontside, always
        _set_total_irradiance_per_side(front=True)
        # then do backside only if needed
        if not (self.plant_parameters_processed["bifaciality_factor"] == 0).all():
            # we have at least one bifacial module location, look at backside irradiance, too
            _set_total_irradiance_per_side(front=False)

        return self


    def estimate_snow_coverage_loss(
            self, 
            num_strings=None, 
            format: str = "portrait", 
            self_cleaning: bool = True, 
            threshold_snowfall : float = 1.0,
            consider_snow_loss : bool | Iterable = True,
            ):
        """
        Calculates and sets as attribute a timeseries of partial module snow 
        cover area ratio, for every timestep and location separately. Estimates 
        snowfall rate in cm per timestep based on empirical equation by Anderson 
        [1].

        Parameters
        ----------
        num_strings: int, optional
            The number of parallel cell strings atop of each other along slant 
            height axis. Will be extracted from self.module["N_p"] if not given 
            and format is 'landscape', else 1. By default None.
        format: str, optional
            Either 'portrait' or 'landscape'. Takes effect only if num_strings 
            is None, then uses the number of parallel strings from module "N_p" 
            parameter as num_strings when format is landscape. By default 
            portrait.
        self_cleaning : bool | np.ndarray, optional
            If True, the modules will be assumed to be tilted to the maximum 
            possible angle to facilitate snow shedding. Will not take effect for
            fixed tilt systems. By default True.
        threshold_snowfall : float, optional
            The snowfall rate threshold in cm/h below which no snow coverage
            is assumed. By default 1.0 cm/h (default in pvlib.snow.coverage_nrel).
        consider_snow_loss : bool | Iterable, optional
            If snow cover losses shall be considered. Can be given as a scalar or 
            per location. By default True.

        Returns
        -------
        obj
            a reference to the invoking SolarWorkflowManager object, with added
            "partial_snowcov" sim_data attribute.

        References
        ----------
        [1] Anderson, Eric A. (1976). A  point energy and mass balance model of a snow cover.
        """
        if not np.any(np.asarray(consider_snow_loss)):
            # no location is affected, set snow coverage to zero for all locs and timesteps
            partial_snowcov = np.zeros(shape=self._sim_shape_full_)
        else:
            # actually calculate the partial snow coverages
            assert format in ["portrait", "landscape"], \
                "format must be 'portrait' or 'landscape'."
            assert num_strings is None or (isinstance(num_strings,int) and num_strings>0),\
                "num_strings must be an integer > 0 if not None."
            assert isinstance(self_cleaning, bool), "self_cleaning must be boolean"
            assert "poa_global" in self.sim_data
            if self.tracking == "singleaxis":
                assert "max_tracking_angle" in self.plant_parameters_processed, \
                    "max_tracking_angle not in self.plant_parameters_processed, required for self-cleaning singleaxis tracking, run preprocess_tracking_angle() first."
            # note that the FULL annual timeseries are needed here since snowfall and melting effects occur also at night
            assert "snowfall_water_equivalent" in self.sim_data_full
            assert "surface_air_temperature" in self.sim_data_full
        
            # save time for non-snow affected locations
            if self.sim_data_full["snowfall_water_equivalent"].max()/50*1000 * 100  <= threshold_snowfall: #m snowfall_water_equivalent is water equiv. in m/h x 100 cm/m, corrected by water over snow density
                # even at lowest possible fresh snow density (50kg/m3 for fresh cold snow based on literature research), we do not ever have 
                # enough snowfall to possibly take effect, can be skipped
                partial_snowcov = np.zeros_like(self.sim_data_full["snowfall_water_equivalent"])
                self.sim_data_full["partial_snowcov"] = partial_snowcov
                self.sim_data["partial_snowcov"] = partial_snowcov[self._time_sel_, :] # reduce to only daylight timesteps of interest
                return self

            # preprocess cleaning info 
            def _format_input_to_locs_array(val, varname, assert_type=None):
                """Formats a value as a 1D array with length equal to No. of locations and checks datatype if required (multiple datatypes as list possible)"""
                TYPE_MAP = {
                    int: np.integer,
                    float: np.floating,
                    bool: np.bool_,
                }
                _val = np.asarray(val)
                if assert_type is not None:
                    if not isinstance(assert_type, list):
                        assert_type = [assert_type]
                    for _type in assert_type:
                        if _type not in TYPE_MAP: 
                            raise KeyError(f"Unknown datatype: {_type}, choose from: {', '.join(TYPE_MAP.keys())}")
                        _typenp = TYPE_MAP[_type]
                        if not np.issubdtype(_val.dtype, _typenp):
                                raise TypeError(f"Expected {assert_type}, got {_val.dtype}")
                if _val.ndim == 0: # scalar
                    return np.full(self.locs.count, _val)
                elif _val.ndim == 1:
                    if not len(_val) == len(self.locs):
                        raise ValueError(f"{varname} was provided as 1D array with length = {len(_val)} but does not match No. of locs: {len(self.locs)}")
                    return _val
                else:
                    raise ValueError(f"Only scalar or 1D arrays are accepted for one-per-location arguments: {varname} = {val}")
            
            self_cleaning = _format_input_to_locs_array(val=self_cleaning, varname="self_cleaning", assert_type=bool) #TODO make self_cleaning a plant_parameters_processed param like all others, allow to pass per location
            consider_snow_loss = _format_input_to_locs_array(val=consider_snow_loss, varname="consider_snow_loss", assert_type=bool)
            
            # get the respective slide angle per tracking mode
            if self.tracking == "singleaxis":
                # get absolute angles from optimized and maximum possible angles first
                current_angles = np.abs(np.asarray(self.sim_data["system_modtilt"])) 
                max_angles = self.plant_parameters_processed["max_tracking_angle"]
                # and replace by maximum backtracking angles where self cleaning is possible
                # max angle at night and for self-cleaning locations
                surface_tilt_full = np.broadcast_to(
                    max_angles[None, :],
                    self._sim_shape_full_,
                ).copy()
                # during daytime, non-self-cleaning locations use actual tracker tilt
                surface_tilt_full[
                    np.ix_(self._time_sel_, ~self_cleaning)
                ] = current_angles[:, ~self_cleaning]
            elif self.tracking == "fixed":
                # set the fixed module tilt angle (must be positive from ground, so use absolute)
                _angles = np.abs(self.plant_parameters_processed["module_tilt"])
                # broadcast to a full annual timeseries
                surface_tilt_full = np.broadcast_to(
                    _angles[None, :],
                    self._sim_shape_full_,
                ).copy()
            else:
                raise ValueError(f"No slide angle defined for self.tracking='{self.tracking}'.")
            
            # calculate the snow fall in "fluffy" cm/h based on water equivalent in mm/h
            # use Anderson (1976) equation: simple. no wind speed, medium values compared to other models, used in SNOWPACK model
            # accept minor deviation due to SURFACE (corrected by fixed 0.065K/10m, International Standard Atmosphere lapse rate) instead of 10m height air temperature to save an additional variable
            fresh_snow_density = 50 + np.maximum(1.7 * np.power(self.sim_data_full["surface_air_temperature"]-0.065 + 15, 1.5), 0)
            snowfall_rate_cm = self.sim_data_full["snowfall_water_equivalent"] * (1000/fresh_snow_density) * 100 # m/h x 100 cm/m

            if np.nanmax(snowfall_rate_cm) <= threshold_snowfall:
                # even at the snowiest location, we do not ever have enough snowfall to possibly take effect, can be skipped
                partial_snowcov = np.zeros_like(self.sim_data_full["snowfall_water_equivalent"])
                self.sim_data_full["partial_snowcov"] = partial_snowcov # set dummy no cover
                self.sim_data["partial_snowcov"] = partial_snowcov[self._time_sel_, :] # reduce to only daylight timesteps of interest
                return self

            # warn in case of snow on near-flat modules - no melting in pvlib and sliding depends on tilt angle!
            if np.any(np.abs(surface_tilt_full) < 10):
                warnings.warn(f"Snow cover may occur on module tilt < 10°. note that only sliding and not melting effects are considered which may lead to unrealistically long snow cover duration.")
            
            # estimate the share of snow-covered surface along the slant height axis
            poa_global_full = np.zeros(self._sim_shape_full_, dtype=float) 
            poa_global_full[self._time_sel_, :] = self.sim_data["poa_global"]
            # calculate partial snow coverage for each location iteratively #TODO make more efficient calculate only for actually snow-affected locations, set to zeros for all others
            partial_snowcov = np.zeros(self._sim_shape_full_) # initialize for all timesteps of the year
            for iloc in range(self.locs.count):
                if not consider_snow_loss[iloc]:
                    # we shall not consider snow loss for this loc, leave snow coverage at the dummy zero value (no effect) and skip
                    continue
                # if the max. snowfall rate is below threshold, set all reduction factors to zero
                if np.nanmax(snowfall_rate_cm[:, iloc]) <= threshold_snowfall:
                    # this particular location does not exceed the threshold ever, leave dummy reduction factors at 0 and skip
                    continue
                # else calculate reduction factors
                partial_snowcov_loc = pvlib.snow.coverage_nrel(
                    snowfall = pd.Series(snowfall_rate_cm[:, iloc], index=self.time_index),
                    poa_irradiance = pd.Series(poa_global_full[:, iloc], index=self.time_index),
                    temp_air = pd.Series(self.sim_data_full["surface_air_temperature"][:, iloc], index=self.time_index),
                    surface_tilt = pd.Series(surface_tilt_full[:, iloc], index=self.time_index), 
                    initial_coverage=0, # always start at zero
                    threshold_snowfall=threshold_snowfall, 
                    can_slide_coefficient=-80.0, # pvlib.snow.coverage_nrel default
                    slide_amount_coefficient=0.197 # pvlib.snow.coverage_nrel default
                    )
                # append to overall factors matrix, still as a full timeseries at this stage
                partial_snowcov[:, iloc] = partial_snowcov_loc.to_numpy()

        # save partial snow coverage as attribute, both in full and clipped version
        self.sim_data["partial_snowcov"] = partial_snowcov[self._time_sel_, :] # reduce to only daylight timesteps of interest
        self.sim_data_full["partial_snowcov"] = partial_snowcov

        return self
    

    def scale_to_unshaded_real_lra(self, max_scaling_factor:float = None):
        """
        Allows to revert the effect of shading on the long-run average 
        irradiance values that are used for spatial disaggregation, e.g. Global 
        Solar Atlas. The hourly values of DNI and GHI are then scaled by the 
        factor of unshaded cumulative irradiance over shaded for the current year,
        following this equation: real_lra = real_lra * (v_all.sum() / v_shaded.sum())
        This allows later reapplication of shading on the affected timesteps 
        without duplicating the terrain shading effects.

        NOTE: The scaling considers only distant horizon shading to avoid potential shading overestimation.

        Assumptions based on horizon shading approach in pvlib-python:
        https://pvlib-python.readthedocs.io/en/stable/gallery/shading/plot_simple_irradiance_adjustment_for_horizon_shading.html 

        Parameters
        ----------
        max_scaling_factor : float, optional
            Limits the ratio between shaded and unshaded real lra to a maximum 
            factor if given, else no effect. By default None.

        Returns
        -------
        obj
            a reference to the invoking SolarWorkflowManager object, with 
            updated 'direct_normal_irradiance' and 'global_horizontal_irradiance' 
            sim_data attributes.
        """
        assert "solar_azimuth" in self.sim_data, f"solar_azimuth data expected in self.sim_data"
        assert "apparent_solar_zenith" in self.sim_data, f"apparent_solar_zenith data expected in self.sim_data"
        assert "direct_normal_irradiance" in self.sim_data, "'direct_normal_irradiance' attribute is expected in self.sim_data"
        assert "global_horizontal_irradiance" in self.sim_data, "'global_horizontal_irradiance' attribute is expected in self.sim_data"

        # first check if horizon has any effect at all, if not return immediately
        if not "distant_horizon_profile" in self.plant_parameters_processed:
            raise AttributeError(f"'distant_horizon_profile' is a mandatory attribute, run self.preprocess_distant_horizon_profile() or self.preprocess_horizon_profile() first.")

        # save time if all horizons are flat and have therefore no impact anyways
        if np.all(self.plant_parameters_processed["distant_horizon_profile"] == 0):
            return self

        # else calculate the shaded timesteps based on the horizon profile
        # interpolate the horizon profile angles for the solar azimuths covered by our irradiance data, iteratively for every location
        horizon_profile = np.vstack([
            np.interp(
                self.sim_data["solar_azimuth"][:, i],
                np.linspace(
                    0,
                    360,
                    self.plant_parameters_processed["distant_horizon_profile"].shape[1],
                    endpoint=False,
                ),
                self.plant_parameters_processed["distant_horizon_profile"][i, :],
            )
            for i in range(self.sim_data["solar_azimuth"].shape[1])
        ]).T
        # calculate the timesteps when the plant is shaded by the horizon
        _unshaded_ts = (90-self.sim_data["apparent_solar_zenith"]) > horizon_profile
        
        # then calculate the DNI rescaling factor as (sum of all DNI values) / (sum of all unshaded DNI values)
        # assume 100% DNI shading when sun behind horizon based on pvlib
        _dni_unshaded_agg = (self.sim_data["direct_normal_irradiance"] * _unshaded_ts).sum(axis=0)
        _dni_total_agg = self.sim_data["direct_normal_irradiance"].sum(axis=0)
        _dni_scaling = _dni_total_agg / _dni_unshaded_agg
        assert (_dni_scaling >= 1).all() # make sure
        if max_scaling_factor is not None:
            # limit the scaling factors per location to a maximum value
            assert isinstance(max_scaling_factor, float) and max_scaling_factor >=1, "max_scaling_factor must be float >= 1.0 if given"
            _dni_scaling = np.minimum(_dni_scaling, max_scaling_factor)
        
        # calculate corrected DNI without shading losses, store in temp variable for now
        _dni_flathorizon = self.sim_data["direct_normal_irradiance"] * _dni_scaling

        # correct GHI based on equation GHI = DNI + cos(teta) * DHI, with teta as solar zenith angle
        # GHI increases simply by the DNI delta since DHI is unaffected by horizon shading by pvlib assumption
        self.sim_data["global_horizontal_irradiance"] = self.sim_data["global_horizontal_irradiance"] + (_dni_flathorizon - self.sim_data["direct_normal_irradiance"])

        # now overwrite DNI as well with corrected value
        self.sim_data["direct_normal_irradiance"] = _dni_flathorizon

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

        Will save backside irradiances as well only when self.plant_parameters_processed["bifaciality_factor"] 
        has non-zero entries in the invoking class instance.

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

            # EXTRACT THE LOCATIONAL DATA FOR IRRADIANCE CALCULATION

            # helper function to extract and preprocess shape for variables either from sim data or placements or defaults
            def _extract_var(var, sim_var=None, fallback=None, time_invariant=False):
                """First tries to get location- and time-variable sim_data, then location-variable placements column, else default."""
                if self.sim_data.get(sim_var) is not None:
                    # get only the timeseries for the representative location
                    return np.asarray(self.sim_data.get(sim_var)[:, iloc], dtype=float).copy()
                elif var is not None and var in self.plant_parameters_processed:
                    # value is not time-variable but in placements df
                    val = self.plant_parameters_processed[var]
                    assert np.asarray(val).ndim == 1 # make sure, needed for below logic
                    if time_invariant:
                        # if parameter is time-invariant, return only the iloc-th value
                        return val[iloc]
                    else:
                        # if time-variant parameter is expected, duplicate for T timesteps
                        return np.full(self._sim_shape_[0], val[iloc])
                elif fallback is not None:
                    # set to variable fallback value
                    return np.full(self._sim_shape_[0], fallback)
                else:
                    # fall back to defaults
                    defaults = {"n_pvrows" : 3, "index_observed_pvrow" : 1} # default width for 2P orientation of large commercial module
                    if var not in defaults:
                        raise KeyError(f"Variable '{var}' is neither a sim_data system variable, nor a column in placements dataframe nor has a default value.")
                    return defaults[var]
            
            # define the base input args for this location
            # define a fallback for the axis azimuth in case of tracked systems where the value does not exist
            # the tracker axis is not used here but value is expected, orientation is always rectangular to module azimuth
            _axazimuth_fallback = _extract_var("module_azimuth", "system_modazimuth") + 90 if self.tracking == "fixed" else None
            pvfts_args = {}
            pvfts_args["solar_azimuth"] = self.sim_data["solar_azimuth"][:, iloc]
            pvfts_args["solar_zenith"] = self.sim_data["apparent_solar_zenith"][:, iloc]
            pvfts_args["surface_azimuth"] = _extract_var("module_azimuth" if self.tracking=="fixed" else None, "system_modazimuth")
            pvfts_args["surface_tilt"] = _extract_var("module_tilt" if self.tracking=="fixed" else None, "system_modtilt")
            pvfts_args["axis_azimuth"] = _extract_var("axazimuth" if self.tracking=="fixed" else "system_axazimuth", "system_axazimuth", _axazimuth_fallback, time_invariant=True)
            pvfts_args["timestamps"] = np.arange(self._sim_shape_[0])
            pvfts_args["dhi"] = self.sim_data["diffuse_horizontal_irradiance"][:, iloc]
            pvfts_args["dni"] = self.sim_data["direct_normal_irradiance"][:, iloc]
            pvfts_args["gcr"] = _extract_var("gcr", time_invariant=True)
            pvfts_args["pvrow_height"] = _extract_var("pvrow_height", time_invariant=True)
            pvfts_args["albedo"] = _extract_var("grdalbedo", "system_grdalbedo")
            pvfts_args["n_pvrows"] = _extract_var("n_pvrows", time_invariant=True)
            pvfts_args["index_observed_pvrow"] = _extract_var("index_observed_pvrow", time_invariant=True)
            pvfts_args["pvrow_width"] = _extract_var("pvrow_width_sloped", time_invariant=True) # pvlib expects the full array width along the sloped edge

            # # CONSIDER IRRADIANCE SHADING BY HORIZON EFFECTS

            # the following equations and assumptions have been adapted to the context herein from pvlib-python read the docs
            # adapted from: https://pvlib-python.readthedocs.io/en/stable/gallery/shading/plot_simple_irradiance_adjustment_for_horizon_shading.html
            # interpolate to hourly solar azimuths indices
            horizon_profile_loc = np.interp( 
                pvfts_args["solar_azimuth"],
                np.linspace(0, 360, len(self.plant_parameters_processed["horizon_profile"][iloc, :]), endpoint=False),
                self.plant_parameters_processed["horizon_profile"][iloc, :],
            ) 
            # calculate the timesteps when the plant is shaded by the horizon
            _horizon_shaded = (90-self.sim_data["apparent_solar_zenith"][:, iloc]) <= horizon_profile_loc
            # correct dni by setting it to zero for timesteps when sun is shaded by horizon - DHI is assumed to be practically not affected
            pvfts_args["dni"] = np.where(_horizon_shaded, 0, pvfts_args["dni"])

            # HANDLE KWARGS

            # handle kwargs for this location - first define those args that pvlib expects to be time-variant
            timeseries_args = {
                "solar_azimuth",
                "solar_zenith",
                "surface_azimuth",
                "surface_tilt",
                "dni",
                "dhi",
            }
            # extract the number of timesteps and locations from sim shape (N_t, N_loc)
            n_timesteps, n_locations = self._sim_shape_
            # iterate over the kwargs and set/overwrite them one by one - note that this is always just for ONE location (iteration above)
            for k, v in kwargs.items():
                arr = np.asarray(v)

                # scalar input
                if arr.ndim == 0:
                    # duplicate into timeseries if expected as time-variant, else set scalar
                    if k in timeseries_args:
                        pvfts_args[k] = np.full(
                            n_timesteps,
                            arr.item(),
                            dtype=float,
                        )
                    else:
                        pvfts_args[k] = arr.item()
                # 1d input, could be per location (time-invariant) or per timestep (same for all locs)
                elif arr.ndim == 1:
                    n_values = arr.shape[0]
                    # avoid ambiguity and demand explicit 2d input if number of locations and timesteps are the same
                    if n_values == n_timesteps == n_locations:
                        raise ValueError(
                            f"Number of timesteps and locations are equal "
                            f"({n_timesteps}). Therefore, 1D kwarg '{k}' with "
                            f"length {n_values} is ambiguous. Provide a 2D array "
                            f"with shape {self._sim_shape_} (N_t, N_loc) for time- and "
                            f"location-dependent data."
                        )
                    # time-dependent data
                    elif n_values == n_timesteps:
                        # make sure it is actually a timeseries argument, then set it
                        if k not in timeseries_args:
                            raise ValueError(
                                f"kwarg '{k}' was provided as a 1D time series, "
                                f"but '{k}' is not a time-varying pvfactors input."
                            )
                        pvfts_args[k] = arr
                    # location-specific data
                    elif n_values == n_locations:
                        if k in timeseries_args:
                            # we effectively have one scalar per location, must be inflated to a timeseries each
                            pvfts_args[k] = np.full(
                                n_timesteps,
                                arr[iloc],
                                dtype=float,
                            )
                        else:
                            # keep the locational value as a scalar
                            pvfts_args[k] = arr[iloc].item()
                    else:
                        raise ValueError(
                            f"kwarg '{k}' was provided as a 1D array with length "
                            f"{n_values}; expected {n_timesteps} timesteps or "
                            f"{n_locations} locations."
                        )
                # 2d input - must then match time x locs shape of sim data
                elif arr.ndim == 2:
                    if arr.shape != self._sim_shape_:
                        raise ValueError(
                            f"kwarg '{k}' was provided as a 2D array with shape "
                            f"{arr.shape}; expected {self._sim_shape_}."
                        )
                    if k not in timeseries_args:
                        raise ValueError(
                            f"kwarg '{k}' was provided as a time/location array, "
                            f"but '{k}' is not a time-varying pvfactors input."
                        )
                    # set all timesteps for the current loc
                    pvfts_args[k] = arr[:, iloc]
                else:
                    raise ValueError(
                        f"Unknown input dimensionality ({arr.ndim}D) "
                        f"for kwarg '{k}'."
                    )
                
            # PREPROCESS ANGLES TO ALLOW FOR PVLIB LIMITATIONS

            # NOTE: surface_tilt in pvlib is defined in the range 0-180°, so negative values will be handled badly
            if self.tracking == "fixed":
                # in a fixed system, module orientations must not change over time
                assert np.all(pvfts_args["surface_tilt"] == pvfts_args["surface_tilt"][0]), "surface_tilt must be constant over time for tracking = 'fixed'!"
                assert np.all(pvfts_args["surface_azimuth"] == pvfts_args["surface_azimuth"][0]), "surface_azimuth must be constant over time for tracking = 'fixed'!"
                # accept either a scalar or a repeated 1D value for axis azimuth, if the latter - reduce to scalar (can never change, is expected as scalar by pvlib)
                if np.asarray(pvfts_args["axis_azimuth"], dtype=float).ndim == 0:
                    pvfts_args["axis_azimuth"] = np.asarray(pvfts_args["axis_azimuth"], dtype=float).item()
                elif np.asarray(pvfts_args["axis_azimuth"], dtype=float).ndim == 1:
                    assert np.all(
                        np.asarray(pvfts_args["axis_azimuth"], dtype=float) == np.asarray(pvfts_args["axis_azimuth"], dtype=float)[0]
                        ), "axis_azimuth must be constant over time for tracking='fixed'."
                    pvfts_args["axis_azimuth"] = np.asarray(pvfts_args["axis_azimuth"], dtype=float)[0].item()
                else:
                    raise ValueError("axis_azimuth must be scalar or a constant 1D array for tracking='fixed'.")
                # fixed tilt often comes with external values where module tilt may be negative to geometrically invert azimuth
                if pvfts_args["surface_tilt"][0] < 0:
                    # in such cases, invert the tilt and surface azimuth combination so that tilt values are always positive!
                    # can be done for all timeseries values at once because constant values have been asserted above
                    pvfts_args["surface_tilt"] = -pvfts_args["surface_tilt"]
                    pvfts_args["surface_azimuth"] = (pvfts_args["surface_azimuth"] + 180) % 360
                    # pvlib also expects the axis_azimuth to be surface azimuth + 90° for fixed tilt: adapt for flipped and ensure (with tol) for non-flipped
                    pvfts_args["axis_azimuth"] = (pvfts_args["surface_azimuth"][0] + 90) % 360
                axis_azimuth_exp = (pvfts_args["surface_azimuth"] + 90) % 360 # the expected value
                axis_diff = (pvfts_args["axis_azimuth"] - axis_azimuth_exp[0] + 180) % 360 - 180
                assert np.all(np.abs(axis_diff) < 1e-9), "Axis azimuth must always be surface_azimuth + 90° for fixed tilt  in pvlib."
            elif self.tracking == "singleaxis":
                # this should be correct as it usually comes out of pvlib.tracking.singleaxis, but make sure with tolerance
                assert (pvfts_args["surface_tilt"] >= -1e-9).all(), "Negative values in surface_tilt array."
            else:
                raise ValueError(f"Unknown value for self.tracking: {self.tracking}")
            
            # PROJECTION DUE TO HILL SLOPE 
            
            # pvlib's pvfactors_timeseries cannot consider hill slope, especially not when no tracking function is applied (fixed tilt modules)
            # it internally treats the the system as a 2d vertical cross-section (here "profile (section) plane") of an infinite row on flat ground
            # that neglects row elevation differences (and also tilt along the main axis) and therefore introduces significant shading error 
            # for singleaxis tracking, this is covered in combination with pvlib.tracking.singleaxis() where crossaxis tilt is considered, but not in pvfactors_timeseries()
            # for fixed tilt PV, axis and cross axis tilt are not considered at all so far and cannot be fed into pvfactors_timeseries() directly
            # Idea: "Flatten" hill slope (to allow horizontal ground assumption in pvfactors) and at the same time adapt solar position and angular plant parameters by the hillslope to preserve incidence angles
            # Use a coordinate transformation (rigid rotation around main tracking/row axis) on hill plane, solar position, module angles, row distance (-> gcr) and row height

            # first get the groundslope in the "profile section" that pvfactors uses, i.e. the vertical plane through the center point, perpendicular to the horizontal projection of the main axis
            _axis_azimuth_rad = np.radians(pvfts_args["axis_azimuth"] % 360)
            project = False # set a flag that can be overwritten if needed

            # 3D-VERSION
            if self.tracking == "singleaxis" and not (np.isclose(self.plant_parameters_processed["singleaxis_tilt"][iloc], 0) and np.isclose(self.plant_parameters_processed["crossaxis_tilt"][iloc], 0)): 
                project = True # we have a non-horizontal plant plane and need to do something
                # in case of single-axis tracking, we have the relevant axis and cross axis tilts already
                _axis_tilt_rad = np.radians(self.plant_parameters_processed["singleaxis_tilt"][iloc])
                _crossaxis_tilt_rad = np.radians(self.plant_parameters_processed["crossaxis_tilt"][iloc])

                # downhill gradient along the horizontal projection of the main axis, positive axis_tilt = descent toward axis_azimuth
                _axis_gradient = np.tan(_axis_tilt_rad)
                # caution: crossaxis tilt is not in a vertical plane (if row axis tilt is not zero) but perpendicular to main axis
                # so first get ground slope in the vertical profile plane perpendicular to the HORIZONTAL axis projection
                _root_term = np.cos(_axis_tilt_rad)**2 - np.sin(_crossaxis_tilt_rad)**2
                if _root_term < -1e-12: # should not happen for a real terrain but axis and cross axis tilts might have been manipulated individually, so make sure
                    raise ValueError(f"Inconsistent singleaxis_tilt ({np.degrees(_axis_tilt_rad)}°) and crossaxis_tilt ({np.degrees(_crossaxis_tilt_rad)}°) at location {iloc}.")
                _root_term = max(_root_term, 0.0) # must be zero or positive for sqrt, fix potential floating point errors by max(x, 0)
                _profile_groundslope_rad = np.arctan2(
                    np.sin(_crossaxis_tilt_rad),
                    np.cos(_axis_tilt_rad) * np.sqrt(_root_term),
                )
                # positive profile ground slope means terrain rises toward the right, therefore invert sign for DOWNhill gradient
                _profile_gradient = -np.tan(_profile_groundslope_rad)
                # next combine them into the overall slope gradient and radians slope of the 3D plant plane
                _plant_gradient = np.hypot(_axis_gradient, _profile_gradient)
                _plant_slope_rad = np.arctan(_plant_gradient)
                # now get the east and north gradients separately to derive the overall plant plane azimuth
                # for main axis: East = sin(axis_azimuth), North = cos(axis_azimuth)
                # for profile axis (right-hand side to main axis = positive): East = cos(axis_azimuth), North = -sin(axis_azimuth)
                _downhill_gradient_east = (
                    _axis_gradient * np.sin(_axis_azimuth_rad)
                    + _profile_gradient * np.cos(_axis_azimuth_rad)
                )
                _downhill_gradient_north = (
                    _axis_gradient * np.cos(_axis_azimuth_rad)
                    - _profile_gradient * np.sin(_axis_azimuth_rad)
                )
                # last calculate the overall plant plane azimuth from the North and East gradients
                _plant_downhill_azimuth_rad = np.arctan2( # considers negatives correctly
                    _downhill_gradient_east,
                    _downhill_gradient_north,
                )% (2.0 * np.pi) # normalize to 0-2*pi


            elif self.tracking == "fixed" and not np.isclose(self.plant_parameters_processed["general_slope"][iloc], 0.0):
                project = True  # we have an inclined system, need to project
                # for fixed pv, no single axis or cross axis tilts exist, plane is solely defined by the hillslope
                # that makes it easy as plant plane = hill plane, for which we have general slope and downhill azimuth already
                _plant_slope_rad = np.radians(self.plant_parameters_processed["general_slope"][iloc])
                _plant_downhill_azimuth_rad = np.radians(self.plant_parameters_processed["downhill_azimuth"][iloc] % 360.0)
                # but we need to calculate the main row axis tilt, positive means descent toward the directed axis azimuth.
                _axis_tilt_rad = np.arctan(
                    np.tan(_plant_slope_rad)
                    * np.cos(
                        _axis_azimuth_rad
                        - _plant_downhill_azimuth_rad
                    )
                )
                # get the gradient in the crossaxis plane perpendicular to the main axis
                _crossaxis_tilt_rad = np.arctan2(
                    np.tan(_plant_slope_rad)
                    * np.sin(_plant_downhill_azimuth_rad - _axis_azimuth_rad),
                    np.cos(_axis_tilt_rad),
                )


            if project:
                # we have an inclined plant plane in 3D space, apply a 3D rigid coordinate transformation to module and solar angles alike
                # this allows the horizontal-ground-assumption enforced by pvlib whilst correctly considering self-shadowing at unchanged POAI

                # start with calculating the AOI and check again later to make sure that this central parameter is indeed not affected by the transformation
                _aoi_before_transf = pvlib.irradiance.aoi(
                    surface_tilt=pvfts_args["surface_tilt"],
                    surface_azimuth=pvfts_args["surface_azimuth"],
                    solar_zenith=pvfts_args["solar_zenith"],
                    solar_azimuth=pvfts_args["solar_azimuth"],
                )
                
                # first get the upward-facing normal on the plant plane as a 3D vector in East-North-Up coordinates
                _terrain_normal = np.array(
                    [
                        np.sin(_plant_slope_rad) * np.sin(_plant_downhill_azimuth_rad), # positive slope angle means horizontal components point in plant slope azimuth direction
                        np.sin(_plant_slope_rad) * np.cos(_plant_downhill_azimuth_rad), # same as above
                        np.cos(_plant_slope_rad),
                    ],
                    dtype=float,
                )
                # also define the target normal on horizontal ground
                _vertical_normal = np.array(
                    [0.0, 0.0, 1.0], # points only up
                    dtype=float,
                )

                # The axis of the required rotation to "flatten" the terrain is therefore perpendicular to both the real terrain normal and the target vertical normal, allowing a "single rigid rotation"
                _rotation_axis = np.cross(_terrain_normal, _vertical_normal)
                _rotation_axis_norm = np.linalg.norm(_rotation_axis) # get its length for normalization
                # make sure that we have a sufficient magnitude in our planned transformation
                if _rotation_axis_norm > 1e-12:
                    # normalize to 1
                    _rotation_axis /= _rotation_axis_norm
                else:
                    # else it is probably numeric difference only, should not occur after above if statements and project = true
                    # if still the case, define a dummy rotation axis with zero rotation angle = without effect to allow further processing
                    _rotation_axis = np.array([1.0, 0.0, 0.0], dtype=float)
                # the rotation angle is known already, it must be _plant_slope_rad
                # but its cosine is also the dot product of the 2 unit vectors on the real and target ground surfaces
                # get it from the latter just to be safe to be consistent (e.g. in above else case)
                _rotation_angle_rad = np.arccos(
                    np.clip(
                        np.dot(
                            _terrain_normal,
                            _vertical_normal,
                        ),
                        -1.0,
                        1.0,
                    )
                )
                if not np.isclose(_rotation_angle_rad, _plant_slope_rad):
                    raise RuntimeError(f"_plant_slope_rad ({_plant_slope_rad}) does not match the arccos of the dot product of the plant sloped plane and horizontal plane ({_rotation_angle_rad})!")

                def _rotate_angle_and_azimuth_to_level_plant_plane(
                    inclination_deg,
                    azimuth_deg,
                ):
                    """
                    Rotate one or more direction vectors so that the complete inclined plant plane 
                    becomes horizontal. Will be applied to every direction, i.e. both main and cross
                    axis tilts will be 0° in the output coordinates. Can be applied to solar vectors, 
                    module surface vectors etc. The same rigid 3D rotation is applied to every 
                    direction so that relative angles such as solar-module AOI are preserved exactly.

                    Parameters
                    ----------
                    inclination_deg : scalar or array-like
                        Angle from upward vertical (!) in degrees:
                        0° upwards, 90° horizontal, 180° downwards.
                        Examples: Solar zenith, module surface inclination = module tilt
                    azimuth_deg : scalar or array-like
                        Compass azimuth in degrees clockwise from North 0°.
                        Examples: Solar azimuth, module (surface) azimuth

                    Returns
                    -------
                    inclination_flat_deg : ndarray
                        Direction angle in degrees from upward vertical after leveling.
                    azimuth_flat_deg : ndarray
                        Direction azimuth 0-360° in degrees from North = 0° after leveling.
                    """
                    # convert input angles to radians and make sure the formats/shapes are matching
                    inclination_rad = np.radians(np.asarray(inclination_deg, dtype=float))
                    azimuth_rad = np.radians(np.asarray(azimuth_deg, dtype=float))
                    inclination_rad, azimuth_rad = np.broadcast_arrays(
                        inclination_rad,
                        azimuth_rad,
                    )
                    # generate an East-North-Up unit vector normal on plant plane
                    vectors = np.stack(
                        [
                            np.sin(inclination_rad) * np.sin(azimuth_rad), # East
                            np.sin(inclination_rad) * np.cos(azimuth_rad), # North
                            np.cos(inclination_rad), # Up
                        ],
                        axis=-1,
                    )

                    # use the Euler-Rodrigues equation to transform the coordinates:
                    # v_rot = v*cos(theta) + (k x v)*sin(theta) + k*(k dot v)*(1 - cos(theta))
                    # with k as the 3D leveling-rotation axis and theta as rotation angle
                    # the 3 rows below correspond to the 3 above terms (downscale, skew toward new orientation, re-add height)
                    vectors_flat = (
                        vectors *  np.cos(_rotation_angle_rad)
                        + np.cross(_rotation_axis, vectors) * np.sin(_rotation_angle_rad)
                        + _rotation_axis * np.sum(vectors * _rotation_axis, axis=-1, keepdims=True) * (1.0 - np.cos(_rotation_angle_rad))
                    )
                    # then extract the East, North, up components from the rotated vectors
                    vector_east_flat = vectors_flat[..., 0] # East component = first entry of the last dimension
                    vector_north_flat = vectors_flat[..., 1] # North component = first entry of the last dimension
                    vector_up_flat = vectors_flat[..., 2] # Up component = first entry of the last dimension

                    # convert rotated/flattened vectors back to the pvlib angular/spherical convention
                    vector_horizontal_flat = np.hypot(vector_east_flat, vector_north_flat)
                    inclination_flat_deg = np.degrees(np.arctan2(vector_horizontal_flat, vector_up_flat)) # arctan2 is sign-true
                    azimuth_flat_deg = np.degrees(np.arctan2(vector_east_flat, vector_north_flat)) % 360.0 # same

                    return (
                        inclination_flat_deg,
                        azimuth_flat_deg,
                    )
                
                # adapt solar position (zenith and azimuth), module surfaces (tilt and azimuth), gcr and pv row height
                
                # solar position and module surface orientation need to be rotated with the same rigid 3D transformation
                pvfts_args["solar_zenith"], pvfts_args["solar_azimuth"] = _rotate_angle_and_azimuth_to_level_plant_plane(
                    inclination_deg=pvfts_args["solar_zenith"],
                    azimuth_deg=pvfts_args["solar_azimuth"],
                )
                pvfts_args["surface_tilt"], pvfts_args["surface_azimuth"] = _rotate_angle_and_azimuth_to_level_plant_plane(
                    inclination_deg=pvfts_args["surface_tilt"],
                    azimuth_deg=pvfts_args["surface_azimuth"],
                )
                # axis azimuth is relevant for pvfactors_timeseries and might be affected, so transform as well 
                _axis_inclination_flat, _axis_azimuth_flat = _rotate_angle_and_azimuth_to_level_plant_plane(
                    inclination_deg=90.0 + np.degrees(_axis_tilt_rad), # make sure we use an inclination angle from vertical for above transformation function, not simply the default axis tilt
                    azimuth_deg=np.degrees(_axis_azimuth_rad),
                )
                # assert that all worked as expected and set as new axis tilt only for the coming pvfactors_timeseries processing
                if not np.isclose(_axis_inclination_flat, 90.0, atol=1e-8):
                    raise RuntimeError(f"Plant-plane leveling did not produce a horizontal row axis: {_axis_inclination_flat}° at location {iloc}.")
                pvfts_args["axis_azimuth"] = float(np.asarray(_axis_azimuth_flat)) % 360

                # gcr needs to be scaled by the cosine of the cross axis slope due to the increased sloped length of the row width (the defined value is understood a vertical projection)
                # Note that only the cross axis slope is affecting the row spacing, not the overall plant slope
                pvfts_args["gcr"] *= np.cos(_crossaxis_tilt_rad) # sloped row width increases over horizontal (defined) row width, so gcr is reduced

                # Note that pvfts_args["pvrow_height"] is not corrected, it is assumed not as vertical but as (minimum) ground distance either to level or sloped ground
                # Steeper slope changes the module angle relative to the sloped ground, steeper relative angles may reduce the ground clearance and cause ground penetration
                # This problem is avoided by definition as minimum distance normal to the ground, plus the relevant dimension is preserved/constant independent of slope

                # finish by recalculating the AOI after transformation and compare it to before to make sure it was indeed not changed
                _aoi_after_transf = pvlib.irradiance.aoi(
                    surface_tilt=pvfts_args["surface_tilt"],
                    surface_azimuth=pvfts_args["surface_azimuth"],
                    solar_zenith=pvfts_args["solar_zenith"],
                    solar_azimuth=pvfts_args["solar_azimuth"],
                )
                np.testing.assert_allclose(
                    _aoi_before_transf,
                    _aoi_after_transf,
                    rtol=1e-4, # 0.1% allowed
                    atol=1e-3, # absolute 0.001 degrees allowed
                    equal_nan=True,
                    err_msg=("Transformation changed solar-to-module AOI!")
                )

            # SIMULATE THE ACTUALLY ABSORBED IRRADIANCES

            # simulate and append locational output to total results
            assert (np.atleast_1d(pvfts_args["pvrow_height"])-0.5*np.atleast_1d(pvfts_args["pvrow_width"])* np.sin(np.radians(np.max(pvfts_args["surface_tilt"]))) > 0).all(),\
                f"pvrow_height must exceed vertical dimension of tilted pvrow_width in all cases. Set pvrow_height to > 0.5*pvrow_width to be safe." # leads to unrealistic results in pvlib.bifacial.pvfactors_timeseries() otherwise
            assert pvfts_args["index_observed_pvrow"] < pvfts_args["n_pvrows"], \
                f"'index_observed_pvrow' ({pvfts_args['index_observed_pvrow']}) must be < 'n_pvrows' {pvfts_args['n_pvrows']}"
            _poa_frontside, _poa_backside, _poa_frontside_absorbed, _poa_backside_absorbed = pvlib.bifacial.pvfactors_timeseries(**pvfts_args) # old version on purpose, deprecation warning is accepted as long as it works since this version also works with elder pvlib versions <= 0.9.0

            # save the outputs to the respective multi-dimensional array columns
            poa_frontside[:, iloc] = _poa_frontside.values
            poa_backside[:, iloc] = _poa_backside.values
            poa_frontside_absorbed[:, iloc] = _poa_frontside_absorbed.values
            poa_backside_absorbed[:, iloc] = _poa_backside_absorbed.values

        
        def _fix_bad_poa_and_set_attr(arr, attr):
            """Sets sim_data attribute with 0 values where raw front POA > 0"""
            sel_bad_poa = (poa_frontside >= 1600) | (arr<0) # pvlib yields negative irradiation in some cases
            self.sim_data[attr] = np.where(
                        sel_bad_poa, 0, arr
                    )


        # finally set the results as sim_data attributes
        _fix_bad_poa_and_set_attr(arr=poa_frontside, attr="poa_global_raw")
        _fix_bad_poa_and_set_attr(arr=poa_frontside_absorbed, attr="poa_global")
        
        # set POA values for backside only when bifacial flag is True 
        _fix_bad_poa_and_set_attr(arr=poa_backside, attr="poa_backside_global_raw")
        _fix_bad_poa_and_set_attr(arr=poa_backside_absorbed, attr="poa_backside_global")

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

        modtilts = self.sim_data.get("system_modtilt", self.plant_parameters_processed["module_tilt"])

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
        database="CEC Modules.csv",
        module_configuration : str = "2P",
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
        module_configuration : str, optional
            The number and orientation in which the modules are mounted, e.g. "2P" for
            two-in-portrait, 2 modules stacked in a portrait orientation along the sloped 
            length. 
        database : str, optional
            The database that shall be loaded, either via a known database in 
            pvlib.pvsystem.retrieve_sam() or as filename of a .csv database in 
            reskit/solar/data. By default "CEC Modules.csv".
        
        NOTE: The bifaciality factor is provided separately as a workflow argument and loaded
        via preprocess_bifaciality_factor(), it can be scalar or location-specific.

        Returns
        -------
        obj
            Returns a reference to the invoking SolarWorkflowManager object

        References
        ----------
        [1] https://pvlib-python.readthedocs.io/en/stable/generated/pvlib.pvsystem.retrieve_sam.html


        """
        if tracking not in [
            "fixed",
            "singleaxis",
        ]:
            raise ValueError("tracking must be either 'fixed' or 'singleaxis'")

        # set tracking type as class attribute
        if tracking not in ["fixed", "singleaxis"]:
            raise NotImplementedError(f"Unknown tracking value ({tracking}). Choose from: 'singleaxis' (HSAT), 'fixed' (tilt)")
        self.tracking = tracking

        def _interpolate_module_params(
            projected_module, original_module_name, tech_year, start_year, database
        ):
            if not isinstance(tech_year, int):
                raise TypeError(
                    "tech_year must be an integer when projected module is selected"
                )
            # avoid extrapolations
            if not start_year <= tech_year <= 2050:
                raise ValueError(f"tech_year must be between {start_year} and 2050 (max. projection) for this module")

            # get the original (unprojected) module parameters
            try:
                # first try to load via pvlib and retrieve_sam()
                db = pvlib.pvsystem.retrieve_sam(database)
            except:
                # else load database from the reskit solar data
                if splitext(database)[-1]=="":
                    # no extension, set it to csv
                    database = database+".csv"
                elif not splitext(database)[-1]==".csv":
                    raise TypeError(f"database must be a csv file if not a known key in retrieve_sam(). Here: {database}")
                db = pd.read_csv(DATA[database]._str, skiprows=[1,2]).set_index("Name", drop=True).T
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
                    module[param] = original_module[param] + (val_proj - original_module[param]) * (
                        tech_year - start_year
                    ) / (2050 - start_year)
                else:
                    assert val_proj == original_module[param], (
                        f"parameter '{param}' is not the same for original ({original_module[param]}) and projected ({val_proj}) modules"
                    )
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
                    database=database,
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
                    database=database,
                )

                module.name = "LG Electronics LG370Q1C-A5"

            elif isinstance(module, str):
                if tech_year is not None:
                    warnings.warn(
                        "NOTE: The tech_year argument is ignored when a specific module is given. Set tech_year to None to silence this warning.\n"
                    )
                # Extract module parameters
                try:
                    # first try to load via pvlib and retrieve_sam()
                    db = pvlib.pvsystem.retrieve_sam(database)
                except:
                    # else load database from the reskit solar data
                    if splitext(database)[-1]=="":
                        # no extension, set it to csv
                        database = database+".csv"
                    elif not splitext(database)[-1]==".csv":
                        raise TypeError(f"database must be a csv file if not a known key in retrieve_sam(). Here: {database}")
                    db = pd.read_csv(DATA[database]._str, skiprows=[1,2]).set_index("Name", drop=True).T
                try:
                    module = getattr(db, module)
                except Exception:
                    raise RuntimeError(
                        f"The module '{module}' is not in the CEC database."
                    )
            else:
                raise TypeError(f"module must be str-formatted module name. Here: {module}")
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

            try:
                module_desc = json.dumps(module)
            except Exception:
                module_desc = "user-configured"
            self.register_workflow_parameter("module_desc", module_desc)
        
        # some modules lack dimensional information, add it if info has been added to length_dict/width_dict manually
        if "Length" not in module.index or pd.isnull(module["Length"]):
            length_dict = {
                    'Trina Solar Co.Ltd TSM-700NEG21C.20' : 2.384
                }
            try:
                module["Length"] = length_dict[module.name]
            except:
                warnings.warn(f"module '{module.name}' from database has no Length information and no additional Length information found in RESKit dictionary. Add Length info manually in length_dict in configure_cec_module()")
        if "Width" not in module.index or pd.isnull(module["Width"]):
            width_dict = {
                    'Trina Solar Co.Ltd TSM-700NEG21C.20' : 1.303
                }
            try:
                module["Width"] = width_dict[module.name]
            except:
                warnings.warn(f"module '{module.name}' from database has no Width information and no additional Width information found in RESKit dictionary. Add Width info manually in width_dict in configure_cec_module()")

        # save module info
        self.module = module

        # ensure a known module orientation and deconstruct it into orientation and number of stacked modules #TODO make this an iterable once module can be defined per loc
        assert isinstance(module_configuration, str), "module_configuration must be a string like '1P' or '3L'"
        pattern = re.compile(r"^([1-9]\d*)([PL])$", re.IGNORECASE)
        match = pattern.fullmatch(module_configuration)
        if match:
            stacked_qty = int(match.group(1))
            orientation = {"L":"landscape", "P": "portrait"}[match.group(2).upper()]
        else:
            raise ValueError(f"Invalid module_configuration format: '{module_configuration}'. Must be a string like '1P' or '3L'")
        # store the module configuration and details as attributes
        self.module_configuration = module_configuration
        self.modules_stacked = stacked_qty
        self.module_orientation = orientation

        # calculate the array width based on the module and orientation info
        crossdim = {"portrait" : "Length", "landscape" : "Width"}[self.module_orientation]
        pvrow_width_sloped = module[crossdim] * self.modules_stacked
        self.plant_parameters_processed["pvrow_width_sloped"] = np.full(self.locs.count, pvrow_width_sloped)

        return self


    def simulate_with_interpolated_single_diode_approximation(self, consider_snow_cover : bool = False, format: str = "portrait", num_strings : int =None):
        """
        Does the simulation with an interpolated single diode approximation 
        using the pvlib.pvsystem.calcparams_desoto() [1] function and the
        pvlib.pvsystem.singlediode() [2] function.

        consider_snow_cover : bool, optional
            If True, 
        format: str, optional
            Either 'portrait' or 'landscape'. Takes effect only if num_strings 
            is None, then uses the number of parallel strings from module "N_p" 
            parameter as num_strings when format is landscape. By default 
            portrait.
        num_strings: int, optional
            The number of parallel cell strings atop of each other along slant 
            height axis. Will be extracted from self.module["N_p"] if not given 
            and format is 'landscape', else 1. By default None.

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
        if np.any(np.asarray(consider_snow_cover)):
            assert "partial_snowcov" in self.sim_data, f"'partial_snowcov' must be calculated first if 'consider_snow_cover' is True for any location."

        assert self.module is not None, "Configure module te be simulated first via configure_cec_module()."

        sel = self.sim_data["poa_global"] > 0
        cell_temp = self.sim_data["cell_temperature"][sel]

        if np.any(np.asarray(consider_snow_cover)): #TODO move this block to the plant (or module?) system setup and save parallel number of strings as attr
            # calculate the number of cell strings in each module parallel to the snow cover line
            if num_strings is None:
                # take data from module if possible
                if format == "landscape" and "N_p" in self.module.index: 
                    # use the No of cell strings parallel to the long side
                    assert isinstance(self.module["N_p"], int) and self.module["N_p"]>0 # make sure
                    num_strings = self.module["N_p"]
                else:
                    # for portrait, all parallel strings are always affected, ergo only 1 parallel string along short side
                    num_strings = 1 # also fall back on binary on/off solution when no N_p available
            else:
                assert isinstance(num_strings, int) and num_strings>0 # make sure
        else:
            # no snow coverage means no difference between strings, assume a single string for calculation efficiency
            num_strings = 1
        
        # iterate over strings in module from bottom to top and simulate yield per each string separately as it may be covered or not
        # Note that bottom/top side are inverted every half day when e.g. singleaxis, but has no effect on overall production as long as partial coverage remains the same
        for _string in range(num_strings):
            # each of the cell strings may produce on both sides, or only backside if frontside is covered
            if np.any(np.asarray(consider_snow_cover)):
                # conservative assumption based on pvlib.snow.dc_loss_nrel(): no production when at least partial coverage of string area
                _production = 1- (self.sim_data["partial_snowcov"] > _string/num_strings)
            else:
                # never covered, set to dummy with every time step exposed
                _production = np.ones(shape=self._sim_shape_)

            #TODO repeat simulation only for THOSE timesteps where the frontside cover boolean is different from cell strings calculated before -> save time when num_strings > 1

            # different front- and backside irradiances would trigger (physically impossible) different electrical reactions of the same cell
            # so reconcile front- and backside parameters: combine front and back POAs at the beginning and use a single interpolator 
            # introduces a marginal rounding error but is much simpler/faster since params need to be calculated only once and are already aligned
            poa = np.multiply(
                self.sim_data["poa_global"],
                _production # _production will be zero for non-production timesteps for this string (e.g. when snow-covered)
            )[sel]
            if not (self.plant_parameters_processed["bifaciality_factor"] == 0).all():
                # add the backside irradiance, reduced by bifaciality factor (simplified)
                poa_back = np.broadcast_to(self.plant_parameters_processed["bifaciality_factor"], sel.shape)[sel] * self.sim_data["poa_backside_global"][sel]
                # special case: Avoid artefacts when simulating vertical modules with snow
                # usually backside is not snow covered but for vertical panels, it is as exposed as the front side , so assume same snow cover
                # NOTE: Artefacts at very steep angles just below to 90° absolute tilt are still possible!
                if self.tracking == "fixed":
                    # apply snow cover reduction to backside POA only of the VERTICAl modules as well
                    vertical_mask = np.isclose(np.abs(self.plant_parameters_processed["module_tilt"]), 90.0)
                    vertical_mask_sel = np.broadcast_to(vertical_mask, sel.shape)[sel]
                    poa_back[vertical_mask_sel] *= _production[sel][vertical_mask_sel]
                # add possible snow-adjusted poa back to total poa
                poa += poa_back

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
                method="lambertw",  # PVLIB v0.7.2 Default
            )
            if num_strings > 1: #TODO power could be added iteratively for more than one string but how to deal with different voltages per string, how to combine into one MODULE_dc_voltage_at_mpp?
                raise NotImplementedError("Define string-wise module_dc_power_at_mpp and module_dc_voltage_at_mpp combination for num_strings > 1 first.")
            interpolator = RectBivariateSpline(
                _temp,
                _poa,
                np.array(gen["p_mp"]).reshape(poaM.shape),
                kx=3,
                ky=3,  # np.array() since type changed between pvlib versions
            )
            self.sim_data["module_dc_power_at_mpp"] = np.zeros_like(self.sim_data["poa_global"])
            self.sim_data["module_dc_power_at_mpp"][sel] = interpolator(cell_temp, poa, grid=False)

            interpolator = RectBivariateSpline(
                _temp,
                _poa,
                np.array(gen["v_mp"]).reshape(poaM.shape),
                kx=3,
                ky=3,  # np.array() since type changed between pvlib versions
            )
            self.sim_data["module_dc_voltage_at_mpp"] = np.zeros_like(self.sim_data["poa_global"])
            self.sim_data["module_dc_voltage_at_mpp"][sel] = interpolator(cell_temp, poa, grid=False)
        
        self.sim_data["capacity_factor"] = self.sim_data["module_dc_power_at_mpp"] / (
            self.module.I_mp_ref * self.module.V_mp_ref
        )

        # Estimate total system generation
        if "capacity" in self.plant_parameters_processed and not self._is_none(self.plant_parameters_processed).any():
            self.sim_data["total_system_generation"] = self.sim_data["capacity_factor"] * np.broadcast_to(
                self.plant_parameters_processed["capacity"], self._sim_shape_
            )

        if "modules_per_string" in self.placements.columns and "strings_per_inverter" in self.placements.columns: #TODO #ISSUE285 replace placements df in this block
            total_modules = (
                self.placements.modules_per_string
                * self.placements.strings_per_inverter
                * getattr(self.placements, "number_of_inverters", 1)
            )

            self.sim_data["total_system_generation"] = self.sim_data["module_dc_power_at_mpp"] * np.broadcast_to(
                total_modules, self._sim_shape_
            )

        return self

    def apply_inverter_losses(
        self,
        inverter,
        method="sandia",
    ):
        """
         apply_inverter_losses(self, inverter, method="sandia", )

         Applies inverter losses using the pvlib.pvsystem.snlinverter() function [1], the pvlib.pvsystem.retrieve_sam() function [2] and the
         pvlib.pvsystem.adrinverter() function [3].


        Parameters
        ----------
         inverter: str
                   Describes the inverter.
                   [TODO: Add a more detailed description following the example of 'configure_cec_module']
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
        assert self.module is not None
        assert "modules_per_string" in self.plant_parameters_processed #TODO #ISSUE285 write preprocessing functions to add modules per string and strings per inverter
        assert "strings_per_inverter" in self.plant_parameters_processed
        assert (
            "capacity" not in self.plant_parameters_processed
        ), "Cannot simultaneously provide 'capacity' and inverter-string parameters"

        if method == "sandia":
            if isinstance(inverter, str):
                db = pvlib.pvsystem.retrieve_sam("SandiaInverter")
                inverter = getattr(db, inverter)

            self.sim_data["inverter_ac_power_at_mpp"] = pvlib.inverter.sandia(
                v_dc=self.sim_data["module_dc_voltage_at_mpp"]
                * np.broadcast_to(self.plant_parameters_processed["modules_per_string"], self._sim_shape_),
                p_dc=self.sim_data["module_dc_power_at_mpp"]
                * np.broadcast_to(
                    self.plant_parameters_processed["modules_per_string"] * self.plant_parameters_processed["strings_per_inverter"],
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
                * np.broadcast_to(self.plant_parameters_processed["modules_per_string"], self._sim_shape_),
                p_dc=self.sim_data["module_dc_power_at_mpp"]
                * np.broadcast_to(
                    self.plant_parameters_processed["modules_per_string"] * self.plant_parameters_processed["strings_per_inverter"],
                    self._sim_shape_,
                ),
                inverter=inverter,
            )

        number_of_inverters = self.plant_parameters_processed.get("number_of_inverters", 1)
        self.sim_data["total_system_generation"] = self.sim_data["inverter_ac_power_at_mpp"] * np.broadcast_to(
            number_of_inverters, self._sim_shape_
        )

        total_capacity = (
            self.module.I_mp_ref
            * self.module.V_mp_ref
            * self.plant_parameters_processed["modules_per_string"]
            * self.plant_parameters_processed["strings_per_inverter"]
            * number_of_inverters
        )

        self.sim_data["capacity_factor"] = self.sim_data["total_system_generation"] / np.broadcast_to(
            total_capacity, self._sim_shape_
        )

        return self
