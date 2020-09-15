import geokit as gk
import pandas as pd
import numpy as np

from os import mkdir, environ
from os.path import join, isfile, isdir
from collections import OrderedDict, namedtuple
from types import FunctionType
from .. import core as rk_wind_core
from ...workflow_manager import WorkflowManager


class WindWorkflowManager(WorkflowManager):
    """
    Helps managing the logical workflow for simulations relating to wind turbines.

    Initialization:

    Parameters
    ----------
    placements : pandas.DataFrame
        A Pandas DataFrame describing the wind turbine placements to be simulated.
        It must include the following columns:
            'geom' or 'lat' and 'lon'
            'hub_height'
            'capacity'
            'rotor_diam' or 'powerCurve'

    synthetic_power_curve_cut_out : int, optional
        cut out wind speed, by default 25

    synthetic_power_curve_rounding : int, optional
        rounding floor, by default 1

    Returns
    -------
    numpy array
        A corrected power curve.

    """

    def __init__(self, placements, synthetic_power_curve_cut_out=25, synthetic_power_curve_rounding=1):
        # Do basic workflow construction
        super().__init__(placements)

        # Check for basics
        assert 'capacity' in self.placements.columns, "Placement dataframe needs 'capacity' column"
        assert 'hub_height' in self.placements.columns, "Placement dataframe needs 'hub_height' column"

        # Check for power curve. If not found, make it!
        self.powerCurveLibrary = dict()

        # Should we automatically generate synthetic power curves?
        if not "powerCurve" in self.placements.columns:
            assert 'rotor_diam' in self.placements.columns, "Placement dataframe needs 'rotor_diam' or 'powerCurve' column"

            specificPower = rk_wind_core.power_curve.compute_specific_power(
                self.placements['capacity'],
                self.placements['rotor_diam'])

            if synthetic_power_curve_rounding is not None:
                specificPower = np.round(
                    specificPower / synthetic_power_curve_rounding) * synthetic_power_curve_rounding
                specificPower = specificPower.astype(int)

            powerCurve = []
            for sppow in specificPower:
                pcid = "SPC:%d,%d" % (sppow, synthetic_power_curve_cut_out)
                powerCurve.append(pcid)

            self.placements['powerCurve'] = powerCurve

        # Put power curves into the power curve library
        for pc in self.placements.powerCurve.values:
            assert isinstance(pc, str), \
                "Power curve value needs to be a string, not " + type(pc)

            if pc in self.powerCurveLibrary:
                continue

            if pc[:4] == "SPC:":
                sppow, cutout = pc.split(":")[1].split(",")
                self.powerCurveLibrary[pc] = rk_wind_core.power_curve.PowerCurve.from_specific_power(
                    specific_power=float(sppow),
                    cutout=float(cutout))
            else:
                self.powerCurveLibrary[pc] = rk_wind_core.turbine_library.TurbineLibrary().loc[pc].PowerCurve

    def set_roughness(self, roughness):
        """
        Sets the 'roughness' column in the placements DataFrame.

        Parameters
        ----------
        roughness : numeric, iterable
            If a numeric is given, sets the same roughness values to all placements.
            If an iterable is given, sets the corresponding roughness value in the iterable to the placements.
            The length of the iterable must match the number of placements 

        Return
        ------
            A reference to the invoking WindWorkflowManager
        """
        self.placements['roughness'] = roughness
        return self

    def estimate_roughness_from_land_cover(self, path, source_type):
        """
        Estimates the 'roughness' value column in the placements DataFrame from a given land cover classification raster file.

        Parameters
        ----------
        path : str 
            path to the raster file
        source_type : str
            string value to get the corresponding key-value pairs. Accepted types 'clc', 'clc-code', 'globCover', 'modis', or 'cci', by default 'clc'

        See also
        --------
            roughness_from_land_cover_classification

        Return
        --------
            A reference to the invoking WindWorkflowManager
        """
        num = gk.raster.interpolateValues(path, self.locs, mode='near')
        self.placements['roughness'] = rk_wind_core.logarithmic_profile.roughness_from_land_cover_classification(
            num, source_type)
        return self

    def logarithmic_projection_of_wind_speeds_to_hub_height(self, consider_boundary_layer_height=False):
        """
        Projects the wind speed values to the hub height.

        Return
        ------
            A reference to the invoking WindWorkflowManager
        """

        assert "roughness" in self.placements.columns
        assert hasattr(self, "elevated_wind_speed_height")

        if consider_boundary_layer_height:
            # When the hub height is above the PBL, then only project to the PBL
            target_height = np.minimum(
                    self.sim_data['boundary_layer_height'],
                    self.placements['hub_height'].values)

            # When the PBL is below the elevated_wind_speed_height, then no projection
            # should be performed. This can be effectlvely accomplished by setting the 
            # target height to that of the elevated_wind_speed_height
            sel = target_height > self.elevated_wind_speed_height
            target_height[sel] = self.elevated_wind_speed_height

        else:
            target_height = self.placements['hub_height'].values

        tmp = rk_wind_core.logarithmic_profile.apply_logarithmic_profile_projection(
            self.sim_data['elevated_wind_speed'],
            measured_height=self.elevated_wind_speed_height,
            target_height=target_height,
            roughness=self.placements['roughness'].values
        )
        
        self.sim_data['elevated_wind_speed'] = tmp
        
        self.elevated_wind_speed_height = self.placements['hub_height'].values

        return self

    def apply_air_density_correction_to_wind_speeds(self):
        """
        Applies air density corrections to the wind speeds at the hub height.

        Return
        ------
            A reference to the invoking WindWorkflowManager


        """

        assert "surface_air_temperature" in self.sim_data, "surface_air_temperature has not been read from a source"
        assert "surface_pressure" in self.sim_data, "surface_pressure has not been read from a source"
        assert hasattr(self, "elevated_wind_speed_height")

        self.sim_data['elevated_wind_speed'] = rk_wind_core.air_density_adjustment.apply_air_density_adjustment(
            self.sim_data['elevated_wind_speed'],
            pressure=self.sim_data['surface_pressure'],
            temperature=self.sim_data['surface_air_temperature'],
            height=self.elevated_wind_speed_height)

        return self

    def convolute_power_curves(self, scaling, base, **kwargs):
        """
        Convolutes a turbine power curve from a normal distribution function with wind-speed-dependent standard deviation.

        Parameters
        ----------
        scaling : float, optional
            scaling factor, by default 0.06
        base : float, optional
            base value, by default 0.1

        Return
        ------
            A reference to the invoking WindWorkflowManager

        """
        assert hasattr(self, "powerCurveLibrary")

        for key in self.powerCurveLibrary.keys():
            self.powerCurveLibrary[key] = self.powerCurveLibrary[key].convolute_by_gaussian(
                scaling=scaling,
                base=base,
                **kwargs
            )

        return self

    def simulate(self):
        """
        Applies the invoking power curve to the given wind speeds.

        Return
        ------
            A reference to the invoking WindWorkflowManager
        """

        gen = np.zeros_like(self.sim_data['elevated_wind_speed'])

        for pckey, pc in self.powerCurveLibrary.items():
            sel = self.placements.powerCurve == pckey
            gen[:, sel] = pc.simulate(self.sim_data['elevated_wind_speed'][:, sel])

        self.sim_data['capacity_factor'] = gen

        return self

    def interpolate_raster_vals_to_hub_height(self, name: str, height_to_raster_dict: dict, **kwargs):
        """Given several raster datasets which correspond to a desired value (e.g. average wind speed) at 
        different altitudes, this function will read values for each placement location from each of these 
        datasets, and will then linearly interpolate them to the hub height of each turbine

        Parameters
        ----------
        name : str 
            The name of the variable to create (will be placed in the `self.placements` member)

        height_to_raster_dict : dict
            A dictionary which maps altitude values to raster datasets

        Return
        ------
            A reference to the invoking WindWorkflowManager
        """
        known_heights = sorted(height_to_raster_dict.keys())
        known_vals = []
        for h in known_heights:
            known_vals.append(
                self.extract_raster_values_at_placements(
                    height_to_raster_dict[h],
                    **kwargs
                ))

        interpolated_vals = np.full_like(known_vals[0], np.nan)
        hh = self.placements['hub_height'].values
        for hi in range(len(known_heights) - 1):
            sel = np.logical_and(
                hh >= known_heights[hi],
                hh < known_heights[hi + 1])
            if sel.any():
                interpolated_vals[sel] = (hh[sel] - known_heights[hi]) / (known_heights[hi + 1] - known_heights[hi]) * (known_vals[hi + 1] - known_vals[hi]) + known_vals[hi]

        if np.isnan(interpolated_vals).any():
            raise RuntimeError("Could not determine interpolation for all hub heights")

        self.placements[name] = interpolated_vals
        return self
