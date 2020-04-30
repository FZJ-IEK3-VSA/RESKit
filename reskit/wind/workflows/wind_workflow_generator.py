import geokit as gk
import reskit as rk

import pandas as pd
import numpy as np
from os import mkdir, environ
from os.path import join, isfile, isdir
from collections import OrderedDict, namedtuple
from types import FunctionType
from ...workflow_generator import WorkflowGenerator


class WindWorkflowGenerator(WorkflowGenerator):
    """
        I am a doc string
    """

    def __init__(self, placements, synthetic_power_curve_cut_out=25, synthetic_power_curve_rounding=1, hub_height_name="hub_height"):
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

            specificPower = rk.core.wind.compute_specific_power(
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
                self.powerCurveLibrary[pc] = rk.core.wind.PowerCurve.from_specific_power(
                    specific_power=float(sppow),
                    cutout=float(cutout))
            else:
                self.powerCurveLibrary[pc] = rk.core.wind.TurbineLibrary().loc[pc].PowerCurve

    def set_roughness(self, roughness):
        self.placements['roughness'] = roughness
        return self

    def estimate_roughness_from_land_cover(self, path, source_type):
        num = gk.raster.interpolateValues(path, self.locs, mode='near')
        self.placements['roughness'] = rk.core.wind.roughness_from_land_cover_classification(
            num, source_type)
        return self

    def logarithmic_projection_of_wind_speeds_to_hub_height(self):
        assert "roughness" in self.placements.columns
        assert hasattr(self, "elevated_wind_speed_height")

        self.sim_data['elevated_wind_speed'] = rk.core.wind.apply_logarithmic_profile_projection(
            self.sim_data['elevated_wind_speed'],
            measured_height=self.elevated_wind_speed_height,
            target_height=self.placements['hub_height'].values,
            roughness=self.placements['roughness'].values
        )
        self.wind_speed_height = self.placements['hub_height'].values

        return self

    def apply_air_density_correction_to_wind_speeds(self):
        assert "surface_air_temperature" in self.sim_data, "surface_air_temperature has not been read from a source"
        assert "surface_pressure" in self.sim_data, "surface_pressure has not been read from a source"
        assert hasattr(self, "elevated_wind_speed_height")

        self.sim_data['elevated_wind_speed'] = rk.core.wind.apply_air_density_adjustment(
            self.sim_data['elevated_wind_speed'],
            pressure=self.sim_data['surface_pressure'],
            temperature=self.sim_data['surface_air_temperature'],
            height=self.wind_speed_height)

        return self

    def convolute_power_curves(self, scaling, base, **kwargs):
        assert hasattr(self, "powerCurveLibrary")

        for key in self.powerCurveLibrary.keys():
            self.powerCurveLibrary[key] = self.powerCurveLibrary[key].convolute_by_guassian(
                scaling=scaling,
                base=base,
                **kwargs
            )

        return self

    def simulate(self):
        gen = np.zeros_like(self.sim_data['elevated_wind_speed'])

        for pckey, pc in self.powerCurveLibrary.items():
            sel = self.placements.powerCurve == pckey
            gen[:, sel] = pc.simulate(self.sim_data['elevated_wind_speed'][:, sel])

        self.sim_data['capacity_factor'] = gen

        return self
