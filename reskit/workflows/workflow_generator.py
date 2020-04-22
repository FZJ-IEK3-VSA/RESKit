import geokit as gk
import reskit as rk
from reskit import windpower

import pandas as pd
import numpy as np
from os import mkdir, environ
from os.path import join, isfile, isdir
from collections import OrderedDict, namedtuple
from types import FunctionType


class WorkflowGenerator():

    self.required_data = []

    def __init__(self, placements):
        # arrange placements, locs, and extent
        assert isinstance(placements, pd.DataFrame)
        self.placements = placements
        self.locs = None

        if 'geom' in placements.columns:
            self.locs = gk.LocationSet(placements.geom)
            self.placements['lon'] = self.locs.lons
            self.placements['lat'] = self.locs.lats
            del self.placements['geom']

        assert 'lon' in placements.columns
        assert 'lat' in placements.columns

        if self.locs is None:
            self.locs = gk.LocationSet(self.placements)

        self.ext = gk.Extent.fromLocationSet(self.locs)

        # Initialize simulation data
        self.sim_data = OrderedDict()
        self.variables = None
        self.sources = OrderedDict()
        self.main_source = None
        self.source_interpolation_mode = OrderedDict()

    # STAGE 1: configuring
    def for_onshore_wind_energy(self, synthetic_power_curve_cut_out=25, synthetic_power_curve_rounding=1):
        from reskit import windpower

        # Check for basics
        assert 'capacity' in self.placements.columns, "Placement dataframe needs 'capacity' column"
        assert 'hubHeight' in self.placements.columns, "Placement dataframe needs 'hubHeight' column"

        # Check for power curve. If not found, make it!
        self.powerCurveLibrary = dict()

        # Should we automatically generate synthetic power curves?
        if not "powerCurve" in self.placements.columns:
            assert 'rotordiam' in self.placements.columns, "Placement dataframe needs 'rotordiam' or 'powerCurve' column"

            specificPower = windpower.specificPower(self.placements['capacity'],
                                                    self.placements['rotordiam'])

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
            assert isinstance(
                pc, str), "Power curve value needs to be a string, not " + type(pc)

            if pc in self.powerCurveLibrary:
                continue

            if pc[:4] == "SPC:":
                pc = pc.split(":")[1]
                sppow, cutout = pc.split(",")
                self.powerCurveLibrary[pc] = windpower.SyntheticPowerCurve(
                    specificCapacity=float(sppow),
                    cutout=float(cutout))
            else:
                self.powerCurveLibrary[pc] = windpower.TurbineLibrary[pc].PowerCurve

        return self

    # STAGE 2: weather data reading and adjusting
    def with_source(self, source_type, path, interpolation_mode="bilinear", **kwargs):
        if source_type == "ERA5":
            self.sources[source_type] = rk.weather.sources.Era5Source(
                path, bounds=self.ext, **kwargs)
        elif source_type == "SARAH":
            self.sources[source_type] = rk.weather.sources.SarahSource(
                path, bounds=self.ext, **kwargs)
        elif source_type == "MERRA":
            self.sources[source_type] = rk.weather.sources.MerraSource(
                path, bounds=self.ext, **kwargs)
        else:
            raise RuntimeError("Unknown source_type")

        if not "times" in self.sim_data:
            self.sim_data['times'] = self.sources[source_type].timeindex
            self.main_source = source_type

        self.source_interpolation_mode[source_type] = interpolation_mode

        return self

    def read(self, var, source=None):
        if from_source is None:
            from_source = self.main_source

        self.sim_data[var] = self.sources[source].get(
            var,
            self.locs,
            interpolation=self.source_interpolation_mode[source],
            forceDataFrame=True)

        if not source == self.main_source:
            self.sim_data[var] = self.sim_data[var].reindex(
                self.sim_data['times'],
                method='ffill')

        # Special check for wind speed height
        if var == "wind_speed_for_wind_energy":
            self.wind_speed_height = self.sources[source].WIND_SPEED_HEIGHT_FOR_WIND_ENERGY

        return self

    # Stage 3: Weather data adjusting & other intermediate steps
    def apply_long_run_average(self, variable, source_lra, real_lra, interpolation="linear-spline"):

        real_lra = gk.raster.interpolateValues(real_lra, self.locs)
        assert np.isnan(real_lra).any() and (real_lra > 0).all()

        source_lra = gk.raster.interpolateValues(source_lra, self.locs)
        assert np.isnan(source_lra).any() and (source_lra > 0).all()

        self.sim_data[variable] *= real_lra / source_lra
        return self

    def estimate_roughness_from_land_cover(self, path, source_type):
        num = gk.raster.interpolateValues(path, self.locs, mode='near')
        self.roughness = rk.weather.windutil.roughnessFromLandCover(
            num, source_type)
        return self

    def project_wind_speeds_to_hub_height_with_log_law(self):
        assert "roughness" in self.placements.columns
        assert hasattr(self, "wind_speed_height")

        self.sim_data['windspeed'] = rk.weather.windutil.projectByLogLaw(
            self.sim_data['wind_speed'],
            measuredHeight=self.wind_speed_height,
            targetHeight=self.placements['hubHeight'].values,
            roughness=self.placements['roughness'].values
        )
        self.wind_speed_height = self.placements['hubHeight'].values

        return self

    def apply_air_density_correction_to_wind_speeds(self):
        assert "air_temp" in self.sim_data, "air_temp has not been read from a source"
        assert "pressure" in self.sim_data, "pressure has not been read from a source"
        assert hasattr(self, "wind_speed_height")

        self.sim_data['windspeed'] = rk.weather.windutil.densityAdjustment(
            self.sim_data['windspeed'],
            pressure=self.sim_data['pressure'],
            temperature=self.sim_data['temperature'],
            height=self.wind_speed_height)

        return self

    def convolute_power_curves(self, stdScaling, stdBase, **kwargs):
        from reskit import windpower

        assert hasattr(self, "powerCurveLibrary")

        for key in self.powerCurveLibrary.keys():
            self.powerCurveLibrary[key] = windpower.convolutePowerCurveByGuassian(
                self.powerCurveLibrary[key],
                stdScaling=stdScaling,
                stdBase=stdBase,
                **kwargs
            )

        return self

    # Stage 4: Do simulation!
    def simulate_onshore_wind_energy(self):
        gen = pd.DataFrame(np.nan,
                           index=self.sim_data['times'],
                           columns=self.locs)

        for pckey, pc in self.powerCurveLibrary.items():
            sel = self.placements.powerCurve == pckey
            placements = self.placements[sel]

            # Do simulation
            gen_ = rk.windpower.simulateTurbine(
                self.sim_data['windspeed'].values[:, sel],
                powerCurve=pc,
                capacity=placements['capacity'].values,
                rotordiam=placements['rotordiam'].values,
                hubHeight=placements['hubHeight'].values,
                loss=0.00)

            gen.values[:, sel] = gen_.values

        self.sim_data['capacity_factor'] = gen

        return self

    # Stage 5: post processing
    def apply_loss_factor(self, loss):
        if isinstance(loss, FunctionType):
            self.sim_data['capacity_factor'] = loss(
                self.sim_data['capacity_factor'])
        else:
            self.sim_data['capacity_factor'] *= loss

        return self


