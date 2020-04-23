import geokit as gk
import reskit as rk

import pandas as pd
import numpy as np
from os import mkdir, environ
from os.path import join, isfile, isdir
from collections import OrderedDict, namedtuple
from types import FunctionType


class WorkflowGenerator():
    def __init__(self, placements):
        # arrange placements, locs, and extent
        assert isinstance(placements, pd.DataFrame)
        self.placements = placements.copy()
        self.locs = None

        if 'geom' in placements.columns:
            self.locs = gk.LocationSet(placements.geom)
            self.placements['lon'] = self.locs.lons
            self.placements['lat'] = self.locs.lats
            del self.placements['geom']

        assert 'lon' in placements.columns
        assert 'lat' in placements.columns

        if self.locs is None:
            self.locs = gk.LocationSet(placements[['lon', 'lat']].values)

        self.ext = gk.Extent.fromLocationSet(self.locs)

        # Initialize simulation data
        self.sim_data = OrderedDict()
        self.time_index = None
        self.workflow_parameters = OrderedDict()

    # STAGE 2: weather data reading and adjusting

    def set_time_index(self, times):
        self.time_index = times

        self._time_sel_ = None
        self._time_index_ = self.time_index.copy()
        self._set_sim_shape()

        return self

    def _set_sim_shape(self):
        self._sim_shape_ = len(self._time_index_), self.locs.count

    def read(self, variables, source_type, path, set_time_index=False, spatial_interpolation_mode="bilinear", temporal_reindex_method="ffill", **kwargs):
        if not set_time_index and self.time_index is None:
            raise RuntimeError("Time index is not available")

        if source_type == "ERA5":
            source_constructor = rk.weather.sources.Era5Source
        elif source_type == "SARAH":
            source_constructor = rk.weather.sources.SarahSource
        elif source_type == "MERRA":
            source_constructor = rk.weather.sources.MerraSource
        else:
            raise RuntimeError("Unknown source_type")

        source = source_constructor(
            path,
            bounds=self.ext,
            **kwargs)

        if set_time_index:
            self.set_time_index(source.timeindex)

        # read variables
        if not isinstance(variables, list):
            variables = [variables, ]

        source.sload(*variables)

        for var in variables:
            self.sim_data[var] = source.get(
                var,
                self.locs,
                interpolation=spatial_interpolation_mode,
                forceDataFrame=True)

            if not set_time_index:
                self.sim_data[var] = self.sim_data[var].reindex(
                    self.time_index,
                    method=temporal_reindex_method)

            self.sim_data[var] = self.sim_data[var].values

            # Special check for wind speed height
            if var == "elevated_wind_speed":
                self.elevated_wind_speed_height = source.ELEVATED_WIND_SPEED_HEIGHT

            if var == "surface_wind_speed":
                self.surface_wind_speed_height = source.SURFACE_WIND_SPEED_HEIGHT

        return self

    # Stage 3: Weather data adjusting & other intermediate steps
    def adjust_variable_to_long_run_average(self, variable, source_long_run_average, real_long_run_average, real_lra_scaling=1, spatial_interpolation="linear-spline"):

        if isinstance(real_long_run_average, str):
            real_lra = gk.raster.interpolateValues(
                real_long_run_average,
                self.locs,
                mode=spatial_interpolation)
            assert not np.isnan(real_lra).any() and (real_lra > 0).all()
        else:
            real_lra = real_long_run_average

        if isinstance(source_long_run_average, str):
            source_lra = gk.raster.interpolateValues(
                source_long_run_average,
                self.locs,
                mode=spatial_interpolation)
            assert not np.isnan(source_lra).any() and (source_lra > 0).all()
        else:
            source_lra = source_long_run_average

        self.sim_data[variable] *= real_lra * real_lra_scaling / source_lra
        return self

    # Stage 5: post processing
    def apply_loss_factor(self, loss, variables=["capacity_factor"]):
        for var in variables:
            if isinstance(loss, FunctionType):
                self.sim_data[var] *= 1 - loss(
                    self.sim_data[var])
            else:
                self.sim_data[var] *= 1 - loss

        return self

    def register_workflow_parameter(self, key, value):
        self.workflow_parameters[key] = value

    def to_xarray(self, output_netcdf_path=None, _intermediate_dict=False):

        import xarray

        times = self.time_index
        if times[0].tz is not None:
            times = [np.datetime64(dt.tz_convert("UTC").tz_convert(None)) for dt in times]
        xds = OrderedDict()
        encoding = dict()

        for c in self.placements.columns:
            if np.issubdtype(self.placements[c].dtype, np.number):
                write = True
            else:
                write = True
                for element in self.placements[c]:
                    if not isinstance(element, str):
                        write = False
                        break
            if write:
                xds[c] = xarray.DataArray(self.placements[c],
                                          dims=["location"],
                                          coords=dict(location=range(self.locs.count)))

        for key in self.sim_data.keys():
            tmp = np.full((len(self.time_index), self.locs.count), np.nan)
            tmp[self._time_sel_, :] = self.sim_data[key]

            xds[key] = xarray.DataArray(tmp,
                                        dims=["time", "location"],
                                        coords=dict(time=times,
                                                    location=range(self.locs.count)))
            encoding[key] = dict(zlib=True)

        if _intermediate_dict:
            return xds

        xds = xarray.Dataset(xds)

        for k, v in self.workflow_parameters.items():
            xds.attrs[k] = v

        if output_netcdf_path is not None:
            xds.to_netcdf(output_netcdf_path, encoding=encoding)

        return xds
