import geokit as gk

import pandas as pd
import numpy as np
from os import mkdir, environ
from os.path import join, isfile, isdir
from collections import OrderedDict, namedtuple
from types import FunctionType

from . import weather as rk_weather


class WorkflowManager():
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

        assert 'lon' in self.placements.columns
        assert 'lat' in self.placements.columns

        if self.locs is None:
            self.locs = gk.LocationSet(self.placements[['lon', 'lat']].values)

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

    def _set_sim_shape(self):
        self._sim_shape_ = len(self._time_index_), self.locs.count

    def read(self, variables, source_type, path, set_time_index=False, spatial_interpolation_mode="bilinear", temporal_reindex_method="nearest", **kwargs):
        if not set_time_index and self.time_index is None:
            raise RuntimeError("Time index is not available")

        if source_type == "ERA5":
            source_constructor = rk_weather.Era5Source
        elif source_type == "SARAH":
            source_constructor = rk_weather.SarahSource
        elif source_type == "MERRA":
            source_constructor = rk_weather.MerraSource
        else:
            raise RuntimeError("Unknown source_type")

        source = source_constructor(
            path,
            bounds=self.ext,
            **kwargs)

        if set_time_index:
            self.set_time_index(source.time_index)

        # read variables
        if not isinstance(variables, list):
            variables = [variables, ]

        source.sload(*variables)

        for var in variables:
            self.sim_data[var] = source.get(
                var,
                self.locs,
                interpolation=spatial_interpolation_mode,
                force_as_data_frame=True)

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

    def to_xarray(self, output_netcdf_path=None, output_variables=None, _intermediate_dict=False):

        import xarray

        times = self.time_index
        if times[0].tz is not None:
            times = [np.datetime64(dt.tz_convert("UTC").tz_convert(None)) for dt in times]
        xds = OrderedDict()
        encoding = dict()

        if "location_id" in self.placements.columns:
            location_coords = self.placements['location_id'].copy()
            del self.placements['location_id']
        else:
            location_coords = np.arange(self.placements.shape[0])

        for c in self.placements.columns:
            if output_variables is not None:
                if c not in output_variables:
                    continue
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
                                          coords=dict(location=location_coords))

        for key in self.sim_data.keys():
            if output_variables is not None:
                if key not in output_variables:
                    continue

            tmp = np.full((len(self.time_index), self.locs.count), np.nan)
            tmp[self._time_sel_, :] = self.sim_data[key]

            xds[key] = xarray.DataArray(tmp,
                                        dims=["time", "location"],
                                        coords=dict(time=times,
                                                    location=location_coords))
            encoding[key] = dict(zlib=True)

        if _intermediate_dict:
            return xds

        xds = xarray.Dataset(xds)

        for k, v in self.workflow_parameters.items():
            xds.attrs[k] = v

        if output_netcdf_path is not None:
            xds.to_netcdf(output_netcdf_path, encoding=encoding)
            return output_netcdf_path
        else:
            return xds


def _split_locs(placements, groups):
    if groups == 1:
        yield placements
    else:
        locs = gk.LocationSet(placements.index)
        for loc_group in locs.splitKMeans(groups=groups):
            yield placements.loc[loc_group[:]]


def distribute_workflow(workflow_function, placements, jobs=2, max_batch_size=None, intermediate_output_dir=None, **kwargs):
    import xarray
    from multiprocessing import Pool

    assert isinstance(placements, pd.DataFrame)
    assert ("lon" in placements.columns and "lat" in placements.columns) or ("geom" in placements.columns)

    # Split placements into groups
    if "geom" in placements.columns:
        locs = gk.LocationSet(placements)
    else:
        locs = gk.LocationSet(np.column_stack([placements.lon.values, placements.lat.values]))

    placements.index = locs
    placements['location_id'] = np.arange(placements.shape[0])

    if max_batch_size is None:
        max_batch_size = int(np.ceil(placements.shape[0] / jobs))

    kmeans_groups = int(np.ceil(placements.shape[0] / max_batch_size))
    placement_groups = []
    for placement_group in _split_locs(placements, kmeans_groups):
        kmeans_groups_level2 = int(np.ceil(placement_group.shape[0] / max_batch_size))

        for placement_sub_group in _split_locs(placement_group, kmeans_groups_level2):
            placement_groups.append(placement_sub_group)

    # Do simulations
    pool = Pool(jobs)

    results = []
    for gid, placement_group in enumerate(placement_groups):

        kwargs_ = kwargs.copy()
        if intermediate_output_dir is not None:
            kwargs_['output_netcdf_path'] = join(intermediate_output_dir, "simulation_group_{:05d}.nc".format(gid))

        results.append(pool.apply_async(
            func=workflow_function,
            args=(placement_group, ),
            kwds=kwargs_
        ))

    xdss = []
    for result in results:
        xdss.append(result.get())

    pool.close()
    pool.join()

    if intermediate_output_dir is None:
        return xarray.concat(xdss, dim="location").sortby('location')
    else:
        return xdss


class WorkflowQueue():
    def __init__(self, workflow, **kwargs):
        self.workflow = workflow
        self.constants = kwargs
        self.queue = OrderedDict()

    def append(self, key, **kwargs):
        self.queue[key] = kwargs

    def execute(self, jobs: int = 1):
        assert jobs >= 1
        jobs = int(jobs)

        if jobs > 1:
            from multiprocessing import Pool
            pool = Pool(jobs)

        results = OrderedDict()
        for key, kwargs in self.queue.items():
            k = self.constants.copy()
            k.update(kwargs)

            if jobs == 1:
                results[key] = self.workflow(**k)
            else:
                results[key] = pool.apply_async(self.workflow, (), k)

        if jobs > 1:
            for key, result_ in results.items():
                results[key] = result_.get()

            pool.close()
            pool.join()

        return results
