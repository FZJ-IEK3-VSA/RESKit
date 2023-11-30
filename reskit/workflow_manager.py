import geokit as gk

import pandas as pd
import numpy as np
import os
from os import mkdir, environ
from os.path import join, isfile, isdir
from collections import OrderedDict, namedtuple
from types import FunctionType
import xarray
from typing import Union, List, OrderedDict
from . import weather as rk_weather
from glob import glob
from . import util as rk_util
import xarray as xr

# from . import weather as rk_weather


class WorkflowManager:
    """
    The WorkflowManager class assists with the construction of more specialized WorkflowManagers,
    such as the WindWorkflowManager or the SolarWorkflowManager. In addition to providing the
    general structure for simulation workflow management, the WorkflowManager also defines
    functionalities which should be common across all WorkflowManagers.

    This includes:
      - Basic initialization
      - Time domain management
      - Reading weather data
      - Adjusting variables by a long-run-average value
      - Applying simple loss factors
      - Saving the state of WorkflowManagers to XArray datasets, either in memory or on disc


    Initialization:
    ---------------

    WorkflowManager( placements )

    """

    def __init__(self, placements: pd.DataFrame):
        # arrange placements, locs, and extent
        assert isinstance(placements, pd.DataFrame)
        self.placements = placements.copy()
        self.locs = None

        # Check if input file contains a geometry collumn
        ispoint = False
        if "geom" in placements.columns:
            if self.placements["geom"].iloc[0].GetGeometryName() == "POINT":
                ispoint = True

        if ispoint:
            self.locs = gk.LocationSet(placements.geom)
            self.placements["lon"] = self.locs.lons
            self.placements["lat"] = self.locs.lats
            del self.placements["geom"]
        else:
            assert (
                "lon" in self.placements.columns
            ), "if geom are not point geometries, dataframe must contain lon columns"
            assert (
                "lat" in self.placements.columns
            ), "if geom are not point geometries, dataframe must contain lat columns"

        if self.locs is None:
            self.locs = gk.LocationSet(self.placements[["lon", "lat"]].values)

        self.ext = gk.Extent.fromLocationSet(self.locs)

        # Initialize simulation data
        self.sim_data = OrderedDict()
        self.time_index = None
        self.workflow_parameters = OrderedDict()

    # STAGE 2: weather data reading and adjusting

    def set_time_index(self, times: pd.DatetimeIndex):
        """Sets the time index of the WorkflowManager


        Parameters
        ----------
            times : pd.DatetimeIndex
                The timesteps to use throughout the WorkflowManager's life cycle. The
                length of this dataset must match the shape of data which is loaded into
                the WorkflorManager.sim_data member.
        """
        self.time_index = times

        self._time_sel_ = None
        self._time_index_ = self.time_index.copy()
        self._set_sim_shape()

    def _set_sim_shape(self):
        self._sim_shape_ = len(self._time_index_), self.locs.count

    def extract_raster_values_at_placements(self, raster, **kwargs):
        """Extracts pixel values at each of the configured placements from the specified raster file"""
        return gk.raster.interpolateValues(raster, points=self.locs, **kwargs)

    def read(
        self,
        variables: Union[str, List[str]],
        source_type: str,
        source: str,
        set_time_index: bool = False,
        spatial_interpolation_mode: str = "bilinear",
        temporal_reindex_method: str = "nearest",
        time_index_from=None,
        **kwargs,
    ):
        """Reads the specified variables from the NetCDF4-style weather dataset, and then extracts
        those variables for each of the coordinates configured in `.placements`. The resulting
        data is then available in `.sim_data`.

        Parameters
        ----------
        variables : str or list of strings
            The variables (or variables) to be read from the specified source
            - If a path to a weather source is given, then only the 'standard' variables
                configured for that source type are available (see the doc string for the
                weather source you are interested in)
            - If either 'elevated_wind_speed' or 'surface_wind_speed' is included in the
                variable list, then the members `.elevated_wind_speed_height` and
                `.surface_wind_speed_height`, respectfully, are also added. These are constants
                which specify what the 'native' wind speed height is, which depends on the source
            - A pre-loaded NCSource can also be given, thus allowing for any variable in the
                source to be specifed in the `variables` list. But the user needs to take care
                of initializing the NCSource and loading the data they want

        source_type : str
            The type of weather datasource which is to be loaded. Can be one of:
              "ERA5", "SARAH", "MERRA", or 'user'
            - If a pre-loaded NCSource is given for the `source` object, then the `source_type`
              should be "user"

        source : str or rk.weather.NCSource
            The source to read weather variables from

        set_time_index : bool, optional
            If True, instructs the workflow manager to set the time index to that which is read
              from the weather source
            - By default False

        spatial_interpolation_mode : str, optional
            The spatial interpolation mode to use while reading data from the weather source at
            each of the placement coordinates
            - By default "bilinear"

        temporal_reindex_method : str, optional
            In the event of missing data, this algorithm is used to fill in the missing data.
            - Can be, for example, "nearest", "ffill", "bfill", "interpolate"
            - By default "nearest"

        Returns
        -------
        WorkflowManager
            Returns the invoking WorkflowManager (for chaining)

        Raises
        ------
        RuntimeError
            If set_time_index is False but no `.time_index` exists
        RuntimeError
            If source_type is unknown
        """
        if not set_time_index and self.time_index is None:
            raise RuntimeError("Time index is not available")

        if isinstance(source, str) and source_type != "user":
            if source_type == "ERA5":
                source_constructor = rk_weather.Era5Source
            elif source_type == "SARAH":
                source_constructor = rk_weather.SarahSource
            elif source_type == "MERRA":
                source_constructor = rk_weather.MerraSource
            else:
                raise RuntimeError("Unknown source_type")

            if source_type == "ERA5":
                source = source_constructor(
                    source, bounds=self.ext, time_index_from=time_index_from, **kwargs
                )
            else:
                source = source_constructor(source, bounds=self.ext, **kwargs)

            # Load the requested variables
            source.sload(*variables)

        else:  # Assume source is already an initialized NCSource Object
            for var in variables:
                assert var in source.data

        if set_time_index:
            self.set_time_index(source.time_index)

        # read variables
        if not isinstance(variables, list):
            variables = [
                variables,
            ]

        for var in variables:
            self.sim_data[var] = source.get(
                var,
                self.locs,  # Manipulate locs here
                interpolation=spatial_interpolation_mode,
                force_as_data_frame=True,
            )

            if not set_time_index:
                self.sim_data[var] = self.sim_data[var].reindex(
                    self.time_index, method=temporal_reindex_method
                )

            self.sim_data[var] = self.sim_data[var].values

            # Special check for wind speed height
            if var == "elevated_wind_speed":
                self.elevated_wind_speed_height = source.ELEVATED_WIND_SPEED_HEIGHT

            if var == "surface_wind_speed":
                self.surface_wind_speed_height = source.SURFACE_WIND_SPEED_HEIGHT

        return self

    # Stage 3: Weather data adjusting & other intermediate steps
    def adjust_variable_to_long_run_average(
        self,
        variable: str,
        source_long_run_average: Union[str, float, np.ndarray],
        real_long_run_average: Union[str, float, np.ndarray],
        real_lra_scaling: float = 1,
        spatial_interpolation: str = "linear-spline",
        nodata_fallback: str = "nan",
    ):
        """Adjusts the average mean of the specified variable to a known long-run-average

        Note:
        -----
        uses the equation: variable[t] = variable[t] * real_long_run_average / source_long_run_average

        Parameters
        ----------
        variable : str
            The variable to be adjusted

        source_long_run_average : Union[str, float, np.ndarray]
            The variable's native long run average (the average in the weather file)
            - If a string is given, it is expected to be a path to a raster file which can be
              used to look up the average values from using the coordinates in `.placements`
            - If a numpy ndarray (or derivative) is given, the shape must be one of (time, placements)
              or at least (placements)

        real_long_run_average : Union[str, float, np.ndarray]
            The variables 'true' long run average
            - If a string is given, it is expected to be a path to a raster file which can be
              used to look up the average values from using the coordinates in `.placements`
            - If a numpy ndarray (or derivative) is given, the shape must be one of (time, placements)
              or at least (placements)

        real_lra_scaling : float, optional
            An optional scaling factor to apply to the values derived from `real_long_run_average`.
            - This is primarily useful when `real_long_run_average` is a path to a raster file
            - By default 1

        spatial_interpolation : str, optional
            When either `source_long_run_average` or `real_long_run_average` are a path to a raster
            file, this input specifies which interpolation algorithm should be used
            - Options are: "near", "linear-spline", "cubic-spline", "average"
            - By default "linear-spline"
            - See for more info: geokit.raster.interpolateValues

        nodata_fallback: str, optional
            When real_long_run_average has no data, it can be decided between fallbackoptions:
            -'source': use source data
            - 'nan': return np.nan for missing values
        Returns
        -------
        WorkflowManager
            Returns the invoking WorkflowManager (for chaining)
        """

        if isinstance(real_long_run_average, str):
            real_lra = gk.raster.interpolateValues(
                real_long_run_average, self.locs, mode=spatial_interpolation
            )
            # if getting values fails, it could be because of interpolation method.
            # thise values will be replaced with the nearest interpolation method
            if np.isnan(real_lra).any():
                real_lra_near = gk.raster.interpolateValues(
                    real_long_run_average, self.locs, mode="near"
                )
                real_lra[np.isnan(real_lra)] = real_lra_near[np.isnan(real_lra)]

            # assert not np.isnan(real_lra).any() and (real_lra > 0).all()
        else:
            real_lra = real_long_run_average

        if isinstance(source_long_run_average, str):
            source_lra = gk.raster.interpolateValues(
                source_long_run_average, self.locs, mode=spatial_interpolation
            )
            # if getting values fails, it could be because of interpolation method.
            # thise values will be replaced with the nearest interpolation method
            if np.isnan(source_lra).any():
                source_lra_nearest = gk.raster.interpolateValues(
                    source_long_run_average, self.locs, mode="near"
                )
                source_lra[np.isnan(source_lra)] = source_lra_nearest[
                    np.isnan(source_lra)
                ]
            # assert not np.isnan(source_lra).any() and (source_lra > 0).all()
        else:
            source_lra = source_long_run_average

        # calulate scaling factor:
        # nan result will stay nan results, as these placements cannot be calculated any more
        factors = real_lra * real_lra_scaling / source_lra

        # write info with missing values to sim_data:
        self.placements[
            f"missing_values_{os.path.basename(real_long_run_average)}"
        ] = np.isnan(factors)

        if nodata_fallback.lower() == "source":
            factors[np.isnan(factors)] = 1

        self.sim_data[variable] = factors * self.sim_data[variable]
        self.placements[f"LRA_factor_{variable}"] = factors
        return self

    def spatial_disaggregation(
        self,
        variable: str,
        source_high_resolution: Union[str, float, np.ndarray],
        source_low_resolution: Union[str, float, np.ndarray],
        real_lra_scaling: float = 1,
        spatial_interpolation: str = "linear-spline",
    ):
        """[summary]

        Parameters
        ----------
        variable : str
            [description]
        source_long_run_average : Union[str, float, np.ndarray]
            [description]
        real_long_run_average : Union[str, float, np.ndarray]
            [description]
        real_lra_scaling : float, optional
            [description], by default 1
        spatial_interpolation : str, optional
            [description], by default "linear-spline"
        """
        # Get values from high resolution tiff file
        if isinstance(source_high_resolution, str):
            correction_values_high_res = (
                gk.raster.interpolateValues(  # TODO change here
                    source_high_resolution, self.locs, mode=spatial_interpolation
                )
            )
            # assert not np.isnan(correction_values_high_res).any() and (correction_values_high_res > 0).all()
        else:
            correction_values_high_res = source_high_resolution

        # Get values from low resolution tiff file (meaned over eg. ERA5)
        if isinstance(source_low_resolution, str):
            correction_values_low_res = gk.raster.interpolateValues(  # TODO change here
                source_low_resolution, self.locs, mode=spatial_interpolation
            )
            # assert not np.isnan(correction_values_low_res).any() and (correction_values_low_res > 0).all()
        else:
            correction_values_low_res = source_low_resolution

        # correction factors:
        factors = correction_values_high_res / correction_values_low_res
        factors = np.nan_to_num(factors, nan=1 / real_lra_scaling)
        assert (factors > 0).all()

        # update values
        self.sim_data[variable] = self.sim_data[variable] * factors * real_lra_scaling
        return self

    # Stage 5: post processing
    def apply_loss_factor(
        self,
        loss: Union[float, np.ndarray, FunctionType],
        variables: Union[str, List[str]] = ["capacity_factor"],
    ):
        """Applies a loss factor onto a specified variable

        Parameters
        ----------
        loss : Union[float, np.ndarray, FunctionType]
            The loss factor(s) to be applied
            - If a float or a numpy ndarray is given, then the following operation is performed:
                > variable = variable * (1 - loss)
            - If a function is given, then  the following operation is performed:
                > variable = variable * (1 - loss(variable) )
            - If a numpy ndarray is given, it must be broadcastable to the variable's shape in
              `.sim_data`

        variables : Union[str, List[str]], optional
            The vairable or variables to apply the loss factor to
            - By default ["capacity_factor"]


        Returns
        -------
        WorkflowManager
            Returns the invoking WorkflowManager (for chaining)
        """

        for var in variables:
            if isinstance(loss, FunctionType):
                self.sim_data[var] *= 1 - loss(self.sim_data[var])
            else:
                self.sim_data[var] *= 1 - loss

        return self

    def register_workflow_parameter(self, key: str, value: Union[str, float]):
        """Add a parameter to the WorkflowManager which will be included in the output XArray dataset

        Parameters
        ----------
        key : str
            The workflow parameter's access key

        value : Union[str,float]
            The workflow parameter's value. Only strings and floats are allowed
        """
        self.workflow_parameters[key] = value

    def to_xarray(
        self,
        output_netcdf_path: str = None,
        output_variables: List[str] = None,
        _intermediate_dict=False,
    ) -> xarray.Dataset:
        """Generates an XArray dataset from the data currently contained in the WorkflowManager

        Note:
        - The `.placements` data is automatically added to the XArray dataset along the 'locations' dimension
        - The `workflow_parameters` data is autmatically added as dimensionless variables
        - The `.sim_data` is automatically added along the dimensions (time, locations)
        - The `.time_index` is automatically added along the dimension 'time'

        Parameters
        ----------
        output_netcdf_path : str, optional
            If given, the XArray dataset will be written to disc at the specified path
            - By default None

        output_variables : List[str], optional
            If given, specifies the variables which should be included in the resulting
            dataset. Otherwise all suitable variables found in `.placements`, `.workflow_parameters`,
            `.sim_data`, and `.time_index` will be included
            - Only variables of numeric or string type are suitable due to NetCDF4 limitations
            - By default None

        Returns
        -------
        xarray.Dataset
            The resulting XArray dataset
        """
        if isinstance(output_variables, str):
            output_variables = [output_variables]
        if isinstance(output_variables, list) and not "RESKit_sim_order" in output_variables:
            output_variables.append('RESKit_sim_order')

        times = self.time_index
        if times[0].tz is not None:
            times = [
                np.datetime64(dt.tz_convert("UTC").tz_convert(None)) for dt in times
            ]
        times_days = np.unique(pd.DatetimeIndex(times).date).astype("datetime64")
        if times_days[0].astype("datetime64[Y]") != times_days[-1].astype(
            "datetime64[Y]"
        ):
            # old tiles where shifted by 1 hour, so the last day of the previous year also appears. catch this problem whti this if clause
            times_days = times_days[1:]
        xds = OrderedDict()
        encoding = dict()

        if "location_id" in self.placements.columns:
            location_coords = self.placements["location_id"].copy()
            del self.placements["location_id"]
        else:
            location_coords = np.arange(self.placements.shape[0])

        # write placements
        for c in self.placements.columns:
            # check if c in requestet output_variables
            if output_variables is not None:
                if c not in output_variables:
                    continue
            if np.issubdtype(self.placements[c].dtype, np.number):
                write = True
            else:
                write = True
                for element in self.placements[c]:
                    if not isinstance(element, (str, bytearray)):
                        write = False
                        break
            if write:
                xds[c] = xarray.DataArray(
                    self.placements[c],
                    dims=["location"],
                    coords=dict(location=location_coords),
                )

        # write sim_data
        for key in self.sim_data.keys():
            # check if key in requestet output_variables
            if output_variables is not None:
                if key not in output_variables:
                    continue

            tmp = np.full((len(self.time_index), self.locs.count), np.nan)
            tmp[self._time_sel_, :] = self.sim_data[key]

            xds[key] = xarray.DataArray(
                tmp,
                dims=["time", "location"],
                coords=dict(time=times, location=location_coords),
            )
            encoding[key] = dict(zlib=True)

        # write sim_data_daily, only if exists
        if hasattr(self, "sim_data_daily"):
            for key in self.sim_data_daily.keys():
                # check if key in requestet output_variables
                if output_variables is not None:
                    if key not in output_variables:
                        continue

                tmp = np.full((len(times_days), self.locs.count), np.nan)
                tmp[:, :] = self.sim_data_daily[key]

                xds[key] = xarray.DataArray(
                    tmp,
                    dims=["time_days", "location"],
                    coords=dict(time_days=times_days, location=location_coords),
                )
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

    def to_netcdf(
        self,
        xds: xarray.Dataset,
        output_netcdf_path: str = None,
        output_variables: List[str] = None,
        _intermediate_dict=False,
    ) -> str:
        """Saves an XArray dataset to netCDF4 format

        Note:
        - The `.placements` data is automatically added to the XArray dataset along the 'locations' dimension
        - The `workflow_parameters` data is autmatically added as dimensionless variables
        - The `.sim_data` is automatically added along the dimensions (time, locations)
        - The `.time_index` is automatically added along the dimension 'time'

        Parameters
        ----------
        output_netcdf_path : str
            If given, the XArray dataset will be written to disc at the specified path
            - By default None

        output_variables : List[str], optional
            If given, specifies the variables which should be included in the resulting
            dataset. Otherwise all suitable variables found in `.placements`, `.workflow_parameters`,
            `.sim_data`, and `.time_index` will be included
            - Only variables of numeric or string type are suitable due to NetCDF4 limitations
            - By default None

        Returns
        -------
        output_netcdf_path
            The resulting output_netcdf_path
        """
        encoding = dict()

        # write sim_data
        for key in self.sim_data.keys():
            # check if key in requestet output_variables
            if output_variables is not None:
                if key not in output_variables:
                    continue
            encoding[key] = dict(zlib=True)

        # FIXME: WHAT IS THIS FOR?
        # #write sim_data_daily, only if exists
        # if hasattr(self, 'sim_data_daily'):
        #     for key in self.sim_data_daily.keys():
        #         #check if key in requestet output_variables
        #         if output_variables is not None:
        #             if key not in output_variables:
        #                 continue
        #         encoding[key] = dict(zlib=True)

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


def distribute_workflow(
    workflow_function: FunctionType,
    placements: pd.DataFrame,
    jobs: int = 2,
    max_batch_size: int = None,
    intermediate_output_dir: str = None,
    **kwargs,
) -> xarray.Dataset:
    """Distributes a RESKit simulation workflow across multiple CPUs

    Parallelism is achieved by breaking up the placements dataframe into placement groups via
      KMeans grouping

    Parameters
    ----------
    workflow_function : FunctionType
        The workflow function to be parallelized
        - All RESKit workflow functions should be suitable here
        - If you want to make your own function, the only requirement is that its first argument
          should be a pandas DataFrame in the form of a placements table (i.e. has a 'lat' and
          'lon' column)
        - Don't forget that that all inputs required for the workflow function are still required,
          and are passed on as constants through any specified `kwargs`

    placements : pandas.DataFrame
        A DataFrame describing the placements to be simulated
        For example, if you are simulating wind turbines, the following columns are likely required:
            ['lon','lat','capacity','hub_height','rotor_diam',]

    jobs : int, optional
        The number of parallel jobs
        - By default 2

    max_batch_size : int, optional
        If given, limits the maximum number of total placements which are simulated in parallel
        - Use this to reduce the memory requirements of the simulations (in turn increasing
          overall simulation time)
        - By default None

    intermediate_output_dir : str, optional
        In case of very large outputs (which are too large to be joined into a singular XArray dataset),
          use this to write the individual simulation results to the specified directory
        - By default None

    **kwargs:
        All all key word arguments are passed on as constants to each simulation
        - Use these to set the required arguments for the given `workflow_function`

    Returns
    -------
    xarray.Dataset
        An XArray Dataset which contains the combined results of the distributed simulations

    """
    import xarray
    from multiprocessing import Pool

    assert isinstance(placements, pd.DataFrame)
    assert ("lon" in placements.columns and "lat" in placements.columns) or (
        "geom" in placements.columns
    )

    # Split placements into groups
    if "geom" in placements.columns:
        locs = gk.LocationSet(placements)
        placements["lat"] = locs.lats
        placements["lon"] = locs.lons
        del placements["geom"]
    else:
        locs = gk.LocationSet(
            np.column_stack([placements.lon.values, placements.lat.values])
        )

    placements.index = locs
    placements["location_id"] = np.arange(placements.shape[0])

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
            kwargs_["output_netcdf_path"] = join(
                intermediate_output_dir, "simulation_group_{:05d}.nc".format(gid)
            )

        results.append(
            pool.apply_async(
                func=workflow_function, args=(placement_group,), kwds=kwargs_
            )
        )
        # results.append(workflow_function(placement_group, **kwargs_ ))

    xdss = []
    for result in results:
        xdss.append(result.get())

    pool.close()
    pool.join()

    if intermediate_output_dir is None:
        return xarray.concat(xdss, dim="location").sortby("location")
    else:
        # return load_workflow_result(xdss)
        return xdss


def load_workflow_result(datasets, loader=xarray.load_dataset, sortby="location"):
    if isinstance(datasets, str):
        if isdir(datasets):
            datasets = glob(join(datasets, "*.nc"))
        else:
            datasets = glob(datasets)

    if len(datasets) == 1:
        ds = xarray.load_dataset(datasets[0]).sortby("locations")
    else:
        ds = xarray.concat(map(loader, datasets), dim="location")

    if sortby is not None:
        ds = ds.sortby(sortby)

    return ds


def execute_workflow_iteratively(
        workflow,
        weather_path_varname,
        zoom=None,
        **workflow_args,
):
    """
    The function executes the indicated workflow iteratively, iterating over weather tiles. The appropriate weather
    tile per placement is extracted automatically and placements are batched together based on weather tile.

    workflow : RESkit workflow
        Callable workflow function, e.g. reskit.wind.wind_era5_2023 
    weather_path_varname : str
        Str formatted name of the weather path variable in this workflow, e.g. 'era5_path' for 
        reskit.wind.wind_era5_2023. Must must be a key of workflow_args.
    zoom : int, optional
        The zoom level of the weather tiles, required only if <X-TILE> or <Y-TILE> in weather path.
    **workflow_args 
        Passed on to the workflow specified above. Must contain 'placements' and the above 
        weather_path_varname as keys.
    """
    # check key inputs
    assert callable(
        workflow), f"workflow must be a callable RESkit workflow function."
    assert 'placements' in workflow_args.keys(
    ), f"'placements' is a mandatory argument/key in workflow_args"
    assert weather_path_varname in workflow_args.keys(
    ), f"weather_path_varname ('{weather_path_varname}')  must be a key in workflow_args."

    # extract data needed for placement preparation
    placements = workflow_args['placements']
    weather_path = workflow_args[weather_path_varname]
    if 'output_netcdf_path' in workflow_args.keys():
        output_netcdf_path = workflow_args['output_netcdf_path']
    else:
        output_netcdf_path = None
    if 'output_variables' in workflow_args.keys():
        output_variables = workflow_args['output_variables']
    else:
        output_variables = None

    # possibly generate dataframe from single locations and add actual weather filepath where needed
    if not weather_path_varname in placements.columns:
        placements = rk_util.get_dataframe_with_weather_tilepaths(
            placements=placements,
            weather_path=weather_path,
            zoom=zoom)

    # remove output saving for the iterative function execution of sub dfs
    workflow_args.update({'output_netcdf_path': None})

    # iterate over weather tiles
    xrds_list = []
    for i, tilepath in enumerate(placements['source'].unique()):
        # reduce placements to subset within the current tile and update function arguments with subset of placements and current weather path
        placements_tile = placements[placements['source'] == tilepath]
        workflow_args.update(
            {'placements': placements_tile, weather_path_varname: tilepath}, )
        # execute workflow with subset and add to list of results
        print(
            f"Now processing tile {i+1}/{len(placements['source'].unique())} with {len(placements_tile)} locations: {tilepath}")
        xrds = workflow(**workflow_args)
        xrds = xrds.set_index(location="RESKit_sim_order")
        xrds_list.append(xrds)

    reskit_xr = xr.concat(xrds_list, dim="location")
    # create a dummy wfm instance for saving
    wfm = WorkflowManager(
        placements=placements.drop(columns='RESKit_sim_order'))
    wfm.to_netcdf(
        xds=reskit_xr,
        output_netcdf_path=output_netcdf_path,
        output_variables=output_variables,
    )

    return reskit_xr


class WorkflowQueue:
    """The WorkflowQueue object allows for the queueing of multiple RESKit workflow simulations
    which are then executed in parallel

    Initialize:
    -----------
    WorkflowFunction( workflow:FunctionType, **kwargs )

    Parameters:
    -----------
    workflow : FunctionType
        The workflow function to be parallelized
        - All RESKit workflow functions should be suitable here
        - Don't forget that that all inputs required for the workflow function are still required,
          and are passed on either as constants through `kwargs` specified in the initializer, or
          else in the subsequent '.append(...)' calls

    **kwargs:
        All key word arguments are passed on as constants to each simulation
        - Use these to set the required arguments for the given `workflow`

    """

    def __init__(self, workflow: FunctionType, **kwargs):
        self.workflow = workflow
        self.constants = kwargs
        self.queue = OrderedDict()

    def append(self, key: str, **kwargs):
        """Appends a simulation set the current queue

        Parameters
        ----------
        key : str
            The access key to use for this simulation set

        **kwargs:
            All other keyword arguments are passed on to the simulation
            for only this simulation
        """
        self.queue[key] = kwargs

    def execute(self, jobs: int = 1) -> OrderedDict[str, xarray.Dataset]:
        """Executes all of the simulation sets that are currently in the queue

        Parameters
        ----------
        jobs : int, optional
            The number of parallel jobs, by default 1

        Returns
        -------
        OrderedDict[xarray.Dataset]
            The results of each simulation set, accessable via their access keys
        """
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
