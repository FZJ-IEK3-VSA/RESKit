# import base packages
import datetime
import warnings
from collections import OrderedDict
from collections.abc import Iterable
from glob import glob
from itertools import compress
from os.path import basename, isdir, isfile, join
from types import FunctionType
from typing import Any, List, Union

# import third party packages
import geokit as gk
import numpy as np
import pandas as pd
import xarray
from pandas.api.types import is_numeric_dtype

from reskit import weather as rk_weather

# import other modules
from reskit.util.paths import as_path_string, is_path_like
from reskit.util.weather_tile import get_location_specific_weather_paths


# The smallest half width, in SRS units, which is added to a zero-width extent.
_MIN_EXTENT_HALF_WIDTH = 1e-5


def _check_coordinate_range(placements, column, minimum, maximum):
    """Check that every coordinate of a placements column is finite and in range.

    Parameters
    ----------
    placements : pandas.DataFrame
        The placements table to check.

    column : str
        The name of the coordinate column, i.e. 'lon' or 'lat'.

    minimum, maximum : float
        The inclusive limits of the valid range.

    Raises
    ------
    ValueError
        If one or more values are outside the range, or are NaN, or are infinite.
    """
    values = pd.to_numeric(placements[column], errors="coerce")
    invalid = ~values.between(minimum, maximum, inclusive="both")
    if invalid.any():
        offenders = ", ".join(f"{index}: {value}" for index, value in values[invalid].head(10).items())
        raise ValueError(
            f"All '{column}' values must be finite and between {minimum} and {maximum}. "
            f"{int(invalid.sum())} of {len(values)} placements are invalid "
            f"(index: value): {offenders}"
        )


def _zarr_regions(locs: gk.LocationSet, dataset: xarray.Dataset, index_pad: int) -> list[np.ndarray]:
    """Group the locations by the regions of the spatial chunk grid of a Zarr store.

    A Zarr source reads the rectangle around the locations it is given, so locations far apart
    read everything between them -- most of the globe for one location per continent. Grouping
    them by chunk-sized regions (Earth Data Hub: 60 cells = 15 degrees) keeps each read to the
    chunks around its own locations, while locations in one region still share their reads.
    The regions are counted from the first grid point of the store, like its chunks, so that
    each region covers whole chunks, see _zarr_region_index().

    Parameters
    ----------
    locs : geokit.LocationSet
        The locations to group.

    dataset : xarray.Dataset
        The opened store, which determines the grid and the size of a region, see
        Era5ZarrSource.spatial_chunk_cells().

    index_pad : int
        The number of cells each region reads beyond its locations on every side, i.e. the
        index_pad of the Era5ZarrSource reading it (by default
        Era5ZarrSource.DEFAULT_INDEX_PAD). A region spans at least twice as many cells, see
        _zarr_region_index().

    Returns
    -------
    list of numpy.ndarray
        One array per region, holding the positions of its locations in `locs`.
    """
    latitude_chunk_cells, longitude_chunk_cells = rk_weather.Era5ZarrSource.spatial_chunk_cells(dataset)

    longitudes = np.asarray(locs.lons)
    latitudes = np.asarray(locs.lats)

    min_region_cells = 2 * index_pad
    longitude_region_index = _zarr_region_index(
        longitudes, dataset, "longitude", longitude_chunk_cells, min_region_cells, wrap_around_globe=True
    )
    latitude_region_index = _zarr_region_index(
        latitudes, dataset, "latitude", latitude_chunk_cells, min_region_cells, wrap_around_globe=False
    )
    region_index_per_axis = np.stack([longitude_region_index, latitude_region_index])

    _, region_of_each_location = np.unique(region_index_per_axis, axis=1, return_inverse=True)
    region_of_each_location = np.asarray(region_of_each_location).ravel()

    region_count = region_of_each_location.max() + 1
    positions_per_region = []
    for region in range(region_count):
        positions_in_region = np.flatnonzero(region_of_each_location == region)
        positions_per_region.append(positions_in_region)

    return positions_per_region


def _zarr_region_index(
    location_coordinates: np.ndarray,
    dataset: xarray.Dataset,
    dimension: str,
    chunk_cells: int | None,
    min_region_cells: int,
    wrap_around_globe: bool,
) -> np.ndarray:
    """Determine the region of each location along one spatial dimension of a Zarr store.

    Each location is assigned the grid cell nearest to it, counted from the first grid point of
    the store in the direction of its coordinates (e.g. north to south for ERA5 latitudes), as
    the chunks of the store are. The cells are then grouped into regions of whole chunks: one
    chunk, or as many chunks as needed for a region of at least min_region_cells.

    Parameters
    ----------
    location_coordinates : numpy.ndarray
        The coordinates of the locations along the dimension, in degrees.

    dataset : xarray.Dataset
        The opened store, see _zarr_regions().

    dimension : str
        The spatial dimension, i.e. 'latitude' or 'longitude'.

    chunk_cells : int or None
        The number of cells in one chunk along the dimension, see
        Era5ZarrSource.spatial_chunk_cells(). None if it is unknown, i.e. for data in memory,
        which is then not split along the dimension.

    min_region_cells : int
        The smallest number of cells in a region. Each region reads some cells beyond its
        locations, see _zarr_regions(): regions smaller than twice that padding would read
        mostly the same chunks as their neighbours.

    wrap_around_globe : bool
        Whether the dimension wraps around the globe, i.e. for longitude. The cells are then
        counted modulo one full circle, so that e.g. a location at -10 degrees falls into the
        last cells of a store running from 0 to 360 degrees.

    Returns
    -------
    numpy.ndarray
        The index of the region of each location along the dimension. All locations are in
        region 0 if the chunk size is unknown or the store has fewer than two coordinates along
        the dimension.
    """
    if chunk_cells is None or dimension not in dataset.coords or dataset[dimension].size < 2:
        return np.zeros(location_coordinates.shape, dtype=int)

    grid_coordinates = dataset[dimension]
    first_coordinate = float(grid_coordinates[0])
    coordinate_step = float(grid_coordinates[1] - grid_coordinates[0])
    resolution = abs(coordinate_step)

    chunks_per_region = max(1, -(-min_region_cells // chunk_cells))  # the ceiling of the division
    region_cells = chunks_per_region * chunk_cells

    # dividing by the signed step counts the cells in the direction of the store's coordinates
    offset_cells = (location_coordinates - first_coordinate) / coordinate_step
    nearest_cell = np.round(offset_cells).astype(int)
    if wrap_around_globe:
        cells_around_globe = round(360 / resolution)
        nearest_cell = nearest_cell % cells_around_globe

    return nearest_cell // region_cells


def _expand_degenerate_bound(value):
    """Expand a zero-width extent bound around one coordinate value.

    The expansion is additive. A multiplicative expansion keeps a zero coordinate at zero.
    This gives a degenerate extent which GeoKit rejects, e.g. for a single placement at
    (0, 0). A multiplicative expansion also inverts the bounds of a negative coordinate.

    Parameters
    ----------
    value : float
        The coordinate value which is both the lower and the upper bound.

    Returns
    -------
    tuple of float
        The new lower bound and the new upper bound.
    """
    half_width = max(abs(value) * 1e-5, _MIN_EXTENT_HALF_WIDTH)
    return value - half_width, value + half_width


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

        # Check if input file contains a geometry column
        ispoint = False
        if "geom" in placements.columns:
            if self.placements["geom"].iloc[0].GetGeometryName() == "POINT":
                ispoint = True
            _srs = placements.geom.iloc[0].GetSpatialReference()
        else:
            # assume lat/lon values in EPSG:4326
            _srs = gk.srs.loadSRS(4326)

        if ispoint:
            self.locs = gk.LocationSet(placements.geom)
            self.placements["lon"] = self.locs.lons
            self.placements["lat"] = self.locs.lats
            del self.placements["geom"]
        else:
            assert "lon" in self.placements.columns, (
                "if geom are not point geometries, dataframe must contain lon columns"
            )
            assert "lat" in self.placements.columns, (
                "if geom are not point geometries, dataframe must contain lat columns"
            )

        if self.locs is None:
            self.locs = gk.LocationSet(self.placements[["lon", "lat"]].values)

        # limit the input placements longitude to range of -180...180
        _check_coordinate_range(self.placements, "lon", -180, 180)
        # limit the input placements latitude to range of -90...90
        _check_coordinate_range(self.placements, "lat", -90, 90)

        # get bounds of the extent
        _bounds = list(self.locs.getBounds())
        # if no extension in lon and/or lat direction, create incremental artificial width
        if _bounds[0] == _bounds[2]:
            _bounds[0], _bounds[2] = _expand_degenerate_bound(_bounds[0])
        if _bounds[1] == _bounds[3]:
            _bounds[1], _bounds[3] = _expand_degenerate_bound(_bounds[1])
        # create extent attribute
        self.ext = gk.Extent(_bounds, srs=_srs)

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
        time_slice: slice | None = None,
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
            source to be specified in the `variables` list. But the user needs to take care
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

        time_slice : slice, optional
            Restricts the simulation to the time steps between `time_slice.start` and
            `time_slice.stop`, both inclusive, e.g. slice("2015-03-01", "2015-03-31 23:30")
            - Only these time steps are read from the weather source
            - Not available for an already initialized source; pass it to its constructor
            - By default None, i.e. all time steps of the source

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

        if not isinstance(variables, list):
            variables = [
                variables,
            ]

        if is_path_like(source) and source_type != "user":
            source = as_path_string(source)
            storage_format = kwargs.pop("storage_format", None)
            is_zarr = storage_format == "zarr" or source.endswith(".zarr") or source.startswith("gs://")
            if source_type == "ERA5":
                source_constructor = rk_weather.Era5ZarrSource if is_zarr else rk_weather.Era5Source
            elif source_type == "SARAH":
                source_constructor = rk_weather.SarahSource
            elif source_type == "MERRA":
                source_constructor = rk_weather.MerraSource
            elif source_type == "ICON-LAM":
                source_constructor = rk_weather.IconlamSource
            else:
                raise RuntimeError("Unknown source_type")

            if source_type == "ERA5":
                if is_zarr:
                    # A Zarr source reads the rectangle around all placements: read placements far
                    # apart (e.g. on other continents) region by region instead, see _read_by_region()
                    dataset = rk_weather.Era5ZarrSource._open_dataset(
                        source,
                        kwargs.get("chunks"),
                        kwargs.get("consolidated", True),
                        kwargs.get("storage_options"),
                    )
                    # the padding the sources of the regions read with: as passed, or their default
                    index_pad = kwargs.get("index_pad", rk_weather.Era5ZarrSource.DEFAULT_INDEX_PAD)
                    regions = _zarr_regions(self.locs, dataset, index_pad)
                    if len(regions) > 1:
                        first_cutout_source, frames = self._read_by_region(
                            source_constructor,
                            dataset,
                            regions,
                            variables,
                            spatial_interpolation_mode,
                            time_index_from=time_index_from,
                            time_slice=time_slice,
                            **kwargs,
                        )
                        return self._store_read_variables(
                            first_cutout_source, frames, set_time_index, temporal_reindex_method
                        )
                    source = dataset  # opened once
                source = source_constructor(
                    source, bounds=self.ext, time_index_from=time_index_from, time_slice=time_slice, **kwargs
                )
            else:
                source = source_constructor(source, bounds=self.ext, time_slice=time_slice, **kwargs)

            # Load the requested variables
            source.sload(*variables)

        else:  # Assume source is already an initialized NCSource-like object
            if time_slice is not None:
                raise ValueError(
                    "'time_slice' cannot be applied to an already initialized source. "
                    "Pass it to the constructor of the source instead."
                )
            missing_variables = [var for var in variables if var not in source.data]
            if missing_variables:
                if hasattr(source, "sload"):
                    source.sload(*missing_variables)
                else:
                    raise AssertionError(
                        "The given source has no '.sload()' method and is missing the variable(s): "
                        + ", ".join(missing_variables)
                    )

        frames = {
            var: source.get(var, self.locs, interpolation=spatial_interpolation_mode, force_as_data_frame=True)
            for var in variables
        }
        return self._store_read_variables(source, frames, set_time_index, temporal_reindex_method)

    def _read_by_region(
        self,
        source_constructor: type[rk_weather.NCSource],
        dataset: xarray.Dataset,
        regions: list[np.ndarray],
        variables: list[str],
        spatial_interpolation_mode: str,
        **source_kwargs: Any,
    ) -> tuple[rk_weather.NCSource, dict[str, pd.DataFrame]]:
        """Read the variables region by region.

        For each region one source is created, a cutout of the store, which reads only the
        rectangle around the placements in the region.

        Parameters
        ----------
        source_constructor : type
            The class of the source to create as the cutout of each region. It has to accept an
            opened dataset as its source, i.e. Era5ZarrSource.

        dataset : xarray.Dataset
            The opened store, which is shared by all regions.

        regions : list of numpy.ndarray
            The positions of the placements in `.locs` per region, see _zarr_regions().

        variables : list of str
            The variables to read.

        spatial_interpolation_mode : str
            The spatial interpolation mode to use at each of the placement coordinates.

        **source_kwargs
            Passed on to `source_constructor`.

        Returns
        -------
        tuple
            The source of the first cutout, which provides the time index and the wind speed
            heights that all cutouts share, and per variable a DataFrame with a column per
            placement, in the order of `.locs`.
        """
        first_cutout_source: rk_weather.NCSource | None = None
        placement_count = self.locs.count

        # one entry per placement, which is filled in as soon as the region of the placement is read
        columns_per_variable: dict[str, list[Any]] = {variable: [None] * placement_count for variable in variables}

        for placement_positions in regions:
            region_placements = [self.locs[position] for position in placement_positions]
            region_locs = gk.LocationSet(region_placements)

            lon_min, lat_min, lon_max, lat_max = region_locs.getBounds()
            if lon_min == lon_max:
                lon_min, lon_max = _expand_degenerate_bound(lon_min)
            if lat_min == lat_max:
                lat_min, lat_max = _expand_degenerate_bound(lat_min)
            cutout_bounds = [lon_min, lat_min, lon_max, lat_max]
            cutout_extent = gk.Extent(cutout_bounds, srs=4326)

            # a shallow copy: the source adds its derived variables to the dataset it is given
            dataset_copy = dataset.copy()
            cutout_source = source_constructor(dataset_copy, bounds=cutout_extent, **source_kwargs)
            cutout_source.sload(*variables)

            for variable in variables:
                region_frame = cutout_source.get(
                    variable,
                    region_locs,
                    interpolation=spatial_interpolation_mode,
                    force_as_data_frame=True,
                )
                for column_position, placement_position in enumerate(placement_positions):
                    placement_column = region_frame.iloc[:, column_position]
                    placement_values = placement_column.to_numpy()
                    columns_per_variable[variable][placement_position] = placement_values

            if first_cutout_source is None:
                first_cutout_source = cutout_source

        if first_cutout_source is None:
            raise ValueError("'regions' must contain at least one region.")

        time_index = first_cutout_source.time_index
        frames = {}
        for variable in variables:
            values_per_placement = np.column_stack(columns_per_variable[variable])
            frames[variable] = pd.DataFrame(values_per_placement, index=time_index)

        return first_cutout_source, frames

    def _store_read_variables(
        self,
        source: rk_weather.NCSource,
        frames: dict[str, pd.DataFrame],
        set_time_index: bool,
        temporal_reindex_method: str,
    ) -> "WorkflowManager":
        """Put the read variables into `.sim_data`.

        Parameters
        ----------
        source : rk_weather.NCSource
            The source which the variables were read from. It provides the time index and the
            wind speed heights.

        frames : dict of str to pandas.DataFrame
            The values per variable, with a column per placement.

        set_time_index : bool
            If True, the time index of the source is set as the time index of the workflow.
            Otherwise the values are reindexed to the existing time index.

        temporal_reindex_method : str
            The method used to reindex the values to the time index of the workflow, e.g.
            "nearest" or "ffill". Not used if `set_time_index` is True.

        Returns
        -------
        WorkflowManager
            Returns the invoking WorkflowManager (for chaining)
        """
        if set_time_index:
            self.set_time_index(source.time_index)

        for variable, frame in frames.items():
            if set_time_index:
                aligned_frame = frame
            else:
                aligned_frame = frame.reindex(self.time_index, method=temporal_reindex_method)

            self.sim_data[variable] = aligned_frame.values

            # Special check for wind speed height
            if variable == "elevated_wind_speed":
                self.elevated_wind_speed_height = source.ELEVATED_WIND_SPEED_HEIGHT

            if variable == "surface_wind_speed":
                self.surface_wind_speed_height = source.SURFACE_WIND_SPEED_HEIGHT

        return self

        # Stage 3: Weather data adjusting & other intermediate steps

    def get_scalar_values_from_raster(self, fp, spatial_interpolation, points=None):
        """
        Auxiliary function to extract raster values with NaN fallback options.
        """
        # geokit before 1.7 opens only string paths; a pathlib.Path (what reskit.data
        # returns) would be handed back unopened and fail inside rasterInfo().
        fp = as_path_string(fp)
        assert isfile(fp), f"File '{fp}' in adjust_variable_to_long_run_average() does not exist."
        # execute with warnings filter since values outside of source data would trigger geokit UserWarning every time
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if points is None:
                points = [(loc.lon, loc.lat) for loc in self.locs._locations]
            else:
                assert isinstance(points, list) and all([isinstance(x, tuple) and len(x) == 2 for x in points]), (
                    "points must be a list of (lon, lat) tuples."
                )

            # interpolateValues returns a scalar for a single location; ensure a 1-D array
            # so the nan handling below (and callers) work regardless of the number of points
            _lra = np.atleast_1d(gk.raster.interpolateValues(fp, points, mode=spatial_interpolation))
            # if getting values fails, it could be because of interpolation method.
            # these values will be replaced with the nearest interpolation method
            if np.isnan(_lra).any():
                _lra_near = np.atleast_1d(gk.raster.interpolateValues(fp, self.locs, mode="near"))
                _lra[np.isnan(_lra)] = _lra_near[np.isnan(_lra)]
            # still nans, i.e. the cell itself is nan, but maybe its neighbors are not
            # try the (nan)median of the surrounding cells
            if np.isnan(_lra).any():

                def _nanmedian(vals, xOff, yOff):
                    """Aux function to mimic the 3 expected inputs in interpolateValues()"""
                    return np.nanmedian(vals)

                points = [(loc.lon, loc.lat) for loc in self.locs._locations]
                _lra_near = np.atleast_1d(gk.raster.interpolateValues(fp, points, mode="func", func=_nanmedian))
                _lra[np.isnan(_lra)] = _lra_near[np.isnan(_lra)]
        return _lra

    def adjust_variable_to_long_run_average(
        self,
        variable: str,
        source_long_run_average: Union[str, float, np.ndarray],
        real_long_run_average: Union[str, float, np.ndarray],
        real_lra_scaling: float = 1,
        spatial_interpolation: str = "linear-spline",
        nodata_fallback: str = "nan",
        nodata_fallback_scaling: float = 1,
        allow_nans: bool = True,
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

        nodata_fallback: float, str, callable, optional
            When real_long_run_average has no data, one can decide between different fallback options, by default np.nan:
            - np.nan or None : return np.nan for missing values in real_long_run_average
            - float : Apply this float value as a scaling factor for all no-data locations only: source_long_run_average * nodata_fallback.
            NOTE: A value of 1.0 will return the source lra value in case of missing real lra values (no additional nodata_fallback_scaling applied).
            - str : Will be interpreted as a filepath to a raster with alternative real_long_run_average values, scaled by nodata_fallback_scaling.
            - callable : any callable method taking the arguments (all iterables): 'locs' and 'source_long_run_average_value'
            (the locations as gk.geom.point objects and original value from source data). The output values will be considered as
            the new real_long_run_average for missing locations only (absolute data, no additional nodata_fallback_scaling applied).

            NOTE: np.nan will also be returned in case that the nodata fallback does not yield values either.

        nodata_fallback_scaling: float
            An optional scaling factor to apply to the values derived from `nodata_fallback`.
            - This is primarily useful when `nodata_fallback` is a path to a raster file
            - By default 1

        allow_nans : boolean, optional
            If True, NaN values may remain after scaling, else an error will raised. By default True.

        Returns
        -------
        WorkflowManager
            Returns the invoking WorkflowManager (for chaining)
        """
        if not (
            nodata_fallback is None
            or callable(nodata_fallback)
            or isinstance(nodata_fallback, (float, int))
            or is_path_like(nodata_fallback)
        ):
            raise TypeError(f"'nodata_fallback' must be a float or a Callable.")

        # first get source values
        if is_path_like(source_long_run_average):
            # assume raster fp
            source_lra = self.get_scalar_values_from_raster(
                fp=source_long_run_average, spatial_interpolation="linear-spline"
            )
        else:
            source_lra = source_long_run_average

        # then get lng-run average values for scaling
        if is_path_like(real_long_run_average):
            # assume a raster path
            real_lra = self.get_scalar_values_from_raster(
                fp=real_long_run_average, spatial_interpolation=spatial_interpolation
            )
        else:
            real_lra = real_long_run_average

        # replace missing values with no-data fallback if needed
        if isinstance(nodata_fallback, str) and nodata_fallback.lower() == "source":
            warnings.warn(
                "'source' value for 'nodata_fallback' is deprecated and will be removed soon. Use 1.0 instead.",
                DeprecationWarning,
            )
            nodata_fallback = 1.0
        if isinstance(nodata_fallback, str) and nodata_fallback.lower() == "nan":
            warnings.warn(
                "'nan' value for 'nodata_fallback' is deprecated and will be removed soon. Use np.nan instead.",
                DeprecationWarning,
            )
            nodata_fallback = np.nan
        if any(np.isnan(real_lra)):  # TODO currently all real_lra are replaced by fallback, is this intentional?
            # we are lacking long-run average values
            if nodata_fallback is None or (isinstance(nodata_fallback, float) and np.isnan(nodata_fallback)):
                # nans will be returned for missing lra values
                fallback_lra = np.array([np.nan] * len(real_lra))
            elif isinstance(nodata_fallback, (int, float)):
                # apply factor to source_lra to scale missing values
                fallback_lra = nodata_fallback * source_lra  # no additional scaling
            elif callable(nodata_fallback):
                # apply function to calculate missing values
                fallback_lra = nodata_fallback(
                    locs=self.locs, source_long_run_average_value=source_lra
                )  # no additional scaling
            elif is_path_like(nodata_fallback):
                # assume this is yet another raster path as fallback and extract missing values
                fallback_lra = (
                    self.get_scalar_values_from_raster(fp=nodata_fallback, spatial_interpolation=spatial_interpolation)
                    * nodata_fallback_scaling
                )

            # divide by real_lra_scaling once to compensate later multiplication below for
            # scaling factor, nodata_fallback should not be multiplied by real_lra_scaling
            fallback_lra = fallback_lra / real_lra_scaling
            # set fallback values where real_lra is nan
            real_lra[np.isnan(real_lra)] = fallback_lra[np.isnan(real_lra)]

        # save LRA as attribute
        self.real_lra = real_lra

        # calculate scaling factor:
        # nan result will stay nan results, as these placements cannot be calculated any more
        factors = real_lra * real_lra_scaling / source_lra
        if any(np.isnan(real_lra)):
            if allow_nans:
                warnings.warn(f"NaN values remaining in real lra after application of nodata_fallback.")
            else:
                raise ValueError(f"Missing values for variable '{variable}' and NaNs not allowed.")

        # write info with missing values to sim_data:
        self.placements[f"missing_values_{basename(real_long_run_average)}_nodata_fallback{nodata_fallback}"] = (
            np.isnan(factors)
        )

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
        if is_path_like(source_high_resolution):
            points = [(loc.lon, loc.lat) for loc in self.locs._locations]
            correction_values_high_res = gk.raster.interpolateValues(  # TODO change here
                source_high_resolution, points, mode=spatial_interpolation
            )
            # assert not np.isnan(correction_values_high_res).any() and (correction_values_high_res > 0).all()
        else:
            correction_values_high_res = source_high_resolution

        # Get values from low resolution tiff file (meant over eg. ERA5)
        if is_path_like(source_low_resolution):
            points = [(loc.lon, loc.lat) for loc in self.locs._locations]
            correction_values_low_res = gk.raster.interpolateValues(  # TODO change here
                source_low_resolution, points, mode=spatial_interpolation
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
            The variable or variables to apply the loss factor to
            - By default ["capacity_factor"]

        Returns
        -------
        WorkflowManager
            Returns the invoking WorkflowManager (for chaining)
        """
        # filter only existing variables
        _variables = [_var for _var in variables if _var in self.sim_data.keys()]
        if len(_variables) < len(variables):
            warnings.warn(
                f"Loss factor could not be applied to the following requested variables because variables are not in sim_data: {', '.join(sorted(set(variables) - set(_variables)))}"
            )

        for var in _variables:
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
        custom_attributes: dict = None,
        _intermediate_dict=False,
    ) -> xarray.Dataset:
        """Generates an XArray dataset from the data currently contained in the WorkflowManager

        Note:
        - The `.placements` data is automatically added to the XArray dataset along the 'locations' dimension
        - The `workflow_parameters` data is automatically added as dimensionless variables
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

        custom_attributes : dict, optional
            If given, adds the key-value pairs as attributes to the XArray dataset
            - These will be added in addition to the workflow_parameters
            - By default None

        Returns
        -------
        xarray.Dataset
            The resulting XArray dataset
        """
        if isinstance(output_variables, str):
            output_variables = [output_variables]
        elif isinstance(output_variables, list):
            # copy the list, the caller's list must not get "RESKit_sim_order" appended
            output_variables = list(output_variables)
        if isinstance(output_variables, list) and "RESKit_sim_order" not in output_variables:
            output_variables.append("RESKit_sim_order")

        times = self.time_index
        if times[0].tz is not None:
            times = [np.datetime64(dt.tz_convert("UTC").tz_convert(None)) for dt in times]
        times_days = np.unique(pd.DatetimeIndex(times).date).astype("datetime64")
        xds = OrderedDict()
        encoding = dict()

        # work on a copy, exporting must not change the state of the WorkflowManager
        placements = self.placements
        if "location_id" in placements.columns:
            location_coords = placements["location_id"].copy()
            placements = placements.drop(columns=["location_id"])
        else:
            location_coords = np.arange(placements.shape[0])

        # write placements
        for c in placements.columns:
            # check if c in requestet output_variables
            if output_variables is not None and c not in output_variables:
                continue

            column = placements[c]

            if not is_numeric_dtype(column):
                if not all(isinstance(x, (str, bytearray)) for x in column):
                    continue

            xds[c] = xarray.DataArray(
                column.to_numpy(),
                dims=["location"],
                coords=dict(location=location_coords),
            )

        # write sim_data
        for key in self.sim_data.keys():
            # check if key in requestet output_variables
            if output_variables is not None:
                if key not in output_variables:
                    continue

            tmp = np.full((len(self.time_index), self.locs.count), 0.0, dtype=float)
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

        # Add custom attributes if provided
        if custom_attributes is not None:
            for k, v in custom_attributes.items():
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
        custom_attributes: dict = None,
        _intermediate_dict=False,
    ) -> str:
        """Saves an XArray dataset to netCDF4 format

        Note:
        - The `.placements` data is automatically added to the XArray dataset along the 'locations' dimension
        - The `workflow_parameters` data is automatically added as dimensionless variables
        - The `.sim_data` is automatically added along the dimensions (time, locations)
        - The `.time_index` is automatically added along the dimension 'time'

        Parameters
        ----------
        xds : xarray.Dataset
            The XArray dataset to save

        output_netcdf_path : str
            If given, the XArray dataset will be written to disc at the specified path
            - By default None

        output_variables : List[str], optional
            If given, specifies the variables which should be included in the resulting
            dataset. Otherwise all suitable variables found in `.placements`, `.workflow_parameters`,
            `.sim_data`, and `.time_index` will be included
            - Only variables of numeric or string type are suitable due to NetCDF4 limitations
            - By default None

        custom_attributes : dict, optional
            If given, adds the key-value pairs as attributes to the XArray dataset before saving
            - These will be added in addition to existing attributes
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

        # Add custom attributes if provided
        if custom_attributes is not None:
            for k, v in custom_attributes.items():
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
            # splitKMeans() returns a LocationSet with coordinates very close to placements.index,
            # but not guaranteed to be exact due to floating-point precision.
            # Therefore, its rounded to 15 decimal to ensure an exact match for use in .loc[].
            rounded_keys = [(round(loc.lon, 15), round(loc.lat, 15)) for loc in loc_group[:]]
            yield placements.loc[rounded_keys]


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
        - Use these to set the required arguments for the given ``workflow_function``

    Returns
    -------
    xarray.Dataset
        An XArray Dataset which contains the combined results of the distributed simulations

    """
    from multiprocessing import Pool

    import xarray

    assert isinstance(placements, pd.DataFrame)
    assert ("lon" in placements.columns and "lat" in placements.columns) or ("geom" in placements.columns)

    # work on a copy, the caller's placements table must not be changed by this function
    placements = placements.copy()

    # Split placements into groups
    if "geom" in placements.columns:
        locs = gk.LocationSet(placements)
        placements["lat"] = locs.lats
        placements["lon"] = locs.lons
        del placements["geom"]
    else:
        locs = gk.LocationSet(np.column_stack([placements.lon.values, placements.lat.values]))
    # placements.index is used in the _split_locs function, where exact key matching is required.
    # Therefore, the coordinates are rounded to 15 decimal places to ensure consistency with the
    # LocationSet keys returned by splitKMeans().
    placements.index = [(round(loc.lon, 15), round(loc.lat, 15)) for loc in locs._locations]
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
            kwargs_["output_netcdf_path"] = join(intermediate_output_dir, "simulation_group_{:05d}.nc".format(gid))

        results.append(pool.apply_async(func=workflow_function, args=(placement_group,), kwds=kwargs_))
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

    if len(datasets) == 0:
        raise ValueError("No workflow result files were found to load.")
    elif len(datasets) == 1:
        ds = loader(datasets[0])
    else:
        ds = xarray.concat(map(loader, datasets), dim="location")

    if sortby is not None:
        ds = ds.sortby(sortby)

    return ds


def execute_workflow_iteratively(
    workflow,
    weather_path_varname,
    zoom=None,
    location_specific_workflow_args={},
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

    location_workflow_args : dict, optional
        Dict with location-specific arguments of the "workflow" as keys, and the respective arg
        value as values. The values are expected to be at least 1d iterables, with the first
        dimension matching the number of locations in length. The values of this iterable will
        then be applied per location along this axis.
        # NOTE: This does not apply to a "placements" dataframe; pass as "workflow_arg" below if required

    **workflow_args
        Passed on to the workflow specified above. Must contain ''placements'' and the above
        weather_path_varname as keys.
    """
    # CHECK INPUTS

    assert callable(workflow), "workflow must be a callable RESkit workflow function."
    assert isinstance(location_specific_workflow_args, dict), "location_specific_workflow_args must be a dict."
    assert isinstance(weather_path_varname, str), f"weather_path_varname ({weather_path_varname}) must be str."

    assert "placements" in workflow_args, "'placements' is a mandatory argument/key in workflow_args."
    placements = workflow_args["placements"]
    assert isinstance(placements, pd.DataFrame), f"placements must be a pd.DataFrame, here: {type(placements)}"
    # generate and write copy back into argy to not manipulate the original
    placements = placements.copy()
    workflow_args["placements"] = placements

    # make sure that the location specific args are ordered iterables and match the locations in shape
    location_specific_workflow_args = location_specific_workflow_args.copy()  # may be manipulated later
    for _arg, _val in location_specific_workflow_args.items():
        if isinstance(_val, set):
            raise TypeError("A set cannot be a positional argument because it is unordered.")
        if isinstance(_val, str) or not isinstance(_val, Iterable):
            raise TypeError(
                f"Location-specific workflow arg '{_arg}' must be an iterable with length equal to placements, pass scalar values as workflow arg."
            )
        try:
            n = len(_val)
        except TypeError:
            raise TypeError(
                f"Location-specific workflow arg '{_arg}' must be an iterable with a defined first dimension."
            )
        if n != len(placements):
            raise ValueError(
                f"Location-specific workflow arg '{_arg}' has first-dimension length {n}, expected {len(placements)}."
            )

    # avoid duplicate argument values
    workflow_keys = set(workflow_args)
    location_specific_keys = set(location_specific_workflow_args)
    placement_keys = set(placements.columns)
    dups = (
        (workflow_keys & location_specific_keys)
        | (workflow_keys & placement_keys)
        | (location_specific_keys & placement_keys)
    )
    if dups:
        raise KeyError(
            "Workflow arguments must be defined only once across workflow_args, "
            "location_specific_workflow_args, and placements.columns. "
            f"Duplicates: {', '.join(sorted(dups))}"
        )

    # ADD INDEX TO PRESERVE/RESTORE ORDER AFTER BATCHWISE ITERATION

    # add an iterable with the current order so it can be restored afterwards
    if "RESKit_sim_order" in placements.columns:
        # make sure it is a consecutive integer sequence
        if not np.array_equal(placements["RESKit_sim_order"], np.arange(len(placements))):
            raise ValueError(
                "If placements dataframe has a 'RESKit_sim_order' column, it must contain a consecutive integer sequence."
            )
    else:
        # add, is mandatory for later recombination of placements and results
        with pd.option_context("mode.chained_assignment", None):
            placements.loc[placements.index, "RESKit_sim_order"] = range(len(placements))

    # REMOVE ARGS WHICH ARE NOT MEANT FOR THE ITERATIVE WORKFLOW EXECUTION

    # extract the overall save_args of to_netcdf() before iteration over tiles
    save_args = {}
    for k in ["output_netcdf_path", "output_variables", "custom_attributes"]:
        assert k not in location_specific_keys and k not in placement_keys, (
            f"'{k}' must be a workflow arg if defined, cannot be a location-specific arg or a placements column name."
        )
        # remove the saving-related args (which should not be passed to individual iterations over tiles) and store them in save args instead
        save_args[k] = workflow_args.pop(k, None)

    # PREPROCESS THE WEATHER TILE PATHS

    assert weather_path_varname in workflow_keys | location_specific_keys | placement_keys, (
        f"weather_path_varname '{weather_path_varname}' must be either a key in workflow_args or location_specific_workflow_args, or a placements df column."
    )
    # get the weather path data
    containers = {
        "location_specific_workflow_args": location_specific_workflow_args,
        "workflow_args": workflow_args,
        "placements": placements,
    }
    for weather_path_source, cont in containers.items():
        if weather_path_varname in cont:
            weather_path = cont.pop(weather_path_varname)
            break
    assert is_path_like(weather_path) or (
        isinstance(weather_path, (list, tuple, np.ndarray, pd.Series)) and all(is_path_like(x) for x in weather_path)
    ), "weather_path must be a str or an ordered iterable of str."
    # Strings from here on: the tile spacers in them are completed by string replacement.
    if is_path_like(weather_path):
        weather_path = as_path_string(weather_path)
    else:
        weather_path = [as_path_string(x) for x in weather_path]
    # broadcast it to one value per location if not provided as such
    try:
        weather_paths = np.broadcast_to(
            weather_path,
            (len(placements),),
        ).tolist()
    except ValueError:
        raise ValueError(f"'{weather_path_varname}' must be scalar or have length {len(placements)}.")
    # get a locations iterable
    if "geom" in placements:
        locs = placements["geom"].to_list()
    elif "lon" in placements and "lat" in placements:
        locs = list(zip(placements.lon, placements.lat))
    else:
        raise AttributeError(f"placements is expected to have a 'geom' column or both 'lat' and 'lon'columns.")
    # now complete the paths by replacing potential spacers based on the respective locations and zoom value
    tilepaths = np.asarray(get_location_specific_weather_paths(weather_paths=weather_paths, locs=locs, zoom=zoom))

    # ITERATIVELY SIMULATE FOR EVERY WEATHER TILEPATH

    # extract unique weather tiles and iterate over them
    unique_tilepaths = sorted(np.unique(tilepaths))
    for i, tilepath in enumerate(unique_tilepaths):
        # generate a mask for the placements covered by this tile
        tilemask = tilepaths == tilepath
        # mask the placements dataframe to filter affected rows
        _placements = placements.loc[tilemask].copy()
        # create a copy of the workflow args for this tilepath only
        _workflow_args = workflow_args.copy()
        # add the tilepath for the current iteration
        if weather_path_source == "placements":
            # keep weather data in placements if it was extracted from there, workflow may expect that
            _placements[weather_path_varname] = tilepath
        else:
            # else write into workflow args
            _workflow_args[weather_path_varname] = tilepath
        # iterate over the global workflow args and change only the "special cases"
        for _arg, _val in workflow_args.items():
            if _arg == "placements":
                # reduce placements to subset within the current tile
                _workflow_args[_arg] = _placements
            else:
                # we can use it as it is, it is a standard "global" arg across all locs
                pass
        # now iterate over the location-specific args and select only those values that apply to the tile subset of the placements df
        for _arg, _val in location_specific_workflow_args.items():
            # mask the iterable just like the placements
            if isinstance(_val, np.ndarray):
                _val = _val[tilemask]  # apply mask on first dimension
            elif isinstance(_val, list):
                _val = list(compress(_val, tilemask))
            elif isinstance(_val, tuple):
                _val = tuple(compress(_val, tilemask))
            elif isinstance(_val, (pd.Series, pd.DataFrame)):
                _val = _val.iloc[tilemask]
            else:
                raise TypeError(f"Unknown iterable type for location-specific workflow arg '{_arg}': {type(_val)}")
            # add the reduced iterable to the final workflow args
            _workflow_args[_arg] = _val

        # execute workflow with subset and add to list of results
        print(
            datetime.datetime.now(),
            f"Now processing tile {i + 1}/{len(unique_tilepaths)} with {len(_placements)} locations: {tilepath}",
        )
        xrds = workflow(**_workflow_args)
        xrds = xrds.set_index(location="RESKit_sim_order")
        if i == 0:
            reskit_xr = xrds
        else:
            reskit_xr = xarray.concat([reskit_xr, xrds], dim="location")

    # SAVE OR COMPLETE XARRAY

    # create a dummy wfm instance for saving
    reskit_xr = reskit_xr.sortby("location")
    wfm = WorkflowManager(placements=placements.drop(columns="RESKit_sim_order"))
    reskit_xr = wfm.to_netcdf(
        xds=reskit_xr,
        **save_args,  # pass output path and variables if given
    )

    return reskit_xr


class WorkflowQueue:
    """The WorkflowQueue object allows for the queueing of multiple RESKit workflow simulations
    which are then executed in parallel

    Initialize:
    -----------
    WorkflowFunction( workflow:FunctionType, **kwargs )

    Parameters
    ----------
    workflow : FunctionType
    The workflow function to be parallelized
    - All RESKit workflow functions should be suitable here
    - Don't forget that that all inputs required for the workflow function are still required,
    and are passed on either as constants through ``kwargs`` specified in the initializer, or
    else in the subsequent ''.append(...)'' calls

    **kwargs:
        All key word arguments are passed on as constants to each simulation
        Use these to set the required arguments for the given ``workflow``

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
            The results of each simulation set, accessible via their access keys
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
