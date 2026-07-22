from collections import OrderedDict
from datetime import timedelta
from typing import Any
import warnings

import geokit as gk
import numpy as np
import pandas as pd
import xarray as xr

from ...util import ResError
from ..NCSource import NCSource
from .Era5Source import Era5Source


class Era5ZarrSource(Era5Source):
    """ERA5 source backed by a regular lat/lon Zarr dataset."""

    # The ERA5 time convention of RESKit (see Era5Source): the timestamps of the store
    # are shifted by this amount to obtain the time index.
    TIME_OFFSET = timedelta(minutes=-30)

    def __init__(
        self,
        source,
        bounds=None,
        index_pad=5,
        time_index_from=None,
        time_slice=None,
        chunks=None,
        consolidated=True,
        storage_options=None,
        verbose=True,
        forward_fill=True,
        **kwargs,
    ):
        """Initialize an ERA5 source from a regular latitude/longitude Zarr store.

        Compared to Era5Source, the data is not read from netCDF4 files but from a local or
        cloud hosted Zarr store, e.g. the Earth Data Hub ERA5 single-level dataset. Flattened
        'values' grids as used by some stores are not supported.

        Stores which only provide the raw accumulated solar radiation ('ssrd', 'fdir') are
        supplemented with the processed variants ('ssrd_t_adj', 'fdir_t_adj') on the fly, see
        _derive_solar_variables. The RESKit ERA5 time convention (TIME_OFFSET) is applied to
        the timestamps of the store, so that 'time_slice' and 'time_index' share one convention.

        Parameters
        ----------
        source : str or xarray.Dataset
            The Zarr store to read, either as an already opened dataset or as a path/URL.
            * Cloud stores are recognized by their protocol, i.e. 'https://' or 'gs://'
            * 'https://' stores are read with the credentials of '~/.netrc', which is where
              the credentials of the Earth Data Hub have to be saved

        bounds : Anything acceptable to geokit.Extent.load(), optional
            The boundaries of the data which is needed
              * Usage of this will help with memory management
              * If None, the full spatial extent of the store is used

        index_pad : int, optional
            The padding to apply to the boundaries
              * Useful in case of interpolation

        time_index_from : str, optional
            The variable which has to exist in the store, given either as an ERA5 name or as
            one of the clear names of Era5Source, e.g. 'elevated_wind_speed'.
            * Only validates the store, the time convention is independent of this variable

        time_slice : slice, optional
            Limit the time span which is loaded from the store, given in the time convention of
            RESKit, i.e. at half hours. Strongly recommended for multi-year cloud stores.
            * The first requested timestep of derived solar variables still uses the
              accumulation preceding it in the store

        chunks : dict, optional
            The chunk sizes to load the store with, e.g. {"valid_time": 48}. Passed on to
            xarray.open_dataset(), by default the chunking of the store is used.

        consolidated : bool, optional
            If True, the consolidated metadata of the store is used, by default True

        storage_options : dict, optional
            Additional options for the storage backend, passed on to xarray.open_dataset().
            Only needed to overwrite the defaults for the recognized cloud protocols.

        verbose : bool, optional
            If True, then status outputs are printed when reading weather data

        forward_fill : bool, optional
            If True, then a single missing time step at the end of the data is forward-filled

        See Also
        --------
        Era5Source
        """
        if kwargs:
            unexpected = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments for Era5ZarrSource: {unexpected}")

        self.fill = forward_fill
        self._flip_lat = True
        self._flip_lon = False
        self._maximal_lon_difference = self.MAX_LON_DIFFERENCE
        self._maximal_lat_difference = self.MAX_LAT_DIFFERENCE
        self.dependent_coordinates = False
        self._verbose = verbose

        ds = self._open_dataset(
            source=source,
            chunks=chunks,
            consolidated=consolidated,
            storage_options=storage_options,
        )

        self.time_name, ds = self._normalise_time_axis(ds)
        ds, self._derived_variables = self._derive_solar_variables(ds, self.time_name)

        # Clear names may map onto several store conventions; the first entry is the
        # canonical name and is used when none of the candidates exist.
        era5_names = {
            "global_horizontal_irradiance": ("ssrd_t_adj", "ssrd"),
            "direct_horizontal_irradiance": ("fdir_t_adj", "fdir"),
            "surface_wind_speed": ("w10", "ws10", "u10"),
            "elevated_wind_speed": ("w100", "ws100", "u100"),
        }
        if time_index_from in era5_names:
            candidates = era5_names[time_index_from]
            time_index_from = next((candidate for candidate in candidates if candidate in ds.data_vars), candidates[0])

        if time_index_from is not None and time_index_from not in ds.data_vars:
            raise ResError(
                f"ERA5 key '{time_index_from}' not known. Check variable 'time_index_from' and store {source}"
            )

        if time_slice is not None:
            if isinstance(time_slice, slice):
                raw_start = pd.Timestamp(time_slice.start) - self.TIME_OFFSET if time_slice.start is not None else None
                raw_stop = pd.Timestamp(time_slice.stop) - self.TIME_OFFSET if time_slice.stop is not None else None
                time_slice = slice(raw_start, raw_stop)
            ds = ds.sel({self.time_name: time_slice})

        if "values" in ds.dims:
            raise ResError(
                "This Era5ZarrSource implementation only supports regular latitude/longitude Zarr stores, not flattened 'values' grids."
            )

        self._dataset = ds
        self.variables = self._build_variable_table(ds, source, self._derived_variables)

        self._allLats = np.asarray(ds["latitude"].values)
        self._allLons = np.asarray(ds["longitude"].values)
        self._lonN = self._allLons.size
        self._latN = self._allLats.size
        self._longitude_360 = self._allLons.min() >= 0 and self._allLons.max() > 180

        self._configure_spatial_selection(bounds=bounds, index_pad=index_pad)
        self._apply_dataset_spatial_subset()

        self.extent = gk.Extent(
            self.lons.min(),
            self.lats.min(),
            self.lons.max(),
            self.lats.max(),
            srs=gk.srs.EPSG4326,
        )

        timeindex = pd.DatetimeIndex(pd.to_datetime(self._dataset[self.time_name].values)) + self.TIME_OFFSET
        self._timeindex_raw = timeindex
        self.time_index = timeindex
        self.data = OrderedDict()

        if verbose:
            print(f"Opened ERA5 Zarr source: {source}")

    @staticmethod
    def _open_dataset(source, chunks, consolidated, storage_options):
        if isinstance(source, xr.Dataset):
            return source

        storage_options_ = dict(storage_options or {})
        if isinstance(source, str):
            if source.startswith("https://"):
                # picks up the credentials of ~/.netrc, e.g. for the Earth Data Hub
                storage_options_.setdefault("client_kwargs", {"trust_env": True})
            elif source.startswith("gs://"):
                storage_options_.setdefault("token", "anon")

        return xr.open_dataset(
            source,
            chunks=chunks,
            engine="zarr",
            consolidated=consolidated,
            storage_options=storage_options_ or None,
            decode_timedelta=False,
        )

    @staticmethod
    def _normalise_time_axis(ds: xr.Dataset) -> tuple[str, xr.Dataset]:
        """Return the temporal data dimension and attach its datetime coordinate."""
        time_names = ("valid_time", "time")

        time_dim = None
        datetime_coordinate = None
        for name in time_names:
            if time_dim is None and name in ds.dims and any(name in data.dims for data in ds.data_vars.values()):
                time_dim = name
            if datetime_coordinate is None and name in ds.coords and np.issubdtype(ds[name].dtype, np.datetime64):
                datetime_coordinate = name

        if time_dim is None:
            raise ResError("ERA5 Zarr store variables must use either a 'valid_time' or 'time' dimension")
        if datetime_coordinate is None or ds[datetime_coordinate].size != ds.sizes[time_dim]:
            raise ResError("ERA5 Zarr store must provide a datetime 'valid_time' or 'time' coordinate")

        # Stores exist where the time dimension carries no (or a non datetime) coordinate,
        # while a matching datetime coordinate is available under the other name.
        if datetime_coordinate == time_dim:
            return time_dim, ds
        return time_dim, ds.assign_coords({time_dim: np.asarray(ds[datetime_coordinate].values)})

    @classmethod
    def _derive_solar_variables(cls, ds: xr.Dataset, time_name: str) -> tuple[xr.Dataset, dict]:
        """Add the processed solar variables to stores that only provide the raw accumulations.

        Mirrors the CDO pipeline ``-divc,3600 -shifttime,+1hour`` which produces the
        '*_t_adj' variables: adj[i] = raw[i-1] / 3600 (J/m² per hour -> W/m²). This is done
        lazily on the full dataset before any time slice is applied, so that the first
        requested timestep can still use the accumulation preceding it in the store.

        Returns the dataset and a mapping of the derived variables onto their raw origin.
        """
        derived = {}
        for raw_name, adjusted_name in (("ssrd", "ssrd_t_adj"), ("fdir", "fdir_t_adj")):
            if raw_name in ds.data_vars and adjusted_name not in ds.data_vars:
                ds[adjusted_name] = ds[raw_name].shift({time_name: 1}) / 3600.0
                ds[adjusted_name].attrs.update(units="W m**-2", long_name=f"Derived on the fly from '{raw_name}'")
                derived[adjusted_name] = raw_name
        return ds, derived

    @staticmethod
    def _build_variable_table(ds: xr.Dataset, source: Any, derived_variables: dict) -> pd.DataFrame:
        index = list(ds.data_vars) + [name for name in ds.coords if name not in ds.data_vars]
        rows = [
            {
                "name": ds[var].attrs.get("standard_name", ds[var].attrs.get("long_name", "Unknown")),
                "units": ds[var].attrs.get("units", "Unknown"),
                "shape": tuple(ds[var].shape),
                "path": str(source),
                "derived_from": derived_variables.get(var),
            }
            for var in index
        ]
        return pd.DataFrame(rows, index=index, columns=["name", "units", "shape", "path", "derived_from"])

    def _wrap_longitudes(self, lons):
        if self._longitude_360:
            return np.mod(lons, 360.0)
        return lons

    def _configure_spatial_selection(self, bounds=None, index_pad=0):
        if bounds is not None:
            self.bounds = gk.Extent.load(bounds).castTo(4326)
            x_min = self._wrap_longitudes(np.array([self.bounds.xMin]))[0]
            x_max = self._wrap_longitudes(np.array([self.bounds.xMax]))[0]

            if abs(x_min - x_max) <= self.MAX_LON_DIFFERENCE:
                x_min -= self.MAX_LON_DIFFERENCE / 2
                x_max += self.MAX_LON_DIFFERENCE / 2

            y_min = self.bounds.yMin
            y_max = self.bounds.yMax
            if abs(y_min - y_max) <= self.MAX_LAT_DIFFERENCE:
                y_min -= self.MAX_LAT_DIFFERENCE / 2
                y_max += self.MAX_LAT_DIFFERENCE / 2

            lon_mask = np.logical_and(self._allLons >= x_min, self._allLons <= x_max)
            lat_mask = np.logical_and(self._allLats >= y_min, self._allLats <= y_max)

            self._lonStart = np.argmax(lon_mask) - 1
            self._lonStop = self._lonStart + 1 + np.argmin(lon_mask[self._lonStart + 1 :]) + 1

            self._latStart = np.argmax(lat_mask) - 1
            self._latStop = self._latStart + 1 + np.argmin(lat_mask[self._latStart + 1 :]) + 1

            self._lonStart = max(0, self._lonStart - index_pad)
            self._lonStop = min(self._allLons.size, self._lonStop + index_pad)
            self._latStart = max(0, self._latStart - index_pad)
            self._latStop = min(self._allLats.size, self._latStop + index_pad)
        else:
            self.bounds = None
            self._lonStart = 0
            self._lonStop = self._allLons.size
            self._latStart = 0
            self._latStop = self._allLats.size

        self.lats = self._allLats[self._latStart : self._latStop]
        self.lons = self._allLons[self._lonStart : self._lonStop]

        if self._flip_lat:
            self.lats = self.lats[::-1]
        if self._flip_lon:
            self.lons = self.lons[::-1]

    def _apply_dataset_spatial_subset(self):
        self._dataset = self._dataset.isel(
            latitude=slice(self._latStart, self._latStop),
            longitude=slice(self._lonStart, self._lonStop),
        )
        self._latStart = 0
        self._latStop = self._dataset.sizes["latitude"]
        self._lonStart = 0
        self._lonStop = self._dataset.sizes["longitude"]

    def var_info(self, var):
        assert var in self.variables.index
        print(self._dataset[var])

    def load(self, variable, name=None, height_idx=None, processor=None, overwrite=False):
        if name is None:
            name = variable

        if not overwrite and name in self.data:
            return

        if variable not in self.variables.index:
            raise ResError(f"Variable '{variable}' not found in ERA5 Zarr store")

        data = self._dataset[variable]

        if height_idx is not None:
            if len(data.dims) < 4:
                raise ResError(f"Variable '{variable}' does not have a height dimension")
            height_dim = [dim for dim in data.dims if dim not in {self.time_name, "latitude", "longitude"}][0]
            data = data.isel({height_dim: height_idx})

        if "latitude" in data.dims:
            data = data.isel(latitude=slice(self._latStart, self._latStop))
        if "longitude" in data.dims:
            data = data.isel(longitude=slice(self._lonStart, self._lonStop))

        expected_dims = (self.time_name, "latitude", "longitude")
        if data.dims != expected_dims:
            raise ResError(f"Variable '{variable}' is expected to have dimensions {expected_dims}, got {data.dims}")

        tmp = np.asarray(data.values)

        if processor is not None:
            tmp = processor(tmp)

        if tmp.shape[0] != self._timeindex_raw.shape[0]:
            if not self.fill:
                raise ResError(
                    "Time mismatch with variable %s. Expected %d, got %d"
                    % (variable, self.time_index.shape[0], tmp.shape[0])
                )
            if tmp.shape[0] + 1 != self._timeindex_raw.shape[0]:
                raise ResError("Filling is only intended to fill the last missing step")
            tmp = np.append(tmp, tmp[np.newaxis, -1, :, :], axis=0)

        self.data[name] = tmp[:, :: -1 if self._flip_lat else 1, :: -1 if self._flip_lon else 1]

        if self._verbose:
            shape = tuple(self.data[name].shape)
            print(f"Loaded ERA5 Zarr variable '{variable}' as '{name}' with shape {shape}")

    def _load_with_fallback(self, preferred_variable, fallback_variable, target_name, derived_warning=None):
        for variable in (preferred_variable, fallback_variable):
            if variable not in self.variables.index:
                continue
            if variable in self._derived_variables and derived_warning is not None:
                warnings.warn(derived_warning, stacklevel=2)
            self.load(variable, name=target_name)
            if variable in self._derived_variables and np.all(np.isnan(self.data[target_name][0])):
                warnings.warn(
                    f"The first timestep of '{target_name}' ({self.time_index[0]}) is NaN because the raw "
                    f"accumulation preceding the start of the store is not available. Drop or fill this "
                    f"timestep, or start the requested 'time_slice' one hour later.",
                    stacklevel=2,
                )
            return
        raise RuntimeError(
            f"Cannot load {target_name}: neither '{preferred_variable}' nor '{fallback_variable}' exist in the ERA5 Zarr store"
        )

    def sload_boundary_layer_height(self):
        return self._load_with_fallback(
            preferred_variable="blh",
            fallback_variable="boundary_layer_height",
            target_name="boundary_layer_height",
        )

    def sload_direct_horizontal_irradiance(self):
        return self._load_with_fallback(
            preferred_variable="fdir_t_adj",
            fallback_variable="fdir",
            target_name="direct_horizontal_irradiance",
            derived_warning=(
                "Processed ERA5 direct horizontal irradiance ('fdir_t_adj') is not available in this Zarr store; "
                "computing on the fly from raw 'fdir' (J/m² → W/m², time-shifted +1 h)."
            ),
        )

    def sload_global_horizontal_irradiance(self):
        return self._load_with_fallback(
            preferred_variable="ssrd_t_adj",
            fallback_variable="ssrd",
            target_name="global_horizontal_irradiance",
            derived_warning=(
                "Processed ERA5 global horizontal irradiance ('ssrd_t_adj') is not available in this Zarr store; "
                "computing on the fly from raw 'ssrd' (J/m² → W/m², time-shifted +1 h)."
            ),
        )

    def get(
        self, variable, locations, interpolation="near", force_as_data_frame=False, outside_okay=False, _indices=None
    ):
        if self._longitude_360:
            original_locs = gk.LocationSet(locations)
            wrapped_locs = [(lon % 360.0, lat) for lon, lat in zip(original_locs.lons, original_locs.lats)]
            result = NCSource.get(
                self,
                variable=variable,
                locations=wrapped_locs,
                interpolation=interpolation,
                force_as_data_frame=force_as_data_frame,
                outside_okay=outside_okay,
                _indices=_indices,
            )
            # Restore original location labels (NCSource names columns/series from the locations passed to it)
            original_names = [f"({loc.lon}, {loc.lat})" for loc in original_locs._locations]
            if isinstance(result, pd.DataFrame):
                result.columns = original_names
            else:
                result.name = original_names[0]
            return result
        return NCSource.get(
            self,
            variable=variable,
            locations=locations,
            interpolation=interpolation,
            force_as_data_frame=force_as_data_frame,
            outside_okay=outside_okay,
            _indices=_indices,
        )

    loc_to_index = NCSource._loc_to_index_rect(0.25, 0.25)
