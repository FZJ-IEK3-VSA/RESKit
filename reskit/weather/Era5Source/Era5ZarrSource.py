from collections import OrderedDict
from datetime import timedelta
from time import perf_counter
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
        if kwargs:
            unexpected = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments for Era5ZarrSource: {unexpected}")

        timings = []

        def _mark(label, tic):
            if verbose:
                timings.append((label, perf_counter() - tic))

        self.fill = forward_fill
        self._flip_lat = True
        self._flip_lon = False
        self._maximal_lon_difference = self.MAX_LON_DIFFERENCE
        self._maximal_lat_difference = self.MAX_LAT_DIFFERENCE
        self.dependent_coordinates = False
        self._source_descriptor = source
        self._verbose = verbose

        tic = perf_counter()
        ds = self._open_dataset(
            source=source,
            chunks=chunks,
            consolidated=consolidated,
            storage_options=storage_options,
        )
        _mark("open_zarr", tic)

        if "valid_time" in ds.coords:
            self.time_name = "valid_time"
        elif "time" in ds.coords:
            self.time_name = "time"
        else:
            raise ResError("ERA5 Zarr store must provide either 'valid_time' or 'time' as a coordinate")

        if time_slice is not None:
            tic = perf_counter()
            if isinstance(time_slice, slice):
                offset = timedelta(minutes=30)
                raw_start = pd.Timestamp(time_slice.start) + offset if time_slice.start is not None else None
                raw_stop = pd.Timestamp(time_slice.stop) + offset if time_slice.stop is not None else None
                raw_time_slice = slice(raw_start, raw_stop)
            else:
                raw_time_slice = time_slice
            ds = ds.sel({self.time_name: raw_time_slice})
            _mark("apply_time_slice", tic)

        if "values" in ds.dims:
            raise ResError(
                "This Era5ZarrSource implementation only supports regular latitude/longitude Zarr stores, not flattened 'values' grids."
            )

        self._dataset = ds

        tic = perf_counter()
        self.variables = self._build_variable_table(ds, source)
        _mark("build_variable_table", tic)

        era5_names = {
            "global_horizontal_irradiance": "ssrd_t_adj",
            "direct_horizontal_irradiance": "fdir_t_adj",
            "surface_wind_speed": "w10",
            "elevated_wind_speed": "w100",
        }
        if time_index_from in era5_names:
            time_index_from = era5_names[time_index_from]

        if time_index_from is not None and time_index_from not in self.variables.index:
            raise ResError(
                f"ERA5 key '{time_index_from}' not known. Check variable 'time_index_from' and store {source}"
            )

        tic = perf_counter()
        self._allLats = np.asarray(ds["latitude"].values)
        self._allLons = np.asarray(ds["longitude"].values)
        self._lonN = self._allLons.size
        self._latN = self._allLats.size
        self._longitude_360 = self._allLons.min() >= 0 and self._allLons.max() > 180
        _mark("load_coordinates", tic)

        tic = perf_counter()
        self._configure_spatial_selection(bounds=bounds, index_pad=index_pad)
        _mark("configure_spatial_selection", tic)

        tic = perf_counter()
        self._apply_dataset_spatial_subset()
        _mark("apply_dataset_spatial_subset", tic)

        self.extent = gk.Extent(
            self.lons.min(),
            self.lats.min(),
            self.lons.max(),
            self.lats.max(),
            srs=gk.srs.EPSG4326,
        )

        tic = perf_counter()
        timeindex = pd.DatetimeIndex(pd.to_datetime(self._dataset[self.time_name].values)) + timedelta(minutes=-30)
        self._timeindex_raw = timeindex
        self.time_index = timeindex
        _mark("build_time_index", tic)
        self.data = OrderedDict()

        if verbose:
            print(f"Opened ERA5 Zarr source: {source}")
            for label, dt in timings:
                print(f"  {label:<28} {dt:8.2f} s")

    @staticmethod
    def _open_dataset(source, chunks, consolidated, storage_options):
        if isinstance(source, xr.Dataset):
            return source

        storage_options_ = dict(storage_options or {})
        if isinstance(source, str) and source.startswith("https://") and "client_kwargs" not in storage_options_:
            storage_options_["client_kwargs"] = {"trust_env": True}
        if isinstance(source, str) and source.startswith("gs://") and "token" not in storage_options_:
            storage_options_["token"] = "anon"

        return xr.open_dataset(
            source,
            chunks=chunks,
            engine="zarr",
            consolidated=consolidated,
            storage_options=storage_options_ or None,
            decode_timedelta=False,
        )

    @staticmethod
    def _build_variable_table(ds: xr.Dataset, source: Any) -> pd.DataFrame:
        index = list(ds.data_vars) + [name for name in ds.coords if name not in ds.data_vars]
        table = pd.DataFrame(columns=["name", "units", "shape", "path"], index=index)

        for var in index:
            data = ds[var]
            table.loc[var, "name"] = data.attrs.get("standard_name", data.attrs.get("long_name", "Unknown"))
            table.loc[var, "units"] = data.attrs.get("units", "Unknown")
            table.loc[var, "shape"] = tuple(data.shape)
            table.loc[var, "path"] = str(source)

        return table

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

        timings = []

        def _mark(label, tic):
            if self._verbose:
                timings.append((label, perf_counter() - tic))

        if variable not in self.variables.index:
            raise ResError(f"Variable '{variable}' not found in ERA5 Zarr store")

        tic = perf_counter()
        data = self._dataset[variable]
        _mark("select_variable", tic)

        if height_idx is not None:
            if len(data.dims) < 4:
                raise ResError(f"Variable '{variable}' does not have a height dimension")
            tic = perf_counter()
            height_dim = [dim for dim in data.dims if dim not in {self.time_name, "latitude", "longitude"}][0]
            data = data.isel({height_dim: height_idx})
            _mark("select_height_index", tic)

        if "latitude" in data.dims or "longitude" in data.dims:
            tic = perf_counter()
        if "latitude" in data.dims:
            data = data.isel(latitude=slice(self._latStart, self._latStop))
        if "longitude" in data.dims:
            data = data.isel(longitude=slice(self._lonStart, self._lonStop))
        if "latitude" in data.dims or "longitude" in data.dims:
            _mark("apply_spatial_subset", tic)

        expected_dims = (self.time_name, "latitude", "longitude")
        if data.dims != expected_dims:
            raise ResError(f"Variable '{variable}' is expected to have dimensions {expected_dims}, got {data.dims}")

        tic = perf_counter()
        tmp = np.asarray(data.values)
        _mark("materialize_values", tic)

        if processor is not None:
            tic = perf_counter()
            tmp = processor(tmp)
            _mark("processor", tic)

        if tmp.shape[0] != self._timeindex_raw.shape[0]:
            if not self.fill:
                raise ResError(
                    "Time mismatch with variable %s. Expected %d, got %d"
                    % (variable, self.time_index.shape[0], tmp.shape[0])
                )
            if tmp.shape[0] + 1 != self._timeindex_raw.shape[0]:
                raise ResError("Filling is only intended to fill the last missing step")
            tmp = np.append(tmp, tmp[np.newaxis, -1, :, :], axis=0)

        if self._flip_lat and not self._flip_lon:
            self.data[name] = tmp[:, ::-1, :]
        elif not self._flip_lat and self._flip_lon:
            self.data[name] = tmp[:, :, ::-1]
        elif self._flip_lat and self._flip_lon:
            self.data[name] = tmp[:, ::-1, ::-1]
        else:
            self.data[name] = tmp

        if self._verbose:
            shape = tuple(self.data[name].shape)
            print(f"Loaded ERA5 Zarr variable '{variable}' as '{name}' with shape {shape}")
            for label, dt in timings:
                print(f"  {label:<28} {dt:8.2f} s")

    def _load_with_fallback(self, preferred_variable, fallback_variable, target_name, warning_message=None, fallback_processor=None):
        if preferred_variable in self.variables.index:
            return self.load(preferred_variable, name=target_name)
        if fallback_variable in self.variables.index:
            if warning_message is not None:
                warnings.warn(warning_message, stacklevel=2)
            return self.load(fallback_variable, name=target_name, processor=fallback_processor)
        raise RuntimeError(
            f"Cannot load {target_name}: neither '{preferred_variable}' nor '{fallback_variable}' exist in the ERA5 Zarr store"
        )

    @staticmethod
    def _solar_accumulation_to_mean_power(arr):
        """Convert ERA5 accumulated solar radiation (J/m²) to mean power (W/m²).

        Mirrors the CDO pipeline ``-divc,3600 -shifttime,+1hour``:
        - divide by 3600 (J m⁻² per hour → W m⁻²)
        - shift by one time step so that adj[i] = raw[i-1]  (CDO shifttime +1 h
          re-labels each record's timestamp one hour later, which in array terms
          means the adjusted value at position i comes from the raw position i-1)
        - fill the first step with 0 (no preceding accumulation available)
        """
        out = np.empty_like(arr, dtype=np.float64)
        out[1:] = arr[:-1] / 3600.0
        out[0] = 0.0
        return out

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
            warning_message=(
                "Processed ERA5 direct horizontal irradiance ('fdir_t_adj') is not available in this Zarr store; "
                "computing on the fly from raw 'fdir' (J/m² → W/m², time-shifted +1 h)."
            ),
            fallback_processor=self._solar_accumulation_to_mean_power,
        )

    def sload_global_horizontal_irradiance(self):
        return self._load_with_fallback(
            preferred_variable="ssrd_t_adj",
            fallback_variable="ssrd",
            target_name="global_horizontal_irradiance",
            warning_message=(
                "Processed ERA5 global horizontal irradiance ('ssrd_t_adj') is not available in this Zarr store; "
                "computing on the fly from raw 'ssrd' (J/m² → W/m², time-shifted +1 h)."
            ),
            fallback_processor=self._solar_accumulation_to_mean_power,
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
