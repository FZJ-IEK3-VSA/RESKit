import os
import cdsapi
import geokit as gk
import netCDF4 as nc4
import numpy as np
import pandas as pd
import xarray as xr
from typing import Union, List, Optional, Tuple
from reskit.util.weather_tile import get_tile_XY


# Coordinate names that must not be treated as data variables when listing/tiling.
_ERA5_COORD_NAMES = {"time", "valid_time", "latitude", "longitude"}

# Encoding keys tied to the source file's on-disk layout. They must be dropped before
# writing a reshaped/subset dataset, otherwise xarray may fail (e.g. chunk sizes larger
# than the new dimensions) or carry over a stale shape.
_LAYOUT_ENCODING_KEYS = {"source", "original_shape", "chunksizes", "preferred_chunks"}


def _open_era5_dataset(nc_path: str) -> xr.Dataset:
    """Open an ERA5 NetCDF file, normalising the time coordinate name to ``time``.

    ERA5 ``netcdf_legacy`` downloads use ``time``; the newer (non-legacy) export uses
    ``valid_time``. Downstream code (Era5Source/NCSource) expects ``time``.
    """
    ds = xr.open_dataset(nc_path)
    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    return ds


def _nc_data_var_names(nc_path: str) -> list[str]:
    """Return the data (non-coordinate) variable names in a NetCDF file."""
    with nc4.Dataset(nc_path) as ds:
        return [v for v in ds.variables if v not in _ERA5_COORD_NAMES]


def _nc_years(nc_path: str) -> list[str]:
    """Return the sorted unique 4-digit years present in the time coordinate."""
    with nc4.Dataset(nc_path) as ds:
        tv = ds.variables["time"]
        dts = nc4.num2date(
            tv[:],
            tv.units,
            getattr(tv, "calendar", "standard"),
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=False,
        )
    return sorted({f"{d.year}" for d in np.atleast_1d(np.asarray(dts)).ravel()})


def _nc_file_has_vars(nc_path: str, required_vars) -> bool:
    """Check whether a NetCDF file exists and contains required data variables."""
    if not os.path.exists(nc_path):
        return False
    try:
        return set(required_vars) <= set(_nc_data_var_names(nc_path))
    except Exception:
        return False


def _spatial_subset(
    ds: xr.Dataset,
    lon_west: float,
    lon_east: float,
    lat_south: float,
    lat_north: float,
) -> xr.Dataset:
    """Inclusive lat/lon box subset (CDO ``sellonlatbox`` equivalent).

    Uses integer-index selection via boolean masks rather than ``.sel(slice(...))`` so it
    is robust to the axis ordering: ERA5 latitudes are descending and longitudes may be
    ascending or shifted, and a wrongly-directed slice would silently return nothing.
    """
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    lat_idx = np.where((lat >= lat_south) & (lat <= lat_north))[0]
    lon_idx = np.where((lon >= lon_west) & (lon <= lon_east))[0]
    return ds.isel(latitude=lat_idx, longitude=lon_idx)


def _write_netcdf(ds: xr.Dataset, path: str) -> None:
    """Write a dataset to NetCDF, preserving CF/time encoding for the netCDF4 consumer.

    Strips on-disk-layout encoding (chunk sizes, original shape) that does not transfer to
    a reshaped dataset, and disables ``_FillValue`` on the coordinate variables to match
    the previous CDO output convention.
    """
    ds = ds.copy()
    for name in ds.variables:
        enc = {k: v for k, v in ds[name].encoding.items() if k not in _LAYOUT_ENCODING_KEYS}
        if name in _ERA5_COORD_NAMES:
            enc["_FillValue"] = None
        ds[name].encoding = enc
    ds.to_netcdf(path)


"""
Running this module will automatically download ERA5 data from CDS for the specified date range and boundary box,
and preprocess the data to adjust solar radiation variables and compute wind speed variables.

Processing is done with xarray/netCDF4 (no external CDO binary required), so it runs on
Linux, macOS and Windows.

To make it happen, you need an account at the Copernicus Climate Data Store (CDS) with
your API key set up in the ~/.cdsapirc file. See: https://cds.climate.copernicus.eu/how-to-api
"""


era5_variables = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "2m_temperature",
    "surface_pressure",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "surface_solar_radiation_downwards",
    "total_sky_direct_solar_radiation_at_surface",
    "boundary_layer_height",
    "forecast_surface_roughness",
]
era5_dataset = "reanalysis-era5-single-levels"

# Maps NetCDF short variable names to the tile filename label segment, matching the
# existing shared data convention: reanalysis-era5-single-levels.z<z>.x<x>.y<y>.y<year>.<label>.nc
_ERA5_NC_TO_TILE_LABEL = {
    # derived wind speeds (from preprocess_era5_data)
    "ws100": "100m_wind_speed.processed",
    "wd100": "100m_wind_direction.processed",
    "ws10": "10m_wind_speed.processed",
    "wd10": "10m_wind_direction.processed",
    # time-adjusted solar (from preprocess_era5_data)
    "ssrd_t_adj": "surface_solar_radiation_downwards.processed.t_adjusted",
    "fdir_t_adj": "total_sky_direct_solar_radiation_at_surface.processed.t_adjusted",
    # unit-converted solar without time adjustment (from preprocess_era5_data)
    "ssrd": "surface_solar_radiation_downwards.processed",
    "fdir": "total_sky_direct_solar_radiation_at_surface.processed",
    # raw variables passed through unchanged
    "t2m": "2m_temperature",
    "d2m": "2m_dewpoint_temperature",
    "sp": "surface_pressure",
    "blh": "boundary_layer_height",
    "fsr": "forecast_surface_roughness",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "u100": "100m_u_component_of_wind",
    "v100": "100m_v_component_of_wind",
}


def _normalize_lon(lon: float) -> float:
    """Wrap a longitude into the [-180, 180] interval."""
    lon = float(lon)
    wrapped = ((lon + 180.0) % 360.0) - 180.0
    # Preserve +180 for positive inputs so padded edge tiles can be represented cleanly.
    if wrapped == -180.0 and lon > 0:
        return 180.0
    return wrapped


def _tile_lookup_lon(lon: float) -> float:
    """Return a normalized longitude safe for slippy-tile index lookup."""
    wrapped = _normalize_lon(lon)
    # deg2num-style tile conversion treats exactly +180 as the next tile column.
    return wrapped if wrapped < 180.0 else 180.0 - 1e-9


def _iter_tile_x_indices(zoom_level: int, lon_west: float, lon_east: float) -> list[int]:
    """Return tile X indices covered by a lon span, including antimeridian wraps."""
    west = _tile_lookup_lon(lon_west)
    east = _tile_lookup_lon(lon_east)
    x_west, _ = get_tile_XY(zoom_level, lon=west, lat=0.0)
    x_east, _ = get_tile_XY(zoom_level, lon=east, lat=0.0)

    if _normalize_lon(lon_west) <= _normalize_lon(lon_east):
        return list(range(x_west, x_east + 1))

    n_tiles = 2**zoom_level
    return list(range(x_west, n_tiles)) + list(range(0, x_east + 1))


def _split_lon_boxes(lon_west: float, lon_east: float) -> list[tuple[float, float]]:
    """Split a lon interval into one or two non-wrapping boxes in [-180, 180]."""
    lon_west = float(lon_west)
    lon_east = float(lon_east)

    if lon_east - lon_west >= 360.0:
        return [(-180.0, 180.0)]

    if -180.0 <= lon_west <= 180.0 and -180.0 <= lon_east <= 180.0 and lon_west <= lon_east:
        return [(lon_west, lon_east)]

    west = _normalize_lon(lon_west)
    east = _normalize_lon(lon_east)
    if west <= east:
        return [(west, east)]

    return [(west, 180.0), (-180.0, east)]


def _shift_lon_to_dataset_window(lon: float, lon_min: float, lon_max: float) -> float:
    """Shift a longitude by +/-360 to match the source dataset's longitude convention."""
    candidates = (float(lon) - 360.0, float(lon), float(lon) + 360.0)
    midpoint = 0.5 * (float(lon_min) + float(lon_max))
    return min(candidates, key=lambda candidate: abs(candidate - midpoint))


def _get_source_lon_boxes(
    lon_west: float,
    lon_east: float,
    source_lon_min: float,
    source_lon_max: float,
) -> list[tuple[float, float]]:
    """Return extraction lon boxes aligned to the source file's longitude convention."""
    if source_lon_min < -180.0 or source_lon_max > 180.0:
        west = _shift_lon_to_dataset_window(lon_west, source_lon_min, source_lon_max)
        east = _shift_lon_to_dataset_window(lon_east, source_lon_min, source_lon_max)
        if west <= east:
            return [(west, east)]

    return _split_lon_boxes(lon_west=lon_west, lon_east=lon_east)


def _align_longitudes_to_source_convention(target_file: str, source_lon_min: float, source_lon_max: float) -> None:
    """Shift written tile longitudes by +/-360 so they match the source file convention."""
    if not (source_lon_min < -180.0 or source_lon_max > 180.0):
        return

    with nc4.Dataset(target_file, "r+") as ds:
        lon_var = ds.variables.get("longitude")
        if lon_var is None:
            return

        lon_values = lon_var[:]
        if lon_values.size == 0:
            return

        source_midpoint = 0.5 * (float(source_lon_min) + float(source_lon_max))
        lon_midpoint = 0.5 * (float(lon_values[0]) + float(lon_values[-1]))
        shift = min((-360.0, 0.0, 360.0), key=lambda delta: abs((lon_midpoint + delta) - source_midpoint))
        if shift != 0.0:
            lon_var[:] = lon_values + shift


def _tile_variable_to_file(
    source_file: str,
    var: str,
    year: str,
    lat_south: float,
    lat_north: float,
    lon_boxes: list[tuple[float, float]],
    target_file: str,
    source_lon_min: float,
    source_lon_max: float,
) -> None:
    """Extract one variable/year tile, merging antimeridian-split lon boxes when needed."""
    with _open_era5_dataset(source_file) as ds:
        da = ds[[var]]
        da = da.isel(time=np.where(da["time"].dt.year.values == int(year))[0])

        parts = [_spatial_subset(da, lw, le, lat_south, lat_north) for lw, le in lon_boxes]
        parts = [p for p in parts if p.sizes["longitude"] > 0]

        if len(parts) <= 1:
            out = parts[0] if parts else _spatial_subset(da, *lon_boxes[0], lat_south, lat_north)
        else:
            # Merge antimeridian-split boxes (CDO ``mergegrid`` equivalent): concatenate
            # along longitude, order, and drop any shared boundary column.
            out = xr.concat(parts, dim="longitude").sortby("longitude")
            _, unique_idx = np.unique(out["longitude"].values, return_index=True)
            out = out.isel(longitude=np.sort(unique_idx))

        out = out.load()
    _write_netcdf(out, target_file)
    _align_longitudes_to_source_convention(target_file, source_lon_min, source_lon_max)


def era5_downloader(
    target_filename: str,
    year: Union[str, List[str]],
    month: Union[str, List[str]],
    variables: list[str],
    area: tuple[float, float, float, float],
    day: Optional[List[str]] = None,
    time: Optional[List[str]] = None,
    grid: tuple[float, float] = (0.25, 0.25),
    data_format: str = "netcdf_legacy",
):
    # default to every day / every hour of the requested month(s); CDS ignores days that
    # do not exist in a given month (e.g. day 31 in February)
    if day is None:
        day = [f"{d:02d}" for d in range(1, 32)]
    if time is None:
        time = [f"{h:02d}:00" for h in range(24)]

    client = cdsapi.Client()

    request = {
        "product_type": "reanalysis",
        "format": data_format,
        "variable": variables,
        "year": year,
        "month": month,
        "day": day,
        "time": time,
        "area": area,  # [north, west, south, east]
        "grid": f"{grid[0]}/{grid[1]}",
    }

    client.retrieve(
        era5_dataset,
        request,
        target_filename,
    )


def preprocess_era5_data(focus_nc: str, processed_dir: Optional[str] = None):
    """
    Ssrd and fdir are hourly backward accumulated quantities in ERA5 with the unit: J m⁻².
    Each value at time t represents the accumulated energy over the previous hour.

    When you divide it by 3600 seconds, you convert the accumulated energy (J m⁻²)
    into an average power flux (W m⁻²) over that hour.

    However, in solar observation and other models,
    this value represents the instantaneous mean over the next hour.
    This matches:
        PV modeling conventions
        Many energy system models
        atlite / PyPSA conventions

    So, we need to do two things:
    1. Convert the accumulated quantity to an average power flux by dividing by 3600
    2. Shift the time axis forward by one hour to represent the average over the next hour.

    Parameters
    ----------
    focus_nc : str
        Path to the raw ERA5 NetCDF file.
    processed_dir : str, optional
        Directory to write processed output files. Defaults to the same directory
        as focus_nc.
    """
    out_dir = processed_dir or os.path.dirname(focus_nc)
    if processed_dir:
        os.makedirs(processed_dir, exist_ok=True)
    f_name = os.path.basename(focus_nc)

    with _open_era5_dataset(focus_nc) as ds:
        varset = set(ds.data_vars)

        # process for solar radiation variables (time adjusted)
        # Build the variable list dynamically from whichever of ssrd/fdir is present so that
        # workflows requesting only one of them (e.g. CSP needs fdir but not ssrd) still work.
        solar_t_out = os.path.join(out_dir, f"{f_name.split('.')[0]}_processed_solar_t_adjusted.nc")
        solar_vars = [v for v in ("ssrd", "fdir") if v in varset]
        if solar_vars:
            out_names = [f"{v}_t_adj" for v in solar_vars]
            if not _nc_file_has_vars(solar_t_out, out_names):
                # 1. accumulated J m**-2 -> mean power flux W m**-2 (divide by 3600s)
                # 2. shift time axis +1h so each value is the mean over the *next* hour
                time_encoding = dict(ds["time"].encoding)
                out = ds[solar_vars] / 3600.0
                out = out.assign_coords(time=out["time"] + pd.Timedelta(hours=1))
                out["time"].encoding = time_encoding
                out = out.rename({v: f"{v}_t_adj" for v in solar_vars})
                for v in solar_vars:
                    attrs = dict(ds[v].attrs)
                    attrs["units"] = "W m**-2"
                    out[f"{v}_t_adj"].attrs = attrs
                _write_netcdf(out.load(), solar_t_out)
            else:
                print(f"Skipping process time-adjusted solar (exists): {solar_t_out}")

        # process for wind speed variables
        ws100_out = os.path.join(out_dir, f"{f_name.split('.')[0]}_processed_ws100.nc")
        if {"u100", "v100"} <= varset:
            if not _nc_file_has_vars(ws100_out, ["ws100"]):
                ws100 = np.sqrt(ds["u100"] ** 2 + ds["v100"] ** 2)
                ws100.name = "ws100"
                ws100.attrs = {"long_name": "100 metre wind speed", "units": "m s**-1"}
                _write_netcdf(ws100.to_dataset().load(), ws100_out)
            else:
                print(f"Skipping process ws100 (exists): {ws100_out}")

        ws10_out = os.path.join(out_dir, f"{f_name.split('.')[0]}_processed_ws10.nc")
        if {"u10", "v10"} <= varset:
            if not _nc_file_has_vars(ws10_out, ["ws10"]):
                ws10 = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)
                ws10.name = "ws10"
                ws10.attrs = {"long_name": "10 metre wind speed", "units": "m s**-1"}
                _write_netcdf(ws10.to_dataset().load(), ws10_out)
            else:
                print(f"Skipping process ws10 (exists): {ws10_out}")


def era5_tiler(
    processed_dir: str,
    tile_output_dir: str,
    zoom_level: int = 4,
    raw_nc: Optional[str] = None,
    raw_variables: Optional[List[str]] = None,
) -> str:
    """
    Splits processed ERA5 NetCDF files into the tiled directory structure expected
    by Era5Source and execute_workflow_iteratively().

    Output follows the shared-data naming convention:
        <tile_output_dir>/<zoom>/<xi>/<yi>/<year>/
            reanalysis-era5-single-levels.z<z>.x<xi>.y<yi>.y<year>.<label>.nc
    where <label> is taken from _ERA5_NC_TO_TILE_LABEL (e.g. "100m_wind_speed.processed").
    One output file is written per variable per tile per year.

    Parameters
    ----------
    processed_dir : str
        Directory containing processed NC files (output of preprocess_era5_data).
    tile_output_dir : str
        Root directory for the tiled output.
    zoom_level : int
        Web Mercator zoom level (default: 4 → 16×16 global grid).
    raw_nc : str, optional
        Path to the raw ERA5 download file, used to tile raw_variables.
    raw_variables : list of str, optional
        NC short names to tile from raw_nc (e.g. ['t2m', 'sp', 'blh']).
        Ignored if raw_nc is None.
    """
    source_group = "reanalysis-era5-single-levels"

    # (source_file, [nc_var_names_to_tile]) pairs
    file_var_pairs: list[tuple[str, list[str]]] = []

    for f in sorted(os.listdir(processed_dir)):
        if not f.endswith(".nc") or "_processed_" not in f:
            continue
        path = os.path.join(processed_dir, f)
        vars_in_file = _nc_data_var_names(path)
        vars_to_tile = [v for v in vars_in_file if v in _ERA5_NC_TO_TILE_LABEL]
        if vars_to_tile:
            file_var_pairs.append((path, vars_to_tile))

    if raw_nc and raw_variables:
        vars_in_raw = set(_nc_data_var_names(raw_nc))
        vars_to_tile = [v for v in raw_variables if v in vars_in_raw and v in _ERA5_NC_TO_TILE_LABEL]
        if vars_to_tile:
            file_var_pairs.append((raw_nc, vars_to_tile))

    for source_file, variables in file_var_pairs:
        with nc4.Dataset(source_file) as ds:
            lats = ds.variables["latitude"][:]
            lons = ds.variables["longitude"][:]
        lon_min, lon_max = float(lons.min()), float(lons.max())
        lat_min, lat_max = float(lats.min()), float(lats.max())
        years = _nc_years(source_file)

        # NW corner → SE corner (tile Y increases southward)
        xi_values = _iter_tile_x_indices(zoom_level=zoom_level, lon_west=lon_min, lon_east=lon_max)
        _, yi_nw = get_tile_XY(zoom_level, lon=_tile_lookup_lon(lon_min), lat=lat_max)
        _, yi_se = get_tile_XY(zoom_level, lon=_tile_lookup_lon(lon_max), lat=lat_min)

        for xi in xi_values:
            for yi in range(yi_nw, yi_se + 1):
                extent = gk.Extent.fromTile(xi, yi, zoom_level).castTo(gk.srs.EPSG4326).pad(2)
                lon_west, lon_east, lat_south, lat_north = extent.xXyY
                lon_boxes = _get_source_lon_boxes(
                    lon_west=lon_west,
                    lon_east=lon_east,
                    source_lon_min=lon_min,
                    source_lon_max=lon_max,
                )

                for year in years:
                    target_dir = os.path.join(tile_output_dir, str(zoom_level), str(xi), str(yi), str(year))
                    os.makedirs(target_dir, exist_ok=True)

                    for var in variables:
                        label = _ERA5_NC_TO_TILE_LABEL[var]
                        target_file = os.path.join(
                            target_dir,
                            f"{source_group}.z{zoom_level}.x{xi}.y{yi}.y{year}.{label}.nc",
                        )
                        if os.path.exists(target_file):
                            print(f"Skipping tile (exists): {target_file}")
                            continue

                        _tile_variable_to_file(
                            source_file=source_file,
                            var=var,
                            year=year,
                            lat_south=lat_south,
                            lat_north=lat_north,
                            lon_boxes=lon_boxes,
                            target_file=target_file,
                            source_lon_min=lon_min,
                            source_lon_max=lon_max,
                        )

    return tile_output_dir


def _era5_download_jobs(start_date: str, end_date: str) -> List[Tuple[str, List[str], List[str]]]:
    """Split the inclusive ``[start_date, end_date]`` range into CDS download jobs.

    A CDS request selects data as the Cartesian product of its ``year`` x ``month`` x ``day``
    lists, so a job is kept to a single year and a single day-set to describe its dates exactly.
    Months that are fully covered by the range are batched together per year using the canonical
    ``01..31`` day list (CDS ignores days that do not exist in a month); the partial first/last
    month of the range (if any) each become their own job with only the days actually requested.

    Parameters
    ----------
    start_date, end_date : str
        Inclusive date bounds (``"YYYY-MM-DD"``). The whole ``end_date`` day is included.

    Returns
    -------
    list of (year, months, days)
        ``year`` is a 4-digit string, ``months`` and ``days`` are lists of 2-digit strings.
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if end < start:
        raise ValueError(f"end_date ({end_date}) is before start_date ({start_date}).")

    full_days = [f"{d:02d}" for d in range(1, 32)]

    # group (year, day-set) -> months, preserving chronological order
    grouped: dict = {}
    for period in pd.period_range(start=start, end=end, freq="M"):
        month_start = max(start, period.start_time)
        month_end = min(end, period.end_time)
        if month_start.day == 1 and month_end.day == period.days_in_month:
            days = tuple(full_days)  # whole month -> batch with other full months
        else:
            days = tuple(f"{d:02d}" for d in range(month_start.day, month_end.day + 1))
        grouped.setdefault((period.year, days), []).append(f"{period.month:02d}")

    return [(str(year), sorted(months), list(days)) for (year, days), months in grouped.items()]


def prepare_era5(
    start_date: str,
    end_date: str,
    boundary_box: dict,
    output_dir: str,
    variables: Optional[List[str]] = None,
    tiling: bool = False,
    zoom_level: int = 4,
    tile_output_dir: Optional[str] = None,
    raw_variables: Optional[List[str]] = None,
):
    # 1. download ERA5 data for the given date range and boundary box
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    if variables is None:
        variables = era5_variables

    # exact, day-level download jobs covering [start_date, end_date] inclusively
    jobs = _era5_download_jobs(start_date, end_date)

    area = (
        boundary_box["north"],
        boundary_box["west"],
        boundary_box["south"],
        boundary_box["east"],
    )

    bbox_tag = f"N{boundary_box['north']}_S{boundary_box['south']}_W{boundary_box['west']}_E{boundary_box['east']}"
    # name the file after the actual day-level extent it contains
    date_tag = f"{pd.Timestamp(start_date).strftime('%Y%m%d')}-{pd.Timestamp(end_date).strftime('%Y%m%d')}"
    output_file = os.path.join(raw_dir, f"{era5_dataset}_{date_tag}_{bbox_tag}_raw.nc")

    if not os.path.exists(output_file):
        if len(jobs) == 1:
            # single request
            year, months, days = jobs[0]
            era5_downloader(
                target_filename=output_file,
                year=year,
                month=months,
                day=days,
                variables=variables,
                area=area,
            )
        else:
            # multiple requests (partial months and/or multiple years): download into temp
            # files, then merge along time
            tmp_files = []
            for i, (year, months, days) in enumerate(jobs):
                tmp_file = os.path.join(raw_dir, f"_tmp_{era5_dataset}_{i:02d}_{bbox_tag}_raw.nc")
                tmp_files.append(tmp_file)
                if not os.path.exists(tmp_file):
                    era5_downloader(
                        target_filename=tmp_file,
                        year=year,
                        month=months,
                        day=days,
                        variables=variables,
                        area=area,
                    )
            datasets = [_open_era5_dataset(f) for f in tmp_files]
            try:
                time_encoding = dict(datasets[0]["time"].encoding)
                merged = xr.concat(datasets, dim="time").sortby("time")
                merged["time"].encoding = time_encoding
                merged = merged.load()
            finally:
                for d in datasets:
                    d.close()
            _write_netcdf(merged, output_file)
            for f in tmp_files:
                os.remove(f)
    else:
        print(f"ERA5 data already exists at {output_file}, skipping download.")

    # 2. preprocess ERA5 data into processed_dir (sibling of raw/)
    processed_dir = os.path.join(output_dir, "processed")
    preprocess_era5_data(focus_nc=output_file, processed_dir=processed_dir)
    print("ERA5 preprocessing done.")

    # 3. optionally tile into <zoom>/<xi>/<yi>/<year>/ structure
    if tiling:
        _tile_out = tile_output_dir or os.path.join(output_dir, "tiles")
        era5_tiler(
            processed_dir=processed_dir,
            tile_output_dir=_tile_out,
            zoom_level=zoom_level,
            raw_nc=output_file,
            raw_variables=raw_variables,
        )
        print("ERA5 tiling done.")
        return _tile_out

    return processed_dir
