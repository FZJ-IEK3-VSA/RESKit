"""
Create a long-run average (LRA) datasets for e.g. wind speeds and solar radiation.
Typical usage (Python):

    import reskit as rk

    ds_lra = rk.create_LRA(
        base_path="/path/to/ERA5",
        variable="surface_solar_radiation_downwards.processed.t_adjusted",
        start_year=1994,
        end_year=2018,
        zoom_level=4,
        out_dir="./output",
        temp_dir=None,
    )

It will write:
  - per-year merged NetCDFs (optional cache)
  - a final LRA NetCDF
  - optionally, a GeoTIFF (requires rioxarray + rasterio)


Input layouts supported:
    - tiled:    <base>/<zoom>/<x-tile>/<y-tile>/<year>/*.{variable}.nc
    - non-tiled (per-year): <base>/<year>/*.{variable}.nc (or similar)
    - non-tiled (misc): searches common patterns including files with the year in
        the filename.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import logging
from pathlib import Path
from typing import Iterable, Literal, Optional
import xarray as xr
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import geokit as gk

LOG = logging.getLogger(__name__)

_HAVE_RIOXARRAY = importlib.util.find_spec("rioxarray") is not None


CombineMode = Literal["auto", "merge", "combine_by_coords"]

def _list_tiled_nc_files(base_path: Path, year: int, variable: str, zoom_level: int) -> list[Path]:
    """List files in the tiled RESKit layout for a given year."""

    base_path = Path(base_path)
    zoom_dir = base_path / str(zoom_level)
    if not zoom_dir.exists():
        return []

    pattern = f"{zoom_dir}/**/**/{year}/*.{variable}.nc"

    return list(dict.fromkeys(sorted(Path(p) for p in glob.glob(pattern, recursive=True))))


def _find_single_year_nc_file(base_path: Path, year: int, variable: str) -> Path:
    """Find exactly one per-year file for a non-tiled layout.

    In the non-tiled case, the expectation is: one NetCDF per year.
    """

    base_path = Path(base_path)
    patterns = [
        f"{base_path}/{year}/*.{variable}.nc",
        f"{base_path}/**/{year}/*.{variable}.nc",
        f"{base_path}/*{year}*.{variable}.nc",
        f"{base_path}/**/*{year}*.{variable}.nc",
    ]

    hits: list[Path] = []
    for pattern in patterns:
        hits = sorted(Path(p) for p in glob.glob(pattern, recursive=True))
        if hits:
            break

    if len(hits) == 1:
        return hits[0]

    if not hits:
        raise FileNotFoundError(
            "No per-year NetCDF found for non-tiled layout. "
            f"year={year}, variable={variable}, base_path={base_path}"
        )

    preview = "\n".join(f"  - {p}" for p in hits[:10])
    raise RuntimeError(
        "Non-tiled layout expects exactly one NetCDF per year, but multiple candidates were found.\n"
        f"year={year}, variable={variable}, base_path={base_path}\n"
        f"First matches:\n{preview}"
    )


def _mean_over_time(ds: xr.Dataset) -> xr.Dataset:
    if "time" not in ds.dims and "time" not in ds.coords:
        return ds
    return ds.mean(dim="time", keep_attrs=True)


def load_era5_year(
    base_path: str | Path,
    year: int,
    variable: str,
    zoom_level: int = 4,
    combine_mode: CombineMode = "auto",
    mean_over_time: bool = True,
) -> xr.Dataset:
    """Load a year of processed ERA5 data.

    Logic:
    - If the input is tiled (zoom directory exists and contains matching tiles): merge tiles.
    - If not tiled: expect exactly one NetCDF for the year; nothing is merged.

    Notes on combining:
    - Some RESKit processing pipelines write tiles with identical variable names,
      but disjoint lat/lon coordinates. In that case, combining by coords is the
      correct operation.
    - If tiles contain distinct data variables (less common), xarray merge is ok.
    """

    base_path = Path(base_path)

    tiled_files = _list_tiled_nc_files(base_path, year, variable, zoom_level)
    # remove duplicates
    if not tiled_files:
        fp = _find_single_year_nc_file(base_path, year, variable)
        with xr.open_dataset(fp) as ds:
            if mean_over_time:
                ds = _mean_over_time(ds)
            return ds.load()

    all_datasets: list[xr.Dataset] = []
    for fp in tqdm(tiled_files, desc=f"Reading tiles {year}"):
        with xr.open_dataset(fp) as ds:
            if mean_over_time:
                ds = _mean_over_time(ds)
            all_datasets.append(ds.load())

    all_datasets = [
        ds.sortby(["latitude", "longitude"])
        for ds in all_datasets
    ]
    ds_merged = xr.merge(all_datasets, compat="no_conflicts")

    return ds_merged


def create_long_run_average(
    base_path: str | Path,
    start_year: int,
    end_year: int,
    variable: str,
    out_dir: str | Path,
    zoom_level: int = 4,
    cache_yearly: bool = True,
    combine_mode: CombineMode = "auto",
    weather_source_prefix: Optional[str] = None,
) -> xr.Dataset:
    """Create a long-run average dataset across years (inclusive)."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    year_range = list(range(start_year, end_year + 1))
    LOG.info("Computing LRA for %s, years=%s", variable, year_range)

    all_years: list[xr.Dataset] = []
    for year in year_range:
        yearly_fp = out_dir / f"{weather_source_prefix}merged_{variable}_{year}.nc"
        if cache_yearly and yearly_fp.exists():
            LOG.info("Using cached yearly merge: %s", yearly_fp)
            ds_year = xr.open_dataset(yearly_fp)
        else:
            ds_year = load_era5_year(
                base_path=base_path,
                year=year,
                variable=variable,
                zoom_level=zoom_level,
                combine_mode=combine_mode,
            )
            if cache_yearly:
                ds_year.to_netcdf(yearly_fp)

        ds_year = ds_year.assign_coords(year=year).expand_dims("year")
        all_years.append(ds_year)

    ds_lra = xr.concat(all_years, dim="year").mean(dim="year", keep_attrs=True)
    
    return ds_lra

def _calculate_DNI(
    solar_elevation_angle: xr.DataArray,
    direct_horizontal_irradiance: xr.DataArray
):
    """Calculate Direct Normal Irradiance (DNI) from Direct Horizontal Irradiance and SZA."""
    # Convert SZA from degrees to radians
    
    # solar_elevation_angle = 90 - solar_zenith_angle
    
    _sea_cleaned = np.where(solar_elevation_angle > 1, solar_elevation_angle, np.nan)

    sin_z = np.sin(np.radians(_sea_cleaned))

    # Calculate DNI using the formula: DNI = DHI / cos(SZA)
    dni = direct_horizontal_irradiance / sin_z
    dni = dni.fillna(0)

    return dni


def calc_solar_elevation_angle(times, lats, lons, temps, pressures):
    """Calculate solar elevation angle using pvlib's spa module.

    Parameters
    ----------
    times : array-like of datetime64, shape (T,)
    lats  : np.ndarray, shape (Y,)
    lons  : np.ndarray, shape (X,)
    temps : np.ndarray, shape (Y, X)  — surface temperature in °C
    pressures : np.ndarray, shape (Y, X)  — surface pressure in Pa

    Returns
    -------
    np.ndarray, shape (T, Y, X)
    """
    from pvlib import spa

    # ERA5 timestamps mark the end of the accumulation period; shift back 30 min
    # to get the representative solar time for each interval.
    unixtime = pd.to_datetime(times).astype(np.int64).values // 10**9
    unixtime = unixtime - 1800

    lat_broad       = lats[:, np.newaxis, np.newaxis]   # (Y, 1, 1)
    lon_broad       = lons[np.newaxis, :, np.newaxis]   # (1, X, 1)
    temp_broad      = temps[:, :, np.newaxis]            # (Y, X, 1)
    pressures_broad = pressures[:, :, np.newaxis]        # (Y, X, 1)

    # spa returns shape (Y, X, T); transpose to (T, Y, X)
    spa_tuple = spa.solar_position_numpy(
        unixtime=unixtime,
        lat=lat_broad,
        lon=lon_broad,
        elev=0,
        pressure=pressures_broad / 100,  # Pa → hPa
        temp=temp_broad,
        delta_t=67.0,
        atmos_refract=0.5667,
        numthreads=1,
        sst=False,
    )
    return spa_tuple[2].transpose(2, 0, 1)

def compute_dni_year(
    base_path: str | Path,
    year: int,
    direct_horizontal_irradiance_variable: str,
    surface_temperature_variable: str,
    surface_pressure_variable: str,
    zoom_level: int = 4,
    combine_mode: CombineMode = "auto",
) -> xr.Dataset:
    """Compute time-averaged DNI for a single year, tile by tile.

    Temperature and pressure are loaded as annual means (memory-efficient).
    Direct horizontal irradiance tiles are streamed one at a time with the full
    time axis so the solar elevation angle can be computed per timestep before
    averaging — loading the whole year at once would exceed memory limits.
    """
    base_path = Path(base_path)

    surface_temperature_year = load_era5_year(
        base_path=base_path,
        year=year,
        variable=surface_temperature_variable,
        zoom_level=zoom_level,
        combine_mode=combine_mode,
        mean_over_time=True,
    )
    data_var_temp = list(surface_temperature_year.data_vars)[0]
    surface_temperature_year = surface_temperature_year[data_var_temp] - 273.15  # K → °C

    surface_pressure_year = load_era5_year(
        base_path=base_path,
        year=year,
        variable=surface_pressure_variable,
        zoom_level=zoom_level,
        combine_mode=combine_mode,
        mean_over_time=True,
    )
    data_var_pres = list(surface_pressure_year.data_vars)[0]
    surface_pressure_year = surface_pressure_year[data_var_pres]

    def _tile_to_dni(
        tile_ds: xr.Dataset,
        temp: xr.DataArray,
        pressure: xr.DataArray,
    ) -> xr.DataArray:
        _temp = temp.sel(latitude=tile_ds["latitude"], longitude=tile_ds["longitude"])
        _pres = pressure.sel(latitude=tile_ds["latitude"], longitude=tile_ds["longitude"])

        sea = calc_solar_elevation_angle(
            times=tile_ds["time"].values,
            lats=tile_ds["latitude"].values,
            lons=tile_ds["longitude"].values,
            temps=_temp.values,
            pressures=_pres.values,
        )
        sea_da = xr.DataArray(
            sea,
            coords={
                "time": tile_ds["time"],
                "latitude": tile_ds["latitude"],
                "longitude": tile_ds["longitude"],
            },
            dims=["time", "latitude", "longitude"],
        )

        data_var = list(tile_ds.data_vars)[0]
        dni = _calculate_DNI(
            solar_elevation_angle=sea_da,
            direct_horizontal_irradiance=tile_ds[data_var],
        )
        dni.attrs.pop("long_name", None)
        return _mean_over_time(dni)

    all_datasets: list[xr.DataArray] = []
    tiled_files = _list_tiled_nc_files(base_path, year, direct_horizontal_irradiance_variable, zoom_level)

    if not tiled_files:
        fp = _find_single_year_nc_file(base_path, year, direct_horizontal_irradiance_variable)
        with xr.open_dataset(fp) as ds:
            all_datasets.append(_tile_to_dni(ds, surface_temperature_year, surface_pressure_year))
    else:
        for fp in tqdm(tiled_files, desc=f"DNI tiles {year}"):
            with xr.open_dataset(fp) as ds:
                all_datasets.append(_tile_to_dni(ds, surface_temperature_year, surface_pressure_year))

    return xr.merge(all_datasets)


def create_long_run_average_DNI(
    base_path: str | Path,
    start_year: int,
    end_year: int,
    variable: str,
    direct_horizontal_irradiance_variable: str,
    surface_temperature_variable: str,
    surface_pressure_variable: str,
    out_dir: str | Path,
    zoom_level: int = 4,
    cache_yearly: bool = True,
    combine_mode: CombineMode = "auto",
    weather_source_prefix: Optional[str] = None,
) -> xr.Dataset:
    """Create a long-run average dataset across years (inclusive)."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    year_range = list(range(start_year, end_year + 1))
    LOG.info("Computing DNI LRA for %s, years=%s", direct_horizontal_irradiance_variable, year_range)

    all_years: list[xr.Dataset] = []
    for year in year_range:
        yearly_fp = out_dir / f"{weather_source_prefix}merged_{variable}_{year}.nc"
        if cache_yearly and yearly_fp.exists():
            LOG.info("Using cached yearly merge: %s", yearly_fp)
            direct_normal_irradiance = xr.open_dataset(yearly_fp)
        else:
            direct_normal_irradiance = compute_dni_year(
                base_path=base_path,
                year=year,
                direct_horizontal_irradiance_variable=direct_horizontal_irradiance_variable,
                surface_temperature_variable=surface_temperature_variable,
                surface_pressure_variable=surface_pressure_variable,
                zoom_level=zoom_level,
                combine_mode=combine_mode,
            )
            if cache_yearly:
                direct_normal_irradiance.to_netcdf(yearly_fp)

        direct_normal_irradiance = direct_normal_irradiance.assign_coords(year=year).expand_dims("year")
        all_years.append(direct_normal_irradiance)

    return xr.concat(all_years, dim="year").mean(dim="year", keep_attrs=True)

def pick_data_var(ds: xr.Dataset, data_var: Optional[str] = None) -> xr.DataArray:
    if data_var is not None:
        if data_var not in ds.data_vars:
            raise KeyError(f"Requested data_var={data_var!r} not found. Available: {list(ds.data_vars)}")
        return ds[data_var]

    if len(ds.data_vars) == 1:
        return ds[next(iter(ds.data_vars))]

    raise ValueError(
        "Dataset contains multiple data variables; specify --data-var. "
        f"Available: {list(ds.data_vars)}"
    )


def write_geotiff_file(
    da: xr.DataArray,
    output_tiff_path: str | Path,
    crs: str = "EPSG:4326",
    sort_lat_ascending: bool = False,
) -> None:
    """Write a DataArray to GeoTIFF using rioxarray.

    Requires optional dependencies: rioxarray, rasterio.
    """

    if not _HAVE_RIOXARRAY:  # pragma: no cover
        raise RuntimeError(
            "GeoTIFF export requires rioxarray (and rasterio). "
            "Install e.g. `pip install rioxarray rasterio`."
        )

    da_out = da
    if sort_lat_ascending:
        if any(coord in da_out.coords for coord in ("lat", "latitude")):
            da_out = da_out.sortby("latitude", ascending=True)
        else:
            raise ValueError("Cannot sort latitude ascending: no 'lat' or 'latitude' coordinate found.")
    else:
        if any(coord in da_out.coords for coord in ("lat", "latitude")):
            da_out = da_out.sortby("latitude", ascending=False)
        else:
            raise ValueError("Cannot sort latitude descending: no 'lat' or 'latitude' coordinate found.")

    da_out = da_out.rio.write_crs(crs)
    output_tiff_path = Path(output_tiff_path)
    output_tiff_path.parent.mkdir(parents=True, exist_ok=True)
    da_out.rio.to_raster(output_tiff_path)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base-path",
        type=Path,
        required=True,
        help=(
            "Base folder containing RESKit-processed ERA5 NetCDFs. "
            "Supports tiled layouts (e.g. .../ERA5_global_processed_V2022.02) and non-tiled per-year layouts."
        ),
    )
    p.add_argument(
        "--variable",
        type=str,
        required=True,
        help="Variable folder/identifier in the filenames, e.g. 100m_wind_speed.processed",
    )
    p.add_argument("--start-year", type=int, required=True)
    p.add_argument("--end-year", type=int, required=True)
    p.add_argument("--zoom-level", type=int, default=4)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output"),
        help="Output directory (will create subfolders)",
    )
    p.add_argument(
        "--no-cache-yearly",
        action="store_false",
        dest="cache_yearly",
        help="Disable caching per-year merged NetCDFs (default: caching is enabled)",
    )
    p.add_argument(
        "--combine-mode",
        choices=("auto", "merge", "combine_by_coords"),
        default="auto",
        help="How to combine tile datasets",
    )
    p.add_argument(
        "--data-var",
        type=str,
        default=None,
        help="If output dataset contains multiple variables, pick which one to export as GeoTIFF",
    )
    p.add_argument(
        "--write-geotiff",
        action="store_true",
        dest="write_geotiff",
        help="Also write GeoTIFF of the selected data variable (requires rioxarray+rasterio)",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    p.add_argument(
        "--temp-dir",
        type=Path,
        default=None,
        help="If set, use this temporary directory for intermediate files instead of out-dir/yearly/variable",
    )
    p.add_argument(
        "--weather-source-prefix",
        type=str,
        default=None,
        help="If set, prepend this prefix to the output file names",
    )
    return p


def create_LRA(
    *,
    base_path: str | Path,
    variable: str,
    start_year: int,
    end_year: int,
    zoom_level: int = 4,
    out_dir: str | Path = Path("output"),
    cache_yearly: bool = True,
    combine_mode: CombineMode = "auto",
    data_var: Optional[str] = None,
    variable_name_output: Optional[str] = None,
    write_geotiff: bool = True,
    write_netcdf: bool = False,
    log_level: str = "INFO",
    temp_dir: Optional[str | Path] = None,
    weather_source_prefix: Optional[str] = None,
) -> xr.Dataset:
    """Compute and write a long-run average (LRA) for a RESKit-processed weather variable.

    This is the programmatic entry point equivalent to the CLI in this module.
    It computes an annual-mean value per year (by averaging over the ``time``
    dimension if present), then averages these annual means across
    ``start_year..end_year``.

    Input data can be stored in two layouts:
    - **Tiled layout** (RESKit default):
      ``<base_path>/<zoom_level>/<x-tile>/<y-tile>/<year>/*.{variable}.nc``
      For each year, all matching tile files are loaded, time-averaged, and
      merged into a global dataset.
    - **Non-tiled layout** (one file per year):
      One matching NetCDF must exist for each year (e.g.
      ``<base_path>/<year>/*.{variable}.nc`` or similar). In this case nothing
      is merged; the single file is loaded and time-averaged.

    Intermediate yearly results:
    - If ``cache_yearly=True`` (default), per-year results are written to
      ``<out_dir>/yearly/<variable>/`` unless ``temp_dir`` is provided.
    - If ``temp_dir`` is provided, intermediates are written to
      ``<temp_dir>/<variable>/`` instead.

    Final outputs (always written under ``out_dir``):
    - NetCDF: ``<out_dir>/LRA/<prefix>long_run_avg_<variable>_<start>_<end>.nc``
    - Optional GeoTIFF (if ``write_geotiff=True``): same naming with ``.tif``.

    Parameters
    ----------
    base_path:
        Root directory containing the processed NetCDF files.
    variable:
        Variable identifier used in filenames (expects ``*.{variable}.nc``).
    start_year, end_year:
        Inclusive year range used for the long-run average.
    zoom_level:
        Zoom level used for tiled layouts. Ignored for non-tiled layouts.
    out_dir:
        Directory where final outputs are written and (by default) where yearly
        intermediate files are cached.
    cache_yearly:
        If True, cache per-year merged/loaded datasets to NetCDF.
    combine_mode:
        How to combine tiled datasets. Use ``"auto"`` (default) unless you have
        a specific reason.
    data_var:
        If the resulting LRA dataset contains multiple data variables, specify
        which one to export when writing a GeoTIFF.
    variable_name_output:
        If provided, use this name for the variable in the output filenames
        instead of the input ``variable``.
    write_geotiff:
        If True, also export a GeoTIFF (requires optional dependencies
        ``rioxarray`` + ``rasterio``).
    write_netcdf:
        If True, write the NetCDF output (default: always False).
    log_level:
        Logging verbosity (``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``).
    temp_dir:
        Optional directory to store intermediate yearly files.
    weather_source_prefix:
        Optional prefix prepended to output filenames. If provided,
        ``"<weather_source_prefix>_"`` is used.

    Returns
    -------
    xarray.Dataset
        The computed long-run average dataset.
    """

    logging.basicConfig(level=getattr(logging, log_level), format="%(levelname)s %(message)s")

    out_dir = Path(out_dir)

    if not temp_dir:
        var_out_dir = out_dir / "yearly" / variable
        var_out_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = Path(temp_dir)
        LOG.info("Using temporary directory for intermediate files: %s", temp_dir)
        var_out_dir = temp_dir / variable

    if weather_source_prefix is None:
        weather_source_prefix = ""
    else:
        weather_source_prefix = f"{weather_source_prefix}_"

    ds_lra = create_long_run_average(
        base_path=base_path,
        start_year=start_year,
        end_year=end_year,
        variable=variable,
        out_dir=var_out_dir,
        zoom_level=zoom_level,
        cache_yearly=cache_yearly,
        combine_mode=combine_mode,
        weather_source_prefix=weather_source_prefix,
    )
    if variable_name_output is not None:
        variable = variable_name_output
        
    output_file_name = f"{weather_source_prefix}{variable}_{start_year}_{end_year}_mean"

    if write_netcdf:
        lra_nc = out_dir / f"{output_file_name}.nc"
        lra_nc.parent.mkdir(parents=True, exist_ok=True)
        ds_lra.to_netcdf(lra_nc)
        LOG.info("Wrote LRA NetCDF: %s", lra_nc)

    if write_geotiff:
        da = pick_data_var(ds_lra, data_var)
        tiff_fp = out_dir / f"{output_file_name}.tiff"
        write_geotiff_file(da, tiff_fp)
        LOG.info("Wrote LRA GeoTIFF: %s", tiff_fp)

    return ds_lra


def create_DNI_LRA(
    *,
    base_path: str | Path,
    variable: str,
    direct_horizontal_irradiance_variable: str,
    surface_temperature_variable: str,
    surface_pressure_variable: str,
    start_year: int,
    end_year: int,
    zoom_level: int = 4,
    out_dir: str | Path = Path("output"),
    cache_yearly: bool = True,
    combine_mode: CombineMode = "auto",
    variable_name_output: Optional[str] = None,
    write_geotiff: bool = True,
    write_netcdf: bool = False,
    log_level: str = "INFO",
    temp_dir: Optional[str | Path] = None,
    weather_source_prefix: Optional[str] = None,
    ):
    """Compute and write a long-run average (LRA) for Direct Normal Irradiance (DNI) using existing files for GHI and DHI.
    
    """
    logging.basicConfig(level=getattr(logging, log_level), format="%(levelname)s %(message)s")
    
    out_dir = Path(out_dir)

    if not temp_dir:
        var_out_dir = out_dir / "yearly" / variable
        var_out_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = Path(temp_dir)
        LOG.info("Using temporary directory for intermediate files: %s", temp_dir)
        var_out_dir = temp_dir / variable
        
    if weather_source_prefix is None:
        weather_source_prefix = ""
    else:
        weather_source_prefix = f"{weather_source_prefix}_"
        
    dni_lra = create_long_run_average_DNI(
        base_path=base_path,
        start_year=start_year,
        end_year=end_year,
        variable=variable,
        direct_horizontal_irradiance_variable=direct_horizontal_irradiance_variable,
        surface_temperature_variable=surface_temperature_variable,
        surface_pressure_variable=surface_pressure_variable,
        out_dir=var_out_dir,
        zoom_level=zoom_level,
        cache_yearly=cache_yearly,
        combine_mode=combine_mode,
        weather_source_prefix=weather_source_prefix,
    )
        
    output_file_name = f"{weather_source_prefix}{variable_name_output}_{start_year}_{end_year}_mean"
    if write_netcdf:
        lra_nc = out_dir / f"{output_file_name}.nc"
        lra_nc.parent.mkdir(parents=True, exist_ok=True)
        dni_lra.to_netcdf(lra_nc)
        LOG.info("Wrote DNI LRA NetCDF: %s", lra_nc)
    if write_geotiff:
        tiff_fp = out_dir / f"{output_file_name}.tiff"
        write_geotiff_file(dni_lra, tiff_fp)
        LOG.info("Wrote DNI LRA GeoTIFF: %s", tiff_fp)
    return dni_lra


# main idea: generate a 3x3 array of the raster data 
# interpolate between the outmost data cell and the first data cell "on the othe side of the world"
# "polar wrap" the top and bottom rows (mirror and center on antimeridian) so that interpolation can cross the poles correctly
# the interpolate missing data in the 3x3 array
# last clip to the bounds of interest

def world_3x3_wrap(arr_center: np.ndarray, rInfo : object):
    """
    Build a 3x3 tiled array with correct pole-wrap and (optionally) return mosaic bounds.
    All maps are arranged such that they align with the antimeridian as center column,
    slight mismatches by resolution/bounds shift are accounted for.
    Missing cells to the +/-180°  longitude and +/-90° latitude bounds are set to NaN.
    Center map cells overwride other data as long as not NaN.

    Inputs
    ------
    arr_center : np.ndarray
        array of raster data for the center tile.
    rInfo : object
        Raster info object with attributes:
        - pixelWidth : float
        - pixelHeight : float
        - bounds : tuple of (xmin, ymin, xmax, ymax)

    Returns
    -------
    arr_3x3 : (3H, 3W) ndarray
    bounds_3x3 : (xmin, ymin, xmax, ymax) 
        Outer-edge bounds of the returned 3x3 array, consistent with the padded grid
        (including any extra/excess cells induced by the shift/resolution).
    """
    xmin, ymin, xmax, ymax = rInfo.bounds
    nrows_c, ncols_c = arr_center.shape

    # Pixel-center coordinates for arr_center (assuming bounds are OUTER EDGES)
    x0 = xmin + rInfo.pixelWidth / 2.0
    y0 = ymax - rInfo.pixelHeight / 2.0
    dy = -rInfo.pixelHeight  # north-up (row index increases southward)

    # Choose nearest achievable world pixel-center indices to +/-180 and +/-90
    i_left  = int(np.rint((-180.0 - x0) / rInfo.pixelWidth))
    i_right = int(np.rint(( 180.0 - x0) / rInfo.pixelWidth))
    if i_right < i_left:
        i_left, i_right = i_right, i_left

    j_top    = int(np.rint(( 90.0 - y0) / dy))
    j_bottom = int(np.rint((-90.0 - y0) / dy))
    if j_bottom < j_top:
        j_top, j_bottom = j_bottom, j_top

    world_cols = i_right - i_left + 1
    world_rows = j_bottom - j_top + 1

    # Pixel-center coordinate of world tile top-left cell
    x_world0 = x0 + i_left * rInfo.pixelWidth
    y_world0 = y0 + j_top * dy

    # Embed center into padded world tile
    padded_center = np.full((world_rows, world_cols), np.nan, dtype=arr_center.dtype)

    col0 = int(np.rint((x0 - x_world0) / rInfo.pixelWidth))
    row0 = int(np.rint((y0 - y_world0) / dy))

    r1, c1 = row0, col0
    r2, c2 = r1 + nrows_c, c1 + ncols_c

    rr1, cc1 = max(r1, 0), max(c1, 0)
    rr2, cc2 = min(r2, world_rows), min(c2, world_cols)

    if rr1 < rr2 and cc1 < cc2:
        src_r1 = rr1 - r1
        src_c1 = cc1 - c1
        src_r2 = src_r1 + (rr2 - rr1)
        src_c2 = src_c1 + (cc2 - cc1)
        padded_center[rr1:rr2, cc1:cc2] = arr_center[src_r1:src_r2, src_c1:src_c2]

    H, W = padded_center.shape
    half_shift = W // 2  # discrete "180°" in index space for this padded tile

    # Correct pole-wrap tile: lat reflection + lon reversal + antimeridian shift
    pole_tile = np.roll(np.fliplr(np.flipud(padded_center)), half_shift, axis=1)

    # Compose 3x3 mosaic
    out = np.full((3 * H, 3 * W), np.nan, dtype=arr_center.dtype)

    def write_tile(dst, tile, top, left):
        sub = dst[top:top + H, left:left + W]
        m = ~np.isnan(tile)
        sub[m] = tile[m]

    # Non-center tiles first
    write_tile(out, padded_center, H, 0)       # middle-left
    write_tile(out, padded_center, H, 2 * W)   # middle-right

    write_tile(out, pole_tile, 0, 0)           # top-left
    write_tile(out, pole_tile, 0, W)           # top-center
    write_tile(out, pole_tile, 0, 2 * W)       # top-right

    write_tile(out, pole_tile, 2 * H, 0)       # bottom-left
    write_tile(out, pole_tile, 2 * H, W)       # bottom-center
    write_tile(out, pole_tile, 2 * H, 2 * W)   # bottom-right

    # Center last so it wins where both have data
    write_tile(out, padded_center, H, W)

    # ---- Bounds for the 3x3 mosaic (outer edges) ----
    # Bounds for single padded world tile (padded_center)
    tile_xmin = x_world0 - rInfo.pixelWidth / 2.0
    tile_xmax = x_world0 + (W - 1) * rInfo.pixelWidth + rInfo.pixelWidth / 2.0
    tile_ymax = y_world0 + rInfo.pixelHeight / 2.0
    tile_ymin = y_world0 + (H - 1) * dy - rInfo.pixelHeight / 2.0  # dy < 0

    # Each additional tile shifts by exactly W columns and H rows in index space
    tile_width_deg  = W * rInfo.pixelWidth
    tile_height_deg = H * rInfo.pixelHeight

    mosaic_xmin = tile_xmin - tile_width_deg
    mosaic_xmax = tile_xmax + tile_width_deg
    mosaic_ymax = tile_ymax + tile_height_deg
    mosaic_ymin = tile_ymin - tile_height_deg

    bounds_3x3 = (mosaic_xmin, mosaic_ymin, mosaic_xmax, mosaic_ymax)
    return out, bounds_3x3


def interp_vertical_1d(arr3x3: np.ndarray, *, max_gap: int | None = None) -> np.ndarray:
    """
    Fill NaNs by 1D linear interpolation along the vertical axis for each column.

    Rules
    -----
    - Only fills NaNs that lie strictly between two valid samples in the same column.
    - Leaves leading/trailing NaNs untouched (not bracketed).
    - Optionally limits filling to gaps of length <= max_gap.

    Parameters
    ----------
    arr3x3 : 2D ndarray
    max_gap : int or None
        If set, only fill NaN runs up to this length (in rows). Larger gaps remain NaN.

    Returns
    -------
    out : 2D ndarray
    """
    if arr3x3.ndim != 2:
        raise ValueError("arr3x3 must be 2D")

    out = arr3x3.astype(float, copy=True)
    nrows, ncols = out.shape
    x = np.arange(nrows)

    for c in range(ncols):
        col = out[:, c]
        nan = np.isnan(col)
        if not nan.any():
            continue

        valid_idx = np.flatnonzero(~nan)
        if valid_idx.size < 2:
            continue  # cannot bracket anything

        # Candidate fill indices: NaNs between first and last valid
        fill_idx = np.flatnonzero(nan & (x > valid_idx[0]) & (x < valid_idx[-1]))
        if fill_idx.size == 0:
            continue

        # If max_gap is set, exclude fill indices that belong to too-long NaN runs
        if max_gap is not None:
            # Find NaN runs in this column
            nan_idx = np.flatnonzero(nan)
            # Split into contiguous runs
            splits = np.where(np.diff(nan_idx) != 1)[0] + 1
            runs = np.split(nan_idx, splits)
            allowed = np.zeros_like(nan, dtype=bool)
            for run in runs:
                if run.size <= max_gap:
                    allowed[run] = True
            fill_idx = fill_idx[allowed[fill_idx]]
            if fill_idx.size == 0:
                continue

        out[fill_idx, c] = np.interp(fill_idx, valid_idx, col[valid_idx])

    return out


def _snap_to_grid(value, origin, res, mode="round"):
    """
    Snap 'value' to the grid defined by origin + k*res.
    mode: 'round'|'floor'|'ceil'
    """
    k = (value - origin) / res
    if mode == "round":
        kk = np.round(k)
    elif mode == "floor":
        kk = np.floor(k)
    elif mode == "ceil":
        kk = np.ceil(k)
    else:
        raise ValueError("mode must be one of: round, floor, ceil")
    return origin + kk * res


def extract_bbox_from_mosaic(mosaic, *, bounds_3x3, pixel_width, pixel_height, bbox, snap_edges=True):
    """
    Cut out a bbox (xmin,ymin,xmax,ymax) from the mosaic.

    Assumptions:
      - north-up grid: row 0 is at ymax, increasing rows go south
      - pixel_height is positive

    If snap_edges=True, bbox edges are snapped to the pixel grid of the mosaic.
    """
    mxmin, mymin, mxmax, mymax = bounds_3x3
    bxmin, bymin, bxmax, bymax = bbox

    if snap_edges:
        # snap x edges to grid anchored at mxmin
        bxmin = _snap_to_grid(bxmin, mxmin, pixel_width, mode="round")
        bxmax = _snap_to_grid(bxmax, mxmin, pixel_width, mode="round")
        # snap y edges to grid anchored at mymax (top edge), moving down by pixel_height
        # so y = mymax - r*pixel_height  => r = (mymax - y)/pixel_height
        bymin = _snap_to_grid(bymin, mymax, -pixel_height, mode="round")  # negative step in y
        bymax = _snap_to_grid(bymax, mymax, -pixel_height, mode="round")

    # Ensure proper ordering
    if bxmax <= bxmin or bymax <= bymin:
        raise ValueError("Invalid bbox after snapping: must satisfy xmax>xmin and ymax>ymin")

    # Convert bbox to pixel indices in mosaic
    # cols: x = mxmin + c*pw at left edge of pixel c
    c0 = int(round((bxmin - mxmin) / pixel_width))
    c1 = int(round((bxmax - mxmin) / pixel_width))

    # rows: y = mymax - r*ph at top edge of pixel r
    r0 = int(round((mymax - bymax) / pixel_height))  # bymax is the top of requested bbox
    r1 = int(round((mymax - bymin) / pixel_height))  # bymin is the bottom

    # Slice (note: Python slicing excludes end)
    cut = mosaic[r0:r1, c0:c1]

    cut_bounds = (bxmin, bymin, bxmax, bymax)
    return cut, cut_bounds

def expand_to_global_coverage(
        rstr : str, 
        target_bounds : tuple, 
        as_array: bool = True,
        output_path: str | None = None,
        ) -> np.ndarray:
    """
    Expands a near global raster to full global coverage by interpolating edge 
    cells across the antimeridian/poles

    Parameters
    ----------
    rstr : str
        Path to the raster file that shall be expanded.
    target_bounds : tuple
        (xmin, ymin, xmax, ymax) bounds of the output raster. Must align with 
        the input bounds plus/minus an integer number of pixels.
    as_array : bool, optional
        If True, the data will be returned as array, if False, a raster will be 
        written to disk, then the output_path must be given, by default True.
    output_path : str | None, optional
        Path to write the output raster if as_array is False, by default None.

    Returns
    -------
    np.ndarray
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    # first get the numpy data array and the raster info
    arr = gk.raster.extractMatrix(rstr)
    rInfo = gk.raster.rasterInfo(rstr)
    if not (rInfo.bounds[0] <= -180.0 and rInfo.bounds[2] >= 180.0):
        # if this is needed, the interpolation function needs to be expanded
        raise ValueError(f"Current version of this function allows only latitudinal expansion, the input raster does not extend over the full longitude range. Raster bounds are: {rInfo.bounds}")

    # then arrange it into a special 3x3 array that wraps around the globe correctly
    arr3x3, bounds3x3 = world_3x3_wrap(rInfo=rInfo, arr_center=arr)
    
    # interpolate missing data in the 3x3 array
    arr3x3_interp = interp_vertical_1d(arr3x3=arr3x3, max_gap=None)

    # now clip it to the bounds of interest, enforce matching the pixel grid
    arr_out, bounds_out = extract_bbox_from_mosaic(
        arr3x3_interp,
        bounds_3x3=bounds3x3,
        pixel_width=rInfo.pixelWidth,
        pixel_height=rInfo.pixelHeight,
        bbox=target_bounds,
        snap_edges=True, # snap bounding box edges to pixel grid
    )
    assert bounds_out == target_bounds # sanity check

    if as_array:
        return arr_out  
    else:
        # write to disk
        output_path = gk.raster.createRaster(
            data=arr_out,
            bounds=target_bounds,
            pixelWidth=rInfo.pixelWidth,
            pixelHeight=rInfo.pixelHeight,
            output=output_path,
            srs=rInfo.srs,
            dtype=rInfo.dtype,
        )
        return output_path



def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)

    create_LRA(
        base_path=args.base_path,
        variable=args.variable,
        start_year=args.start_year,
        end_year=args.end_year,
        zoom_level=args.zoom_level,
        out_dir=args.out_dir,
        cache_yearly=args.cache_yearly,
        combine_mode=args.combine_mode,
        data_var=args.data_var,
        write_geotiff=args.write_geotiff,
        log_level=args.log_level,
        temp_dir=args.temp_dir,
        weather_source_prefix=args.weather_source_prefix,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

