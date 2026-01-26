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
import pvlib

LOG = logging.getLogger(__name__)

_HAVE_RIOXARRAY = importlib.util.find_spec("rioxarray") is not None


CombineMode = Literal["auto", "merge", "combine_by_coords"]

def getSZA_grid(
    latitude: np.array, 
    longitude: np.array,
    time_index: pd.DatetimeIndex,
    utc_offset: float = 0
):
    """
    Calculates SZA for a GRID of latitudes and longitudes over time.
    
    Inputs:
        latitude:   1D Array of shape (Y,)
        longitude:  1D Array of shape (X,)
        time_index: pd.DatetimeIndex of length T
        
    Returns:
        np.array of shape (T, Y, X) -> (Time, Latitude, Longitude)
    """
    # ---------------------------------------------------------
    # 1. Reshape Inputs for 3D Broadcasting (Time, Lat, Lon)
    # ---------------------------------------------------------
    
    # Time (Axis 0): Shape (T, 1, 1)
    # Allows broadcasting across both Lat and Lon dimensions
    doy = time_index.dayofyear.values.reshape(-1, 1, 1)
    h = time_index.hour.values.reshape(-1, 1, 1)
    m = time_index.minute.values.reshape(-1, 1, 1)
    
    # Latitude (Axis 1): Shape (1, Y, 1)
    # Broadcasts across Time and Lon
    lat_rad = np.radians(np.array(latitude)).reshape(1, -1, 1)
    
    # Longitude (Axis 2): Shape (1, 1, X)
    # Broadcasts across Time and Lat
    lon = np.array(longitude).reshape(1, 1, -1)

    # ---------------------------------------------------------
    # 2. Time-dependent calculations (Result shape: T, 1, 1)
    # ---------------------------------------------------------
    
    hour_minute = (h + m / 60.0) - utc_offset
    
    g = (360 / 365.25) * (doy + hour_minute / 24.0)
    g_rad = np.radians(g)

    # Solar Declination
    declination = (0.396372 
                   - 22.91327 * np.cos(g_rad) 
                   + 4.02543 * np.sin(g_rad) 
                   - 0.387205 * np.cos(2 * g_rad) 
                   + 0.051967 * np.sin(2 * g_rad) 
                   - 0.154527 * np.cos(3 * g_rad) 
                   + 0.084798 * np.sin(3 * g_rad))
    
    d_rad = np.radians(declination)

    # Equation of Time
    time_correction = (0.004297 
                       + 0.107029 * np.cos(g_rad) 
                       - 1.837877 * np.sin(g_rad) 
                       - 0.837378 * np.cos(2 * g_rad) 
                       - 2.340475 * np.sin(2 * g_rad))

    # ---------------------------------------------------------
    # 3. 3D Space-Time Calculation
    # ---------------------------------------------------------
    
    # SHA Calculation
    # (T,1,1) + (1,1,X) + (T,1,1) -> Result Shape (T, 1, X)
    SHA = (hour_minute - 12) * 15 + lon + time_correction
    SHA_rad = np.radians(SHA)

    # Cosine SZA Calculation
    # Term 1: sin(lat)*sin(d) 
    #         Shape: (1, Y, 1) * (T, 1, 1) -> (T, Y, 1)
    # Term 2: cos(lat)*cos(d)*cos(SHA)
    #         Shape: (1, Y, 1) * (T, 1, 1) * (T, 1, X) -> (T, Y, X)
    
    term1 = np.sin(lat_rad) * np.sin(d_rad)
    term2 = np.cos(lat_rad) * np.cos(d_rad) * np.cos(SHA_rad)
    
    # Adding (T, Y, 1) + (T, Y, X) broadcasts the 1 to X -> (T, Y, X)
    cos_sza = term1 + term2

    # Clip for arccos safety
    cos_sza = np.clip(cos_sza, -1.0, 1.0)

    SZA = np.degrees(np.arccos(cos_sza))

    return SZA

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
    Inputs:
    times: pd.DatetimeIndex of shape (T,)
    lats: np.array of shape (Y,)
    lons: np.array of shape (X,)
    temps: temperatures at lats/lons 
    Returns:
    np.array of shape (T, Y, X) -> (Time, Latitude, Longitude)
    """
    from pvlib import spa
    import time
    
    unixtime = pd.to_datetime(times).astype(np.int64).values // 10**9 # Shape (T,)
    unixtime = unixtime - 1800  # Subtract 1800 seconds (30 mins)
    
    lat_broad = lats[:, np.newaxis, np.newaxis]   # Shape (Y, 1, 1)
    lon_broad = lons[np.newaxis, :, np.newaxis]   # Shape (1, X, 1)

    temp_broad = temps[:, :, np.newaxis]          # Shape (Y, X, 1)
    pressures_broad = pressures[:, :, np.newaxis]  # Shape (Y, X, 1)
    
    # DEBUG: Verify shapes before running
    print(f"Time Shape: {unixtime.shape}")
    print(f"Lat  Shape: {lat_broad.shape}")
    print(f"Lon  Shape: {lon_broad.shape}")

    # --- 3. Run Calculation ---
    # This results in a shape of (Y, X, T)
    # because numpy aligns the (1)s on the right with (T)
    start = time.time()
    spa_tuple = spa.solar_position_numpy(
        unixtime=unixtime,
        lat=lat_broad,
        lon=lon_broad,
        elev=0,
        pressure=pressures_broad/100,
        temp=temp_broad,
        delta_t=67.0,
        atmos_refract=0.5667,
        numthreads=1,
        sst=False
    )

    elevation_angle = spa_tuple[2].transpose(2, 0, 1)
    print("Solar elevation angle calculation time:", time.time() - start)
    print(f"---"*20)
    
    return elevation_angle

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
    LOG.info("Computing LRA for %s, years=%s", direct_horizontal_irradiance_variable, year_range)

    all_years: list[xr.Dataset] = []
    for year in year_range:
        yearly_fp = out_dir / f"{weather_source_prefix}merged_{variable}_{year}.nc"
        if cache_yearly and yearly_fp.exists():
            LOG.info("Using cached yearly merge: %s", yearly_fp)
            direct_normal_irradiance = xr.open_dataset(yearly_fp)    

        else:   
            # NOTE: this does not work due to memory issues
            # direct_horizontal_irradiance_year = load_era5_year(
            #     base_path=base_path,
            #     year=year,
            #     variable=direct_horizontal_irradiance_variable,
            #     zoom_level=zoom_level,
            #     combine_mode=combine_mode,
            #     mean_over_time=False
            # )
            
            surface_temperature_year = load_era5_year(
                base_path=base_path,
                year=year,
                variable=surface_temperature_variable,
                zoom_level=zoom_level,
                combine_mode=combine_mode,
                mean_over_time=True)
            data_var_temp = list(surface_temperature_year.data_vars)[0]
            surface_temperature_year = surface_temperature_year[data_var_temp]
            surface_temperature_year = surface_temperature_year - 273.15 # convert to Degree Celsius
            
            surface_pressure_year = load_era5_year(
                base_path=base_path,
                year=year,
                variable=surface_pressure_variable,
                zoom_level=zoom_level,
                combine_mode=combine_mode,
                mean_over_time=True)
            data_var_pres = list(surface_pressure_year.data_vars)[0]
            surface_pressure_year = surface_pressure_year[data_var_pres]
                
            
            
            def _DNI_ds_calculator(
                direct_horizontal_irradiance_tile: xr.Dataset,
                surface_temperature_year: xr.DataArray,
                surface_pressure_year: xr.DataArray,
            ) -> xr.DataArray:
                
                solar_elevation_angle = calc_solar_elevation_angle(
                    times=direct_horizontal_irradiance_tile['time'].values,
                    lats=direct_horizontal_irradiance_tile['latitude'].values,
                    lons=direct_horizontal_irradiance_tile['longitude'].values,
                    temps=surface_temperature_year.values,
                    pressures=surface_pressure_year.values
                    
                )
                    
                solar_elevation_angle_ds = xr.DataArray(
                    solar_elevation_angle,
                    coords={
                        "time": direct_horizontal_irradiance_tile['time'],
                        "latitude": direct_horizontal_irradiance_tile['latitude'],
                        "longitude": direct_horizontal_irradiance_tile['longitude'],
                    },
                    dims=["time", "latitude", "longitude"]
                )
                
                data_var = list(direct_horizontal_irradiance_tile.data_vars)[0]
                
                direct_normal_irradiance = _calculate_DNI(
                    solar_elevation_angle=solar_elevation_angle_ds,
                    direct_horizontal_irradiance=direct_horizontal_irradiance_tile[data_var]
                )
                
                # drop long_name from attributes if exists
                if 'long_name' in direct_normal_irradiance.attrs:
                    direct_normal_irradiance.attrs.pop('long_name')
                
                direct_normal_irradiance = _mean_over_time(direct_normal_irradiance)
            
                return direct_normal_irradiance
            
            all_datasets: list[xr.Dataset] = []
            tiled_files = _list_tiled_nc_files(base_path, year, direct_horizontal_irradiance_variable, zoom_level)
            # remove duplicates
            if not tiled_files:
                fp = _find_single_year_nc_file(base_path, year, direct_horizontal_irradiance_variable)
                with xr.open_dataset(fp) as direct_horizontal_irradiance:
                    # clip surface_temperature_year to lat/lon bounds of direct_horizontal_irradiance
                    _surface_temperature_year = surface_temperature_year.sel(
                        latitude=direct_horizontal_irradiance["latitude"],
                        longitude=direct_horizontal_irradiance["longitude"]
                    )
                    _surface_pressure_year = surface_pressure_year.sel(
                        latitude=direct_horizontal_irradiance["latitude"],
                        longitude=direct_horizontal_irradiance["longitude"]
                    )
                    
                    all_datasets.append(_DNI_ds_calculator(direct_horizontal_irradiance, _surface_temperature_year, _surface_pressure_year))
                
            else:
                for fp in tqdm(tiled_files, desc=f"Reading tiles {year}"):
                    with xr.open_dataset(fp) as direct_horizontal_irradiance:
                        
                        
                        # clip surface_temperature_year to lat/lon bounds of direct_horizontal_irradiance
                        _surface_temperature_year = surface_temperature_year.sel(
                            latitude=direct_horizontal_irradiance["latitude"],
                            longitude=direct_horizontal_irradiance["longitude"]
                        )
                        _surface_pressure_year = surface_pressure_year.sel(
                            latitude=direct_horizontal_irradiance["latitude"],
                            longitude=direct_horizontal_irradiance["longitude"]
                        )

                            
                        all_datasets.append(_DNI_ds_calculator(direct_horizontal_irradiance, _surface_temperature_year, _surface_pressure_year))


            direct_normal_irradiance = xr.merge(all_datasets)#, compat="override")

            # From this, calculate Direct Normal Irradiance (DNI)
            # solar_zenith_angle = getSZA_grid(
            #     latitude=direct_horizontal_irradiance_year['latitude'].values,
            #     longitude=direct_horizontal_irradiance_year['longitude'].values,
            #     time_index=direct_horizontal_irradiance_year['time'].to_index(),
            #     utc_offset=0
            # )
                        
            if cache_yearly:
                direct_normal_irradiance.to_netcdf(yearly_fp)
        
        direct_normal_irradiance = direct_normal_irradiance.assign_coords(year=year).expand_dims("year")
        
        all_years.append(direct_normal_irradiance)

    DNI_LRA = xr.concat(all_years, dim="year").mean(dim="year", keep_attrs=True)
    
    return DNI_LRA

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
    data_var: Optional[str] = None,
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

