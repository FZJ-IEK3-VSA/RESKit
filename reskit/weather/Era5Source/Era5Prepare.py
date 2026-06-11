import os
import tempfile
import cdsapi
import geokit as gk
import netCDF4 as nc4
from cdo import Cdo
import pandas as pd
from typing import Union, List, Optional
from reskit.util.weather_tile import get_tile_XY


def _nc_file_has_vars(cdo, nc_path, required_vars):
    """Check whether a NetCDF file exists and contains required variables."""
    if not os.path.exists(nc_path):
        return False
    try:
        vars_in_file = set(cdo.showname(input=nc_path)[0].split())
        return set(required_vars) <= vars_in_file
    except Exception:
        return False


"""
Running this module will automatically download ERA5 data from CDS for the specified date range and boundary box,
and preprocess the data to adjust solar radiation variables and compute wind speed variables.

To make it happen, you need to have:
1. An account at the Copernicus Climate Data Store (CDS) and have your API key set up in the ~/.cdsapirc file. See here for more details: https://cds.climate.copernicus.eu/how-to-api
2. The CDO (Climate Data Operators) installed on your system. Install cdo and python-cdo from channel conda-forge will do the job, for example: conda install -c conda-forge cdo python-cdo
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
    cdo: Cdo,
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

    def _subset_input(lon_west: float, lon_east: float) -> str:
        return (
            f"-selname,{var} -selyear,{year} -sellonlatbox,{lon_west},{lon_east},{lat_south},{lat_north} {source_file}"
        )

    if len(lon_boxes) == 1:
        lon_west, lon_east = lon_boxes[0]
        cdo.copy(input=_subset_input(lon_west, lon_east), output=target_file)
        _align_longitudes_to_source_convention(target_file, source_lon_min, source_lon_max)
        return

    with tempfile.TemporaryDirectory(dir=os.path.dirname(target_file)) as tmp_dir:
        tmp_files = []
        for idx, (lon_west, lon_east) in enumerate(lon_boxes):
            tmp_file = os.path.join(tmp_dir, f"tile_part_{idx}.nc")
            cdo.copy(input=_subset_input(lon_west, lon_east), output=tmp_file)
            tmp_files.append(tmp_file)
        cdo.mergegrid(input=" ".join(tmp_files), output=target_file)
    _align_longitudes_to_source_convention(target_file, source_lon_min, source_lon_max)


def era5_downloader(
    target_filename: str,
    year: Union[str, List[str]],
    month: Union[str, List[str]],
    variables: list[str],
    area: tuple[float, float, float, float],
    grid: tuple[float, float] = (0.25, 0.25),
    data_format: str = "netcdf_legacy",
):
    client = cdsapi.Client()

    request = {
        "product_type": "reanalysis",
        "format": data_format,
        "variable": variables,
        "year": year,
        "month": month,
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
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
    # prepare cdo instance
    cdo = Cdo()

    # detect variables in the nc file
    varnames = cdo.showname(input=focus_nc)[0].split()
    varset = set(varnames)

    out_dir = processed_dir or os.path.dirname(focus_nc)
    if processed_dir:
        os.makedirs(processed_dir, exist_ok=True)
    f_name = os.path.basename(focus_nc)
    # process for solar radiation variables
    # solar_out = os.path.join(dir, f"{f_name.split('.')[0]}_processed_solar.nc")
    # if {"ssrd", "fdir"} & varset:
    #     if not nc_file_has_vars(cdo, solar_out, ["ssrd", "fdir"]):
    #         unit = "W m**-2"
    #         cdo.copy(
    #             input=(
    #                 f"-setattribute,ssrd@units=\"{unit}\" "
    #                 f"-setattribute,fdir@units=\"{unit}\" "
    #                 f"-divc,3600 "
    #                 f"-selname,ssrd,fdir "
    #                 f"{focus_nc}"
    #             ),
    #             output=solar_out,
    #         )
    #     else:
    #         print(f"Skipping solar preprocessing (exists): {solar_out}")

    # process for solar radiation variables (time adjusted)
    # Build the variable list dynamically from whichever of ssrd/fdir is present so that
    # workflows requesting only one of them (e.g. CSP needs fdir but not ssrd) still work.
    solar_t_out = os.path.join(out_dir, f"{f_name.split('.')[0]}_processed_solar_t_adjusted.nc")
    solar_vars = [v for v in ("ssrd", "fdir") if v in varset]
    if solar_vars:
        out_names = [f"{v}_t_adj" for v in solar_vars]
        if not _nc_file_has_vars(cdo, solar_t_out, out_names):
            unit = "W m**-2"
            rename = ",".join(f"{v},{v}_t_adj" for v in solar_vars)
            attrs = " ".join(f'-setattribute,{v}@units="{unit}"' for v in solar_vars)
            sel = ",".join(solar_vars)
            cdo.copy(
                input=(
                    f"-chname,{rename} "
                    f"{attrs} "
                    f"-shifttime,+1hour "
                    f"-divc,3600 "
                    f"-selname,{sel} "
                    f"{focus_nc}"
                ),
                output=solar_t_out,
            )
        else:
            print(f"Skipping process time-adjusted solar (exists): {solar_t_out}")

    # process for wind speed variables
    ws100_out = os.path.join(out_dir, f"{f_name.split('.')[0]}_processed_ws100.nc")
    if {"u100", "v100"} <= varset:
        if not _nc_file_has_vars(cdo, ws100_out, ["ws100"]):
            unit = "m s**-1"
            long_name = "100 metre wind speed"
            cdo.copy(
                input=(
                    f'-setattribute,ws100@long_name="{long_name}" '
                    f'-setattribute,ws100@units="{unit}" '
                    f"-expr,'ws100=sqrt(u100*u100+v100*v100)' "
                    f"{focus_nc}"
                ),
                output=ws100_out,
            )
        else:
            print(f"Skipping process ws100 (exists): {ws100_out}")

    ws10_out = os.path.join(out_dir, f"{f_name.split('.')[0]}_processed_ws10.nc")
    if {"u10", "v10"} <= varset:
        if not _nc_file_has_vars(cdo, ws10_out, ["ws10"]):
            unit = "m s**-1"
            long_name = "10 metre wind speed"
            cdo.copy(
                input=(
                    f'-setattribute,ws10@long_name="{long_name}" '
                    f'-setattribute,ws10@units="{unit}" '
                    f"-expr,'ws10=sqrt(u10*u10+v10*v10)' "
                    f"{focus_nc}"
                ),
                output=ws10_out,
            )
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
    cdo = Cdo()
    source_group = "reanalysis-era5-single-levels"

    # (source_file, [nc_var_names_to_tile]) pairs
    file_var_pairs: list[tuple[str, list[str]]] = []

    for f in sorted(os.listdir(processed_dir)):
        if not f.endswith(".nc") or "_processed_" not in f:
            continue
        path = os.path.join(processed_dir, f)
        vars_in_file = cdo.showname(input=path)[0].split()
        vars_to_tile = [v for v in vars_in_file if v in _ERA5_NC_TO_TILE_LABEL]
        if vars_to_tile:
            file_var_pairs.append((path, vars_to_tile))

    if raw_nc and raw_variables:
        vars_in_raw = set(cdo.showname(input=raw_nc)[0].split())
        vars_to_tile = [v for v in raw_variables if v in vars_in_raw and v in _ERA5_NC_TO_TILE_LABEL]
        if vars_to_tile:
            file_var_pairs.append((raw_nc, vars_to_tile))

    for source_file, variables in file_var_pairs:
        with nc4.Dataset(source_file) as ds:
            lats = ds.variables["latitude"][:]
            lons = ds.variables["longitude"][:]
        lon_min, lon_max = float(lons.min()), float(lons.max())
        lat_min, lat_max = float(lats.min()), float(lats.max())
        years = cdo.showyear(input=source_file)[0].split()

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
                            cdo=cdo,
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

    dates = pd.date_range(start=start_date, end=end_date, freq="MS")

    # group months per year to avoid over-downloading on partial multi-year ranges
    months_by_year: dict[str, list[str]] = {}
    for d in dates:
        months_by_year.setdefault(str(d.year), []).append(f"{d.month:02d}")

    area = (
        boundary_box["north"],
        boundary_box["west"],
        boundary_box["south"],
        boundary_box["east"],
    )

    bbox_tag = f"N{boundary_box['north']}_S{boundary_box['south']}_W{boundary_box['west']}_E{boundary_box['east']}"
    output_file = os.path.join(
        raw_dir,
        f"{era5_dataset}_{dates[0].strftime('%Y%m')}-{dates[-1].strftime('%Y%m')}_{bbox_tag}_raw.nc",
    )
    if not os.path.exists(output_file):
        if len(months_by_year) == 1:
            # single year: one request
            year, months = next(iter(months_by_year.items()))
            era5_downloader(
                target_filename=output_file,
                year=year,
                month=months,
                variables=variables,
                area=area,
            )
        else:
            # multi-year: download per year into temp files, then merge with CDO
            cdo = Cdo()
            yearly_files = []
            for year, months in months_by_year.items():
                yearly_file = os.path.join(raw_dir, f"_tmp_{era5_dataset}_{year}_{bbox_tag}_raw.nc")
                yearly_files.append(yearly_file)
                if not os.path.exists(yearly_file):
                    era5_downloader(
                        target_filename=yearly_file,
                        year=year,
                        month=months,
                        variables=variables,
                        area=area,
                    )
            cdo.mergetime(input=" ".join(yearly_files), output=output_file)
            for f in yearly_files:
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
