"""Tests for the LRA (long-run average) preprocessing functions in ``reskit.util.create_LRA``.

The heavy lifting in that module is a chain of small, deterministic steps: find the
input NetCDFs, average them over time and over years, and -- for the global rasters --
wrap the near-global result around the antimeridian and the poles so that no grid cell
is left without data. Each of those steps is tested here on synthetic inputs.
"""

import numpy as np
import pytest
import xarray as xr

import geokit as gk

from reskit.util.create_LRA import (
    _calculate_DNI,
    _find_single_year_nc_file,
    _list_tiled_nc_files,
    _mean_over_time,
    _snap_to_grid,
    _world_index_range,
    build_arg_parser,
    create_LRA,
    create_long_run_average,
    expand_to_global_coverage,
    extract_bbox_from_mosaic,
    interp_vertical_1d,
    load_era5_year,
    pick_data_var,
    world_3x3_wrap,
    write_geotiff_file,
)

VARIABLE = "100m_wind_speed.processed"


# ---------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------


def _make_dataset(lats, lons, value, n_times=3, var_name="ws100"):
    """A small ERA5-like dataset, constant in space and time at ``value``."""
    data = np.full((n_times, len(lats), len(lons)), float(value))
    return xr.Dataset(
        {var_name: (("time", "latitude", "longitude"), data)},
        coords={
            "time": np.arange(n_times),
            "latitude": np.asarray(lats, dtype=float),
            "longitude": np.asarray(lons, dtype=float),
        },
    )


def _write_tiled(base_path, year, value, variable=VARIABLE, zoom_level=4):
    """Write two tiles that are disjoint in longitude, in the tiled RESKit layout."""
    for tile_index, lons in enumerate([[0.0, 1.0], [2.0, 3.0]]):
        tile_dir = base_path / str(zoom_level) / str(tile_index) / "0" / str(year)
        tile_dir.mkdir(parents=True, exist_ok=True)
        ds = _make_dataset(lats=[50.0, 51.0], lons=lons, value=value)
        ds.to_netcdf(tile_dir / f"tile.{variable}.nc")


class _RasterInfo:
    """Stand-in for ``geokit.raster.rasterInfo``, which is all ``world_3x3_wrap`` needs."""

    def __init__(self, pixel_width, pixel_height, bounds):
        self.pixelWidth = pixel_width
        self.pixelHeight = pixel_height
        self.bounds = bounds


# ---------------------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------------------


def test_list_tiled_nc_files_finds_all_tiles_of_a_year(tmp_path):
    _write_tiled(tmp_path, year=2000, value=1.0)
    _write_tiled(tmp_path, year=2001, value=2.0)

    hits = _list_tiled_nc_files(tmp_path, year=2000, variable=VARIABLE, zoom_level=4)

    assert len(hits) == 2
    # the tiled layout puts each file directly below its year, so compare path components
    # rather than substrings of the path -- the separator differs between platforms
    assert all(p.parent.name == "2000" for p in hits)
    # sorted and free of duplicates
    assert hits == sorted(set(hits))


def test_list_tiled_nc_files_returns_empty_for_missing_zoom_level(tmp_path):
    _write_tiled(tmp_path, year=2000, value=1.0, zoom_level=4)

    assert _list_tiled_nc_files(tmp_path, year=2000, variable=VARIABLE, zoom_level=5) == []


def test_list_tiled_nc_files_ignores_other_variables(tmp_path):
    _write_tiled(tmp_path, year=2000, value=1.0, variable=VARIABLE)
    _write_tiled(tmp_path, year=2000, value=1.0, variable="surface_pressure.processed")

    hits = _list_tiled_nc_files(tmp_path, year=2000, variable=VARIABLE, zoom_level=4)

    assert len(hits) == 2
    assert all(p.name.endswith(f".{VARIABLE}.nc") for p in hits)


def test_find_single_year_nc_file(tmp_path):
    year_dir = tmp_path / "2000"
    year_dir.mkdir()
    expected = year_dir / f"whatever.{VARIABLE}.nc"
    _make_dataset([50.0], [0.0], 1.0).to_netcdf(expected)

    assert _find_single_year_nc_file(tmp_path, year=2000, variable=VARIABLE) == expected


def test_find_single_year_nc_file_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        _find_single_year_nc_file(tmp_path, year=2000, variable=VARIABLE)


def test_find_single_year_nc_file_raises_when_ambiguous(tmp_path):
    year_dir = tmp_path / "2000"
    year_dir.mkdir()
    for name in ["a", "b"]:
        _make_dataset([50.0], [0.0], 1.0).to_netcdf(year_dir / f"{name}.{VARIABLE}.nc")

    with pytest.raises(RuntimeError, match="exactly one NetCDF per year"):
        _find_single_year_nc_file(tmp_path, year=2000, variable=VARIABLE)


# ---------------------------------------------------------------------------------------
# Averaging
# ---------------------------------------------------------------------------------------


def test_mean_over_time_collapses_the_time_dimension():
    ds = _make_dataset([50.0], [0.0], value=1.0, n_times=4)
    ds["ws100"][:, 0, 0] = [1.0, 2.0, 3.0, 4.0]

    out = _mean_over_time(ds)

    assert "time" not in out.dims
    assert np.isclose(out["ws100"].values.item(), 2.5)


def test_mean_over_time_is_a_no_op_without_a_time_dimension():
    ds = _make_dataset([50.0], [0.0], value=1.0).mean(dim="time")

    assert _mean_over_time(ds).equals(ds)


def test_load_era5_year_merges_tiles_into_one_grid(tmp_path):
    _write_tiled(tmp_path, year=2000, value=7.0)

    ds = load_era5_year(tmp_path, year=2000, variable=VARIABLE, zoom_level=4)

    # the two tiles cover disjoint longitudes and are merged onto a common grid
    assert list(ds["longitude"].values) == [0.0, 1.0, 2.0, 3.0]
    assert "time" not in ds.dims
    assert np.allclose(ds["ws100"].values, 7.0)


def test_load_era5_year_reads_non_tiled_layout(tmp_path):
    year_dir = tmp_path / "2000"
    year_dir.mkdir()
    _make_dataset([50.0, 51.0], [0.0, 1.0], value=3.0).to_netcdf(year_dir / f"x.{VARIABLE}.nc")

    ds = load_era5_year(tmp_path, year=2000, variable=VARIABLE, zoom_level=4)

    assert "time" not in ds.dims
    assert np.allclose(ds["ws100"].values, 3.0)


def test_load_era5_year_can_keep_the_time_dimension(tmp_path):
    _write_tiled(tmp_path, year=2000, value=7.0)

    ds = load_era5_year(tmp_path, year=2000, variable=VARIABLE, zoom_level=4, mean_over_time=False)

    assert ds.sizes["time"] == 3


def test_create_long_run_average_averages_over_years(tmp_path):
    # yearly means of 1, 2, 3 -> LRA of 2
    for offset, year in enumerate([2000, 2001, 2002]):
        _write_tiled(tmp_path, year=year, value=float(offset + 1))

    ds_lra = create_long_run_average(
        base_path=tmp_path,
        start_year=2000,
        end_year=2002,
        variable=VARIABLE,
        out_dir=tmp_path / "out",
        zoom_level=4,
        cache_yearly=False,
    )

    assert "year" not in ds_lra.dims
    assert np.allclose(ds_lra["ws100"].values, 2.0)


def test_create_long_run_average_caches_and_reuses_yearly_files(tmp_path):
    _write_tiled(tmp_path, year=2000, value=5.0)
    out_dir = tmp_path / "out"

    create_long_run_average(
        base_path=tmp_path,
        start_year=2000,
        end_year=2000,
        variable=VARIABLE,
        out_dir=out_dir,
        zoom_level=4,
        cache_yearly=True,
        weather_source_prefix="ERA5_",
    )
    cached = out_dir / f"ERA5_merged_{VARIABLE}_2000.nc"
    assert cached.exists()

    # overwrite the cache with a different value and delete the inputs: if the cache is
    # honoured, the result must come from the cached file rather than from the tiles
    _make_dataset([50.0], [0.0], value=99.0).mean(dim="time").to_netcdf(cached)
    ds_lra = create_long_run_average(
        base_path=tmp_path / "does_not_exist",
        start_year=2000,
        end_year=2000,
        variable=VARIABLE,
        out_dir=out_dir,
        zoom_level=4,
        cache_yearly=True,
        weather_source_prefix="ERA5_",
    )

    assert np.allclose(ds_lra["ws100"].values, 99.0)


# ---------------------------------------------------------------------------------------
# Variable selection and output
# ---------------------------------------------------------------------------------------


def test_pick_data_var_returns_the_only_variable():
    ds = _make_dataset([50.0], [0.0], 1.0)

    assert pick_data_var(ds).name == "ws100"


def test_pick_data_var_selects_by_name():
    ds = _make_dataset([50.0], [0.0], 1.0)
    ds["other"] = ds["ws100"] * 2

    assert pick_data_var(ds, data_var="other").name == "other"


def test_pick_data_var_raises_on_ambiguity():
    ds = _make_dataset([50.0], [0.0], 1.0)
    ds["other"] = ds["ws100"] * 2

    with pytest.raises(ValueError, match="multiple data variables"):
        pick_data_var(ds)


def test_pick_data_var_raises_on_unknown_name():
    ds = _make_dataset([50.0], [0.0], 1.0)

    with pytest.raises(KeyError):
        pick_data_var(ds, data_var="nope")


def test_write_geotiff_file_writes_a_north_up_raster(tmp_path):
    da = _make_dataset([50.0, 51.0], [0.0, 1.0], value=1.0).mean(dim="time")["ws100"]
    da[:] = [[1.0, 2.0], [3.0, 4.0]]  # latitude 50 first, i.e. south first
    out = tmp_path / "lra.tiff"

    write_geotiff_file(da, out)

    assert out.exists()
    # written north-up, so the row for latitude 51 (values 3, 4) must come first
    assert np.allclose(gk.raster.extractMatrix(str(out))[0, :], [3.0, 4.0])


def test_write_geotiff_file_raises_without_a_latitude_coordinate(tmp_path):
    da = xr.DataArray(np.ones((2, 2)), dims=("y", "x"))

    with pytest.raises(ValueError, match="Cannot determine the latitude axis"):
        write_geotiff_file(da, tmp_path / "lra.tiff")


def test_write_geotiff_file_writes_the_correct_bounds(tmp_path):
    # coordinates are pixel centers, so the written extent is half a pixel wider on each side
    da = _make_dataset([50.0, 51.0], [0.0, 1.0], value=1.0).mean(dim="time")["ws100"]
    out = tmp_path / "lra.tiff"

    write_geotiff_file(da, out)

    info = gk.raster.rasterInfo(str(out))
    assert info.bounds == (-0.5, 49.5, 1.5, 51.5)
    assert np.isclose(info.pixelWidth, 1.0)
    assert np.isclose(info.pixelHeight, 1.0)


def test_write_geotiff_file_is_north_up_regardless_of_input_order(tmp_path):
    ascending = _make_dataset([50.0, 51.0], [0.0, 1.0], value=1.0).mean(dim="time")["ws100"]
    ascending[:] = [[1.0, 2.0], [3.0, 4.0]]
    descending = ascending.sortby("latitude", ascending=False)

    write_geotiff_file(ascending, tmp_path / "a.tiff")
    write_geotiff_file(descending, tmp_path / "d.tiff")

    assert np.allclose(
        gk.raster.extractMatrix(str(tmp_path / "a.tiff")),
        gk.raster.extractMatrix(str(tmp_path / "d.tiff")),
    )


def test_write_geotiff_file_rejects_an_irregular_grid(tmp_path):
    da = _make_dataset([50.0, 51.0, 55.0], [0.0, 1.0], value=1.0).mean(dim="time")["ws100"]

    with pytest.raises(ValueError, match="not evenly spaced"):
        write_geotiff_file(da, tmp_path / "lra.tiff")


def test_create_LRA_writes_a_geotiff_of_the_long_run_average(tmp_path):
    for offset, year in enumerate([2000, 2001]):
        _write_tiled(tmp_path, year=year, value=float(offset + 1))
    out_dir = tmp_path / "out"

    create_LRA(
        base_path=tmp_path,
        variable=VARIABLE,
        start_year=2000,
        end_year=2001,
        zoom_level=4,
        out_dir=out_dir,
        cache_yearly=False,
        write_geotiff=True,
    )

    tiff = out_dir / f"{VARIABLE}_2000_2001_mean.tiff"
    assert tiff.exists()
    # mean of the yearly means 1 and 2
    assert np.allclose(gk.raster.extractMatrix(str(tiff)), 1.5)


def test_create_LRA_honours_variable_name_output_and_prefix(tmp_path):
    _write_tiled(tmp_path, year=2000, value=1.0)
    out_dir = tmp_path / "out"

    create_LRA(
        base_path=tmp_path,
        variable=VARIABLE,
        start_year=2000,
        end_year=2000,
        out_dir=out_dir,
        cache_yearly=False,
        variable_name_output="100m_wind_speed",
        weather_source_prefix="ERA5",
        write_geotiff=True,
    )

    assert (out_dir / "ERA5_100m_wind_speed_2000_2000_mean.tiff").exists()


def test_create_LRA_can_write_netcdf(tmp_path):
    _write_tiled(tmp_path, year=2000, value=1.0)
    out_dir = tmp_path / "out"

    create_LRA(
        base_path=tmp_path,
        variable=VARIABLE,
        start_year=2000,
        end_year=2000,
        out_dir=out_dir,
        cache_yearly=False,
        write_geotiff=False,
        write_netcdf=True,
    )

    assert (out_dir / f"{VARIABLE}_2000_2000_mean.nc").exists()


# ---------------------------------------------------------------------------------------
# DNI derivation
# ---------------------------------------------------------------------------------------


def test_calculate_DNI_divides_by_the_sine_of_the_elevation_angle():
    sea = xr.DataArray([30.0, 90.0])
    dhi = xr.DataArray([100.0, 100.0])

    dni = _calculate_DNI(sea, dhi)

    # DNI = DHI / sin(elevation): sin(30) = 0.5, sin(90) = 1
    assert np.allclose(dni.values, [200.0, 100.0])


def test_calculate_DNI_is_zero_for_the_sun_at_or_below_the_horizon():
    sea = xr.DataArray([-10.0, 0.0, 1.0])
    dhi = xr.DataArray([100.0, 100.0, 100.0])

    dni = _calculate_DNI(sea, dhi)

    # elevations <= 1 degree are masked out and filled with 0 rather than blowing up
    assert np.allclose(dni.values, 0.0)


# ---------------------------------------------------------------------------------------
# Global expansion: 3x3 world wrap, interpolation, and clipping
# ---------------------------------------------------------------------------------------


def test_world_3x3_wrap_tiles_the_centre_map_three_by_three():
    arr = np.arange(180 * 360, dtype=float).reshape(180, 360)
    rinfo = _RasterInfo(pixel_width=1.0, pixel_height=1.0, bounds=(-180.0, -90.0, 180.0, 90.0))

    out, bounds = world_3x3_wrap(arr, rinfo)

    height, width = out.shape[0] // 3, out.shape[1] // 3
    assert out.shape == (3 * height, 3 * width)
    # the centre tile is written last and therefore wins over the wrapped tiles
    centre = out[height : 2 * height, width : 2 * width]
    assert np.allclose(centre[: arr.shape[0], : arr.shape[1]], arr)
    # the mosaic spans three tiles in each direction
    xmin, ymin, xmax, ymax = bounds
    assert np.isclose(xmax - xmin, 3 * width * rinfo.pixelWidth)
    assert np.isclose(ymax - ymin, 3 * height * rinfo.pixelHeight)


def test_world_3x3_wrap_mirrors_the_map_across_the_poles():
    # a map that only covers the northern hemisphere, so the polar wrap is visible
    arr = np.arange(2 * 4, dtype=float).reshape(2, 4)
    rinfo = _RasterInfo(pixel_width=90.0, pixel_height=45.0, bounds=(-180.0, 0.0, 180.0, 90.0))

    out, _ = world_3x3_wrap(arr, rinfo)

    # the tiles above and below the centre carry data mirrored over the pole, which is
    # what lets a later vertical interpolation cross the pole instead of running off the map
    height, width = out.shape[0] // 3, out.shape[1] // 3
    above = out[0:height, width : 2 * width]
    assert np.isfinite(above).any()


def test_world_index_range_brackets_the_targets():
    # pixel centers at 0.5, 1.5, ... 9.5 -> indices 0..9 lie within [0, 10]
    assert _world_index_range(0.0, 10.0, origin=0.5, step=1.0) == (0, 9)


def test_world_index_range_handles_a_negative_step():
    # north-up rows: centers at 89.5, 88.5, ... -89.5 for a global 1 degree grid
    assert _world_index_range(90.0, -90.0, origin=89.5, step=-1.0) == (0, 179)


def test_world_index_range_does_not_invent_indices_outside_the_data():
    # the target sits exactly halfway between two centers, which is where rounding to
    # nearest used to reach past the edge of the data and produce an all-NaN column
    lo, hi = _world_index_range(-180.0, 180.0, origin=-179.5, step=1.0)

    assert (lo, hi) == (0, 359)
    assert hi - lo + 1 == 360


def test_interp_vertical_1d_fills_gaps_between_valid_samples():
    arr = np.array([[1.0], [np.nan], [np.nan], [4.0]])

    out = interp_vertical_1d(arr)

    assert np.allclose(out[:, 0], [1.0, 2.0, 3.0, 4.0])


def test_interp_vertical_1d_leaves_unbracketed_nans_alone():
    arr = np.array([[np.nan], [1.0], [2.0], [np.nan]])

    out = interp_vertical_1d(arr)

    assert np.isnan(out[0, 0])
    assert np.isnan(out[3, 0])


def test_interp_vertical_1d_respects_max_gap():
    arr = np.array([[1.0], [np.nan], [np.nan], [np.nan], [5.0]])

    assert np.isnan(interp_vertical_1d(arr, max_gap=2)[1:4, 0]).all()
    assert np.isfinite(interp_vertical_1d(arr, max_gap=3)[1:4, 0]).all()


def test_interp_vertical_1d_needs_two_valid_samples():
    arr = np.array([[np.nan], [1.0], [np.nan]])

    assert np.isnan(interp_vertical_1d(arr)[[0, 2], 0]).all()


def test_interp_vertical_1d_rejects_non_2d_input():
    with pytest.raises(ValueError, match="must be 2D"):
        interp_vertical_1d(np.ones(3))


@pytest.mark.parametrize(
    "mode, expected",
    [("round", 10.0), ("floor", 10.0), ("ceil", 12.0)],
)
def test_snap_to_grid(mode, expected):
    # grid is 0, 2, 4, ... and the value sits just above 10
    assert np.isclose(_snap_to_grid(10.4, origin=0.0, res=2.0, mode=mode), expected)


def test_snap_to_grid_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode must be one of"):
        _snap_to_grid(1.0, origin=0.0, res=1.0, mode="nearest")


def test_extract_bbox_from_mosaic_cuts_the_requested_window():
    mosaic = np.arange(4 * 4, dtype=float).reshape(4, 4)
    bounds_3x3 = (0.0, 0.0, 4.0, 4.0)  # 1 degree pixels

    cut, cut_bounds = extract_bbox_from_mosaic(
        mosaic,
        bounds_3x3=bounds_3x3,
        pixel_width=1.0,
        pixel_height=1.0,
        bbox=(1.0, 1.0, 3.0, 3.0),
    )

    assert cut_bounds == (1.0, 1.0, 3.0, 3.0)
    # rows 1..2 (from the top, north-up) and columns 1..2
    assert np.allclose(cut, mosaic[1:3, 1:3])


def test_extract_bbox_from_mosaic_rejects_a_degenerate_bbox():
    mosaic = np.zeros((4, 4))

    with pytest.raises(ValueError, match="Invalid bbox after snapping"):
        extract_bbox_from_mosaic(
            mosaic,
            bounds_3x3=(0.0, 0.0, 4.0, 4.0),
            pixel_width=1.0,
            pixel_height=1.0,
            bbox=(2.0, 1.0, 2.0, 3.0),
        )


def _write_near_global_raster(path, bounds, pixel=1.0):
    """A raster spanning the full longitude range but only part of the latitude range."""
    xmin, ymin, xmax, ymax = bounds
    n_rows = int(round((ymax - ymin) / pixel))
    n_cols = int(round((xmax - xmin) / pixel))
    data = np.tile(np.arange(n_rows, dtype=float).reshape(-1, 1), (1, n_cols))
    gk.raster.createRaster(
        data=data,
        bounds=bounds,
        pixelWidth=pixel,
        pixelHeight=pixel,
        output=str(path),
        srs=gk.srs.EPSG4326,
        dtype="float32",
    )
    return data


def test_expand_to_global_coverage_fills_the_missing_latitude_band(tmp_path):
    src = tmp_path / "near_global.tif"
    _write_near_global_raster(src, bounds=(-180.0, -60.0, 180.0, 60.0))

    out = expand_to_global_coverage(str(src), target_bounds=(-180.0, -90.0, 180.0, 90.0))

    assert out.shape == (180, 360)
    # this is the point of the whole exercise: no grid cell is left without a value
    assert np.isfinite(out).all()


def test_expand_to_global_coverage_preserves_the_original_data(tmp_path):
    src = tmp_path / "near_global.tif"
    original = _write_near_global_raster(src, bounds=(-180.0, -60.0, 180.0, 60.0))

    out = expand_to_global_coverage(str(src), target_bounds=(-180.0, -90.0, 180.0, 90.0))

    # the original 120 rows sit in the middle of the 180 output rows, unchanged
    assert np.allclose(out[30:150, :], original)


def test_expand_to_global_coverage_can_write_a_raster(tmp_path):
    src = tmp_path / "near_global.tif"
    _write_near_global_raster(src, bounds=(-180.0, -60.0, 180.0, 60.0))
    dst = tmp_path / "global.tif"

    expand_to_global_coverage(
        str(src),
        target_bounds=(-180.0, -90.0, 180.0, 90.0),
        as_array=False,
        output_path=str(dst),
    )

    assert dst.exists()
    assert gk.raster.rasterInfo(str(dst)).bounds == (-180.0, -90.0, 180.0, 90.0)


def test_expand_to_global_coverage_rejects_incomplete_longitude_coverage(tmp_path):
    src = tmp_path / "regional.tif"
    _write_near_global_raster(src, bounds=(-90.0, -60.0, 90.0, 60.0))

    with pytest.raises(ValueError, match="only latitudinal expansion"):
        expand_to_global_coverage(str(src), target_bounds=(-180.0, -90.0, 180.0, 90.0))


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------


def test_arg_parser_defaults():
    args = build_arg_parser().parse_args(
        ["--base-path", "/data", "--variable", VARIABLE, "--start-year", "2000", "--end-year", "2001"]
    )

    assert args.zoom_level == 4
    assert args.cache_yearly is True
    assert args.combine_mode == "auto"
    assert args.write_geotiff is False
    assert args.start_year == 2000


def test_arg_parser_no_cache_yearly_flag():
    args = build_arg_parser().parse_args(
        [
            "--base-path",
            "/data",
            "--variable",
            VARIABLE,
            "--start-year",
            "2000",
            "--end-year",
            "2001",
            "--no-cache-yearly",
            "--write-geotiff",
        ]
    )

    assert args.cache_yearly is False
    assert args.write_geotiff is True


def test_arg_parser_requires_the_core_arguments():
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(["--variable", VARIABLE])
