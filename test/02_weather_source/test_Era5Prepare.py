import os
import shutil
import netCDF4 as nc4
import numpy as np
import pandas as pd
import xarray as xr
import pytest
from reskit import TEST_DATA
from reskit.weather.Era5Source.Era5Prepare import (
    _ERA5_NC_TO_TILE_LABEL,
    _align_longitudes_to_source_convention,
    _era5_download_jobs,
    _get_source_lon_boxes,
    _iter_tile_x_indices,
    _normalize_lon,
    _split_lon_boxes,
    _tile_variable_to_file,
    era5_tiler,
    preprocess_era5_data,
)

# era5-like test data: lat=[49,52], lon=[5,7.5], year=2015
# At zoom 4, this falls entirely within tile (x=8, y=5)
ZOOM = 4
TILE_X, TILE_Y, TILE_YEAR = 8, 5, 2015

EXPECTED_TILE_DIR = os.path.join(str(ZOOM), str(TILE_X), str(TILE_Y), str(TILE_YEAR))

ERA5_DATASET = "reanalysis-era5-single-levels"


def tile_filename(zoom, x, y, year, label):
    return f"{ERA5_DATASET}.z{zoom}.x{x}.y{y}.y{year}.{label}.nc"


@pytest.fixture
def era5_like_tile_input(tmp_path):
    """Temp dir with era5-like files renamed to match era5_tiler's expected naming.
    Returns (processed_dir, raw_nc_path).
    """
    era5_like = TEST_DATA["era5-like"]
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    shutil.copy(
        os.path.join(era5_like, "100m_wind_speed.processed.nc"),
        processed_dir / "era5_test_processed_ws100.nc",
    )
    raw_nc = tmp_path / "raw" / "era5_test_raw.nc"
    raw_nc.parent.mkdir()
    shutil.copy(os.path.join(era5_like, "boundary_layer_height.nc"), raw_nc)
    return processed_dir, raw_nc


def test_era5_tiler_creates_tile_directory(era5_like_tile_input, tmp_path):
    processed_dir, raw_nc = era5_like_tile_input
    tile_out = tmp_path / "tiles"
    era5_tiler(
        processed_dir=str(processed_dir),
        tile_output_dir=str(tile_out),
        zoom_level=ZOOM,
        raw_nc=str(raw_nc),
        raw_variables=["blh"],
    )
    assert (tile_out / EXPECTED_TILE_DIR).is_dir()


def test_era5_tiler_ws100_filename(era5_like_tile_input, tmp_path):
    processed_dir, _ = era5_like_tile_input
    tile_out = tmp_path / "tiles"
    era5_tiler(processed_dir=str(processed_dir), tile_output_dir=str(tile_out), zoom_level=ZOOM)
    expected = (
        tile_out / EXPECTED_TILE_DIR / tile_filename(ZOOM, TILE_X, TILE_Y, TILE_YEAR, "100m_wind_speed.processed")
    )
    assert expected.exists(), f"Expected tile file not found: {expected}"


def test_era5_tiler_raw_variable_filename(era5_like_tile_input, tmp_path):
    processed_dir, raw_nc = era5_like_tile_input
    tile_out = tmp_path / "tiles"
    era5_tiler(
        processed_dir=str(processed_dir),
        tile_output_dir=str(tile_out),
        zoom_level=ZOOM,
        raw_nc=str(raw_nc),
        raw_variables=["blh"],
    )
    expected = tile_out / EXPECTED_TILE_DIR / tile_filename(ZOOM, TILE_X, TILE_Y, TILE_YEAR, "boundary_layer_height")
    assert expected.exists(), f"Expected tile file not found: {expected}"


def test_era5_tiler_output_contains_correct_variable(era5_like_tile_input, tmp_path):
    processed_dir, _ = era5_like_tile_input
    tile_out = tmp_path / "tiles"
    era5_tiler(processed_dir=str(processed_dir), tile_output_dir=str(tile_out), zoom_level=ZOOM)
    tile_file = (
        tile_out / EXPECTED_TILE_DIR / tile_filename(ZOOM, TILE_X, TILE_Y, TILE_YEAR, "100m_wind_speed.processed")
    )
    with nc4.Dataset(str(tile_file)) as ds:
        assert "ws100" in ds.variables


def test_era5_tiler_skip_existing(era5_like_tile_input, tmp_path):
    processed_dir, _ = era5_like_tile_input
    tile_out = tmp_path / "tiles"
    era5_tiler(processed_dir=str(processed_dir), tile_output_dir=str(tile_out), zoom_level=ZOOM)
    tile_file = (
        tile_out / EXPECTED_TILE_DIR / tile_filename(ZOOM, TILE_X, TILE_Y, TILE_YEAR, "100m_wind_speed.processed")
    )
    mtime_before = tile_file.stat().st_mtime
    era5_tiler(processed_dir=str(processed_dir), tile_output_dir=str(tile_out), zoom_level=ZOOM)
    assert tile_file.stat().st_mtime == mtime_before


def test_era5_tiler_no_raw_variables(era5_like_tile_input, tmp_path):
    processed_dir, _ = era5_like_tile_input
    tile_out = tmp_path / "tiles"
    era5_tiler(processed_dir=str(processed_dir), tile_output_dir=str(tile_out), zoom_level=ZOOM)
    blh_tile = tile_out / EXPECTED_TILE_DIR / tile_filename(ZOOM, TILE_X, TILE_Y, TILE_YEAR, "boundary_layer_height")
    assert not blh_tile.exists()


def _days_covered(start, end):
    """Reconstruct the set of calendar days a job list actually requests (CDS drops days that
    don't exist in a month, so we mirror that here)."""
    covered = set()
    for year, months, days in _era5_download_jobs(start, end):
        for m in months:
            for d in days:
                try:
                    covered.add(pd.Timestamp(f"{year}-{m}-{d}").normalize())
                except ValueError:
                    pass
    return covered


def test_download_jobs_submonth_range_uses_exact_days():
    jobs = _era5_download_jobs("2000-01-01", "2000-01-03")
    assert jobs == [("2000", ["01"], ["01", "02", "03"])]


def test_download_jobs_full_month_uses_canonical_day_list():
    # a fully-covered month (even February) uses 01..31 so full months batch together
    (year, months, days), = _era5_download_jobs("2001-02-01", "2001-02-28")
    assert (year, months) == ("2001", ["02"])
    assert days == [f"{d:02d}" for d in range(1, 32)]


def test_download_jobs_full_year_is_single_request():
    jobs = _era5_download_jobs("2000-01-01", "2000-12-31")
    assert len(jobs) == 1  # one CDS request, not twelve
    year, months, days = jobs[0]
    assert year == "2000"
    assert months == [f"{m:02d}" for m in range(1, 13)]
    assert days == [f"{d:02d}" for d in range(1, 32)]


def test_download_jobs_partial_ends_split_into_three():
    jobs = _era5_download_jobs("2000-01-15", "2000-03-10")
    assert [(m, d[0], d[-1]) for _, m, d in jobs] == [
        (["01"], "15", "31"),
        (["02"], "01", "31"),  # middle month is full -> canonical days
        (["03"], "01", "10"),
    ]


def test_download_jobs_multi_year_split_per_year():
    jobs = _era5_download_jobs("2000-11-01", "2001-02-28")
    assert [y for y, _, _ in jobs] == ["2000", "2001"]


@pytest.mark.parametrize(
    "start,end",
    [
        ("2000-01-01", "2000-01-03"),
        ("2000-01-15", "2000-03-10"),
        ("2000-01-01", "2000-12-31"),
        ("2000-11-01", "2001-02-28"),
    ],
)
def test_download_jobs_cover_exactly_requested_days(start, end):
    assert _days_covered(start, end) == set(pd.date_range(start, end, freq="D").normalize())


def test_download_jobs_end_before_start_raises():
    with pytest.raises(ValueError):
        _era5_download_jobs("2000-02-01", "2000-01-01")


def test_normalize_lon_wraps_out_of_range_values():
    assert _normalize_lon(-182.0) == pytest.approx(178.0)
    assert _normalize_lon(181.0) == pytest.approx(-179.0)
    assert _normalize_lon(180.0) == pytest.approx(180.0)


def test_iter_tile_x_indices_wraps_across_antimeridian():
    assert _iter_tile_x_indices(zoom_level=4, lon_west=-182.0, lon_east=-156.0) == [15, 0, 1]


def test_split_lon_boxes_splits_wrapped_interval():
    assert _split_lon_boxes(lon_west=-181.8, lon_east=-155.5) == pytest.approx([(178.2, 180.0), (-180.0, -155.5)])
    assert _split_lon_boxes(lon_west=156.0, lon_east=181.0) == pytest.approx([(156.0, 180.0), (-180.0, -179.0)])


def test_get_source_lon_boxes_keeps_extended_negative_longitudes():
    assert _get_source_lon_boxes(
        lon_west=178.0,
        lon_east=-155.5,
        source_lon_min=-181.75,
        source_lon_max=-155.5,
    ) == pytest.approx([(-182.0, -155.5)])


def _make_era5_raw(path, *, lat, lon, n_times=4, vars_spec, seed=0):
    """Write a synthetic ERA5-like raw NetCDF (time, latitude, longitude) for testing.

    vars_spec maps variable name -> attrs dict; values are random floats. Time is encoded
    as int32 'hours since 1900-01-01' to mirror the real downloads.
    """
    time = pd.date_range("2015-01-01", periods=n_times, freq="h")
    shape = (len(time), len(lat), len(lon))
    rng = np.random.default_rng(seed)
    coords = {"time": time, "latitude": np.asarray(lat, "f4"), "longitude": np.asarray(lon, "f4")}
    data = {}
    for name, attrs in vars_spec.items():
        arr = rng.uniform(0.0, 1000.0, shape).astype("f4")
        data[name] = xr.DataArray(arr, dims=("time", "latitude", "longitude"), coords=coords, attrs=attrs)
    ds = xr.Dataset(data)
    ds["time"].encoding = {"units": "hours since 1900-01-01 00:00:00.0", "calendar": "gregorian", "dtype": np.int32}
    ds.to_netcdf(path)
    return ds


def test_preprocess_wind_speed_matches_sqrt_and_sets_attrs(tmp_path):
    raw = tmp_path / "raw.nc"
    ds = _make_era5_raw(
        raw,
        lat=[52.0, 51.75, 51.5],
        lon=[5.0, 5.25, 5.5],
        vars_spec={
            "u100": {"units": "m s**-1"},
            "v100": {"units": "m s**-1"},
            "u10": {"units": "m s**-1"},
            "v10": {"units": "m s**-1"},
        },
    )
    proc = tmp_path / "processed"
    preprocess_era5_data(str(raw), str(proc))

    ws100_file = next(p for p in os.listdir(proc) if "ws100" in p)
    with nc4.Dataset(os.path.join(proc, ws100_file)) as out:
        expected = np.sqrt(ds["u100"].values ** 2 + ds["v100"].values ** 2)
        assert np.allclose(out["ws100"][:], expected, atol=1e-4)
        assert out["ws100"].units == "m s**-1"
        assert out["ws100"].long_name == "100 metre wind speed"


def test_preprocess_solar_converts_units_shifts_time_and_preserves_encoding(tmp_path):
    raw = tmp_path / "raw.nc"
    ds = _make_era5_raw(
        raw,
        lat=[52.0, 51.75],
        lon=[5.0, 5.25],
        vars_spec={
            "ssrd": {"units": "J m**-2", "long_name": "Surface solar radiation downwards"},
            "fdir": {"units": "J m**-2", "long_name": "Direct solar radiation at surface"},
        },
    )
    proc = tmp_path / "processed"
    preprocess_era5_data(str(raw), str(proc))

    solar_file = next(p for p in os.listdir(proc) if "solar" in p)
    with nc4.Dataset(os.path.join(proc, solar_file)) as out:
        assert {"ssrd_t_adj", "fdir_t_adj"} <= set(out.variables)
        # accumulated J m**-2 -> mean power flux W m**-2
        assert np.allclose(out["ssrd_t_adj"][:], ds["ssrd"].values / 3600.0, atol=1e-2)
        assert out["ssrd_t_adj"].units == "W m**-2"
        # original descriptive attrs carried through the rename
        assert out["ssrd_t_adj"].long_name == "Surface solar radiation downwards"
        # time shifted +1h, encoding preserved for the netCDF4 consumer
        times = nc4.num2date(out["time"][:], out["time"].units, out["time"].calendar)
        assert times[0].isoformat() == "2015-01-01T01:00:00"
        # hour granularity + integer dtype preserved (xarray canonicalises the trailing
        # "00:00:00.0", which num2date parses identically)
        assert out["time"].units.startswith("hours since 1900-01-01")
        assert np.issubdtype(out["time"].dtype, np.integer)


def test_tile_variable_merges_antimeridian_boxes(tmp_path):
    # Source straddles the antimeridian: high-positive and low-negative longitudes.
    lon = [178.0, 178.5, 179.0, 179.5, -179.5, -179.0]
    raw = tmp_path / "raw.nc"
    ds = _make_era5_raw(raw, lat=[10.0, 9.75], lon=lon, vars_spec={"ws100": {"units": "m s**-1"}})

    target = tmp_path / "tile.nc"
    _tile_variable_to_file(
        source_file=str(raw),
        var="ws100",
        year="2015",
        lat_south=9.0,
        lat_north=11.0,
        lon_boxes=[(178.0, 180.0), (-180.0, -179.0)],
        target_file=str(target),
        source_lon_min=min(lon),
        source_lon_max=max(lon),
    )

    with nc4.Dataset(str(target)) as out:
        out_lons = out["longitude"][:]
        # both boxes merged, sorted ascending, no duplicates
        assert np.allclose(out_lons, sorted(lon))
        assert len(out_lons) == len(set(out_lons.tolist()))
        assert "ws100" in out.variables
        # values preserved from the source for a sampled longitude
        src_col = ds.isel(longitude=lon.index(179.0))["ws100"].values
        out_col = out["ws100"][:, :, list(out_lons).index(179.0)]
        assert np.allclose(out_col, src_col, atol=1e-4)


def test_align_longitudes_to_source_convention_shifts_positive_tile_axis(tmp_path):
    tile_file = tmp_path / "tile.nc"
    with nc4.Dataset(tile_file, "w") as ds:
        ds.createDimension("longitude", 3)
        lon_var = ds.createVariable("longitude", "f4", ("longitude",))
        lon_var[:] = np.array([178.0, 191.0, 204.0], dtype=np.float32)

    _align_longitudes_to_source_convention(
        target_file=str(tile_file),
        source_lon_min=-182.0,
        source_lon_max=-156.0,
    )

    with nc4.Dataset(tile_file) as ds:
        assert ds.variables["longitude"][:].tolist() == pytest.approx([-182.0, -169.0, -156.0])
