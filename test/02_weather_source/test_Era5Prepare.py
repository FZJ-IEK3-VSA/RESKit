import os
import shutil
import netCDF4 as nc4
import pytest
from reskit import TEST_DATA
from reskit.weather.Era5Source.Era5Prepare import era5_tiler, _ERA5_NC_TO_TILE_LABEL

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
    """Temp dir with era5-like files renamed to match era5_tiler's expected naming."""
    era5_like = TEST_DATA["era5-like"]
    shutil.copy(
        os.path.join(era5_like, "100m_wind_speed.processed.nc"),
        tmp_path / "era5_test_processed_ws100.nc",
    )
    shutil.copy(
        os.path.join(era5_like, "boundary_layer_height.nc"),
        tmp_path / "era5_test_raw.nc",
    )
    return tmp_path


def test_era5_tiler_creates_tile_directory(era5_like_tile_input, tmp_path):
    tile_out = tmp_path / "tiles"
    era5_tiler(
        source_dir=str(era5_like_tile_input),
        tile_output_dir=str(tile_out),
        zoom_level=ZOOM,
        raw_variables=["blh"],
    )
    assert (tile_out / EXPECTED_TILE_DIR).is_dir()


def test_era5_tiler_ws100_filename(era5_like_tile_input, tmp_path):
    tile_out = tmp_path / "tiles"
    era5_tiler(
        source_dir=str(era5_like_tile_input),
        tile_output_dir=str(tile_out),
        zoom_level=ZOOM,
    )
    expected = tile_out / EXPECTED_TILE_DIR / tile_filename(ZOOM, TILE_X, TILE_Y, TILE_YEAR, "100m_wind_speed.processed")
    assert expected.exists(), f"Expected tile file not found: {expected}"


def test_era5_tiler_raw_variable_filename(era5_like_tile_input, tmp_path):
    tile_out = tmp_path / "tiles"
    era5_tiler(
        source_dir=str(era5_like_tile_input),
        tile_output_dir=str(tile_out),
        zoom_level=ZOOM,
        raw_variables=["blh"],
    )
    expected = tile_out / EXPECTED_TILE_DIR / tile_filename(ZOOM, TILE_X, TILE_Y, TILE_YEAR, "boundary_layer_height")
    assert expected.exists(), f"Expected tile file not found: {expected}"


def test_era5_tiler_output_contains_correct_variable(era5_like_tile_input, tmp_path):
    tile_out = tmp_path / "tiles"
    era5_tiler(
        source_dir=str(era5_like_tile_input),
        tile_output_dir=str(tile_out),
        zoom_level=ZOOM,
    )
    tile_file = tile_out / EXPECTED_TILE_DIR / tile_filename(ZOOM, TILE_X, TILE_Y, TILE_YEAR, "100m_wind_speed.processed")
    with nc4.Dataset(str(tile_file)) as ds:
        assert "ws100" in ds.variables


def test_era5_tiler_skip_existing(era5_like_tile_input, tmp_path):
    tile_out = tmp_path / "tiles"
    era5_tiler(source_dir=str(era5_like_tile_input), tile_output_dir=str(tile_out), zoom_level=ZOOM)
    tile_file = tile_out / EXPECTED_TILE_DIR / tile_filename(ZOOM, TILE_X, TILE_Y, TILE_YEAR, "100m_wind_speed.processed")
    mtime_before = tile_file.stat().st_mtime
    era5_tiler(source_dir=str(era5_like_tile_input), tile_output_dir=str(tile_out), zoom_level=ZOOM)
    assert tile_file.stat().st_mtime == mtime_before


def test_era5_tiler_no_raw_variables(era5_like_tile_input, tmp_path):
    tile_out = tmp_path / "tiles"
    era5_tiler(source_dir=str(era5_like_tile_input), tile_output_dir=str(tile_out), zoom_level=ZOOM)
    blh_tile = tile_out / EXPECTED_TILE_DIR / tile_filename(ZOOM, TILE_X, TILE_Y, TILE_YEAR, "boundary_layer_height")
    assert not blh_tile.exists()
