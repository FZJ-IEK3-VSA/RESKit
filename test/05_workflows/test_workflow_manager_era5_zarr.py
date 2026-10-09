from os.path import join

import numpy as np
import pandas as pd
import pytest
from era5_encoding import assert_matches_encoding

pytest.importorskip("zarr")

from reskit import data, WorkflowManager
from reskit.csp.workflows.workflows import csp_ptr_era5, csp_ptr_era5_specific_dataset
from reskit.solar.workflows.workflows import openfield_pv_era5
from reskit.wind.workflows.workflows import wind_era5_PenaSanchezDunkelWinklerEtAl2025

FIXTURES = data.paths("test_suite")
ERA5_ZARR = join(FIXTURES["era5_zarr"], "era5.zarr")


# The 140 hours of the 'era5' netCDF4 fixtures, on RESKit's time index
ERA5_HOURS = slice("2014-12-31 23:30", "2015-01-06 18:30")


@pytest.fixture
def pt_placements() -> pd.DataFrame:
    """The placements used by the netCDF4 ERA5 workflow manager tests."""
    placements = pd.DataFrame()
    placements["lon"] = [6.083, 6.183, 6.083, 6.183, 6.083]
    placements["lat"] = [50.475, 50.575, 50.675, 50.775, 50.875]
    placements["hub_height"] = [140, 140, 140, 140, 140]
    placements["capacity"] = [2000, 3000, 4000, 5000, 6000]
    placements["rotor_diam"] = [136, 136, 136, 136, 136]
    return placements


ERA5_COMPARISON_VARIABLES = [
    "elevated_wind_speed",
    "surface_wind_speed",
    "surface_pressure",
    "surface_air_temperature",
    "surface_dew_temperature",
    "global_horizontal_irradiance",
    "direct_horizontal_irradiance",
    "boundary_layer_height",
]


def _read_era5(placements, source, **kwargs):
    man = WorkflowManager(placements)
    man.read(
        variables=ERA5_COMPARISON_VARIABLES,
        source_type="ERA5",
        source=str(source),
        set_time_index=True,
        verbose=False,
        spatial_interpolation_mode="bilinear",
        temporal_reindex_method="nearest",
        **kwargs,
    )
    return man


def test_era5_netcdf_and_zarr_read_alike(pt_placements):
    """WorkflowManager.read() gives from a real online Zarr store what it gives from the netCDF4 fixtures.

    era5.zarr covers the same box and hours as the 'era5' fixtures. Not to the bit --
    the netCDF4 fixtures are packed to 16 bit integers, the Zarr data is bit-rounded -- but
    up to what both encodings round away, see test/era5_encoding.py. Bilinear interpolation
    averages neighbouring cells, so it cannot make the difference larger.
    """
    netcdf_man = _read_era5(pt_placements, FIXTURES["era5"])
    zarr_man = _read_era5(pt_placements, ERA5_ZARR, time_slice=ERA5_HOURS)

    assert zarr_man.time_index.equals(netcdf_man.time_index)

    for variable in ERA5_COMPARISON_VARIABLES:
        assert_matches_encoding(zarr_man.sim_data[variable], netcdf_man.sim_data[variable], variable)

    assert zarr_man.elevated_wind_speed_height == netcdf_man.elevated_wind_speed_height
    assert zarr_man.surface_wind_speed_height == netcdf_man.surface_wind_speed_height


@pytest.fixture
def era5_zarr_workflow_store(tmp_path):
    import xarray as xr

    times = pd.date_range("2020-01-01 00:00:00", periods=3, freq="h")
    latitudes = np.array([51.25, 51.0, 50.75, 50.5, 50.25, 50.0])
    longitudes = np.array([5.75, 6.0, 6.25, 6.5, 6.75, 7.0])

    t_idx = np.arange(times.size)[:, None, None]
    lat_grid = latitudes[None, :, None]
    lon_grid = longitudes[None, None, :]

    ds = xr.Dataset(
        data_vars={
            "u100": (("time", "latitude", "longitude"), np.full((3, 6, 6), 3.0)),
            "v100": (("time", "latitude", "longitude"), np.full((3, 6, 6), 4.0)),
            "sp": (("time", "latitude", "longitude"), 100000.0 + t_idx + lat_grid + 2.0 * lon_grid),
            "t2m": (("time", "latitude", "longitude"), 273.15 + 10.0 + t_idx + lat_grid + lon_grid),
        },
        coords={
            "valid_time": times,
            "latitude": latitudes,
            "longitude": longitudes,
        },
    )

    store = tmp_path / "era5_workflow.zarr"
    ds.to_zarr(store)
    return store


def test_WorkflowManager_read_era5_zarr(era5_zarr_workflow_store):
    placements = pd.DataFrame(
        {
            "lon": [6.375],
            "lat": [50.625],
            "hub_height": [120.0],
            "capacity": [3000.0],
            "rotor_diam": [150.0],
        }
    )

    man = WorkflowManager(placements)
    man.read(
        variables=["elevated_wind_speed", "surface_pressure", "surface_air_temperature"],
        source_type="ERA5",
        source=str(era5_zarr_workflow_store),
        storage_format="zarr",
        set_time_index=True,
        spatial_interpolation_mode="bilinear",
        verbose=False,
    )

    assert man.time_index[0] == pd.Timestamp("2019-12-31 23:30:00")
    assert np.allclose(man.sim_data["elevated_wind_speed"][:, 0], np.array([5.0, 5.0, 5.0]))
    assert np.allclose(man.sim_data["surface_pressure"][:, 0], np.array([100063.375, 100064.375, 100065.375]))
    assert np.allclose(man.sim_data["surface_air_temperature"][:, 0], np.array([67.0, 68.0, 69.0]))


def test_WorkflowManager_read_era5_zarr_applies_time_slice(era5_zarr_workflow_store):
    placements = pd.DataFrame({"lon": [6.375], "lat": [50.625]})

    man = WorkflowManager(placements)
    man.read(
        variables=["surface_pressure"],
        source_type="ERA5",
        source=str(era5_zarr_workflow_store),
        storage_format="zarr",
        time_slice=slice("2020-01-01 00:30:00", "2020-01-01 01:30:00"),
        set_time_index=True,
        spatial_interpolation_mode="bilinear",
        verbose=False,
    )

    assert man.time_index.tolist() == [
        pd.Timestamp("2020-01-01 00:30:00"),
        pd.Timestamp("2020-01-01 01:30:00"),
    ]


def test_WorkflowManager_reads_placements_far_apart_region_by_region(pt_placements, monkeypatch):
    """Placements in different regions of the store's chunk grid are read per region, with the
    same result as one read: a Zarr source reads the rectangle around its placements, so one read
    for placements on two continents would load everything between them.
    """
    from reskit import weather as rk_weather
    from reskit import workflow_manager

    store = ERA5_ZARR
    together = _read_era5(pt_placements, store)
    # regions of one 0.25 degree cell: the placements, 0.1 degree apart, fall into separate regions
    # (without a minimum region size, which the padding of the reads would otherwise set)
    monkeypatch.setattr(rk_weather.Era5ZarrSource, "spatial_chunk_cells", staticmethod(lambda dataset: (1, 1)))
    zarr_regions = workflow_manager._zarr_regions
    monkeypatch.setattr(
        workflow_manager, "_zarr_regions", lambda locs, dataset, index_pad: zarr_regions(locs, dataset, 0)
    )
    regions = []
    read_by_region = WorkflowManager._read_by_region
    monkeypatch.setattr(
        WorkflowManager,
        "_read_by_region",
        lambda self, constructor, dataset, parts, *args, **kwargs: regions.append(len(parts))
        or read_by_region(self, constructor, dataset, parts, *args, **kwargs),
    )

    apart = _read_era5(pt_placements, store)

    assert regions and regions[0] > 1
    assert apart.time_index.equals(together.time_index)
    assert apart.elevated_wind_speed_height == together.elevated_wind_speed_height
    for var in ERA5_COMPARISON_VARIABLES:
        np.testing.assert_allclose(apart.sim_data[var], together.sim_data[var], rtol=1e-6, err_msg=var)


def test_Era5ZarrSource_spatial_chunk_cells():
    import xarray as xr

    from reskit import weather as rk_weather

    store = xr.open_dataset(ERA5_ZARR, engine="zarr")
    lat_chunk = store["u100"].encoding["chunks"][store["u100"].dims.index("latitude")]
    lon_chunk = store["u100"].encoding["chunks"][store["u100"].dims.index("longitude")]

    assert rk_weather.Era5ZarrSource.spatial_chunk_cells(store) == (lat_chunk, lon_chunk)
    assert rk_weather.Era5ZarrSource.spatial_chunk_cells(store.load().drop_encoding()) == (None, None)


def test_Era5ZarrSource_spatial_chunk_cells_from_dask_chunks():
    """Without the stored chunks in the encoding (e.g. after arithmetic), the dask chunks count."""
    pytest.importorskip("dask")
    import xarray as xr

    from reskit import weather as rk_weather

    store = xr.open_dataset(ERA5_ZARR, engine="zarr").drop_encoding().chunk({"latitude": 4, "longitude": 3})

    assert rk_weather.Era5ZarrSource.spatial_chunk_cells(store) == (4, 3)


def _chunked_grid(latitudes, longitudes, chunks):
    """An in-memory store on the given grid, whose variable reports the given chunk sizes."""
    import xarray as xr

    values = np.zeros((latitudes.size, longitudes.size))
    coords = {"latitude": latitudes, "longitude": longitudes}
    store = xr.Dataset({"u100": (("latitude", "longitude"), values)}, coords=coords)
    store["u100"].encoding["chunks"] = chunks
    return store


def _region_sets(locations, store, index_pad=5):
    """The regions of _zarr_regions() as sets of location positions, independent of their order."""
    import geokit as gk

    from reskit.workflow_manager import _zarr_regions

    regions = _zarr_regions(gk.LocationSet(locations), store, index_pad)
    return {frozenset(region.tolist()) for region in regions}


def test_Era5ZarrSource_spatial_chunk_cells_differ_per_axis():
    from reskit import weather as rk_weather

    latitudes = np.arange(60.0, 40.0, -0.25)
    longitudes = np.arange(0.0, 40.0, 0.25)
    store = _chunked_grid(latitudes, longitudes, chunks=(8, 40))

    assert rk_weather.Era5ZarrSource.spatial_chunk_cells(store) == (8, 40)


def test_zarr_regions_are_counted_from_the_first_grid_point_of_the_store():
    """A regional store starting at 72N, 25W with 60 cell (15 degree) chunks: its chunk edges are
    at 57N and 10W, not at the 15 degree multiples counted from 90S and 180W.
    """
    latitudes = 72.0 - 0.25 * np.arange(100)
    longitudes = -25.0 + 0.25 * np.arange(200)
    store = _chunked_grid(latitudes, longitudes, chunks=(60, 60))

    # the first two are in the first chunk (cells 0..59 along both axes), the third is in the
    # chunk south of it (nearest cell 60 along latitude)
    locations = [(-24.0, 71.9), (-11.0, 57.5), (-11.0, 56.9)]

    assert _region_sets(locations, store) == {frozenset({0, 1}), frozenset({2})}


def test_zarr_regions_wrap_longitudes_around_the_store():
    """A store running from 0 to 360 degrees with 100 cell (25 degree) chunks keeps 10W and 1W in
    its last chunk (350E to 360E), and 11W in the chunk before it.
    """
    latitudes = np.array([50.0, 49.75])
    longitudes = 0.25 * np.arange(1440)
    store = _chunked_grid(latitudes, longitudes, chunks=(2, 100))

    locations = [(-10.0, 50.0), (-1.0, 50.0), (-11.0, 50.0)]

    assert _region_sets(locations, store) == {frozenset({0, 1}), frozenset({2})}


def test_zarr_regions_span_twice_the_padding_of_the_reads():
    """Chunks of one cell, reads padded by 5 cells: a region spans 10 cells (2.5 degrees), as
    smaller regions would read mostly the same chunks as their neighbours.
    """
    latitudes = np.array([50.0, 49.75])
    longitudes = 0.25 * np.arange(40)
    store = _chunked_grid(latitudes, longitudes, chunks=(1, 1))

    # cells 0, 9 and 10 along longitude
    locations = [(0.0, 50.0), (2.25, 50.0), (2.5, 50.0)]

    assert _region_sets(locations, store, index_pad=5) == {frozenset({0, 1}), frozenset({2})}
    assert _region_sets(locations, store, index_pad=0) == {frozenset({0}), frozenset({1}), frozenset({2})}


def test_zarr_regions_do_not_split_data_in_memory():
    """Without known chunk sizes, i.e. for data in memory, all locations are read together."""
    latitudes = np.arange(60.0, -60.0, -0.25)
    longitudes = np.arange(-60.0, 60.0, 0.25)
    store = _chunked_grid(latitudes, longitudes, chunks=None)

    locations = [(-50.0, 50.0), (0.0, 0.0), (50.0, -50.0)]

    assert _region_sets(locations, store) == {frozenset({0, 1, 2})}


def test_zarr_region_index_spans_whole_chunks():
    """Chunks of 3 cells are below a minimum of 4 cells: a region spans two chunks."""
    from reskit.workflow_manager import _zarr_region_index

    latitudes = np.arange(10.0, 0.0, -0.25)
    longitudes = np.arange(0.0, 10.0, 0.25)
    store = _chunked_grid(latitudes, longitudes, chunks=(3, 3))

    # cells 0, 5 and 6 along longitude
    location_longitudes = np.array([0.0, 1.25, 1.5])
    region_index = _zarr_region_index(location_longitudes, store, "longitude", 3, 4, wrap_around_globe=True)

    assert region_index.tolist() == [0, 0, 1]
