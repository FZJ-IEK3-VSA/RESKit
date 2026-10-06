import inspect

import numpy as np
import pandas as pd
import pytest
from era5_encoding import assert_matches_encoding

pytest.importorskip("zarr")

from reskit import TEST_DATA, WorkflowManager
from reskit.csp.workflows.workflows import csp_ptr_era5, csp_ptr_era5_specific_dataset
from reskit.solar.workflows.workflows import openfield_pv_era5
from reskit.wind.workflows.workflows import wind_era5_PenaSanchezDunkelWinklerEtAl2025


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
    netcdf_man = _read_era5(pt_placements, TEST_DATA["era5-like"])
    zarr_man = _read_era5(pt_placements, TEST_DATA["era5.zarr"], time_slice=ERA5_HOURS)

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


@pytest.mark.parametrize(
    "workflow",
    [openfield_pv_era5, csp_ptr_era5, csp_ptr_era5_specific_dataset, wind_era5_PenaSanchezDunkelWinklerEtAl2025],
)
def test_era5_workflows_expose_time_slice(workflow):
    assert "time_slice" in inspect.signature(workflow).parameters


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


def test_WorkflowManager_read_era5_netcdf_rejects_time_slice():
    placements = pd.DataFrame({"lon": [6.375], "lat": [50.625]})

    man = WorkflowManager(placements)
    with pytest.raises(RuntimeError, match="only supported for Zarr-backed ERA5 sources"):
        man.read(
            variables=["surface_pressure"],
            source_type="ERA5",
            source="does_not_need_to_exist.nc",
            time_slice=slice("2020-01-01", "2020-01-02"),
            set_time_index=True,
            verbose=False,
        )


def test_WorkflowManager_reads_placements_far_apart_region_by_region(pt_placements, monkeypatch):
    """Placements in different regions of the store's chunk grid are read per region, with the
    same result as one read: a Zarr source reads the rectangle around its placements, so one read
    for placements on two continents would load everything between them.
    """
    from reskit import weather as rk_weather

    store = TEST_DATA["era5.zarr"]
    together = _read_era5(pt_placements, store)
    # 0.1 degree regions: the placements, 0.1 degree apart, fall into separate regions
    monkeypatch.setattr(rk_weather.Era5ZarrSource, "spatial_chunk_degrees", staticmethod(lambda dataset: 0.1))
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


def test_Era5ZarrSource_spatial_chunk_degrees():
    import xarray as xr

    from reskit import weather as rk_weather

    store = xr.open_dataset(TEST_DATA["era5.zarr"], engine="zarr")
    lat_chunk = store["u100"].encoding["chunks"][store["u100"].dims.index("latitude")]

    assert rk_weather.Era5ZarrSource.spatial_chunk_degrees(store) == pytest.approx(max(1.0, lat_chunk * 0.25))
    assert rk_weather.Era5ZarrSource.spatial_chunk_degrees(store.load().drop_encoding()) is None
