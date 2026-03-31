import numpy as np
import pandas as pd
import pytest

pytest.importorskip("zarr")

from reskit import WorkflowManager


def test_WorkflowManager_read_era5_zarr(tmp_path):
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
        source=str(store),
        storage_format="zarr",
        set_time_index=True,
        spatial_interpolation_mode="bilinear",
        verbose=False,
    )

    assert man.time_index[0] == pd.Timestamp("2019-12-31 23:30:00")
    assert np.allclose(man.sim_data["elevated_wind_speed"][:, 0], np.array([5.0, 5.0, 5.0]))
    assert np.allclose(man.sim_data["surface_pressure"][:, 0], np.array([100063.375, 100064.375, 100065.375]))
    assert np.allclose(man.sim_data["surface_air_temperature"][:, 0], np.array([67.0, 68.0, 69.0]))
