import geokit as gk
import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("zarr")

from reskit.weather import Era5ZarrSource


@pytest.fixture
def era5_zarr_store(tmp_path):
    times = pd.date_range("2020-01-01 00:00:00", periods=3, freq="h")
    latitudes = np.array([51.25, 51.0, 50.75, 50.5, 50.25, 50.0])
    longitudes = np.array([5.75, 6.0, 6.25, 6.5, 6.75, 7.0])

    t_idx = np.arange(times.size)[:, None, None]
    lat_grid = latitudes[None, :, None]
    lon_grid = longitudes[None, None, :]

    ds = xr.Dataset(
        data_vars={
            "u100": (("valid_time", "latitude", "longitude"), np.full((3, 6, 6), 3.0)),
            "v100": (("valid_time", "latitude", "longitude"), np.full((3, 6, 6), 4.0)),
            "u10": (("valid_time", "latitude", "longitude"), np.zeros((3, 6, 6))),
            "v10": (("valid_time", "latitude", "longitude"), np.full((3, 6, 6), 2.0)),
            "sp": (("valid_time", "latitude", "longitude"), 100000.0 + t_idx + lat_grid + 2.0 * lon_grid),
            "t2m": (("valid_time", "latitude", "longitude"), 273.15 + 10.0 + t_idx + lat_grid + lon_grid),
            "d2m": (("valid_time", "latitude", "longitude"), 273.15 + 2.0 + t_idx + lat_grid + lon_grid),
            "ssrd": (("valid_time", "latitude", "longitude"), 500.0 + t_idx + lat_grid + lon_grid),
            "fdir": (("valid_time", "latitude", "longitude"), 200.0 + t_idx + lat_grid + lon_grid),
        },
        coords={
            "valid_time": times,
            "latitude": latitudes,
            "longitude": longitudes,
        },
    )

    store = tmp_path / "era5_like.zarr"
    ds.to_zarr(store)
    return store


@pytest.fixture
def pt_Era5ZarrSource(era5_zarr_store):
    return Era5ZarrSource(str(era5_zarr_store), verbose=False)


def test_Era5ZarrSource___init__(era5_zarr_store):
    src = Era5ZarrSource(str(era5_zarr_store), verbose=False)

    assert np.allclose(src.lats, np.array([50.0, 50.25, 50.5, 50.75, 51.0, 51.25]))
    assert np.allclose(src.lons, np.array([5.75, 6.0, 6.25, 6.5, 6.75, 7.0]))
    assert src.time_index[0] == pd.Timestamp("2019-12-31 23:30:00")
    assert src.time_index[-1] == pd.Timestamp("2020-01-01 01:30:00")

    bounded = Era5ZarrSource(
        str(era5_zarr_store),
        bounds=gk.Extent(6.2, 50.45, 6.55, 50.8, srs=gk.srs.EPSG4326),
        index_pad=0,
        verbose=False,
    )
    assert bounded.lats.min() >= 50.0
    assert bounded.lats.max() <= 51.25
    assert bounded.lats.size < src.lats.size
    assert bounded.lons.size < src.lons.size


def test_Era5ZarrSource_wraps_360_longitudes(tmp_path):
    times = pd.date_range("2020-01-01 00:00:00", periods=2, freq="h")
    latitudes = np.array([50.5, 50.25, 50.0])
    longitudes = np.array([359.75, 0.0, 0.25])

    ds = xr.Dataset(
        data_vars={
            "sp": (("valid_time", "latitude", "longitude"), np.arange(18).reshape(2, 3, 3)),
        },
        coords={
            "valid_time": times,
            "latitude": latitudes,
            "longitude": longitudes,
        },
    )

    store = tmp_path / "era5_wrap.zarr"
    ds.to_zarr(store)
    src = Era5ZarrSource(str(store), verbose=False)
    src.sload("surface_pressure")

    out = src.get("surface_pressure", [(-0.1, 50.25)], interpolation="near")
    expected = pd.Series([4, 13], index=src.time_index, name="(-0.1, 50.25)")
    pd.testing.assert_series_equal(out, expected)


def test_Era5ZarrSource_standard_loaders(pt_Era5ZarrSource):
    pt_Era5ZarrSource.sload(
        "surface_pressure",
        "surface_air_temperature",
        "surface_dew_temperature",
        "surface_wind_speed",
        "elevated_wind_speed",
    )

    assert np.allclose(pt_Era5ZarrSource.data["surface_pressure"][0, 4, 2], 100063.5)
    assert np.allclose(pt_Era5ZarrSource.data["surface_air_temperature"][0, 4, 2], 67.25)
    assert np.allclose(pt_Era5ZarrSource.data["surface_dew_temperature"][0, 4, 2], 59.25)
    assert np.allclose(pt_Era5ZarrSource.data["surface_wind_speed"], 2.0)
    assert np.allclose(pt_Era5ZarrSource.data["elevated_wind_speed"], 5.0)


def test_Era5ZarrSource_get_bilinear(pt_Era5ZarrSource):
    pt_Era5ZarrSource.sload("surface_pressure")

    out = pt_Era5ZarrSource.get(
        "surface_pressure",
        [(6.375, 50.625)],
        interpolation="bilinear",
    )

    expected = pd.Series(
        [100063.375, 100064.375, 100065.375],
        index=pt_Era5ZarrSource.time_index,
        name="(6.375, 50.625)",
    )
    pd.testing.assert_series_equal(out, expected)


def test_Era5ZarrSource_solar_fallbacks(pt_Era5ZarrSource):
    with pytest.warns(UserWarning, match="computing on the fly from raw 'ssrd'"):
        pt_Era5ZarrSource.sload("global_horizontal_irradiance")
    with pytest.warns(UserWarning, match="computing on the fly from raw 'fdir'"):
        pt_Era5ZarrSource.sload("direct_horizontal_irradiance")

    # On-the-fly processing: adj[i] = raw[i-1] / 3600, adj[0] = 0
    # At step 0 the first value is filled with 0
    assert np.allclose(pt_Era5ZarrSource.data["global_horizontal_irradiance"][0, 4, 2], 0.0)
    assert np.allclose(pt_Era5ZarrSource.data["direct_horizontal_irradiance"][0, 4, 2], 0.0)
    # At step 1 the value equals raw step 0 / 3600
    # raw ssrd[0, lat=51.0, lon=6.25] = 500 + 0 + 51.0 + 6.25 = 557.25
    assert np.allclose(pt_Era5ZarrSource.data["global_horizontal_irradiance"][1, 4, 2], 557.25 / 3600)
    # raw fdir[0, lat=51.0, lon=6.25] = 200 + 0 + 51.0 + 6.25 = 257.25
    assert np.allclose(pt_Era5ZarrSource.data["direct_horizontal_irradiance"][1, 4, 2], 257.25 / 3600)


def test_Era5ZarrSource_missing_variable_raises(era5_zarr_store):
    ds = xr.open_zarr(str(era5_zarr_store)).drop_vars("ssrd")
    src = Era5ZarrSource(ds, verbose=False)

    with pytest.raises(RuntimeError, match="neither 'ssrd_t_adj' nor 'ssrd'"):
        src.sload("global_horizontal_irradiance")
