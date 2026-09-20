from pathlib import Path

import geokit as gk
import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("zarr")

from reskit.util import ResError
from reskit.weather import Era5ZarrSource


def make_dataset(data_vars, times=None, latitudes=None, longitudes=None, extra_coords=None) -> xr.Dataset:
    """An in-memory store, which Era5ZarrSource accepts in place of a path"""
    times = pd.date_range("2020-01-01 00:00:00", periods=2, freq="h") if times is None else times
    latitudes = np.array([50.5, 50.25, 50.0]) if latitudes is None else latitudes
    longitudes = np.array([6.0, 6.25, 6.5]) if longitudes is None else longitudes

    coords = {"valid_time": times, "latitude": latitudes, "longitude": longitudes}
    coords.update(extra_coords or {})
    return xr.Dataset(data_vars=data_vars, coords=coords)


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
    expected = pd.Series([4, 13], index=src.time_index, name="(-0.1, 50.25)", dtype=out.dtype)
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

    # On-the-fly processing: adj[i] = raw[i-1] / 3600. At the actual
    # beginning of a store, no preceding accumulation is available.
    assert np.isnan(pt_Era5ZarrSource.data["global_horizontal_irradiance"][0, 4, 2])
    assert np.isnan(pt_Era5ZarrSource.data["direct_horizontal_irradiance"][0, 4, 2])
    # At step 1 the value equals raw step 0 / 3600
    # raw ssrd[0, lat=51.0, lon=6.25] = 500 + 0 + 51.0 + 6.25 = 557.25
    assert np.allclose(pt_Era5ZarrSource.data["global_horizontal_irradiance"][1, 4, 2], 557.25 / 3600)
    # raw fdir[0, lat=51.0, lon=6.25] = 200 + 0 + 51.0 + 6.25 = 257.25
    assert np.allclose(pt_Era5ZarrSource.data["direct_horizontal_irradiance"][1, 4, 2], 257.25 / 3600)


def test_Era5ZarrSource_solar_fallback_preserves_slice_boundary(era5_zarr_store):
    src = Era5ZarrSource(
        str(era5_zarr_store),
        time_slice=slice("2020-01-01 00:30:00", "2020-01-01 01:30:00"),
        verbose=False,
    )

    with pytest.warns(UserWarning, match="computing on the fly from raw 'ssrd'"):
        src.sload("global_horizontal_irradiance")

    assert src.time_index.tolist() == [
        pd.Timestamp("2020-01-01 00:30:00"),
        pd.Timestamp("2020-01-01 01:30:00"),
    ]
    # The first requested value comes from raw step 0, outside the selected
    # output range, rather than being replaced with zero.
    assert np.allclose(src.data["global_horizontal_irradiance"][0, 4, 2], 557.25 / 3600)


@pytest.mark.parametrize("time_index_from", [None, "direct_horizontal_irradiance", "elevated_wind_speed"])
def test_Era5ZarrSource_time_index_from_does_not_shift_data(era5_zarr_store, time_index_from):
    """'time_index_from' selects the reference variable, it must not move the time convention."""
    src = Era5ZarrSource(str(era5_zarr_store), time_index_from=time_index_from, verbose=False)

    src.sload("surface_pressure")
    with pytest.warns(UserWarning, match="computing on the fly from raw 'fdir'"):
        src.sload("direct_horizontal_irradiance")

    # ERA5 time convention of RESKit: store timestamp - 30 minutes, independent of the reference variable
    assert src.time_index.tolist() == [
        pd.Timestamp("2019-12-31 23:30:00"),
        pd.Timestamp("2020-01-01 00:30:00"),
        pd.Timestamp("2020-01-01 01:30:00"),
    ]
    # instantaneous variable: sp[valid_time=1, lat=51.0, lon=6.25] = 100000 + 1 + 51.0 + 12.5
    assert np.allclose(src.data["surface_pressure"][1, 4, 2], 100064.5)
    # accumulated variable: adj[i] = raw[i-1] / 3600
    assert np.allclose(src.data["direct_horizontal_irradiance"][1, 4, 2], 257.25 / 3600)


def test_Era5ZarrSource_marks_derived_variables(pt_Era5ZarrSource):
    variables = pt_Era5ZarrSource.variables

    assert variables.loc["ssrd_t_adj", "derived_from"] == "ssrd"
    assert variables.loc["fdir_t_adj", "derived_from"] == "fdir"
    assert pd.isna(variables.loc["ssrd", "derived_from"])
    assert pd.isna(variables.loc["sp", "derived_from"])


def test_Era5ZarrSource_warns_on_nan_first_timestep(pt_Era5ZarrSource):
    with pytest.warns(UserWarning, match="first timestep of 'global_horizontal_irradiance'"):
        pt_Era5ZarrSource.sload("global_horizontal_irradiance")

    assert np.isnan(pt_Era5ZarrSource.data["global_horizontal_irradiance"][0, 4, 2])


def test_Era5ZarrSource_missing_variable_raises(era5_zarr_store):
    ds = xr.open_zarr(str(era5_zarr_store)).drop_vars("ssrd")
    src = Era5ZarrSource(ds, verbose=False)

    with pytest.raises(RuntimeError, match="neither 'ssrd_t_adj' nor 'ssrd'"):
        src.sload("global_horizontal_irradiance")


def test_Era5ZarrSource_rejects_unexpected_kwargs(era5_zarr_store):
    with pytest.raises(TypeError, match="Unexpected keyword arguments"):
        Era5ZarrSource(str(era5_zarr_store), verbose=False, nonsense=1, other=2)


def test_Era5ZarrSource_unknown_time_index_from(era5_zarr_store):
    with pytest.raises(ResError, match="not known"):
        Era5ZarrSource(str(era5_zarr_store), time_index_from="not_a_variable", verbose=False)


def test_Era5ZarrSource_time_slice_as_timestamp_list(era5_zarr_store):
    """Anything but a slice is passed on to xarray as is, i.e. in the convention of the store"""
    src = Era5ZarrSource(
        str(era5_zarr_store),
        time_slice=[pd.Timestamp("2020-01-01 01:00:00")],
        verbose=False,
    )

    assert src.time_index.tolist() == [pd.Timestamp("2020-01-01 00:30:00")]


def test_Era5ZarrSource_verbose_output(era5_zarr_store, capsys):
    src = Era5ZarrSource(str(era5_zarr_store), verbose=True)
    printed = capsys.readouterr().out
    assert "ERA5 Zarr time range: 2019-12-31 23:30:00 to 2020-01-01 01:30:00 (3 time steps)" in printed
    assert "Opened ERA5 Zarr source" in printed

    src.load("sp")
    assert "Loaded ERA5 Zarr variable 'sp' as 'sp' with shape (3, 6, 6)" in capsys.readouterr().out


def test_Era5ZarrSource_verbose_output_on_empty_time_slice(era5_zarr_store, capsys):
    Era5ZarrSource(
        str(era5_zarr_store),
        time_slice=slice("2021-01-01 00:30:00", "2021-01-02 00:30:00"),
        verbose=True,
    )

    assert "empty, the requested 'time_slice' selects no time steps" in capsys.readouterr().out


def test_Era5ZarrSource_rejects_flattened_values_grid():
    ds = xr.Dataset(
        data_vars={"sp": (("valid_time", "values"), np.zeros((2, 4)))},
        coords={
            "valid_time": pd.date_range("2020-01-01 00:00:00", periods=2, freq="h"),
            "values": np.arange(4),
        },
    )

    with pytest.raises(ResError, match="regular latitude/longitude Zarr stores"):
        Era5ZarrSource(ds, verbose=False)


def test_Era5ZarrSource_requires_a_known_time_dimension():
    ds = xr.Dataset(
        data_vars={"sp": (("step", "latitude", "longitude"), np.zeros((2, 3, 3)))},
        coords={
            "step": np.arange(2),
            "valid_time": pd.date_range("2020-01-01 00:00:00", periods=2, freq="h"),
            "latitude": np.array([50.5, 50.25, 50.0]),
            "longitude": np.array([6.0, 6.25, 6.5]),
        },
    )

    with pytest.raises(ResError, match="'valid_time' or 'time' dimension"):
        Era5ZarrSource(ds, verbose=False)


def test_Era5ZarrSource_requires_a_datetime_coordinate():
    ds = make_dataset(
        data_vars={"sp": (("valid_time", "latitude", "longitude"), np.zeros((2, 3, 3)))},
        times=np.arange(2),
    )

    with pytest.raises(ResError, match="datetime 'valid_time' or 'time' coordinate"):
        Era5ZarrSource(ds, verbose=False)


def test_Era5ZarrSource_takes_datetimes_from_the_other_time_name():
    """Stores exist where the data dimension and the datetime coordinate use different names"""
    ds = make_dataset(
        data_vars={"sp": (("valid_time", "latitude", "longitude"), np.zeros((2, 3, 3)))},
        times=np.arange(2),
        extra_coords={"time": ("valid_time", pd.date_range("2020-01-01 00:00:00", periods=2, freq="h"))},
    )

    src = Era5ZarrSource(ds, verbose=False)

    assert src.time_index.tolist() == [
        pd.Timestamp("2019-12-31 23:30:00"),
        pd.Timestamp("2020-01-01 00:30:00"),
    ]


@pytest.mark.parametrize(
    "source, expected_storage_options",
    [
        ("https://edh.example/era5.zarr", {"client_kwargs": {"trust_env": True}}),
        ("gs://bucket/era5.zarr", {"token": "anon"}),
        ("/local/path/era5.zarr", None),
        (Path("/local/path/era5.zarr"), None),  # non-string stores are passed on untouched
    ],
)
def test_Era5ZarrSource_open_dataset_protocol_defaults(monkeypatch, source, expected_storage_options):
    captured = {}

    def fake_open_dataset(source, **kwargs):
        captured["source"] = source
        captured.update(kwargs)
        return "opened"

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

    assert Era5ZarrSource._open_dataset(source=source, chunks=None, consolidated=True, storage_options=None) == "opened"
    assert captured["source"] == source
    assert captured["engine"] == "zarr"
    assert captured["storage_options"] == expected_storage_options


def test_Era5ZarrSource_open_dataset_passes_datasets_through():
    ds = make_dataset(data_vars={"sp": (("valid_time", "latitude", "longitude"), np.zeros((2, 3, 3)))})

    assert Era5ZarrSource._open_dataset(source=ds, chunks=None, consolidated=True, storage_options=None) is ds


def test_Era5ZarrSource_wrap_longitudes(pt_Era5ZarrSource):
    # the store of the fixture uses a [-180, 180) grid, so nothing is wrapped
    assert np.allclose(pt_Era5ZarrSource._wrap_longitudes(np.array([-0.1, 6.0])), np.array([-0.1, 6.0]))

    pt_Era5ZarrSource._longitude_360 = True
    assert np.allclose(pt_Era5ZarrSource._wrap_longitudes(np.array([-0.1, 6.0])), np.array([359.9, 6.0]))


def test_Era5ZarrSource_axis_flipping(pt_Era5ZarrSource):
    """The latitude axis of ERA5 is flipped to be ascending, the longitude axis is not"""
    src = pt_Era5ZarrSource
    src.sload("surface_pressure")
    assert np.allclose(src.lats, np.array([50.0, 50.25, 50.5, 50.75, 51.0, 51.25]))
    assert np.allclose(src.lons, np.array([5.75, 6.0, 6.25, 6.5, 6.75, 7.0]))

    # a store which needs the opposite treatment of both axes
    src._flip_lat = False
    src._flip_lon = True
    src._configure_spatial_selection()
    src.load("sp", name="flipped_pressure")

    assert np.allclose(src.lats, np.array([51.25, 51.0, 50.75, 50.5, 50.25, 50.0]))
    assert np.allclose(src.lons, np.array([7.0, 6.75, 6.5, 6.25, 6.0, 5.75]))
    assert np.allclose(src.data["flipped_pressure"], src.data["surface_pressure"][:, ::-1, ::-1])


def test_Era5ZarrSource_var_info(pt_Era5ZarrSource, capsys):
    pt_Era5ZarrSource.var_info("sp")
    assert "sp" in capsys.readouterr().out

    with pytest.raises(AssertionError):
        pt_Era5ZarrSource.var_info("not_a_variable")


def test_Era5ZarrSource_load_defaults_to_the_variable_name(pt_Era5ZarrSource):
    pt_Era5ZarrSource.load("sp")
    assert "sp" in pt_Era5ZarrSource.data

    # an already loaded variable is not read again unless requested
    loaded = pt_Era5ZarrSource.data["sp"]
    pt_Era5ZarrSource.load("sp")
    assert pt_Era5ZarrSource.data["sp"] is loaded
    pt_Era5ZarrSource.load("sp", overwrite=True)
    assert pt_Era5ZarrSource.data["sp"] is not loaded


def test_Era5ZarrSource_load_unknown_variable(pt_Era5ZarrSource):
    with pytest.raises(ResError, match="not found in ERA5 Zarr store"):
        pt_Era5ZarrSource.load("not_a_variable")


def test_Era5ZarrSource_load_with_height_index():
    ds = make_dataset(
        data_vars={
            "u": (("valid_time", "height", "latitude", "longitude"), np.arange(2 * 2 * 3 * 3).reshape(2, 2, 3, 3)),
            "sp": (("valid_time", "latitude", "longitude"), np.zeros((2, 3, 3))),
        },
        extra_coords={"height": np.array([10.0, 100.0])},
    )
    src = Era5ZarrSource(ds, verbose=False)

    src.load("u", name="wind", height_idx=1)
    assert src.data["wind"].shape == (2, 3, 3)
    assert np.allclose(src.data["wind"], np.asarray(ds["u"].isel(height=1).values)[:, ::-1, :])

    with pytest.raises(ResError, match="does not have a height dimension"):
        src.load("sp", height_idx=0)


@pytest.mark.parametrize("dims", [("valid_time", "longitude"), ("valid_time", "latitude")])
def test_Era5ZarrSource_load_rejects_unexpected_dimensions(dims):
    ds = make_dataset(
        data_vars={
            "odd": (dims, np.zeros((2, 3))),
            "sp": (("valid_time", "latitude", "longitude"), np.zeros((2, 3, 3))),
        }
    )
    src = Era5ZarrSource(ds, verbose=False)

    with pytest.raises(ResError, match="is expected to have dimensions"):
        src.load("odd")


def test_Era5ZarrSource_forward_fills_a_single_missing_last_step(era5_zarr_store):
    src = Era5ZarrSource(str(era5_zarr_store), verbose=False)
    src._timeindex_raw = src._timeindex_raw.append(pd.DatetimeIndex([src._timeindex_raw[-1] + pd.Timedelta("1h")]))
    src.time_index = src._timeindex_raw

    src.load("sp")

    assert src.data["sp"].shape[0] == src.time_index.size
    assert np.allclose(src.data["sp"][-1], src.data["sp"][-2])


def test_Era5ZarrSource_rejects_larger_time_mismatches(era5_zarr_store):
    src = Era5ZarrSource(str(era5_zarr_store), verbose=False)
    src._timeindex_raw = src._timeindex_raw.append(
        pd.DatetimeIndex([src._timeindex_raw[-1] + pd.Timedelta(hours=hour) for hour in (1, 2)])
    )

    with pytest.raises(ResError, match="Filling is only intended to fill the last missing step"):
        src.load("sp")


def test_Era5ZarrSource_time_mismatch_without_forward_fill(era5_zarr_store):
    src = Era5ZarrSource(str(era5_zarr_store), forward_fill=False, verbose=False)
    src._timeindex_raw = src._timeindex_raw.append(pd.DatetimeIndex([src._timeindex_raw[-1] + pd.Timedelta("1h")]))

    with pytest.raises(ResError, match="Time mismatch with variable sp"):
        src.load("sp")


def test_Era5ZarrSource_get_data_frame_on_360_grid(tmp_path):
    times = pd.date_range("2020-01-01 00:00:00", periods=2, freq="h")
    ds = make_dataset(
        data_vars={"sp": (("valid_time", "latitude", "longitude"), np.arange(18).reshape(2, 3, 3).astype(float))},
        times=times,
        longitudes=np.array([359.75, 0.0, 0.25]),
    )

    store = tmp_path / "era5_wrap_frame.zarr"
    ds.to_zarr(store)
    src = Era5ZarrSource(str(store), verbose=False)
    src.sload("surface_pressure")

    out = src.get("surface_pressure", [(-0.1, 50.25), (-0.15, 50.0)], interpolation="near", force_as_data_frame=True)

    assert isinstance(out, pd.DataFrame)
    # the columns keep the labels of the given, unwrapped locations
    assert list(out.columns) == ["(-0.1, 50.25)", "(-0.15, 50.0)"]
    assert np.allclose(out["(-0.1, 50.25)"].values, [4, 13])
    assert np.allclose(out["(-0.15, 50.0)"].values, [6, 15])
