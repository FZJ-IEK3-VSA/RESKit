import numpy as np
from reskit.util.weather_tile import get_dataframe_with_weather_tilepaths, get_tile_XY
from reskit import TEST_DATA
import geokit as gk
import pytest


def test_weather_tilepaths():
    era5_path = f".../<ZOOM>/<X-TILE>/<Y-TILE>/2015"

    df = gk.vector.extractFeatures(TEST_DATA["turbinePlacements.shp"])
    df["hub_height"] = np.linspace(100, 130, df.shape[0])
    df["capacity"] = 3000
    df["rotor_diam"] = 170
    df.loc[::2, "rotor_diam"] = 150
    df_base = df.copy()

    output = get_dataframe_with_weather_tilepaths(
        placements=df,
        weather_path=era5_path,
        zoom=4,
    )
    tile_paths = output.source.unique()[0]
    assert tile_paths == ".../4/8/5/2015"

    # test with iterable of geometry objects instead of dataframe
    output = get_dataframe_with_weather_tilepaths(
        placements=list(df.geom),
        weather_path=era5_path,
        zoom=4,
    )
    tile_paths = output.source.unique()[0]
    assert tile_paths == ".../4/8/5/2015"

    # test with iterable of lat/lon tuples
    df = df_base.copy()
    df["lon"] = df.apply(lambda x: x.geom.GetX(), axis=1)
    df["lat"] = df.apply(lambda x: x.geom.GetY(), axis=1)
    output = get_dataframe_with_weather_tilepaths(
        placements=list(zip(df.lon, df.lat)),
        weather_path=era5_path,
        zoom=4,
    )
    tile_paths = output.source.unique()[0]
    assert tile_paths == ".../4/8/5/2015"

    # test with dataframe yet without weather path
    df = df_base.copy()
    df["lon"] = df.apply(lambda x: x.geom.GetX(), axis=1)
    df["lat"] = df.apply(lambda x: x.geom.GetY(), axis=1)
    with pytest.raises(AssertionError) as e:
        # must not work without "source" column
        output = get_dataframe_with_weather_tilepaths(
            placements=df,
            weather_path=None,
            zoom=4,
        )
    df["source"] = era5_path
    output = get_dataframe_with_weather_tilepaths(
        placements=df,
        weather_path=None,
        zoom=4,
    )
    tile_paths = output.source.unique()[0]
    assert tile_paths == ".../4/8/5/2015"


def test_get_tile_XY():
    df = gk.vector.extractFeatures(TEST_DATA["turbinePlacements.shp"])

    # test geom
    X, Y = get_tile_XY(zoom=4, lon=None, lat=None, geom=df.geom[0])
    assert (X, Y) == (8, 5)
    # test lat/lon
    X, Y = get_tile_XY(zoom=4, lon=df.geom[0].GetX(), lat=df.geom[0].GetY(), geom=None)
    assert (X, Y) == (8, 5)
