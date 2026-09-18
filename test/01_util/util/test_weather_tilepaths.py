import geokit as gk
import numpy as np
import pytest

from reskit import TEST_DATA
from reskit.util.weather_tile import (
    get_tile_xy,
    get_location_specific_weather_paths,
)


def test_weather_tilepaths():
    era5_path = f".../<ZOOM>/<X-TILE>/<Y-TILE>/2015"

    df = gk.vector.extractFeatures(TEST_DATA["turbinePlacements.shp"])
    df["hub_height"] = np.linspace(100, 130, df.shape[0])
    df["capacity"] = 3000
    df["rotor_diam"] = 170
    df.loc[::2, "rotor_diam"] = 150
    df_base = df.copy()

    tile_paths = get_location_specific_weather_paths(
        weather_paths=era5_path,
        locs=df["geom"],
        zoom=4,
    )
    assert tile_paths[0] == ".../4/8/5/2015"

    # test with iterable of geometry objects instead of dataframe
    # tile_paths = output.source.unique()
    tile_paths = get_location_specific_weather_paths(
        weather_paths=era5_path,
        locs=list(df.geom),
        zoom=4,
    )
    assert tile_paths[0] == ".../4/8/5/2015"

    # test with iterable of lat/lon tuples
    df = df_base.copy()
    df["lon"] = df.apply(lambda x: x.geom.GetX(), axis=1)
    df["lat"] = df.apply(lambda x: x.geom.GetY(), axis=1)
    tile_paths = get_location_specific_weather_paths(
        weather_paths=era5_path,
        locs=list(zip(df.lon, df.lat)),
        zoom=4,
    )
    assert tile_paths[0] == ".../4/8/5/2015"

    # test with dataframe yet without weather path
    df = df_base.copy()
    df["lon"] = df.apply(lambda x: x.geom.GetX(), axis=1)
    df["lat"] = df.apply(lambda x: x.geom.GetY(), axis=1)
    with pytest.raises(TypeError) as e:
        # must not work without "source" column
        tile_paths = get_location_specific_weather_paths(
            weather_paths=None,
            locs=list(zip(df.lon, df.lat)),
            zoom=4,
        )


def test_get_tile_XY():
    df = gk.vector.extractFeatures(TEST_DATA["turbinePlacements.shp"])

    # test geom
    X, Y = get_tile_xy(zoom=4, lon=None, lat=None, geom=df.geom[0])
    assert (X, Y) == (8, 5)
    # test lat/lon
    X, Y = get_tile_xy(zoom=4, lon=df.geom[0].GetX(), lat=df.geom[0].GetY(), geom=None)
    assert (X, Y) == (8, 5)


def test_get_location_specific_weather_paths():
    fps = get_location_specific_weather_paths(
        weather_paths="my/path/<ZOOM>/X<X-TILE>/Y<Y-TILE>/myfile.nc",
        locs=gk.LocationSet(
            [
                gk.geom.point(7.0, 51.0, srs=4326),
                gk.geom.point(121.0, 58.0, srs=4326),
            ]
        ),
        zoom=17,
    )
    assert fps == [
        "my/path/17/X68084/Y43879/myfile.nc",
        "my/path/17/X109590/Y39477/myfile.nc",
    ]
