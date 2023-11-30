import numpy as np
from reskit.util.weather_tile import get_dataframe_with_weather_tilepaths
from reskit import TEST_DATA
import geokit as gk


def test_weather_tilepaths():
    era5_path = f".../4/<X-TILE>/<Y-TILE>/2015"

    df = gk.vector.extractFeatures(TEST_DATA["turbinePlacements.shp"])
    df["hub_height"] = np.linspace(100, 130, df.shape[0])
    df["capacity"] = 3000
    df["rotor_diam"] = 170
    df.loc[::2, "rotor_diam"] = 150

    output = get_dataframe_with_weather_tilepaths(
        placements=df,
        weather_path=era5_path,
        zoom=4,
    )
    tile_paths = output.source.unique()[0]
    assert tile_paths == ".../4/8/5/2015"
