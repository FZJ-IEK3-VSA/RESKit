from copy import copy

import geokit as gk
import osgeo
import pandas as pd
import re
from smopy import deg2num
import numpy as np


def get_tile_xy(zoom, lon=None, lat=None, geom=None):
    """
    Returns the X/Y id of the respective tile for a given
    latitude and longitude and tile No.

    Params:
    ======
    zoom : zoom level for tiling
    lon : longitude point to consider, only if geom not given
    lat : latitude point to consider, only if geom not given
    geom : point geom of type osgeo.ogr.Geometry, only if not lat and lon are given, else None

    Return:
    ======
    tuple
        - (X, Y)
    """
    # check zoom level
    if not isinstance(zoom, int):
        raise TypeError(f"zoom must be an int or float")
    elif not zoom > 0:
        raise ValueError(f"zoom must be greater zero")

    # get or assert correctness of lat/lon
    if geom is not None:
        # check geometry details
        assert lon is None and lat is None, f"When geom is not None, lat and lon must be None"
        if not (isinstance(geom, osgeo.ogr.Geometry) and "POINT" in geom.GetGeometryName()):
            raise ValueError(f"geom must be an osgeo.ogr.Geometry point geometry")
        assert geom.GetSpatialReference() is None or geom.GetSpatialReference().IsSame(gk.srs.loadSRS(4326)), (
            f"geom reference system must be EPSG:4326 or None (then EPSG:4326 is assumed)"
        )
        # extract lat and lon
        lon = geom.GetX()
        lat = geom.GetY()
    else:
        assert isinstance(lon, (int, float)) and -180 <= lon <= 180, (
            f"lon must be an integer or float between -180/+180"
        )
        assert isinstance(lat, (int, float)) and -90 <= lat <= 90, f"lat must be an integer or float between -90/+90°"

    # get tile id
    X, Y = deg2num(lat, lon, zoom=zoom)

    # deg_to_num cannot deal with extreme latitudes, set to Y edge tile manually
    if Y < 0:
        print(
            f"Locations (lat={lat}, lon={lon}) below the minimum tile Y-index (0) will be corrected to the lowest available tile index: 0 "
        )
        Y = 0
    elif Y > ((2**zoom) - 1):
        print(
            f"Locations (lat={lat}, lon={lon}) outside the maximum tile Y-index ({Y}) at zoom level {zoom} will be corrected to the outmost available tile index: {zoom**2 - 1} "
        )
        Y = (2**zoom) - 1

    return (X, Y)


def get_tilepath(weather_path, lat=None, lon=None, zoom=None):
    """
    Returns a tilepath with potential <X-TILE> and <Y-TILE> as well 
    as <ZOOM> spacers replaced by the respective values based on 
    latitude, longitude and zoom level.

    weather_path : str
        The base path, may contain '<X-TILE>', '<Y-TILE>' and '<ZOOM>
        spacers which will be replaced.
    lat : int | float | None, optional
        The latitude in degrees, takes effect only if weather_path
        contains spacers. By default None.
    lon : int | float | None, optional
        The longitude in degrees, takes effect only if weather_path
        contains spacers. By default None.
    zoom : int | None, optional
        The zoom level at which the tiling was done, takes effect
        only if weather_path contains spacers. By default None.

    Returns
    -------
        str : weather_path with all spacers replaced by the respective
              values
    """
    if "<X-TILE>" in weather_path or "<Y-TILE>" in weather_path or "<ZOOM>" in weather_path:
        assert isinstance(zoom, int) and zoom > 0, (
            f"zoom must be a positive integer tiling level if weather_path contains X/Y spacers"
        )
        assert isinstance(lat, (int, float, np.number)), (
            f"lat must be a float or integer degree if weather_path contains X/Y spacers"
        )
        assert isinstance(lon, (int, float, np.number)), (
            f"lon must be a float or integer degree if weather_path contains X/Y spacers"
        )
        _X, _Y = get_tile_xy(zoom=zoom, lon=lon, lat=lat, geom=None)
        weather_path = (
            weather_path.replace("<X-TILE>", str(_X)).replace("<Y-TILE>", str(_Y)).replace("<ZOOM>", str(zoom))
        )
    # make sure we got all spacers
    spacers = re.findall(r"<[^>]*>", weather_path)
    if len(spacers) > 0:
        raise ValueError(
            f"weather_path still contains spacer after replacing '<X-TILE>', '<Y-TILE>' and '<ZOOM>': {', '.join(spacers)}"
        )
    return weather_path


def get_location_specific_weather_paths(weather_paths, locs, zoom=None):
    """
    Generate an iterable with one path per location, replacing potential
    spacers with location-specific data.#

    weather_paths : str | list[str]
        A str filepath or a list thereof, with spacers '<X-TILE>',
        '<Y-TILE>' and '<ZOOM> allowed. Length must match the length
        of locs if provided as a list.
    locs : list[tuple] | geokit.LocationSet

    Returns
    -------
        list[str] : List of completed weather paths, specific for and in
        the same order as the locations
    """
    # check inputs
    if isinstance(locs, gk.LocationSet):
        locs = locs._locations
    if isinstance(locs, tuple) or isinstance(locs, str) or not hasattr(locs, "__iter__"):
        raise TypeError(f"weather_paths must be an iterable but not a str or tuple.")
    if isinstance(weather_paths, str):
        weather_paths = [weather_paths] * len(locs)
    elif not isinstance(weather_paths, list):
        raise TypeError("weather_paths must be a list of str if not a str.")
    if not all([isinstance(x, str) for x in weather_paths]):
        raise TypeError("All values in weather_paths must be str.")
    if not len(weather_paths) == len(locs):
        raise ValueError(f"weather_paths and locs must have the same length if weather_paths is given as a list.")

    # define a container for completed weather paths and fill iteratively
    out = []
    for wp, loc in zip(weather_paths, locs):
        if isinstance(loc, tuple):
            # assume we have a (lon, lat) tuple in EPSG:4326
            lon, lat = loc
        elif isinstance(loc, osgeo.ogr.Geometry):
            assert loc.GetGeometryName() == "POINT", (
                f"loc must be a POINT geometry if provided as osgeo.ogr.Geometry, here: {loc.GetGeometryName()}"
            )
            loc = gk.geom.transform(loc, toSRS=4326)
            lon = loc.GetX()
            lat = loc.GetY()
        elif isinstance(loc, gk.Location):
            # is always in EPSG:4326
            lon = loc.lon
            lat = loc.lat
        else:
            raise TypeError(f"Unknown loc type: {type(loc)}")
        # now complete the weather path with location data
        out.append(get_tilepath(weather_path=wp, lat=lat, lon=lon, zoom=zoom))

    return out
