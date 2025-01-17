import geokit as gk
import osgeo
import pandas as pd
from smopy import deg2num
from copy import copy


def get_tile_XY(zoom, lon=None, lat=None, geom=None):
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
        assert (
            lon is None and lat is None
        ), f"When geom is not None, lat and lon must be None"
        if not (
            isinstance(geom, osgeo.ogr.Geometry) and "POINT" in geom.GetGeometryName()
        ):
            raise ValueError(f"geom must be an osgeo.ogr.Geometry point geometry")
        assert geom.GetSpatialReference() is None or geom.GetSpatialReference().IsSame(
            gk.srs.loadSRS(4326)
        ), f"geom reference system must be EPSG:4326 or None (then EPSG:4326 is assumed)"
        # extract lat and lon
        lon = geom.GetX()
        lat = geom.GetY()
    else:
        assert (
            isinstance(lon, (int, float)) and -360 <= lon <= 360
        ), f"lon must be an integer or float between -360/+360"
        assert (
            isinstance(lat, (int, float)) and -180 <= lon <= 180
        ), f"lat must be an integer or float between -180/+180°"

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


def get_dataframe_with_weather_tilepaths(placements, weather_path, zoom):
    """
    This method will generate a dataframe from a list of input placements
    and add the link to the corresponding weather data tile in a new dataframe
    column 'source'.

    placements : pd.DataFrame with 'geom' column and osgeo.ogr.Geometry point
    objects in EPSG:4326 or 'lat' and 'lon' columns with degrees in EPSG:4326.
    weather_path : The path to the tilepath or a dummy path containing '<X-TILE>'
    and '<Y-TILE>', optionally also <ZOOM> as spacers. These will be replaced by
    the actual tile ID in x and y direction, plus zoom value if applicable.
    weather_path can also be None only if 'source' is an existing attribute of the
    placements dataframe, the column values will be assumed as existing filepaths
    of weather data tiles then.
    in placements dataframe or if no '<ZOOM>' spacer in weather_path.
    """
    if not isinstance(placements, pd.DataFrame):
        if not hasattr(placements, "__iter__"):
            raise TypeError(
                f"If placements is not a pd.DataFrame, it must be an iterable."
            )
        # we definitely have an iterator at hand, unpack and check of what types
        if all(
            [
                isinstance(x, tuple)
                and all([isinstance(latlon, (float, int)) for latlon in x])
                for x in placements
            ]
        ):
            # we have lon and lat values at hand, generate two lon/lat columns
            lons, lats = zip(*placements)
            placements = pd.DataFrame()
            placements["lon"] = lons
            placements["lat"] = lats
        elif all([isinstance(x, osgeo.ogr.Geometry) for x in placements]):
            assert all(
                [
                    x.GetSpatialReference() is None
                    or x.GetSpatialReference().IsSame(gk.srs.loadSRS(4326))
                    for x in placements
                ]
            ), f"All srs of objects in placements must be EPSG:4326"
            assert all(
                ["POINT" in x.GetGeometryName() for x in placements]
            ), f"All geometries must be POINT features."
            # we have geometries, create a geom column and extract lat/lon
            _placements = copy(placements)
            placements = pd.DataFrame()
            placements["geom"] = _placements
            placements["lon"] = placements["geom"].apply(lambda x: x.GetX())
            placements["lat"] = placements["geom"].apply(lambda x: x.GetY())
        else:
            raise TypeError(
                f"If placements is an iterator, it must contain either (lon, lat) tuples or osgeo.ogr.Geometry point geometries in EPS:4326."
            )
    else:
        # we have a df, make sure the necessary data is available and add lat/lon where needed
        assert "geom" in placements.columns or all(
            [c in placements.columns for c in ["lon", "lat"]]
        ), f"pd.DataFrame must contain 'geom' or 'lat' and 'lon' columns."
        assert (
            not "RESKit_sim_order" in placements.columns
        ), f"placements must not have 'RESKit_sim_order' attribute"
        if not "lon" in placements.columns:
            placements["lon"] = placements.geom.apply(lambda x: x.GetX())
        if not "lat" in placements.columns:
            placements["lat"] = placements.geom.apply(lambda x: x.GetY())

    # get the actual weather tilepath
    def _get_tilepath(weather_path, zoom, lat, lon):
        if "<X-TILE>" in weather_path or "<Y-TILE>" in weather_path:
            assert isinstance(
                zoom, int
            ), f"zoom must be a positive integer tiling level if weather_path contains X/Y spacers"
            _X, _Y = get_tile_XY(zoom=zoom, lon=lon, lat=lat, geom=None)
            return (
                weather_path.replace("<X-TILE>", str(_X))
                .replace("<Y-TILE>", str(_Y))
                .replace("<ZOOM>", str(zoom))
            )
        else:
            return weather_path

    if weather_path is None:
        # the info must already be in the dataframe then
        assert (
            "source" in placements.columns
        ), f"weather_path is None yet no 'source' attribute in placements dataframe."
        if any(
            [
                _str in _fp
                for _str in ["<X-TILE>", "<Y-TILE>", "<ZOOM>"]
                for _fp in placements.source
            ]
        ):
            # overwrite source attributes with specific tilepaths
            print(
                f"NOTE: 'source' attributes will be overwritten with specific filepath!"
            )
            placements["source"] = placements.apply(
                lambda x: _get_tilepath(
                    weather_path=x.source, zoom=zoom, lon=x.lon, lat=x.lat
                ).replace("<ZOOM>", str(zoom)),
                axis=1,
            )
    else:
        # make sure we have no source column to avoid overwriting data
        assert (
            not "source" in placements.columns
        ), f"If weather_path is given, placements must not have a 'source' attribute already"

        # add source column with the actual tile filepaths
        placements["source"] = placements.apply(
            lambda x: _get_tilepath(
                weather_path=weather_path, zoom=zoom, lon=x.lon, lat=x.lat
            ),
            axis=1,
        )

    # add an id column to ensure correct order preservation
    # placements["RESKit_sim_order"] = range(len(placements)) #TODO remove

    return placements
