import numpy as np
import pandas as pd
import geokit as gk
import glob
import os
from functools import lru_cache
from typing import Optional, Literal
import math
import warnings

from reskit.util.errors import ResError
from reskit.util.paths import as_path_string

#: The collection in ``reskit/data/collections.yaml`` that names the rasters read
#: here: ``water_depth`` for :func:`water_depth_from_location`, ``coast_distance``
#: for :func:`distance_to_coastline`.
COLLECTION = "offshore_siting"


@lru_cache(maxsize=None)
def catalogued_input(handle: str) -> str:
    """
    The raster the ``offshore_siting`` collection names as ``handle``, fetched.

    This is what the two functions below read when they are called without a
    raster path. The path comes from ``reskit.data.paths(COLLECTION)``, which
    fetches the collection's full variant from the ETHOS.Data catalogue on first
    use -- both rasters, several gigabytes -- and answers from the shared cache
    afterwards. It is resolved once per process. Pass a raster path to the
    functions instead to read a private copy, or configure one for the dataset
    with ``ethos-data config set-root``.

    Parameters
    ----------
    handle : str
        ``"water_depth"`` or ``"coast_distance"``.

    Returns
    -------
    str
        Absolute path of the raster file, or of the directory of raster tiles.
    """
    from reskit import data  # local import: ethos_data is needed on this route only

    try:
        return as_path_string(data.paths(COLLECTION)[handle])
    except Exception as error:
        raise ResError(
            f"No raster was given and the default could not be provided. The default is the "
            f"{handle!r} input of the {COLLECTION!r} collection in reskit/data/collections.yaml, "
            f"resolved through the ETHOS.Data catalogue: {error}"
        ) from error


def water_depth_from_location(
    latitude: int | float,
    longitude: int | float,
    waterDepthFilePath: Optional[str] = None,
    consider_only: Literal[False, "negative", "positive"] = False,
):
    """
    Returns the water depth (in meters) at a given geographic location.

    Parameters
    ----------
    latitude : int|float
        Latitude in decimal degrees.
    longitude : int|float
        Longitude in decimal degrees.
    waterDepthFilePath : str or pathlib.Path, optional
       Path or pattern to one or more GeoTIFF water depth files: a single file, a
       directory of tiles, or a pattern with wildcards such as '*'. By default the
       ``water_depth`` input of the ``offshore_siting`` collection is read from the
       ETHOS.Data catalogue (see :func:`catalogued_input`); its ``test`` variant,
       ``reskit.data.paths("offshore_siting", test=True)["water_depth"]``, is a
       small fixture covering part of the German Bight. Relevant files can be
       downloaded from https://www.gebco.net/data-products/gridded-bathymetry-data.
    consider_only : {False, "negative", "positive"}, default False
        Controls how to interpret the sign of the source raster values:
            - False: legacy behavior — return abs(resultDepth) if a value is found.
            - "negative": treat water depth as NEGATIVE below sea level (e.g., GEBCO).
                        Positive values (land) are returned as 0.0. Negative values
                        are returned as their magnitude (e.g., -7.3 -> 7.3).
            - "positive": treat water depth as POSITIVE below sea level (inverted rasters).
                        Negative values (land/invalid) are returned as 0.0. Positive
                        values are returned as-is.

    Returns
    -------
    float or None
        Water depth at the specified location in meters (always positive). Returns None if not found.
    """
    if waterDepthFilePath is None:
        waterDepthFilePath = catalogued_input("water_depth")
    waterDepthFilePath = as_path_string(waterDepthFilePath)

    if os.path.isdir(waterDepthFilePath):
        candidates = sorted(glob.glob(os.path.join(waterDepthFilePath, "*.tif")))
    else:
        candidates = sorted(glob.glob(waterDepthFilePath))

    if not candidates:
        raise ValueError(f"No .tif files found for path or pattern: {waterDepthFilePath}")

    # geokit.raster.interpolateValues expects a single file path, use the first match
    resultDepth = None

    for source_file in candidates:
        with warnings.catch_warnings():
            # A point outside a raster tile is expected while searching
            # through multiple candidate files.
            warnings.filterwarnings(
                "ignore",
                message=r".*exceed/s the source's limits.*",
                category=UserWarning,
            )

            candidateDepth = gk.raster.interpolateValues(
                source=source_file,
                points=(longitude, latitude),
                pointSRS=gk.srs.EPSG4326,
            )

        if candidateDepth is None:
            continue

        candidateDepth = float(candidateDepth)

        if math.isnan(candidateDepth):
            continue

        resultDepth = candidateDepth
        break

    if resultDepth is None:
        return None

    val = resultDepth

    if consider_only == "negative":
        # GEBCO-like: sea depths are negative, land elevations positive
        if val >= 0.0:
            return 0.0
        return abs(val)  # make depth positive
    elif consider_only == "positive":
        # Inverted: sea depths are positive, land/invalid negative
        if val <= 0.0:
            return 0.0
        return val  # already positive depth
    else:
        # Legacy behavior for backward compatibility
        return abs(val)


# if you want to execute the distance to coastline more often, please separete the loading of the taserband to increase execution time


def distance_to_coastline(latitude, longitude, distancetoCoastFilePath=None):
    """
    Computes the distance to the coastline from a given geographic point.

    Parameters
    ----------
    latitude : float
        Latitude in decimal degrees.
    longitude : float
        Longitude in decimal degrees.
    distancetoCoastFilePath : str or pathlib.Path, optional
        File path to the distance-to-coast raster. By default the ``coast_distance``
        input of the ``offshore_siting`` collection is read from the ETHOS.Data
        catalogue (see :func:`catalogued_input`); its ``test`` variant,
        ``reskit.data.paths("offshore_siting", test=True)["coast_distance"]``, is a
        small fixture covering part of the German Bight. The full grid is documented at
        https://oceancolor.gsfc.nasa.gov/resources/docs/distfromcoast/


    Returns
    -------
    float or None
        Distance in kilometers, or None if the point is out of bounds or an error occurs.
    """
    if distancetoCoastFilePath is None:
        distancetoCoastFilePath = catalogued_input("coast_distance")
    distancetoCoastFilePath = as_path_string(distancetoCoastFilePath)

    try:
        value = gk.raster.interpolateValues(distancetoCoastFilePath, (longitude, latitude))

        return value

    except Exception as e:
        print(f"Error at Lat: {latitude}, Lon: {longitude}: {e}")
    return None
