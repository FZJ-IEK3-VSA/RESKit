import numpy as np
import pandas as pd
import geokit as gk
from reskit.default_paths import DEFAULT_PATHS
import glob
import os
from typing import Optional, Literal

from reskit.util.errors import ResError


def waterDepthFromLocation(
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
    waterDepthFilePath : str, optional
       Path or pattern to one or more GeoTIFF water depth files.
       Support wildcards such as '*'.
       Relevant files can be downloaded from https://www.gebco.net/data-products/gridded-bathymetry-data.
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
        if "waterDepthFilePath" not in DEFAULT_PATHS:
            raise KeyError("Add 'waterDepthFilePath' key with filepath value to default_path.yaml")

        waterDepthFilePath = DEFAULT_PATHS["waterDepthFilePath"]

        if not waterDepthFilePath:
            raise ValueError("No waterDepthFilePath provided or found in default_path.yaml.")

    if os.path.isdir(waterDepthFilePath):
        candidates = sorted(glob.glob(os.path.join(waterDepthFilePath, "*.tif")))
    else:
        candidates = sorted(glob.glob(waterDepthFilePath))

    if not candidates:
        raise ValueError(f"No .tif files found for path or pattern: {waterDepthFilePath}")

    # geokit.raster.interpolateValues expects a single file path, use the first match
    source_file = candidates[0]

    resultDepth = gk.raster.interpolateValues(
        source=source_file,
        points=(longitude, latitude),
    )

    if resultDepth is None:
        return None

    val = float(resultDepth)

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


# %% function to calculate the distance to the coastline
# if you want to execute the distance to coastline more often, please separete the loading of the taserband to increase execution time


def distanceToCoastline(latitude, longitude, distancetoCoastFilePath=None):
    """
    Computes the distance to the coastline from a given geographic point.

    Parameters
    ----------
    latitude : float
        Latitude in decimal degrees.
    longitude : float
        Longitude in decimal degrees.
    distancetoCoastFilePath : str, optional
        File path to the distance-to-coast raster. Loaded from defaults if not specified.
        Relevant files can be downloaded from https://oceancolor.gsfc.nasa.gov/resources/docs/distfromcoast/


    Returns
    -------
    float or None
        Distance in kilometers, or None if the point is out of bounds or an error occurs.
    """
    if distancetoCoastFilePath is None:
        if not "distancetoCoastPath" in DEFAULT_PATHS:
            raise KeyError(f"Add 'distancetoCoastFilePath' key with filepath value to default_paths.yaml")
        distancetoCoastFilePath = DEFAULT_PATHS.get("distancetoCoastPath")
        if distancetoCoastFilePath is None:
            raise ValueError("No distaneFilePath is given. Please add it to default_path.yaml.")

    try:
        value = gk.raster.interpolateValues(distancetoCoastFilePath, (longitude, latitude))

        return value

    except Exception as e:
        print(f"Error at Lat: {latitude}, Lon: {longitude}: {e}")
    return None
