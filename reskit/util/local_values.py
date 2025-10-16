import numpy as np
import pandas as pd
import geokit as gk
from reskit.default_paths import DEFAULT_PATHS

from . import ResError


# %%


def waterDepthFromLocation(
    latitude,
    longitude,
    waterDepthFolderPath=None,
):
    """
    Returns the water depth (in meters) at a given geographic location.

    Parameters
    ----------
    latitude : float
        Latitude in decimal degrees.
    longitude : float
        Longitude in decimal degrees.
    waterDepthFolderPath : str, optional
        Path to the folder containing water depth .tif files. Loaded from defaults if not specified.

    Returns
    -------
    float or None
        Water depth at the specified location in meters (always positive). Returns None if not found.
    """

    if waterDepthFolderPath is None:
        waterDepthFolderPath = DEFAULT_PATHS.get("waterdepthFile")
        if waterDepthFolderPath is None:
            raise ValueError(
                "No waterDepthFilePath is given. Please add it to default_path.yaml."
            )

    depthFiles = glob.glob(os.path.join(waterDepthFolderPath, "*.tif"))
    resultDepth = gk.raster.interpolateValues(
        source=depthFiles, points=(longitude, latitude)
    )

    return abs(resultDepth) if resultDepth is not None else None


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

    Returns
    -------
    float or None
        Distance in kilometers, or None if the point is out of bounds or an error occurs.
    """
    if distancetoCoastFilePath is None:
        distancetoCoastFilePath = DEFAULT_PATHS.get("distancetoCoast")
        if distancetoCoastFilePath is None:
            raise ValueError(
                "No distaneFilePath is given. Please add it to default_path.yaml."
            )

    try:
        value = gk.raster.interpolateValues(distancetoCoastFilePath, (longitude, latitude))

        return value

    except Exception as e:
        print(f"Error at Lat: {latitude}, Lon: {longitude}: {e}")
    return None

