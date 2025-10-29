import numpy as np
import geokit as gk
from os.path import isfile
from collections.abc import Iterable

from reskit.util import ResError


def location_to_module_azimuth(
    locs: gk.LocationSet | Iterable, convention: str = "NorthSouth", **kwargs
):
    """
    Simple module surface azimuth estimator based off latitude coordinates.

    Parameters
    ----------
    locs : geokit.LocationSet or iterable of (lon,lat) pairs
        The locations at which to estimate module azimuth angle

    convention : str, optional
        The calculation method used to suggest module surface azimuth angles.
        * "NorthSouth" will assign south-facing modules to the
          Northern hemisphere and vice versa.
        * A path to a raster file from which the location specific
          azimuth (in clockwise degree starting North) is extracted

    kwargs:
        Will be forwarded to geokit.raster.interpolateValues(), only applies
        when `convention` is a path to a raster file.

    Returns
    -------
    np.ndarray
        Suggested module azimuth at each of the provided `locs`. Has the same
        length as the number of `locs`.
    """
    locs = gk.LocationSet(locs)
    if convention == "NorthSouth":
        # assign 0° (north-facing) to Southern hemisphere and 180° to Northern hemisphere
        modazimuths = np.array([180 if loc.lat >= 0 else 0 for loc in locs])
    elif isinstance(convention, str) and isfile(convention):
        # try to extract data from raster
        try:
            modazimuths = gk.raster.interpolateValues(convention, locs, **kwargs)
        except Exception:
            raise OSError(
                f"File cannot be read by gk.raster.interpolateValues(): {convention}."
            )
    else:
        raise ValueError(f"Unknown module azimuth convention '{convention}'.")

    return modazimuths


def location_to_module_tilt(locs, convention: str = "Ryberg2020", **kwargs):
    """
    def location_to_tilt(locs, convention="Ryberg2020", **kwargs)

    Simple system tilt estimator based off latitude and longitude coordinates


    Parameters
    ----------
    locs : geokit.LocationSet or iterable of (lon,lat) pairs
           The locations at which to estimate system tilt angle

    convention : str, optional
                 The calculation method used to suggest system tilts
                 Options are:
                     * "Ryberg2020"
                     * A string consumable by 'eval'
                     - Can use the variable 'latitude'
                     - Ex. "latitude*0.76"
                     * A path to a raster file

    kwargs: Optional keyword arguments to use in geokit.raster.interpolateValues(...).
            Only applies when `convention` is a path to a raster file


    Returns
    -------
    np.ndarray
        Suggested tilt angle at each of the provided `locs`.
        Has the same length as the number of `locs`.

    Notes
    -----
    "Ryberg2020"
        When `convention` equals "Ryberg2020", the following equation is followed:

        .. math:: 42.327719357601396 * arctan( 1.5 * abs(latitude) )

    .. [1] TODO: Cite future Ryberg2020 publication

    """
    locs = gk.LocationSet(locs)

    if convention == "Ryberg2020":
        tilt = 42.327719357601396 * np.arctan(1.5 * np.radians(np.abs(locs.lats)))

    elif isfile(convention):
        tilt = gk.raster.interpolateValues(convention, locs, **kwargs)

    else:
        try:
            tilt = eval(convention, {}, {"latitude": locs.lats})
        except Exception:
            raise ResError("Failed to apply tilt convention")

    return tilt


def location_to_tracker_axis_azimuth(locs, convention:str="North", **kwargs):
    """
    Simple azimuth estimator for the tracker axis in single-axis tracking 
    systems based off latitude coordinates.

    Parameters
    ----------
    locs : geokit.LocationSet or iterable of (lon,lat) pairs
        The locations at which to estimate module azimuth angle

    convention : str, optional
        The calculation method used to suggest module surface azimuth angles.
        * "North" will assign a north-facing azimuth to all locations (typical 
          North-South running axes orientation for single-axis tracking systems)
        * A path to a raster file from which the location specific
          azimuth (in clockwise degree starting North) is extracted

    kwargs: 
        Will be forwarded to geokit.raster.interpolateValues(), only applies 
        when `convention` is a path to a raster file.

    Returns
    -------
    np.ndarray
        Suggested axis azimuth at each of the provided `locs`. Has the same 
        length as the number of `locs`.
    """
    locs = gk.LocationSet(locs)
    if convention == "North":
        # assign 0° (north-facing) to all locs 
        axazimuths = np.full((len(locs), ), 0)
    elif isinstance(convention, str) and isfile(convention):
        # try to extract data from raster
        try:
            axazimuths = gk.raster.interpolateValues(convention, locs, **kwargs)
        except Exception:
            raise OSError(f"File cannot be read by gk.raster.interpolateValues(): {convention}.")
    else:  
        raise ValueError(f"Unknown axis azimuth convention '{convention}'.")
    
    return axazimuths


def location_to_tracker_axis_tilt(locs, convention:str="flat", **kwargs):
    """
    Simple tilt estimator for the tracker axis in single-axis tracking systems 
    based off latitude coordinates. 

    Parameters
    ----------
    locs : geokit.LocationSet or iterable of (lon,lat) pairs
        The locations at which to estimate module azimuth angle

    convention : str, optional #TODO update docstr, seems to be still azimuth
        The calculation method used to suggest tracker axis tilt angles.
        * "flat" will assign a 0° axis tilt to all locations 
        * A path to a raster file from which the location specific axis 
          tilt (in clockwise degree starting North) is extracted

    kwargs: 
        Will be forwarded to geokit.raster.interpolateValues(), only applies 
        when `convention` is a path to a raster file.

    Returns
    -------
    np.ndarray
        Suggested axis tilt at each of the provided `locs`. Has the same 
        length as the number of `locs`.
    """
    locs = gk.LocationSet(locs)
    if convention == "flat":
        # assign 0° slope to all locs 
        axtilts = np.full((len(locs), ), 0)
    elif isinstance(convention, str) and isfile(convention):
        # try to extract data from raster
        try:
            axtilts = gk.raster.interpolateValues(convention, locs, **kwargs)
        except Exception:
            raise OSError(f"Axis tilt file cannot be read by gk.raster.interpolateValues(): {convention}.")
    else:  
        raise ValueError(f"Unknown axis tilt convention '{convention}'.")
    
    return axtilts


def location_to_cross_axis_tilt(locs, convention:str="flat", **kwargs):
    """
    Simple estimator for the cross axis slope in single-axis tracking 
    systems based off latitude coordinates.

    Parameters
    ----------
    locs : geokit.LocationSet or iterable of (lon,lat) pairs
        The locations at which to estimate module tilt angle

    convention : str, optional
        The calculation method used to suggest cross axis tilt angles.
        * "flat" will assign a 0° cross axis tilt to all locations 
        * A path to a raster file from which the location specific
          cross-axis tilt is extracted

    kwargs: 
        Will be forwarded to geokit.raster.interpolateValues(), only applies 
        when `convention` is a path to a raster file.

    Returns
    -------
    np.ndarray
        Estimated cross axis tilt at each of the provided `locs`. Has the same 
        length as the number of `locs`.
    """
    locs = gk.LocationSet(locs)
    if convention == "flat":
        # assign 0° to all locs 
        caxtilts = np.full((len(locs), ), 0)
    elif isinstance(convention, str) and isfile(convention):
        # try to extract data from raster
        try:
            caxtilts = gk.raster.interpolateValues(convention, locs, **kwargs)
        except Exception:
            raise OSError(f"File cannot be read by gk.raster.interpolateValues(): {convention}.")
    else:  
        raise ValueError(f"Unknown cross axis tilt convention '{convention}'.")
    
    return caxtilts