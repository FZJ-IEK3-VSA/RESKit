import numpy as np
import geokit as gk
import pandas as pd
from os.path import isfile
from collections.abc import Iterable
from types import NoneType
import warnings

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


def location_to_module_tilt(
        locs, 
        convention: str = "Ryberg2020", 
        **kwargs
        ):
    """
    Estimates module tilt off location-specific arguments for selected "convention options.


    Parameters
    ----------
    locs : geokit.LocationSet or iterable of (lon,lat) pairs
           The locations at which to estimate system tilt angle

    convention : str, optional
        The calculation method used to suggest system tilts. Options are:
        * "Ryberg2020"
            Calculates tilt based on latitude, developed for Europe by Ryberg et al. [1]
            Equation: 42.327719357601396 * arctan( 1.5 * abs(latitude) ), accepts no kwargs.
        * A path to a raster file
            Must be an existing file readable by geokit.raster.interpolateValues(),
            will then geospatiallyextract the tilt directly from a raster file.
            kwargs will be passed on to geokit.raster.interpolateValues().

    kwargs: Optional keyword arguments for the respective convention core function.


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

    if not isinstance(convention, str):
        raise TypeError(f"convention is expected to be a str: {convention}")

    if convention == "Ryberg2020":
        assert not kwargs, f"No keyword arguments accepted for convention 'Ryberg2020': {kwargs}"
        tilt = 42.327719357601396 * np.arctan(1.5 * np.radians(np.abs(locs.lats)))

    elif isfile(convention):
        try:
            tilt = gk.raster.interpolateValues(convention, locs, **kwargs)
        except Exception as e:
            raise ResError(f"convention must be readable by geokit.raster.interpolateValues() if an existing filepath is given, here: '{convention}'.\n{e}")
    else:
        raise ResError(f"Unknown convention (or non-existing file) for location_to_module_tilt(): '{convention}'")

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


def location_to_tracker_axis_tilt(locs, convention:str="flat", fallback:int|float=None, **kwargs):
    """
    Simple tilt estimator for the tracker axis in single-axis tracking systems 
    based off latitude coordinates. 

    Parameters
    ----------
    locs : geokit.LocationSet or iterable of (lon,lat) pairs
        The locations at which to estimate module azimuth angle#TODO update

    convention : str, optional #TODO update docstr, seems to be still azimuth
        The calculation method used to suggest tracker axis tilt angles.
        * "flat" will assign a 0° axis tilt to all locations 
        * A path to a raster file from which the location specific axis 
          tilt (in clockwise degree starting North) is extracted

    fallback : int | float, optional
        Will replace possible NaN values in the axis tilt iterable after 
        application of the main function if given. By default None, i.e. no effect.

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
            axtilts = np.atleast_1d(gk.raster.interpolateValues(convention, locs, **kwargs))
        except Exception:
            raise OSError(f"Axis tilt file cannot be read by gk.raster.interpolateValues(): {convention}.")
    else:  
        raise ValueError(f"Unknown axis tilt convention '{convention}'.")
    
    if fallback is not None:
        axtilts[np.isnan(axtilts)] = fallback
    
    return axtilts


def location_to_cross_axis_tilt(locs, convention:str="flat", fallback:int|float=None, **kwargs):
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

    fallback : int | float, optional
        Will replace possible NaN values in the cross-axis tilt iterable after 
        application of the main function if given. By default None, i.e. no effect.

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
            caxtilts = np.atleast_1d(gk.raster.interpolateValues(convention, locs, **kwargs))
        except Exception:
            raise OSError(f"File cannot be read by gk.raster.interpolateValues(): {convention}.")
    else:  
        raise ValueError(f"Unknown cross axis tilt convention '{convention}'.")
    
    if fallback is not None:
        caxtilts[np.isnan(caxtilts)] = fallback

    return caxtilts

def location_to_gcr_tonita_et_al_2023(
        lat : int | float, 
        bifaciality_factor : int | float,
        tracking : str,
        shading_loss : float,
        ):
    """
    Returns the optimal Ground Coverage Ratio for a horizontal 
    single-axis tracking (HSAT) plant based on the results 
    by Tonita et al. (2023). For details see [1].

    Parameters
    ----------
    lat : float | int
        The latitude of the plant.
    bifaciality_factor : float
        Tonita et al. provide a mono- and a bifacial equation, 
        set bifaciality_factor to their defaults of 0.0 or 0.96
        to get their exact results. Note that other bifaciality 
        factors will lead to a simplified, interpolated GCR!
    tracking : str
        "fixed", "singleaxis" or "vertical" for fixed tilt or 
        horizontal single axis tracking or vertical systems.
    shading_loss : float, optional
        Select the accepted annual  energy yield loss due to 
        shading, Tonita et al. offer 5-15% (0.05, 0.1 and 0.15).
    
    Return
    ------
    float : Optimal ground coverage ratio allowing the specified shading loss

    References
    ----------
    [1] Tonita, Russel, Validivia, Hinzer (2023): Optimal ground coverage ratios 
        for tracked, fixed-tilt, and vertical photovoltaic systems for latitudes 
        up to 75◦N, https://doi.org/10.1016/j.solener.2023.04.038
    """
    # check if all inputs are scalar to return a scalar again below
    scalar_input = all(np.ndim(x) == 0 for x in [lat, bifaciality_factor, tracking, shading_loss])

    # broadcast scalar inputs to the length of the longest iterable
    inputs = [lat, bifaciality_factor, tracking, shading_loss]
    lengths = [1 if np.ndim(x) == 0 else len(x) for x in inputs]
    n = max(lengths)

    def _broadcast(x):
        x = np.asarray(x)
        if x.ndim == 0:
            return np.full(n, x.item())
        if len(x) == 1:
            return np.full(n, x[0])
        if len(x) != n:
            raise ValueError(
                f"All iterable inputs must have length 1 or {n}, here: {len(x)}."
            )
        return x

    lat = _broadcast(lat)
    bifaciality_factor = _broadcast(bifaciality_factor)
    tracking = _broadcast(tracking)
    shading_loss = _broadcast(shading_loss)

    # check inputs
    if not np.issubdtype(bifaciality_factor.dtype, np.number):
        raise TypeError("bifaciality_factor must be int or float.")
    if np.any((bifaciality_factor < 0) | (bifaciality_factor > 1)):
        raise ValueError(f"bifaciality_factor must be between 0 and 1.0, here: {bifaciality_factor}")
    if not np.issubdtype(lat.dtype, np.number):
        raise TypeError("lat must be int or float.")
    if np.any((lat < -90) | (lat > 90)):
        raise ValueError(f"lat must be between -90° and +90°, here: {lat}")

    # first define the individual paramaters for every case
    params = {
        "singleaxis": {
            0.05 : {
                "bi" : (-2.68E-3, 0.361),
                "mono" : (-2.82E-3, 0.388),
            },
            0.10 : {
                "bi" : (-4.37E-3, 0.575),
                "mono" : (-4.76E-3, 0.621),
            },
            0.15 : {
                "bi" : (-5.76E-3, 0.762),
                "mono" : (-6.33E-3, 0.825),
            }
        },
        "vertical": {
            0.05 : {
                "bi" : (-2.68E-3, 0.361),
                "mono" : (-2.82E-3, 0.388),
            },
            0.10 : {
                "bi" : (-4.37E-3, 0.575),
                "mono" : (-4.76E-3, 0.621),
            },
            0.15 : {
                "bi" : (-5.76E-3, 0.762),
                "mono" : (-6.33E-3, 0.825),
            }
        },
        "fixed": {
            0.05 : {
                "bi" : (-0.560, 0.133, 40.2, 0.70),
                "mono" : (-0.550, 0.138, 43.4, 0.71)
            },
            0.10 : {
                "bi" : (-0.485, 0.171, 46.2, 0.72),
                "mono" : (-0.441, 0.198, 48.7, 0.72),
            },
            0.15 : {
                "bi" : (-0.414, 0.207, 49.9, 0.74),
                "mono" : (-0.371, 0.208, 51.5, 0.75)
            }
        }
    }

    gcrmono = np.empty(lat.shape, dtype=float)
    gcrbifac = np.empty(lat.shape, dtype=float)

    # calculate only actually occurring tracking and shading loss combinations
    for _tracking in np.unique(tracking):
        if _tracking not in params:
            raise ValueError(f"Unknown tracking type '{_tracking}'.")

        tracking_mask = tracking == _tracking

        for _shading_loss in np.unique(shading_loss[tracking_mask]):
            # make sure we have parameters for this shading loss
            if _shading_loss not in params[_tracking]:
                raise KeyError(
                    f"shading_loss = {_shading_loss} is not defined for tracking = '{_tracking}', "
                    f"choose from: {params[_tracking].keys()}"
                )

            mask = tracking_mask & (shading_loss == _shading_loss)

            if _tracking in ["singleaxis", "vertical"]:
                # define the gcr getter for this tracking type and shading loss
                a, b = params[_tracking][_shading_loss]["mono"]
                gcrmono[mask] = a * np.abs(lat[mask]) + b

                a, b = params[_tracking][_shading_loss]["bi"]
                gcrbifac[mask] = a * np.abs(lat[mask]) + b

            elif _tracking == "fixed":
                P, k, a0, g0 = params[_tracking][_shading_loss]["mono"]
                gcrmono[mask] = P/(1+np.exp(-k*(np.abs(lat[mask])-a0)))+g0

                P, k, a0, g0 = params[_tracking][_shading_loss]["bi"]
                gcrbifac[mask] = P/(1+np.exp(-k*(np.abs(lat[mask])-a0)))+g0

    # get mono- and bifacial gcr based on absolute lat to account for Southern hemisphere
    # interpolate them based on the actrual bifaciality factor
    gcrinterp = gcrmono + (gcrbifac - gcrmono) * (bifaciality_factor - 0)/(0.96 - 0)

    if scalar_input:
        return gcrinterp.item()
    return gcrinterp


def location_to_gcr(
        convention: str, 
        module_tilt: int | float | Iterable = None,
        north_slope : int | float | str | Iterable = 0,
        east_slope : int | float | str | Iterable = 0,
        bifaciality_factor : float = None,
        min_gcr : float | NoneType = 0.3,
        **kwargs):
    """
    Estimates optimal gcr off latitude based on a given convention and tracking 
    system. Optional global horizontal irradiance and slope data in tracker and 
    cross axis direction improve accuracy for single-axis tracking. Assumes a 
    North-South-facing azimuth for single-axis tracker axes.

    Parameters
    ----------
    locs : geokit.LocationSet or iterable of (lon,lat) pairs
        The locations at which to estimate module azimuth angle
    tracking : str
        If the system is 'fixed' tilt or 'singleaxis' tracking.
    convention : str, optional
        The calculation method used to suggest module surface azimuth angles.
        Available conventions for single-axis tracking:
        * "winter_solstice_rule" will assign the gcr based on the latitude 
          and possibly North-facing slope for fixed tilt pv facing the equator 
          applicable to equator-facing fixed tilt pv parks.
        * 'tonita_et_al_2023_5perc' will assign the optimal GCR under 5% loss 
          assumption according to the publication by Tonita et al. [1]
        * A path to a raster file from which the location specific
          azimuth (in clockwise degree starting North) is extracted
    module_tilt : int | float | Iterable, optional
        Module tilt angle from ground, mandatory when tracking = 'fixed'.
        By default None.
    north_slope : int | float | str | Iterable, optional
        The north-facing slope angle in degrees, if given as str, a filepath
        with slope raster is expected. By default 0.
    east_slope : int | float | str | Iterable, optional
        The east-facing slope angle in degrees, if given as str, a filepath
        with slope raster is expected. By default 0.
    bifaciality_factor : float | Iterable, optional
        The bifaciality factor of the module as float from 0.0-1.0.
        Is mandatory for some conventions such as 'tonita_et_al_2023_5perc'.
        By default None.
    min_gcr : float | NoneType, optional
        If given as a float, GCR values will be limited to this minimum value.
        Has no effect if None, by default 0.3.
    kwargs: 
        Will be forwarded to geokit.raster.interpolateValues(), only applies 
        when `convention` is a path to a raster file.

    Returns
    -------
    np.ndarray
        Suggested axis azimuth at each of the provided `locs`. Has the same 
        length as the number of `locs`.
    
    References
    ----------
    [1] Tonita et al. (2023): "Optimal ground coverage ratios for tracked, 
        fixed-tilt, and vertical photovoltaic systems for latitudes up to 75°N"
        DOI 10.1016/j.solener.2023.04.038
    """
    locs = gk.LocationSet(locs)
    if min_gcr is not None:
        assert isinstance(min_gcr, float) and 0 <= min_gcr <= 1.0, \
            f"min_gcr must be a float >= 0 and <= 1.0 if not None, here: {min_gcr}."
    
    # first check if we have a given raster from which we only need to extract the gcrs
    if isinstance(convention, str) and isfile(convention):
        # try to extract data from raster
        try:
            gcrs = gk.raster.interpolateValues(convention, locs, **kwargs)
            # apply min gcr
            if min_gcr is not None:
                gcrs[gcrs < min_gcr] = min_gcr
            return gcrs
        except Exception:
            raise OSError(f"File cannot be read by gk.raster.interpolateValues(): {convention}.")
        
    # first try to extract the slopes
    if isinstance(north_slope, str):
        # assume a slope raster
        assert isfile(north_slope), f"north_slope is expected to be an existing tif file if given as str: {north_slope}"
        # try to extract data from raster
        try:
            north_slope = gk.raster.interpolateValues(north_slope, locs, **kwargs)
        except Exception:
            raise OSError(f"north_slope file cannot be read by gk.raster.interpolateValues(): {north_slope}.")
    if isinstance(east_slope, str):
        # assume a slope raster
        assert isfile(east_slope), f"east_slope is expected to be an existing tif file if given as str: {east_slope}"
        # try to extract data from raster
        try:
            east_slope = gk.raster.interpolateValues(east_slope, locs, **kwargs)
        except Exception:
            raise OSError(f"east_slope file cannot be read by gk.raster.interpolateValues(): {east_slope}.")
        assert not pd.isnull(east_slope).any(), \
            "east_slope raster contains NaN values for at least one location."
        
    if tracking == "fixed":
        # a different set of conventions applies for fixed and single-axis tracking
        if convention == "winter_solstice_rule":

            # note that east slope is not used in this convention because it has negligible influence on equator-facing fixed modules
            if east_slope is not None and np.any(np.asarray(east_slope) != 0):
                warnings.warn(f"east_slope ({east_slope}) is not None/zero, but will be neglected by tracking='fixed' and convention='{convention}'.")

            if isinstance(north_slope, str):
                # assume a slope raster
                assert isfile(north_slope), f"north_slope is expected to be an existing tif file if given as str: {north_slope}"
                # try to extract data from raster
                try:
                    north_slope = gk.raster.interpolateValues(north_slope, locs, **kwargs)
                except Exception:
                    raise OSError(f"north_slope file cannot be read by gk.raster.interpolateValues(): {north_slope}.")  
            row_pitches, gcrs = calculate_row_pitch_and_gcr_from_winter_solstice_rule(
                lats = np.array([loc.lat for loc in locs]), 
                module_tilts=module_tilt, 
                north_slopes=north_slope, 
                solar_hour=12, 
                module_area_width=3.3, 
                min_interrow_distance=2.5,
                )
    
    elif tracking == "singleaxis":
        if convention == "tonita_et_al_2023_5perc":
            # Based on Tonita et al. (2023): Optimal ground coverage ratios for tracked, fixed-tilt, and vertical photovoltaic systems for latitudes up to 75◦N
            # separate mono- and bifacial (factor 0.96, see Tonita et al. 2023) lines
            def _interpolate_gcr(lat, bifac):
                # get value for bifaciality factors 0 and 0.96 and interpolate (linear is a simplification due to lack of more detailed data)
                assert np.all((0 <= bifac) & (bifac <= 1.0)) # make sure
                # get mono- and bifacial gcr based on absolute lat to account for Southern hemisphere
                gcrmono = -2.82*0.001 * np.abs(lat) + 0.388
                gcrbifac = -2.68*0.001 * np.abs(lat) + 0.361
                return gcrmono + (gcrbifac - gcrmono) * (bifac - 0)/(0.96 - 0)
            lats = np.array([loc.lat for loc in locs])
            bifacs = np.ones_like(lats) * bifaciality_factor
            # apply function to all lats and bifacs tuples
            gcrs = _interpolate_gcr(lats, bifacs)

    else:
        # None of the above applied, raise error
        raise ValueError(f"Unknown gcr convention '{convention}' for tracking = '{tracking}'.")

    # if requested, apply min gcr to locs with NaN or lower gcr then required
    if min_gcr is not None:
        assert isinstance(min_gcr, float) and 0 <= min_gcr <= 1.0, \
            f"min_gcr must be a float >= 0 and <= 1.0 if not None, here: {min_gcr}."
        gcrs[(gcrs < min_gcr) | np.isnan(gcrs)] = min_gcr

    return gcrs


## winter solstice rule: no shade on winter solstice at either solar noon or any morning hour
def _get_winter_solstice_solar_elevation(
        lats: int | float | np.ndarray, 
        solar_hour: int | float | np.ndarray = 12
        ):
    """
    Returns the solar zenith angle in degrees at winter solstice for a given 
    (solar) time of the day.

    Parameters
    ----------
    lats : int | float | np.ndarray
        The latitude(s) in degrees, positive = North.
    solar_hour : int | float | np.ndarray, optional
        The solar hour relative to true solar noon = 12, 10h30 would become 10.5.
        By default 12 (solar noon).
    
    Returns:
    --------
    float
        Solar elevation at given hour of winter solstice in degrees over horizon.
    """
    # check inputs
    assert isinstance(solar_hour, (int, float)) and 0<= solar_hour <= 24, \
        "solar_hour must be >= 0 and <= 24."
    assert isinstance(lats, (int, float, np.ndarray)), \
        "lats must be int, float or np.ndarray"
    if isinstance(lats, np.ndarray):
        _asarr = True
    else:
        lats = np.atleast_1d(lats)
        _asarr = False
    
    assert (-90 <= lats).all() & (lats <= 90).all(), f"lats must be >= -90 and <= 90, here: {lats}"


    # calculate the hour angle, i.e. horizontal deviation from solar noon
    hour_angle = 15*(solar_hour - 12) * np.pi/180
    # convert to rad values
    tropic = np.where(lats>=0, -23.43472, +23.43472)
    tropic_rad = tropic * np.pi/180
    lats_rad = lats * np.pi/180
    zenith = np.arccos(
            np.sin(lats_rad)*np.sin(tropic_rad) + np.cos(lats_rad)*np.cos(tropic_rad)*np.cos(hour_angle)
        ) * 180/np.pi
    # solar elevation is 90° - zenith
    solar_elevation = 90 - zenith
    
    if not _asarr:
        solar_elevation = solar_elevation[0]
        
    return solar_elevation


def calculate_row_pitch_and_gcr_from_winter_solstice_rule(
        lats: int | float | np.ndarray | pd.Series, 
        module_tilts: int | float | np.ndarray | pd.Series, 
        north_slopes: int | float | np.ndarray | pd.Series = 0, 
        solar_hour: int | float | np.ndarray | pd.Series = 12, 
        module_area_width: int | float | np.ndarray | pd.Series = 3.3,
        min_interrow_distance: int | float | np.ndarray | pd.Series = 2.5
        ):
    """
    Calculates the required row pitches/spacing for one or multiple equator-facing
    PV parks with fixed tilts based on the winter solstice rule such that no 
    shading occurs at a given variable solar hour. Also calculate the resulting 
    ground coverage ratios (gcr).

    Parameters
    ----------
    lats : int | float | np.ndarray | pd.Series
        The latitude(s) in degrees, positive = North.
    module_tilts : int | float | np.ndarray | pd.Series
        The module tilt in equator direction relative to flat ground.
        Negative values are allowed and describe module front facing away
        from the equator.
    north_slopes : int | float | np.ndarray | pd.Series, optional
        The ground slope facing North (i.e. the normal on the slope plane is tilted 
        towards North) when positive, negative values are South slopes, by default 0
    solar_hour : int | float | np.ndarray | pd.Series, optional
        The solar hour relative to true solar noon = 12, 10h30 would become 10.5.
        By default 12 (solar noon).
    module_area_width : int | float | np.ndarray | pd.Series, optional
        The width of the module area per each row in [m], measured along the 
        tilted edge. When a panel is e.g. 2m x 1m and mounted crosswise (1P), or 
        when 2 panels are mounted side by side laterally (2H), the value would 
        be 2 [m] in both cases, by default 3.3 [m] (2x 1.65m).
    min_interrow_distance : int | float | np.ndarray | pd.Series, optional
        The minimum distance to be kept between rows in [m] e.g. to allow for 
        maintenance trucks to pass. Set to 0.0 to ignore, by default 2.5 [m].

    Returns
    -------
    tuple[float, float] | tuple[np.ndarray, np.ndarray]
        A tuple of floats (all scalar inputs) or np.ndarrays (row pitches, ground coverage ratios)
    """
    # adapt/check types and set as array flag
    _asarr = False
    for var in [lats, module_tilts, north_slopes, solar_hour, module_area_width, min_interrow_distance]:
        assert isinstance(var, (int, float, np.ndarray, np.number, pd.Series)),\
            "All input variables must be int, float or np.ndarray/pd.Series types."
        if isinstance(var, (np.ndarray, pd.Series)):
            _asarr = True
    lats = np.atleast_1d(lats)
    module_tilts = np.atleast_1d(module_tilts)
    solar_hour = np.atleast_1d(solar_hour)
    module_area_width = np.atleast_1d(module_area_width)
    north_slopes = np.atleast_1d(north_slopes)
    min_interrow_distance = np.atleast_1d(min_interrow_distance)
    # make sure iterable shapes match and then align them
    arrays = [lats, module_tilts, north_slopes, solar_hour, module_area_width, min_interrow_distance]
    for arr in arrays:
        assert arr.ndim == 1, "Array-like inputs must be one-dimensional."
    lengths = [arr.size for arr in arrays if arr.size > 1]
    assert len(set(lengths)) <= 1, "All non-scalar inputs must have the same length."
    lats, module_tilts, north_slopes, solar_hour, module_area_width, min_interrow_distance = np.broadcast_arrays(
        lats,
        module_tilts,
        north_slopes,
        solar_hour,
        module_area_width,
        min_interrow_distance,
    )
    assert np.all((north_slopes > -90) & (north_slopes < 90)), "north_slopes must be >-90° and <90°"
    assert np.all((lats >= -90) & (lats <= 90)), "lats must be >=-90° and <=90°"
    assert np.all((module_tilts >= -90) & (module_tilts <= 90)), "module_tilts must be >=-90° and <=90°"
    assert np.all((solar_hour >= 0) & (solar_hour <= 24)), "solar_hour must be between 0 and 24"
    assert np.all(module_area_width > 0), "module_area_width must be >0"
    assert np.all(min_interrow_distance >= 0), "min_interrow_distance must be >= 0"

    
    # first get solar elevation
    solelevs = _get_winter_solstice_solar_elevation(lats=lats, solar_hour=solar_hour)
    
    # prep the degree values as rads
    module_tilts_rad = np.deg2rad(module_tilts)
    north_slopes_rad = np.deg2rad(north_slopes)
    solelevs_rad = np.deg2rad(solelevs)
    
    # then calculate the row pitch geometrically

    # start with basic module area width and height, only absolute slope matters so use abs()
    H = module_area_width * np.abs(np.sin(module_tilts_rad)) # vertical module height
    B = module_area_width * np.abs(np.cos(module_tilts_rad)) # horizontal projection length of module

    # geometrically required row spacing/pitches to avoid shading at given solar elevation
    tan_solelev = np.tan(solelevs_rad)
    hemisphere_sign = np.where(lats >= 0, 1.0, -1.0)
    numerator = B * tan_solelev + H
    denom = tan_solelev - hemisphere_sign * np.tan(north_slopes_rad)
    RP = np.full_like(denom, np.inf, dtype=float)
    valid = denom > 0
    RP[valid] = numerator[valid] / denom[valid]

    # enforce minimum inter-row spacing where needed
    _min_pitch = min_interrow_distance + B
    RP = np.maximum(RP, _min_pitch)

    # calculate gcr as module width over row pitch
    GCR = module_area_width / RP
    # set locations which cannot be resolved by winter solstice rule to NaN instead of zero
    GCR[~valid] = np.nan

    if not _asarr:
        RP = RP[0]
        GCR = GCR[0]

    return RP, GCR


def get_park_capacity_density(
      cap_dens_module: float | int, 
      gcrs: int | float | np.ndarray | pd.Series, 
      min_cap_dens_park: float | None = None,
      shape_factor: float | np.ndarray = 1.0,
      ):
    """
    Calculate the solar park capacity density based on module capacity density, 
    Ground Coverage Ratio and area efficiency of plot. Enforce minimum capacity 
    density where required.
    
    Parameters
    ----------
    cap_dens_module : float | int
        Capacity density of the module type in [W/m²]
    gcrs : int | float | np.ndarray | pd.Series
        Ground coverage ratios per location in positive floats <= 1.0.
    min_cap_dens_park : float | int | None, optional
        The minimum allowed capacity density in [MW/ha], will be set if value is 
        below this threshold. Will be ignored if None, by default None
        NOTE: This minimum will be applied to the array area, not to the 
        fenced area. The shape_factor can hence afterwards reduce the effective 
        capacity density across the whole fenced plot when it is not completely 
        covered with solar arrays.
    shape_factor : float | np.ndarray, optional
        The share of the property/plot that is actually built upon, usually not 
        100% due to local shading, inconvenient property shape or crossing roads,
        maintenance and inverter buildings. Will reduce the final park capacity
        density by this very factor, can be given per each location individually.
        By default 1.0.

    Returns
    -------
    float, np.ndarray
       The capacity density of the park in [MW/ha], either as float for a single 
       or as array for multiple locations.
    """
    # check types and set as array flag
    _asarr = False
    for var in [gcrs, shape_factor]:
        if not isinstance(var, (int, float, np.ndarray)):
            raise TypeError("gcrs and shape_factor inputs must be int, float, pd.Series or np.ndarray types.")
        if isinstance(var, (np.ndarray. pd.Series)):
            _asarr = True
    if not isinstance(cap_dens_module, (float, int, np.number)):
        raise TypeError("cap_dens_module must be float or int if not None.")
    if not min_cap_dens_park is None or isinstance(min_cap_dens_park, (float, int, np.number)):
        raise TypeError("min_cap_dens_park must be float or int if not None.")
    
    # scale to park density via gcr
    cap_dens_park = np.atleast_1d(cap_dens_module) * np.atleast_1d(gcrs) *10000/1E6 # MW/ha

    # set min density if applicable
    if min_cap_dens_park is not None:
       cap_dens_park[cap_dens_park<np.atleast_1d(min_cap_dens_park)] = min_cap_dens_park

    # apply the shape factor reduction 
    shape_factor = np.atleast_1d(shape_factor)
    if shape_factor.size not in (1, gcrs.size):
        raise ValueError(f"shape_factor must either be scalar or have the same length as gcrs. Here: {shape_factor}")
    if np.any((gcrs <= 0) | (gcrs > 1)):
        raise ValueError("gcrs must contain values > 0 and <= 1.")
    if np.any((shape_factor <= 0) | (shape_factor > 1)):
        raise ValueError("shape_factor must contain values > 0 and <= 1.")
    cap_dens_park = shape_factor * cap_dens_park

    return cap_dens_park if _asarr else cap_dens_park[0]


def get_gcr_from_capacity_density(
        capacity_density_park : int | float | np.ndarray | pd.Series,
        capacity_density_module : int |float | np.ndarray | pd.Series,
        packing_factor : float | np.ndarray | pd.Series = 0.74,
):
    """
    Calculates the Ground Coverage Ratio (GCR) based on a given
    park and module capacity density and a packing factor.

    Parameters
    ----------
    capacity_density_park : int |float | np.ndarray
        Capacity density of the whole park, relative to the total
        project area (unless packing_factor is 1.0) in MW/km².
    capacity_density_module : int |float | np.ndarray
        Capacity density of the used modules in W/m². 
    packing_factor : float| np.ndarray, optional
        The array (incl. interrow spacing) area over total project 
        ("fenceline") area to account for unused space, space for 
        roads, transformers, inverters etc. By default 0.74 [1]
    
    Returns
    -------
        np.ndarray
        Ground Coverage Ratio of the array field of the PV park 
        (excluding the un- or otherwise used areas of the plot)

    References
    ----------
    [1] Hu, S., Sun, Y., Hernandez, R.R. et al. Quantifying 
        land-use metrics for solar photovoltaic projects in 
        the western United States. Commun Earth Environ 6, 
        1006 (2025). https://doi.org/10.1038/s43247-025-02862-5
    """
    assert isinstance(capacity_density_park, (int, float, np.ndarray, pd.Series)), \
        "capacity_density_park must be int, float, np.ndarray or pd.Series"
    assert isinstance(capacity_density_module, (int, float, np.ndarray, pd.Series)), \
        "capacity_density_module must be int, float, np.ndarray or pd.Series"
    assert isinstance(packing_factor, (float, np.ndarray, pd.Series)), \
        "packing_factor must be float, np.ndarray or pd.Series"
    
    # set a flag if we need to return a scalar result
    if all([isinstance(v, (int, float)) for v in [capacity_density_park, capacity_density_module, packing_factor]]):
        as_scalar = True
    else:
        as_scalar = False

    packing_factor = np.atleast_1d(packing_factor)
    assert all([0<x<=1.0 for x in packing_factor]), f"All packing_factor values must be floats >0 and <= 1.0"
    
    # calculate the GCR for this constellation
    gcr = np.atleast_1d(capacity_density_park)/np.atleast_1d(capacity_density_module) / packing_factor

    if as_scalar:
        return gcr[0]
    else:
        return gcr