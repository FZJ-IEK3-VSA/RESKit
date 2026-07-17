import numpy as np
import geokit as gk
import pandas as pd
from os.path import isfile
from collections.abc import Iterable
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


def location_to_module_tilt(locs, convention: str = "Ryberg2020", **kwargs):
    """
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


def location_to_gcr(
        locs: gk.LocationSet | Iterable, 
        tracking: str, 
        convention: str, 
        module_tilt: int | float | Iterable = None,
        north_slope : int | float | str | Iterable = 0,
        east_slope : int | float | str | Iterable = 0,
        bifaciality_factor : float = None,
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

    # first check if we have a given raster from which we only need to extract the gcrs
    if isinstance(convention, str) and isfile(convention):
        # try to extract data from raster
        try:
            return gk.raster.interpolateValues(convention, locs, **kwargs)
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
        assert not any()
        
    if tracking == "fixed":
        # a different set of conventions applies for fixed and single-axis tracking
        if convention == "winter_solstice_rule":

            # note that east slope is not used in this convention because it has negligible influence on equator-facing fixed modules
            if east_slope not in [None, 0]:
                warnings.warn(f"east_slope ({east_slope}) is not None or zero, but will be neglected by tracking='fixed' and convention='{convention}'.")

            if isinstance(north_slope, str):
                # assume a slope raster
                assert isfile(north_slope), f"north_slope is expected to be an existing tif file if given as str: {north_slope}"
                # try to extract data from raster
                try:
                    north_slope = gk.raster.interpolateValues(north_slope, locs, **kwargs)
                except Exception:
                    raise OSError(f"north_slope file cannot be read by gk.raster.interpolateValues(): {north_slope}.")  
            row_pitches, gcrs = calulate_row_pitch_and_gcr( #add azimuth as arg or reduce function (name) to only equator facing
                lats = np.array([loc.lat for loc in locs]), 
                module_tilts=module_tilt, 
                north_slopes=north_slope, 
                solar_hour=12, 
                module_area_width=3.3, 
                min_interrow_distance=2.5,
                )
            return gcrs
    
    if tracking == "singleaxis":
        if convention == "tonita_et_al_2023_5perc":
            # Based on Tonita et al. (2023): Optimal ground coverage ratios for tracked, fixed-tilt, and vertical photovoltaic systems for latitudes up to 75◦N
            # separate mono- and bifacial (factor 0.96, see Tonita et al. 2023) lines
            def _interpolate_gcr(lat, bifac):
                # get value for bifaciality factors 0 and 0.96 and interpolate (linear is a simplification due to lack of more detailed data)
                assert 0 <= bifac <= 1.0 # make sure
                gcrmono = -2.82*0.001 * lat + 0.388
                gcrbifac = -2.68*0.001 * lat + 0.361
                return gcrmono + (gcrbifac - gcrmono) * (bifac - 0)/(0.96 - 0)
            lats = np.array([loc.lat for loc in locs])
            bifacs = np.ones_like(lats) * bifaciality_factor
            return _interpolate_gcr(lats, bifacs)

    # None of the above applied, raise error
    raise ValueError(f"Unknown gcr convention '{convention}' for tracking = '{tracking}'.")
    


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


def calulate_row_pitch_and_gcr(
        lats: int | float | np.ndarray | pd.Series, 
        module_tilts: int | float | np.ndarray | pd.Series, 
        north_slopes: int | float | np.ndarray | pd.Series = 0, 
        solar_hour: int | float | np.ndarray | pd.Series = 12, 
        module_area_width: int | float | np.ndarray | pd.Series = 3.3, 
        min_interrow_distance: int | float = 2.5
        ):
    """
    Calculates the required row pitches/spacing for one or multiple South-facing
    PV parks with fixed tilts based on the winter solstice rule such that no 
    shading occurs at a given variable solar hour. Also calculate the resulting 
    ground coverage ratios (gcr).

    Parameters
    ----------
    lats : int | float | np.ndarray
        The latitude(s) in degrees, positive = North.
    module_tilts : int | float | np.ndarray
        The module tilt in Southern direction relative to flat ground.
    north_slopes : int | float | np.ndarray, optional
        The ground slope facing North when positive, by default 0
    solar_hour : int | float | np.ndarray, optional
        The solar hour relative to true solar noon = 12, 10h30 would become 10.5.
        By default 12 (solar noon).
    module_area_width : int | float | np.ndarray, optional
        The width of the module area per each row in [m], measured along the 
        tilted edge. When a panel is e.g. 2m x 1m and mounted crosswise (1P), or 
        when 2 panels are mounted side by side laterally (2H), the value would 
        be 2 [m] in both cases, by default 3.3 [m] (2x 1.65m).
    min_interrow_distance : int, float, optional
        The minimum distance to be kept between rows in [m] e.g. to allow for 
        maintanance trucks to pass. Set to None to ignore, by default 2.5 [m].

    Returns
    -------
    np.ndarray
        _description_ #TODO
    """
    # adapt/check types and set as array flag
    _asarr = False
    if isinstance(module_tilts, pd.Series):
        module_tilts = module_tilts.values
    if isinstance(north_slopes, pd.Series):
        north_slopes = north_slopes.values
    if isinstance(solar_hour, pd.Series):
        solar_hour = solar_hour.values
    if isinstance(module_area_width, pd.Series):
        module_area_width = module_area_width.values
    for var in [lats, module_tilts, north_slopes, solar_hour, module_area_width]:
        assert isinstance(var, (int, float, np.ndarray, np.number)),\
            "All input variables must be int, float or np.ndarray/pd.Series types."
        if isinstance(var, np.ndarray):
            _asarr = True
    assert min_interrow_distance is None or isinstance(min_interrow_distance, (int, float)),\
        "min_interrow_distance must be int if not None."
    
    # first get solar elevation
    solelevs = _get_winter_solstice_solar_elevation(lats=np.atleast_1d(lats), solar_hour=solar_hour)
    
    # prep the degree values as rads
    module_tilts_rad = np.atleast_1d(module_tilts) * np.pi/180
    north_slopes_rad = np.atleast_1d(north_slopes) * np.pi/180
    solelevs_rad = solelevs * np.pi/180
    
    # then calculate the row picth geometrically

    # start with basic module area width and height
    B = module_area_width * np.cos(module_tilts_rad) # horizontal projection of module
    H = module_area_width * np.sin(module_tilts_rad) # vertical module height


    # geometrically required row spacing/pitches to avoid shading at given solar elevation
    RP = (B * np.tan(solelevs_rad) + H) / (np.tan(solelevs_rad) + np.tan(-north_slopes_rad))

    # set geometrically infeasible locations to inf row pitch
    RP[(np.tan(solelevs_rad) + np.tan(-north_slopes_rad))<=0]=np.inf
    # set minimum interow spacing where needed
    if min_interrow_distance is not None:
        _min_pitch = min_interrow_distance + np.full(RP.shape, B)
        sel = RP<_min_pitch
        RP[sel]=_min_pitch[sel]

    # calculate gcr as covered area from bird's perspective over row pitch
    GCR = np.cos(module_tilts_rad) * module_area_width / RP

    if not _asarr:
        RP = RP[0]
        GCR = GCR[0]

    return RP, GCR


def get_park_capacity_density(
      cap_dens_module: float | int, 
      module_tilts : int | float | np.ndarray | pd.Series, 
      gcrs: int | float | np.ndarray | pd.Series, 
      min_cap_dens_park: float | None = None,
      shape_factor: float | np.ndarray = 1.0,
      ):
    """
    Calculate the solar park capacity density based on   
    Parameters
    ----------
    cap_dens_module : float | int
        Capacity density of the module type in [W/m²]
    module_tilts : int | float | np.ndarray | pd.Series
        Module tilts between ground and module plane facing the equator, in deg.
    gcrs : int | float | np.ndarray | pd.Series
        Ground coverage ratios per location in positive floats <= 1.0.
    min_cap_dens_park : float | None, optional
        The minimum allowed capacity density in [MW/ha], will be set if value is 
        below this threshold. Will be ignored if None, by default None
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
    if isinstance(module_tilts, pd.Series):
       module_tilts = module_tilts.values
    if isinstance(gcrs, pd.Series):
       gcrs = gcrs.values
    _asarr = False
    for var in [module_tilts, gcrs, shape_factor]:
        assert isinstance(var, (int, float, np.ndarray)),\
            "All input variables must be int, float or np.ndarray types."
        if isinstance(var, np.ndarray):
            _asarr = True
    assert isinstance(cap_dens_module, (float, int)),\
        "cap_dens_module must be float or int if not None."
    assert min_cap_dens_park is None or isinstance(min_cap_dens_park, (float)),\
        "min_cap_dens_park must be float if not None."
    
    # project module density to flat ground
    cap_dens_module_grd = np.atleast_1d(cap_dens_module) / np.cos(np.atleast_1d(module_tilts)*np.pi/180) # W/m2

    # scale to park density via gcr
    cap_dens_park = cap_dens_module_grd * np.atleast_1d(gcrs) *10000/1E6 # MW/ha

    # apply the shape factor reduction 
    shape_factor = np.atleast_1d(shape_factor)
    if len(shape_factor) > 1:
        assert len(shape_factor) == len(cap_dens_park), \
            f"shape_factor must have the same length as module_tilts, or gcrs if the latter is an iterable."
    cap_dens_park = shape_factor * cap_dens_park

    # set min density if applicable
    if min_cap_dens_park is not None:
       cap_dens_park[cap_dens_park<np.atleast_1d(min_cap_dens_park)] = min_cap_dens_park

    return cap_dens_park if _asarr else cap_dens_park[0]