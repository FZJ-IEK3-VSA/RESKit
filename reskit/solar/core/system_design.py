import numpy as np
import geokit as gk
from os.path import isfile

from ...util import ResError


def location_to_tilt(locs, convention="Ryberg2020", **kwargs):
    """
    def location_to_tilt(locs, convention="Ryberg2020", **kwargs)
    
    Simple system tilt estimator based off latitude and longitude coordinates


    Parameters
    ----------
    locs : geokit.LocationSet or iterable of (lat,lon) pairs
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

    if convention == 'Ryberg2020':
        tilt = 42.327719357601396 * np.arctan(1.5 * np.radians(np.abs(locs.lats)))

    elif isfile(convention):
        tilt = gk.raster.interpolateValues(convention, locs, **kwargs)

    else:
        try:
            tilt = eval(convention, {}, {"latitude": locs.lats})
        except:
            raise ResError("Failed to apply tilt convention")

    return tilt
