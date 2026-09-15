import geokit as gk
import numpy as np

from ...util import ResError

# The tilt conventions which `location_to_tilt` accepts
_TILT_CONVENTIONS = ("Ryberg2020",)


def location_to_tilt(locs, convention="Ryberg2020"):
    """
    Simple system tilt estimator based off latitude and longitude coordinates


    Parameters
    ----------
    locs : geokit.LocationSet or iterable of (lon,lat) pairs
           The locations at which to estimate system tilt angle

    convention : str, optional
                 The calculation method used to suggest system tilts.
                 Only "Ryberg2020" is accepted.


    Returns
    -------
    np.ndarray
        Suggested tilt angle at each of the provided `locs`.
        Has the same length as the number of `locs`.

    Raises
    ------
    ResError
        If `convention` is not a string, or if it is not a known convention.

    Notes
    -----
    "Ryberg2020"
        When `convention` equals "Ryberg2020", the following equation is followed:

        .. math:: 42.327719357601396 * arctan( 1.5 * abs(latitude) )

    .. [1] TODO: Cite future Ryberg2020 publication

    """
    if not isinstance(convention, str):
        raise ResError(f"Tilt convention must be a string, but is: {type(convention)}")

    if convention not in _TILT_CONVENTIONS:
        raise ResError(f"Tilt convention must be one of {', '.join(_TILT_CONVENTIONS)}, but is: {convention}")

    locs = gk.LocationSet(locs)

    return 42.327719357601396 * np.arctan(1.5 * np.radians(np.abs(locs.lats)))
