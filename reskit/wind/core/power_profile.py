import numpy as np


def apply_power_profile_projection(measured_wind_speed, measured_height, target_height, alpha=1 / 7):
    """
    Estimates wind speed values at target height based on another measured wind speed at a known height subject to a power-law scaling factor.

    TODO: Follow the same indentation standard you've used for other functions. For example, compare this docstring against "apply_logarithmic_profile_projection"

    Parameters
    ----------
    measured_wind_speed : array_like
      The raw wind speeds to be adjusted.
      If a single dimension array is given, it is assumed to represent timeseries values for a single location
      If a multidimensional array is given, the assumed dimensional context is (time, locations), and 'targetLoc' must be an iterable with the same length as the 'locations' dimension
        TODO: See the note I made for the similar logarithmic function

    measured_height : array_like
      The measurement height of the raw windspeeds.
      Must either be a single value, or an array of values with the same length as the "locations" dimension of `measured_wind_speed`

    target_height : numeric or array_like
      The (hub) height to project each wind speed timeseries to
      Must either be a single value, or an array of values with the same length as the "locations" dimension of `measured_wind_speed`

    alpha : numeric or array_like, optional
      The scaling factor used to project each wind speed timeseries, by default 1/7
      Must either be a single value, or an array of values with the same length as the "locations" dimension of `measured_wind_speed`

    Notes
    -----
      The default scaling factor (alpha = 1/7) corresponds to neutral stability conditions.
      Alpha values an also be computed using the function "alpha_from_levels"

    Returns
    -------
    array_like
        projected wind speeds
        Has the same dimensions as `measured_wind_speed`

    See Also
    --------
        - apply_logarithmic_profile_projection( <wind speeds>, <measured height>, <target height>, <roughness> )

    """
    return measured_wind_speed * np.power(target_height / measured_height, alpha)


def alpha_from_levels(low_wind_speed, low_height, high_wind_speed, high_height):
    """
    Obtains the scaling factor given two wind speeds measured at two different known heights.

    TODO: Check indentation

    Parameters
    ----------
    low_wind_speed : numeric or array_like
      The measured windspeed at the 'lower height'

    low_height : numeric or array_like
      The measured height at the 'lower height'

    high_wind_speed : numeric or array_like
      The measured windspeed at the 'higher height'

    high_height : numeric or array_like
      The measured height at the 'higher height'

    Returns
    -------
    numeric or array-like
      The corresponding scaling factor
      The output dimensionality follows the broadcasting rules of Numpy

    """

    return np.log(low_wind_speed / high_wind_speed) / np.log(low_height / high_height)
