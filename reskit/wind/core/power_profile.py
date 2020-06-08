import numpy as np


def apply_power_profile_projection(measured_wind_speed, measured_height, target_height, alpha = 1/7):
  """
  Estimates wind speed values at target height based on another measured wind speed at a known height subject to a scaling factor.

  Parameters
  ----------
  measured_wind_speed : array_like
    The raw wind speeds to be adjusted.
    If a single dimension array is given, it is assumed to represent timeseries values for a single location
    If a multidimensional array is given, the assumed dimensional context is (time, locations), and 'targetLoc' must be an iterable with the same length as the 'locations' dimension
  measured_height : array_like
    The measurement height of the raw windspeeds.
    If a single dimension array is given for measured_wind_speed, a single value is expected for measured_height.
    If a multidimensional array is given for measured_wind_speed, a array of values is expected for measured_height. One value for each wind speed timeseries

  target_height : numeric or array_like
    The (hub) height to project each wind speed timeseries to
    If a numeric value is given, all windspeed timeseries will be projected to this height.
    If an array is given, each value must match to one wind speed timeseries in measured_wind_speed

  alpha : numeric or array_like, optional
    The scaling factor used to project each wind speed timeseries, by default 1/7
    If a numeric value is given, all windspeed timeseries will be projected using this alpha value
    If an array is given, each value must match to one wind speed timeseries in measured_wind_speed
  
  Notes
  -----
    The default scaling factor (alpha = 1/7) corresponds to neutral stability conditions.
  
  Returns
  -------
  array_like
      projected wind speed

  """
  return measured_wind_speed * np.power(target_height / measured_height, alpha)


def alpha_from_levels(low_wind_speed, low_height, high_wind_speed, high_height):
  """
  Obtaines the scaling factor given two wind speeds measured at two different known heights.

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
  numeric
    The corresponding scaling factor

  """

  return np.log(low_wind_speed / high_wind_speed) / np.log(low_height / high_height)
