import numpy as np


def projectByPowerLaw(measured_wind_speed, measured_height, target_height, alpha=1 / 7):
    """Estimates windspeed at target height ($h_t$) based off a measured windspeed 
    ($u_m$) at a known measurement height ($h_m$) subject to the scaling factor ($a$)

    $ u_t = u_m * (\\frac{h_t}{h_m})^a $


    Parameters:
    -----------
    measured_wind_speed : numpy.ndarray
        The raw windspeeds to be adjusted
        * If an array is given with a single dimension, it is assumed to represent 
          timeseries values for a single location
        * If multidimensional array is given, the assumed dimensional context is 
          (time, locations), and 'targetLoc' must be an iterable with the same 
          length as the 'locations' dimension

    measured_height : numeric or numpy.ndarray
        The measurement height of the raw windspeeds
        * If an array is given for measured_wind_speed with a single dimension, a 
          single value is expected for measured_height
        * If multidimensional array is given for measured_wind_speed, an array of
          values is expected for measured_height. One value for each wind speed
          timeseries

    target_height : numeric or numpy.ndarray
        The height to project each wind speed timeseries to
        * If a numeric value is given, all windspeed timeseries will be projected
          to this height
        * If an array is given for target_height, each value must match to one
          wind speed timeseries in measured_wind_speed

    alpha : numeric or numpy.ndarray, optional
        The alpha value used to project each wind speed timeseries
        * If a numeric value is given, all windspeed timeseries will be projected
          using this alpha value
        * If an array is given, each value must match to one wind speed timeseries
          in measured_wind_speed
        * The default 1/7 value corresponds to neutral stability conditions

    """
    return measured_wind_speed * np.power(target_height / measured_height, alpha)


def alphaFromLevels(low_wind_speed, low_height, high_wind_speed, high_height):
    """Solves for the scaling factor ($a$) given two windspeeds with known heights

    $ a = log(\\frac{u_m}{u_t}) / log(\\frac{h_m}{h_t}) $

    Parameters:
    -----------
    low_wind_speed : numeric or numpy.ndarray
        The measured windspeed at the 'lower height'

    low_height : numeric or numpy.ndarray
        The measured height at the 'lower height'

    high_wind_speed : numeric or numpy.ndarray
        The measured windspeed at the 'lower height'

    high_height : numeric or numpy.ndarray
        The measured height at the 'lower height'

    """
    return np.log(low_wind_speed / high_wind_speed) / np.log(low_height / high_height)
