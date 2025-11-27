import numpy as np


def calculate_wet_bulb_temperature(air_temperature, relative_humidity):
    """
    Function to calculate the wet bulb temperature from air temperature and relative humidity.

    Parameters
    ----------
    air_temperature: float | int
        air temperature in °C
    relative_humidity: float | int
        relative humidity in %

    References
    ----------
    [1] Roland Stull. Wet-bulb temperature from relative humidity and air temperature. Journal of Applied Meteorology and Climatology, 50:2267–2269, 11 2011.
    """
    wet_bulb_temperature = (
        air_temperature * np.arctan(0.151977 * (relative_humidity + 8.313659) ** (0.5))
        + np.arctan(air_temperature + relative_humidity)
        - np.arctan(relative_humidity - 1.676331)
        + 0.00391838 * (relative_humidity**1.5) * np.arctan(0.023101 * relative_humidity)
        - 4.686035
    )

    return wet_bulb_temperature
