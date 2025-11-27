import numpy as np


def calculate_specific_humidity(air_temperature, relative_humidity):
    """
    Function to calculate the specific humidity from air temperature and relative humidity.

    Parameters
    ----------
    air_temperature: float | int
        air temperature in °C
    relative_humidity: float | int
        relative humidity in [0,1] 

    References
    ----------
    [1] John M. Wallace and Peter V. Hobbs. Atmospheric Science: An Introductory Survey. Academic Press, San Diego, 2nd edition, 2006.
    [2] https://en.wikipedia.org/wiki/Goff%E2%80%93Gratch_equation
    """
    air_temperature_kelvin = air_temperature + 273.15 

    p_sat_water = np.exp(23.709 - (4111/(air_temperature_kelvin - 35.44))) # for temperature aove 0°C [1]

    # Goff–Gratch equation for vapor pressure over ice [2]
    T_triple = 273.16  # Triple point of water in Kelvin
    e_triple = 6.1071  # Saturation vapor pressure at the triple point in hPa
    log10_e = -9.09718 * ((T_triple / air_temperature_kelvin) - 1) - 3.56654 * np.log10(T_triple / air_temperature_kelvin) + 0.876793 * (1 - (air_temperature_kelvin / T_triple)) + np.log10(e_triple)
    p_sat_ice = 10**log10_e * 100

    p_sat = np.where(air_temperature_kelvin > 273.15, p_sat_water, p_sat_ice)

    p_v = relative_humidity * p_sat
    # calculate specific humidity
    specific_humidity = 0.622 * (p_v / (101325 - p_v))

    return specific_humidity
