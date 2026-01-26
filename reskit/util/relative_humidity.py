import numpy as np


def calculate_relative_humidity(dewpoint_temperature, air_temperature):
    """
    Function to calculate the relative humidity from dewpoint temperature and air temperature using the Sonntag formula.

    Parameters
    ----------
    dewpoint_temperature: float | int
        dewpoint temperature in °C
    air_temperature: float | int
        air temperature in °C

    References
    ----------
    [1] https://www.npl.co.uk/resources/q-a/dew-point-and-relative-humidity
    """

    def calculate_vapor_pressure(temperature):
        """
        Function to calculate the vapor pressure from the temperature

        temperature: temperature in °C
        """
        temperature_kelvin = temperature + 273.15
        vapor_pressure = np.exp(
            -6096.9385 / temperature_kelvin
            + 21.2409642
            - 2.711193 * 10**-2 * temperature_kelvin
            + 1.673952 * 10**-5 * temperature_kelvin**2
            + 2.433502 * np.log(temperature_kelvin)
        )  # vapor pressure [1]
        return vapor_pressure

    relative_humidity = (
        calculate_vapor_pressure(dewpoint_temperature) / calculate_vapor_pressure(air_temperature) * 100
    )  # [1]

    return relative_humidity
