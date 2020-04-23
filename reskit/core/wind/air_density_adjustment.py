
import numpy as np


def apply_air_density_adjustment(wind_speed, pressure, temperature, height=0):
    """
    Applies wind_speed correction based off calculated aid density

    Notes:
    ------
    * Density calculation from ideal gas
    * Projection using barometric equation
    * Density correction from assuming equal energy in the wind 
     - Suggested by IEC IEC61400-12

    Parameters:
    -----------
    wind_speed : The wind speeds to adjust

    pressure : The pressure at the surface, in Pa

    temperature : Air temperature at the surface, in C

    height : The height to project the air density to, in meters
    """
    g0 = 9.80665  # Gravitational acceleration [m/s2]
    Ma = 0.0289644  # Molar mass of dry air [kg/mol]
    R = 8.3144598  # Universal gas constant [N·m/(mol·K)]
    rhoSTD = 1.225  # Standard air density [kg/m3]

    temperature = (temperature + 273.15)

    # Get surface density
    # NOTE: I could use the equation from above, but its basically exactly equal
    #       to ideal gas for humidity=0, and humidity does not have a significant
    #       impact until high temperatures are considered

    rho = pressure * Ma / (R * temperature)

    # Project rho to the desired height
    if not height is None:
        rho = rho * np.exp((-g0 * Ma * height) / (R * temperature))

    # Adjust wind speeds to standard-air-density-equivalent
    return np.power(rho / rhoSTD, 1 / 3) * wind_speed
