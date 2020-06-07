
import numpy as np


def apply_air_density_adjustment(wind_speed, pressure, temperature, height=0):
    """
    Applies air density corrections to the wind speed as suggested by the IEC 61400-12-1:2017 [1].

    Parameters
    ----------
    wind_speed : float or array-like
        The wind speed in m/s at <(or close to)> the surface/hub height?.                                                           @Sev: "(or close to)" is ok too, right?
    pressure : float or array-like
        Air preassure in Pa at (or close to) the surface.
    temperature : float or array-like
        Air temperature in degree Celsius at (or close to) the surface.
    height : float or array-like, optional
        The height to project the air density to in m, by default 0

    Returns
    -------
    float or array-like
        The air density corrected wind speed in m/s at the given height.

    Notes
    ------
        Air density calculation using the ideal gas equation since it is equivalent for a real-gas ecuaquion for humidity = 0, and humidity does not have a significant impact until high temperatures are considered [2]   @Sev: Reference added, got one better?
        Pressure projection using barometric equation and density correction from assuming equal energy in the wind 
    Sources
    -------
        [1] International Electrotechnical Commision (ICE). (2017). IEC 61400-12-1:2017 (p. 558). https://webstore.iec.ch/publication/26603
        [2] Yue, W., Xue, Y., & Liu, Y. (2017). High Humidity Aerodynamic Effects Study on Offshore Wind Turbine Airfoil/Blade Performance through CFD Analysis. International Journal of Rotating Machinery, 2017, 1–15. https://doi.org/10.1155/2017/7570519
    """
    g0 = 9.80665  # Gravitational acceleration [m/s2]
    Ma = 0.0289644  # Molar mass of dry air [kg/mol]
    R = 8.3144598  # Universal gas constant [N·m/(mol·K)]
    rhoSTD = 1.225  # Standard air density [kg/m3]

    temperature = (temperature + 273.15)

    # Get surface density
    rho = pressure * Ma / (R * temperature)

    # Project rho to the desired height
    if not height is None:
        rho = rho * np.exp((-g0 * Ma * height) / (R * temperature))

    # Adjust wind speeds to standard-air-density-equivalent
    return np.power(rho / rhoSTD, 1 / 3) * wind_speed
