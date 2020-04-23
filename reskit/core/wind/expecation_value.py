from scipy.interpolate import splrep, splev
import numpy as np

from ..util import ResError


def expectated_capacity_factor_from_weibull(power_curve, mean_wind_speed=5, weibull_shape=2):
    """Computes the expected capacity factor of a wind turbine based on an assumed Weibull distribution of observed wind speeds
    """
    from scipy.special import gamma
    from scipy.stats import exponweib

    # Get windspeed distribution
    lam = mean_wind_speed / gamma(1 + 1 / weibull_shape)
    dws = 0.001
    ws = np.arange(0, 40, dws)
    pdf = exponweib.pdf(ws, 1, weibull_shape, scale=lam)

    # Estimate generation
    power_curveInterp = splrep(power_curve.wind_speed, power_curve.capacity_factor)
    gen = splev(ws, power_curveInterp)

    # Do some "just in case" clean-up
    cutin = power_curve.wind_speed.min()  # use the first defined windspeed as the cut in
    cutout = power_curve.wind_speed.max()  # use the last defined windspeed as the cut out

    gen[gen < 0] = 0  # floor to zero

    gen[ws < cutin] = 0  # Drop power to zero before cutin
    gen[ws > cutout] = 0  # Drop power to zero after cutout

    # Done
    totalGen = (gen * pdf).sum() * dws
    return totalGen


def expectated_capacity_factor_from_distribution(power_curve, wind_speed_values, wind_speed_counts):
    """Computes the expected capacity factor of a wind turbine based on an explicitly-provided wind speed distribution
    """
    wind_speed_values = np.array(wind_speed_values)
    wind_speed_counts = np.array(wind_speed_counts)

    if not len(wind_speed_values.shape) == 1:
        raise ResError("wind_speed_values must be 1-dimensional")

    # Handle 2 dimensional counts with 1 dimensional wind speeds
    if len(wind_speed_counts.shape) > 1:
        if not wind_speed_counts.shape[0] == wind_speed_values.shape[0]:
            raise ResError("Dimensional incompatability")

        wind_speed_values = np.reshape(wind_speed_values, (wind_speed_counts.shape[0], 1))

    # Estimate generation distribution
    gen = np.interp(wind_speed_values, power_curve.wind_speed, power_curve.capacity_factor, left=0, right=0) * wind_speed_counts

    meanGen = gen.sum(0) / wind_speed_counts.sum(0)

    # Done
    return meanGen
