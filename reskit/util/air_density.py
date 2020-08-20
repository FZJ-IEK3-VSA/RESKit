import numpy as np


def compute_air_density(temperature=20, pressure=101325, relative_humidity=0, dew_temperature=None):
    """Computes air density, following the apprach of "Revised formula for the density of moist air (CIPM-2007)" by A Picard, R S Davis, M Glaser and K Fujii"""

    if relative_humidity is None and dew_temperature is None:
        relative_humidity = 0

    t = temperature
    T = 273.15 + t
    p = pressure

    A = 1.2378847e-5
    B = -1.9121316e-2
    C = 33.93711047
    D = -6.3431645e3

    a_ = 1.00062
    b_ = 3.14e-8
    y_ = 5.6e-7

    if not dew_temperature is None:
        Td = dew_temperature + 273.15
        psv = np.exp(A * np.power(Td, 2) + B * Td + C + D / Td)
        f = a_ + b_ * p + y_ * np.power(dew_temperature, 2)
        xv = f * psv / p
    else:
        psv = np.exp(A * np.power(T, 2) + B * T + C + D / T)
        f = a_ + b_ * p + y_ * np.power(t, 2)
        xv = relative_humidity * f * psv / p

    a0 = 1.58123e-6
    a1 = -2.9331e-8
    a2 = 1.1043e-10
    b0 = 5.707e-6
    b1 = -2.051e-8
    c0 = 1.9898e-4
    c1 = -2.376e-6
    d = 1.83e-11
    e = -0.765e-8

    Z = 1 - (p / T) * (a0 - a1 * t + a2 * np.power(t, 2) + (b0 + b1 * t) * xv + (c0 + c1 * t) * np.power(xv, 2)) + np.power(p / T, 2) * (d + e * np.power(xv, 2))

    Ma = 28.96546e-3
    Mv = 18.01528e-3
    R = 8.314472

    airden = p * Ma / (Z * R * T) * (1 - xv * (1 - (Mv / Ma)))

    return airden
