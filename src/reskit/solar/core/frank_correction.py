import numpy as np


def frank_correction_factors(ghi, dni_extra, times, solar_elevation):
    """Applies the proposed transmissivity-based irradiance corrections to COSMO
    data based on Frank et al.

    TODO: Add citation to Frank!
    """
    transmissivity = ghi / dni_extra
    sigmoid = 1 / (1 + np.exp(-(transmissivity - 0.5) / 0.03))

    # Adjust cloudy regime
    months = times.month
    cloudyFactors = np.empty(months.shape)

    cloudyFactors[months == 1] = 0.7776553729824053
    cloudyFactors[months == 2] = 0.7897164461247639
    cloudyFactors[months == 3] = 0.8176553729824052
    cloudyFactors[months == 4] = 0.8406805293005672
    cloudyFactors[months == 5] = 0.8761808928311765
    cloudyFactors[months == 6] = 0.9094139886578452
    cloudyFactors[months == 7] = 0.9350856478115459
    cloudyFactors[months == 8] = 0.9191682419659737
    cloudyFactors[months == 9] = 0.912703795259561
    cloudyFactors[months == 10] = 0.8775035625999711
    cloudyFactors[months == 11] = 0.8283158353933402
    cloudyFactors[months == 12] = 0.7651417769376183
    cloudyFactors = np.broadcast_to(
        cloudyFactors.reshape((cloudyFactors.size, 1)), ghi.shape
    )

    cloudyFactors = cloudyFactors * (1 - sigmoid)

    # Adjust clearsky regime
    e = solar_elevation
    clearSkyFactors = np.ones(e.shape)
    clearSkyFactors[np.where((e >= 10) & (e < 20))] = 1.17612920884004
    clearSkyFactors[np.where((e >= 20) & (e < 30))] = 1.1384180020822825
    clearSkyFactors[np.where((e >= 30) & (e < 40))] = 1.1022951259566156
    clearSkyFactors[np.where((e >= 40) & (e < 50))] = 1.0856852748290704
    clearSkyFactors[np.where((e >= 50) & (e < 60))] = 1.0779254457050245
    clearSkyFactors[np.where(e >= 60)] = 1.0715262914980628

    clearSkyFactors *= sigmoid

    # Apply to ghi
    totalCorrectionFactor = clearSkyFactors + cloudyFactors

    del clearSkyFactors, cloudyFactors, e, months, sigmoid, transmissivity

    return totalCorrectionFactor
