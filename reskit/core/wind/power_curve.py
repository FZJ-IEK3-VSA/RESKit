from collections import namedtuple
import numpy as np
from os.path import join, dirname
import pandas as pd
from scipy.interpolate import splrep, splev
from scipy.stats import norm

from ..util import ResError

_P = namedtuple('PowerCurve', 'ws cf')
_synthetic_power_curve_data = None


def synthetic_power_curve_data():
    global _synthetic_power_curve_data

    if _synthetic_power_curve_data is None:
        _synthetic_power_curve_data = pd.read_csv(
            join(dirname(__file__), "synthetic_turbine_params.csv"), header=1)

    return _synthetic_power_curve_data


def compute_specific_power(capacity, rotor_diam, **k):
    """Computes specific power from capacity and rotor diameter"""
    return capacity * 1000 / rotor_diam**2 / np.pi * 4


class PowerCurve(namedtuple('PowerCurve', 'wind_speed capacity_factor')):
    """ 
    A wind turbine's power curve represented by a set of (wind-speed,capacty-factor) pairs:
    """

    def __str__(self):
        out = ""
        for ws, cf in zip(self.wind_speed, self.capacity_factor):
            out += "%6.2f - %4.2f\n" % (ws, cf)
        return out

    def _repr_svg_(self):
        # return str(self)

        import matplotlib.pyplot as plt
        from io import BytesIO

        plt.figure(figsize=(7, 3))
        plt.plot(self.wind_speed, self.capacity_factor, color=(0, 91 / 255, 130 / 255), linewidth=3)
        plt.tick_params(labelsize=12)
        plt.xlabel("wind speed [m/s]", fontsize=13)
        plt.ylabel("capacity output", fontsize=13)
        plt.tight_layout()
        plt.grid()

        f = BytesIO()
        plt.savefig(f, format="svg", dpi=100)
        plt.close()
        f.seek(0)
        return f.read().decode('ascii')

    @staticmethod
    def from_specific_power(specific_power, cutout=25):
        """The synthetic power curve generator creates a wind turbine power curve 
        based off observed relationships between turbine specific power and known
        power curves
        """
        # Create ws
        ws = [0, ]

        spcd = synthetic_power_curve_data()

        ws.extend(np.exp(spcd.const + spcd.scale * np.log(specific_power)))
        ws.extend(np.linspace(ws[-1], cutout, 20)[1:])
        ws = np.array(ws)

        # create capacity factor output
        cf = [0, ]
        cf.extend(spcd.perc_capacity / 100)
        cf.extend([1] * 19)
        cf = np.array(cf)

        # Done!
        return PowerCurve(ws, cf)

    @staticmethod
    def from_capacity_and_rotor_diam(capacity, rotor_diam, cutout=25):
        """The synthetic power curve generator creates a wind turbine power curve 
        based off observed relationships between turbine specific power and known
        power curves
        """
        return PowerCurve.from_specific_power(compute_specific_power(capacity, rotor_diam))

    def simulate(self, wind_speed):
        """apply the invoking power curve to the given wind speeds"""
        powerCurveInterp = splrep(self.wind_speed, self.capacity_factor)
        return splev(wind_speed, powerCurveInterp)

    def convolute_by_guassian(self, scaling=0.06, base=0.1, extend_beyond_cut_out=True, _min_speed=0.01, _max_speed=40, _steps=4000):
        """
        Convolutes a turbine power curve from a normal distribution function with wind-speed-dependent standard deviation.

        * The wind-speed-dependent standard deviation is computed with:
            std = wind_speed * scaling + base
        """
        # Initialize windspeed axis
        ws = np.linspace(_min_speed, _max_speed, _steps)
        dws = ws[1] - ws[0]

        # check if we have enough resolution
        tmp = (scaling * 5 + base) / dws
        if tmp < 1.0:  # manually checked threshold
            if tmp < 0.25:  # manually checked threshold
                raise ResError("Insufficient number of 'steps'")
            else:
                print("WARNING: 'steps' may not be high enough to properly compute the convoluted power curve. Check results or use a higher number of steps")

        # Initialize vanilla power curve
        selfInterp = splrep(ws, np.interp(ws, self.wind_speed, self.capacity_factor))

        cf = np.zeros(_steps)
        sel = ws < self.wind_speed.max()
        cf[sel] = splev(ws[sel], selfInterp)

        cf[ws < self.wind_speed.min()] = 0  # set all windspeed less than cut-in speed to 0
        cf[ws > self.wind_speed.max()] = 0  # set all windspeed greater than cut-out speed to 0 (just in case)
        cf[cf < 0] = 0  # force a floor of 0
        # cf[cf>self[:,1].max()] = self[:,1].max() # force a ceiling of the max capacity

        # Begin convolution
        convolutedCF = np.zeros(_steps)
        for i, ws_ in enumerate(ws):
            convolutedCF[i] = (norm.pdf(ws, loc=ws_, scale=scaling * ws_ + base) * cf).sum() * dws

        # Correct cutoff, maybe
        if not extend_beyond_cut_out:
            convolutedCF[ws > self.wind_speed[-1]] = 0

        # Done!
        ws = ws[::40]
        convolutedCF = convolutedCF[::40]
        return PowerCurve(ws, convolutedCF)
