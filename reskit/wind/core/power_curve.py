from collections import namedtuple
import numpy as np
from os.path import join, dirname
import pandas as pd
from scipy.interpolate import splrep, splev
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm

from ...util import ResError

_P = namedtuple("PowerCurve", "ws cf")
_synthetic_power_curve_data = None


def synthetic_power_curve_data() -> pd.DataFrame:
    """
    Reads the data used for creating a synthetic power curve.

    Returns
    -------
    pandas DataFrame
        DataFrame with the data to create a synthetic power curve.

    """
    global _synthetic_power_curve_data

    if _synthetic_power_curve_data is None:
        _synthetic_power_curve_data = pd.read_csv(
            join(dirname(__file__), "data", "synthetic_turbine_params.csv"), header=1
        )

    return _synthetic_power_curve_data


def compute_specific_power(capacity, rotor_diam, **k):
    """
    Calculates the corresponding specific power for a wind turbine in kW/m2 from values of capacity in kW and rotor diameter in m.

    Parameters
    ----------
    capacity : float or array-like
        Turbine's nominal capacity in kW.

    rotor_diam : float or array-like
        Turbine's hub height in m.

    Returns
    -------
    float or array-like
        Specific power in kW/m2

    """
    return capacity * 1000 / rotor_diam**2 / np.pi * 4


class PowerCurve:
    """
    Creates a wind turbine's power curve represented by a set of (wind-speed,capacity-factor) pairs.

    Initialization:

    Parameters
    ----------
        wind_speed : array-like
            The wind speeds values
        capacity_factor : array-like
            The corresponding capacity factor

    Returns
    -------
    PowerCurve object

    """

    def __init__(self, wind_speed, capacity_factor):
        self.wind_speed = np.array(wind_speed)
        self.capacity_factor = np.array(capacity_factor)

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
        plt.plot(
            self.wind_speed,
            self.capacity_factor,
            color=(0, 91 / 255, 130 / 255),
            linewidth=3,
        )
        plt.tick_params(labelsize=12)
        plt.xlabel("wind speed [m/s]", fontsize=13)
        plt.ylabel("capacity output", fontsize=13)
        plt.tight_layout()
        plt.grid()

        f = BytesIO()
        plt.savefig(f, format="svg", dpi=100)
        plt.close()
        f.seek(0)
        return f.read().decode("ascii")

    @staticmethod
    def from_specific_power(specific_power, cutout=25):
        """
        Creates a synthetic wind turbine power curve based on observed relationships between turbine specific power and known power curves according to Ryberg et al. [1].

        Parameters
        ----------
        specific_power : float
            Turbines's specific power in m/s

        cutout : int, optional
            Cut out wind speed in m/s, by default 25

        Returns
        -------
        PowerCurve

        Sources
        -------
        [1] Ryberg, D. S., Caglayan, D. G., Schmitt, S., Linßen, J., Stolten, D., & Robinius, M. (2019). The future of European onshore wind energy potential: Detailed distribution and simulation of advanced turbine designs. Energy. https://doi.org/10.1016/j.energy.2019.06.052

        See also
        --------
            PowerCurve.from_capacity_and_rotor_diam( <turbine capacity>, <turbine rotor diameter> )
        """
        # Create ws
        ws = [
            0,
        ]

        spcd = synthetic_power_curve_data()

        ws.extend(np.exp(spcd.const + spcd.scale * np.log(specific_power)))
        ws.extend(np.linspace(ws[-1], cutout, 20)[1:])
        ws = np.array(ws)

        # create capacity factor output
        cf = [
            0,
        ]
        cf.extend(spcd.perc_capacity / 100)
        cf.extend([1] * 19)
        cf = np.array(cf)

        # Done!
        return PowerCurve(ws, cf)

    @staticmethod
    def from_capacity_and_rotor_diam(capacity, rotor_diam, cutout=25):
        """
        Creates a synthetic wind turbine power curve based on observed relationships between turbine's capacity, rotor diameter and known power curves according to Ryberg et al. [1].

        Parameters
        ----------
        capacity : numeric
            Baseline turbine capacity in kW.

        rotor_diam : numeric
            turbine rotor diameter in m

        cutout : int, optional
            Cut out wind speed in m/s, by default 25

        Returns
        -------
        PowerCurve

        Sources
        -------
        [1] Ryberg, D. S., Caglayan, D. G., Schmitt, S., Linßen, J., Stolten, D., & Robinius, M. (2019). The future of European onshore wind energy potential: Detailed distribution and simulation of advanced turbine designs. Energy. https://doi.org/10.1016/j.energy.2019.06.052

        See also
        --------
            PowerCurve.from_specific_power( <turbine specific power> )
        """
        return PowerCurve.from_specific_power(
            compute_specific_power(capacity, rotor_diam)
        )

    def simulate(self, wind_speed):
        """
        Applies the invoking power curve to the given wind speeds.

        Parameters
        ----------
        wind_speed : array_like
            Local average wind speed close to or at the hub height.

        Returns
        -------
        array_like
            CorrespongDing capacity fators for the given wind speeds

        """

        powerCurveInterp = PchipInterpolator(self.wind_speed, self.capacity_factor)
        output = powerCurveInterp(wind_speed)

        if isinstance(wind_speed, pd.DataFrame):
            output = pd.DataFrame(
                output, index=wind_speed.index, columns=wind_speed.columns
            )

        return output

    def expected_capacity_factor_from_weibull(self, mean_wind_speed=5, weibull_shape=2):
        """
        Computes the expected average capacity factor of a wind turbine based on a Weibull distribution of wind speeds.

        Parameters
        ----------
        mean_wind_speed : int, optional
            mean wind speed at the location in m/s, by default 5

        weibull_shape : int, optional
            Weibull shape parameter, by default 2

        Returns
        -------
        numeric
            Average capacity factor

        See also
        -------
            PowerCurve.expected_capacity_factor_from_distribution

        """
        from scipy.special import gamma
        from scipy.stats import exponweib

        # Get windspeed distribution
        lam = mean_wind_speed / gamma(1 + 1 / weibull_shape)
        dws = 0.001
        ws = np.arange(0, 40, dws)
        pdf = exponweib.pdf(ws, 1, weibull_shape, scale=lam)

        # Estimate generation
        power_curveInterp = splrep(self.wind_speed, self.capacity_factor)
        gen = splev(ws, power_curveInterp)

        # Do some "just in case" clean-up
        cutin = self.wind_speed.min()  # use the first defined windspeed as the cut in
        cutout = self.wind_speed.max()  # use the last defined windspeed as the cut out

        gen[gen < 0] = 0  # floor to zero

        gen[ws < cutin] = 0  # Drop power to zero before cutin
        gen[ws > cutout] = 0  # Drop power to zero after cutout

        # Done
        meanCapFac = (gen * pdf).sum() * dws
        return meanCapFac

    def expected_capacity_factor_from_distribution(
        self, wind_speed_values, wind_speed_counts
    ):
        """
        Computes the expected average capacity factor of a wind turbine based on an explicitly-provided wind speed distribution

        Parameters
        ----------
        wind_speed_values : numeric or array-like
            wind speed values in m/s

        wind_speed_counts : numeric or array-like
            corresponding counts (number of occurrence) of the given wind speed values.
            Counts will be normalized within the function

        Example
        -------
            pc.expected_capacity_factor_from_distribution(
                wind_speed_values=[  1,   2,   3,   4,   5,      6], # Units of m/s
                wind_speed_counts=[0.1, 0.3, 0.5, 0.3, 0.1, 0.025 ]  # Units of "counts"
                )

        Returns
        -------
        numeric
            Average capacity factor

        See also
        -------
            PowerCurve.expected_capacity_factor_from_weibull

        """
        wind_speed_values = np.array(wind_speed_values)
        wind_speed_counts = np.array(wind_speed_counts)

        if not len(wind_speed_values.shape) == 1:
            raise ResError("wind_speed_values must be 1-dimensional")

        # Handle 2 dimensional counts with 1 dimensional wind speeds
        if len(wind_speed_counts.shape) > 1:
            if not wind_speed_counts.shape[0] == wind_speed_values.shape[0]:
                raise ResError("Dimensional incompatibility")

            wind_speed_values = np.reshape(
                wind_speed_values, (wind_speed_counts.shape[0], 1)
            )

        # Estimate generation distribution
        gen = (
            np.interp(
                wind_speed_values,
                self.wind_speed,
                self.capacity_factor,
                left=0,
                right=0,
            )
            * wind_speed_counts
        )

        meanGen = gen.sum(0) / wind_speed_counts.sum(0)

        # Done
        return meanGen

    def convolute_by_gaussian(
        self,
        scaling=0.06,
        base=0.1,
        extend_beyond_cut_out=True,
        _min_speed=0.01,
        _max_speed=40,
        _steps=4000,
    ):
        """
        Convolutes a turbine power curve by a normal distribution function with wind-speed-dependent standard deviation.

        Parameters
        ----------
        scaling : float, optional
            scaling factor, by default 0.06

        base : float, optional
            base value, by default 0.1

        extend_beyond_cut_out : bool, optional
            extend the estimation beyond the turbine's cut out wind speed, by default True

        _min_speed : float, optional
            minimum wind speed value in m/s to be considered, by default 0.01

        _max_speed : int, optional
            maximum wind speed value in m/s to be considered, by default 40

        _steps : int, optional
            number of steps in between the wind speed range, by default 4000

        Returns
        -------
        PowerCurve
            The resulting convoluted power curve

        Notes
        ------
        The wind-speed-dependent standard deviation is computed with: std = wind_speed * scaling + base

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
                print(
                    "WARNING: 'steps' may not be high enough to properly compute the convoluted power curve. Check results or use a higher number of steps"
                )

        # Initialize vanilla power curve
        selfInterp = splrep(ws, np.interp(ws, self.wind_speed, self.capacity_factor))

        cf = np.zeros(_steps)
        sel = ws < self.wind_speed.max()
        cf[sel] = splev(ws[sel], selfInterp)

        # set all windspeed less than cut-in speed to 0
        cf[ws < self.wind_speed.min()] = 0
        # set all windspeed greater than cut-out speed to 0 (just in case)
        cf[ws > self.wind_speed.max()] = 0
        cf[cf < 0] = 0  # force a floor of 0
        # cf[cf>self[:,1].max()] = self[:,1].max() # force a ceiling of the max capacity

        # Begin convolution
        convolutedCF = np.zeros(_steps)
        for i, ws_ in enumerate(ws):
            convolutedCF[i] = (
                norm.pdf(ws, loc=ws_, scale=scaling * ws_ + base) * cf
            ).sum() * dws

        # Correct cutoff, maybe
        if not extend_beyond_cut_out:
            convolutedCF[ws > self.wind_speed[-1]] = 0

        # Done!
        ws = ws[::40]
        convolutedCF = convolutedCF[::40]
        return PowerCurve(ws, convolutedCF)

    def apply_loss_factor(self, loss):
        """
        Applies a loss factor onto the power curve. It can be a single value, or a function which takes a 'capacity factor' array as input.

        Parameters
        ----------
        loss : numeric or function
            If numeric, the value is applied at all capacity factors with:
                new_capacity_factors = [1-loss] * previous_capacity_factors
            If a function, it must take a numpy array representing capacity factor values as input, resulting in the equation:
                new_capacity_factors = [1-loss(previous_capacity_factors)] * previous_capacity_factors

        Returns
        -------
        PowerCurve
            The corrected power curve.

        """
        try:
            cf = self.capacity_factor * (1 - loss)
        except:
            cf = self.capacity_factor * (1 - loss(self.capacity_factor))

        return PowerCurve(self.wind_speed, cf)
