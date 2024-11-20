import datetime
import geokit as gk
import pandas as pd
import numpy as np
import time
import warnings
import windpowerlib

from os import mkdir, environ
from os.path import join, isfile, isdir
from collections import OrderedDict, namedtuple
from types import FunctionType
from .. import core as rk_wind_core
from ...workflow_manager import WorkflowManager


class WindWorkflowManager(WorkflowManager):
    """
    Helps managing the logical workflow for simulations relating to wind turbines.

    Initialization:

    Parameters
    ----------
    placements : pandas.DataFrame
        A Pandas DataFrame describing the wind turbine placements to be simulated.
        It must include the following columns:
            'geom' or 'lat' and 'lon'
            'hub_height'
            'capacity'
            'rotor_diam' or 'powerCurve'

    synthetic_power_curve_cut_out : int, optional
        cut out wind speed, by default 25

    synthetic_power_curve_rounding : int, optional
        rounding floor, by default 1

    Returns
    -------
    numpy array
        A corrected power curve.

    """

    def __init__(
        self,
        placements,
        synthetic_power_curve_cut_out=25,
        synthetic_power_curve_rounding=1,
    ):
        # Do basic workflow construction
        super().__init__(placements)

        # Check for basics
        assert (
            "capacity" in self.placements.columns
        ), "Placement dataframe needs 'capacity' column"
        assert (
            "hub_height" in self.placements.columns
        ), "Placement dataframe needs 'hub_height' column"

        # Check for power curve. If not found, make it!
        self.powerCurveLibrary = dict()

        # Should we automatically generate synthetic power curves?
        def generate_missing_synthetic_power_curves(self):
            """
            Generates synthetic power curves for all placements that do not have a power curve defined.
            """
            placements_wo_PC = self.placements[self.placements.powerCurve.isna()]
            assert (
                "rotor_diam" in placements_wo_PC.columns
            ), "Placements needs 'rotor_diam' or 'powerCurve' specified"

            if len(placements_wo_PC) == 0:
                return

            specificPower = rk_wind_core.power_curve.compute_specific_power(
                placements_wo_PC["capacity"], placements_wo_PC["rotor_diam"]
            ).astype(float)

            if synthetic_power_curve_rounding is not None:
                specificPower = (
                    np.round(specificPower / synthetic_power_curve_rounding)
                    * synthetic_power_curve_rounding
                )
                specificPower = specificPower.astype(int)

            powerCurve = []
            for sppow in specificPower:
                pcid = "SPC:%d,%d" % (sppow, synthetic_power_curve_cut_out)
                powerCurve.append(pcid)

            self.placements.loc[self.placements.powerCurve.isna(), "powerCurve"] = (
                powerCurve
            )

        if not "powerCurve" in self.placements.columns:
            assert (
                "rotor_diam" in self.placements.columns
            ), "Placement dataframe needs 'rotor_diam' or 'powerCurve' column"
            self.placements["powerCurve"] = None
            generate_missing_synthetic_power_curves(self)
        else:
            generate_missing_synthetic_power_curves(self)

        # Put power curves into the power curve library
        for pc in self.placements.powerCurve.values:
            assert isinstance(
                pc, str
            ), "Power curve value needs to be a string, not " + type(pc)

            if pc in self.powerCurveLibrary:
                continue

            if pc[:4] == "SPC:":
                sppow, cutout = pc.split(":")[1].split(",")
                self.powerCurveLibrary[pc] = (
                    rk_wind_core.power_curve.PowerCurve.from_specific_power(
                        specific_power=float(sppow), cutout=float(cutout)
                    )
                )
            else:
                self.powerCurveLibrary[pc] = (
                    rk_wind_core.turbine_library.TurbineLibrary().loc[pc].PowerCurve
                )

    def set_roughness(self, roughness):
        """
        Sets the 'roughness' column in the placements DataFrame.

        Parameters
        ----------
        roughness : numeric, iterable
            If a numeric is given, sets the same roughness values to all placements.
            If an iterable is given, sets the corresponding roughness value in the iterable to the placements.
            The length of the iterable must match the number of placements

        Return
        ------
            A reference to the invoking WindWorkflowManager
        """
        self.placements["roughness"] = roughness
        return self

    def estimate_roughness_from_land_cover(self, path, source_type):
        """
        Estimates the 'roughness' value column in the placements DataFrame from a given land cover classification raster file.

        Parameters
        ----------
        path : str
            path to the raster file
        source_type : str
            string value to get the corresponding key-value pairs. Accepted types 'clc', 'clc-code', 'globCover', 'modis', or 'cci', by default 'clc'

        See also
        --------
            roughness_from_land_cover_classification

        Return
        --------
            A reference to the invoking WindWorkflowManager
        """
        num = gk.raster.interpolateValues(path, self.locs, mode="near")
        self.placements["roughness"] = (
            rk_wind_core.logarithmic_profile.roughness_from_land_cover_classification(
                num, source_type
            )
        )
        return self

    def logarithmic_projection_of_wind_speeds_to_hub_height(
        self, consider_boundary_layer_height=False
    ):
        """
        Projects the wind speed values to the hub height.

        Return
        ------
            A reference to the invoking WindWorkflowManager
        """

        assert "roughness" in self.placements.columns
        assert hasattr(self, "elevated_wind_speed_height")

        if consider_boundary_layer_height:
            # When the hub height is above the PBL, then only project to the PBL
            target_height = np.minimum(
                self.sim_data["boundary_layer_height"],
                self.placements["hub_height"].values,
            )

            # When the PBL is below the elevated_wind_speed_height, then no projection
            # should be performed. This can be effectlvely accomplished by setting the
            # target height to that of the elevated_wind_speed_height
            sel = target_height > self.elevated_wind_speed_height
            target_height[sel] = self.elevated_wind_speed_height

        else:
            target_height = self.placements["hub_height"].values

        tmp = rk_wind_core.logarithmic_profile.apply_logarithmic_profile_projection(
            self.sim_data["elevated_wind_speed"],
            measured_height=self.elevated_wind_speed_height,
            target_height=target_height,
            roughness=self.placements["roughness"].values,
        )

        self.sim_data["elevated_wind_speed"] = tmp

        self.elevated_wind_speed_height = self.placements["hub_height"].values

        return self

    def apply_air_density_correction_to_wind_speeds(self):
        """
        Applies air density corrections to the wind speeds at the hub height.

        Return
        ------
            A reference to the invoking WindWorkflowManager


        """

        assert (
            "surface_air_temperature" in self.sim_data
        ), "surface_air_temperature has not been read from a source"
        assert (
            "surface_pressure" in self.sim_data
        ), "surface_pressure has not been read from a source"
        assert hasattr(self, "elevated_wind_speed_height")

        self.sim_data["elevated_wind_speed"] = (
            rk_wind_core.air_density_adjustment.apply_air_density_adjustment(
                self.sim_data["elevated_wind_speed"],
                pressure=self.sim_data["surface_pressure"],
                temperature=self.sim_data["surface_air_temperature"],
                height=self.elevated_wind_speed_height,
            )
        )

        return self

    def apply_wake_correction_of_wind_speeds(
        self,
        wake_reduction_curve_name="dena_mean",
    ):
        """
        Applies a wind-speed dependent reduction factor to the wind speeds at elevated height,
        based on

        Parameters
        ----------
        wake_reduction_curve_name : str, optional
            string value to describe the wake reduction method. None will cause no reduction,
            by default "dena_mean". Choose from (see more information here under wind_efficiency_curve_name[1]):
            * "dena_mean",
            * "knorr_mean",
            * "dena_extreme1",
            * "dena_extreme2",
            * "knorr_extreme1",
            * "knorr_extreme2",
            * "knorr_extreme3",

        Return
        ------
            A reference to the invoking WindWorkflowManager
        """
        # return as is if no wake reduction shall be applied
        if wake_reduction_curve_name is None:
            return self

        assert hasattr(self, "elevated_wind_speed_height")
        self.sim_data["elevated_wind_speed"] = (
            windpowerlib.wake_losses.reduce_wind_speed(
                self.sim_data["elevated_wind_speed"],
                wind_efficiency_curve_name=wake_reduction_curve_name,
            )
        )

        return self

    def convolute_power_curves(self, scaling, base, **kwargs):
        """
        Convolutes a turbine power curve from a normal distribution function with wind-speed-dependent standard deviation.

        Parameters
        ----------
        scaling : float, optional
            scaling factor, by default 0.06
        base : float, optional
            base value, by default 0.1

        Return
        ------
            A reference to the invoking WindWorkflowManager

        """
        assert hasattr(self, "powerCurveLibrary")

        for key in self.powerCurveLibrary.keys():
            self.powerCurveLibrary[key] = self.powerCurveLibrary[
                key
            ].convolute_by_gaussian(scaling=scaling, base=base, **kwargs)

        return self

    def simulate(
        self,
        max_batch_size=None,
        cf_correction_factor=1.0,
        tolerance=0.01,
        max_iterations=10,
        verbose=True,
    ):
        """
        Applies the invoking power curve to the given wind speeds.
        A max_batch_size can be set, splitting the simulation in batches.
        If set, cf_correction_factor is applied iteratively to adjust avreage cf output.
        Capacity factors are calculated in the subfunction _sim(), which is called iteratively.

        max_batch_size : int, optional
            The maximum number of locations to be simulated simultaneously.
            If None, no limits will be applied, by default None.

        cf_correction_factor : float, optional
            The average cf output will be adjusted by this ratio
            via wind speed adaptations (no linear scaling). By default 1.0.

        tolerance : float, optional
            The max. deviation of the simulated average cf from the enforced
            corrected value, by default 0.03, i.e. 3% absolute.

        max_iterations : int, optional
            The max. No. of simulation iteratons allowed for iterative
            simulation of one batch until the tolerance is met, else a
            TimeOutError will be raised. By default 10 iterations.

        verbose : bool, optional
            If True, additional status information will be printed, by
            default True.

        Return
        ------
            A reference to the invoking WindWorkflowManager
        """

        def _sim(ws_correction_factors, _batch, max_batch_size, sel):
            """
            Applies the invoking power curve to the given wind speeds.
            """

            _gen = np.zeros_like(
                self.sim_data["elevated_wind_speed"][
                    :, _batch * max_batch_size : (_batch + 1) * max_batch_size
                ]
            )

            for pckey, pc in self.powerCurveLibrary.items():
                _sel = (
                    self.placements.iloc[
                        _batch * max_batch_size : (_batch + 1) * max_batch_size, :
                    ].powerCurve
                    == pckey
                )
                # simulate only intersection of selection (sel) and power curve selection (_sel)
                _sel = np.logical_and(_sel, sel)
                if not _sel.any():
                    continue
                _gen[:, _sel] = np.round(
                    pc.simulate(
                        self.sim_data["elevated_wind_speed"][
                            :, _batch * max_batch_size : (_batch + 1) * max_batch_size
                        ][:, _sel]
                        * ws_correction_factors[_sel]
                    ),
                    3,
                )
                # set values < 0 to zero. Prevents negative values
                _gen[_gen < 0] = 0

            return _gen

        if max_batch_size is not None:
            if not isinstance(max_batch_size, int) and max_batch_size > 0:
                raise TypeError(f"max_batch_size must be an integer > 0")
            if max_batch_size > len(self.locs):
                max_batch_size = len(self.locs)
        else:
            max_batch_size = self.sim_data["elevated_wind_speed"].shape[1]

        # calculate required No. of batches
        _batches = np.ceil(
            self.sim_data["elevated_wind_speed"].shape[1] / max_batch_size
        )

        # get and set correction factor
        self.set_correction_factors(correction_factors=cf_correction_factor)

        # iterate over batches
        for _batch in range(int(_batches)):
            # calculate a starting point generation value
            if _batch == int(_batches) - 1:
                # if the last batch, the length may be shorter if total placements No is not a multiple of max_batch_size
                len_locs = len(self.locs) - (_batch * max_batch_size)
            else:
                # all batches besides possibly the last must be of length max_batch_size
                len_locs = max_batch_size
            if verbose:
                print(
                    datetime.datetime.now(),
                    f"Based on max_batch_size={max_batch_size}, the total of {len(self.locs)} placements were split into {int(_batches)} sub batches. Proceeding with batch {_batch+1}/{int(_batches)} (id={_batch}) with {len_locs} placements.",
                )

            # simulate first time to get the undistorted RESkit cfs
            sel = np.full(len_locs, True)
            gen_last = _sim(
                ws_correction_factors=np.array([1.0] * len_locs),
                _batch=_batch,
                max_batch_size=max_batch_size,
                sel=sel,
            )
            avg_gen_last = np.nanmean(gen_last, axis=0)

            # calculate the target average cf as product of raw RESkit cf and correction factor
            _target_cfs = (
                avg_gen_last
                * self.correction_factors[
                    _batch * max_batch_size : (_batch + 1) * max_batch_size
                ]
            )
            # make sure that the average (usually annual) target cs is realistic in all locs
            if (
                isinstance(cf_correction_factor, (float, int))
                and cf_correction_factor == 1
            ):
                # we have no correction, but cfs may not be > 1
                if (_target_cfs > 1).any():
                    raise ValueError(
                        f"The turbine parameters lead to average capacity factors greater 1.0."
                    )
            else:
                # we correct values, make sure they do not get unrealistically close to 1.0 (needs room for some non-windy hours/year)
                if (_target_cfs > 0.95).any():
                    warnings.warn(
                        f"The current correction factors lead to average target capacity factors greater 0.95. The correction factors at these locations will be adjusted such that the maximum target capacity factor is 0.95."
                    )
                # get values where the target cf is greater than 0.95
                _sel = _target_cfs > 0.95
                # modify the target cfs such that 0.95 is the maximum target cf
                _target_cfs[_sel] = 0.95

            # make sure the target cf is not not NaN, possibly due to missing GWA cell value
            if np.isnan(_target_cfs).any():
                warnings.warn(
                    f"WARNING: {len(self.locs[_batch*max_batch_size:(_batch+1)*max_batch_size][np.isnan(_target_cfs)])} NaNs detected in weather data LRA: {self.locs[_batch*max_batch_size:(_batch+1)*max_batch_size][np.isnan(_target_cfs)]}"
                )

            # set the initial deviation based on initial, undistorted generation vs target generation
            _deviations_last = avg_gen_last / _target_cfs

            # calculate the min. required relative convergence per iteration to achieve tolerance after max. iterations
            min_convergence = 1 - (tolerance / abs(_deviations_last - 1)) ** (
                1 / max_iterations
            )

            # initialize the correction factors as 1.0 everywhere, will be adapted first thing if tolerance is not met by deviations
            _ws_corrs_current = np.array([1.0] * len(_deviations_last))

            # iterate until the target cf average is met, i.e. until absolute deltas of 1.0 and deviations are all less than tolerance
            _itercount = 0
            while (abs(_deviations_last - 1) > tolerance).any():
                # safety fallback - exit in case of infinite loops
                if _itercount > max_iterations:
                    raise TimeoutError(
                        f"{str(datetime.datetime. now())} The simulation did not reach the required tolerance of {tolerance} within the given max. {max_iterations} iterations. Remaining max. absolute deviation is {round(max(abs(_deviations_last - 1)),4)}. Number of placements with deviation > {tolerance}: {sum(abs(_deviations_last - 1)>tolerance)}/{len(_deviations_last)}. Increase tolerance or max_iterations value."
                    )
                # print deviation status for the current iteration
                if verbose:
                    print(
                        datetime.datetime.now(),
                        f"Maximum rel. deviation after {'initial simulation' if _itercount==0 else str(_itercount)+' additional iteration(s)'} is {round(max(abs(_deviations_last - 1)),4)}, Number/share of placements with deviation > tolerance ({tolerance}): {sum(abs(_deviations_last - 1)>tolerance)}/{len(_deviations_last)}. More iterations required.",
                    )

                # update the estimated correction factor for the wind speed for this iteration
                _ws_corrs_current = _ws_corrs_current * np.cbrt(
                    1 / _deviations_last
                )  # power law

                # calculate only off-tolerance locs with an adapted ws correction
                sel = abs(_deviations_last - 1) > tolerance
                # simulate only the placements to be updated
                # Note that gen_current contains zeros where not sel
                gen_current = _sim(
                    ws_correction_factors=_ws_corrs_current,
                    _batch=_batch,
                    max_batch_size=max_batch_size,
                    sel=sel,
                )
                # write the old values into those locations who have met tolerance already (was zero so far)
                gen_current[:, ~sel] = gen_last[:, ~sel]

                # calculate the average cf per location
                avg_gen_current = np.nanmean(gen_current, axis=0)

                # calculate the new preliminary deviation factors from current simulated gen
                _deviations_current = avg_gen_current / _target_cfs

                # identify those locations where the cf does not converge (sufficiently)
                # but exclude those that have reached the tolerance already (no further conversion)
                _non_convs = np.isnan(_target_cfs) | (
                    abs(_deviations_current - 1)
                    > (1 - min_convergence) * abs(_deviations_last - 1)
                ) & (abs(_deviations_current - 1) > tolerance)
                if _non_convs.sum() > 0:
                    print(
                        f"{_non_convs.sum()}/{len(_deviations_current)} placements ({round(_non_convs.sum()/len(_deviations_current)*100, 2)}%) did not converge (sufficiently). Average cf will be enforced.",
                        flush=True,
                    )

                if (
                    (
                        ((gen_current == 0) | (gen_current == 1)).sum(axis=0) / 8760
                        < 0.15
                    )[_non_convs]
                ).any():
                    f"{((((gen_current == 0) | (gen_current == 1)).sum(axis=0) / 8760 < 0.15)[_non_convs]).sum()}non-converging placements with <15% cf=0 or cf=1.0 found: {(((gen_current == 0) | (gen_current == 1)).sum(axis=0) / 8760)[_non_convs]}"

                def correct_cf(arr, target_mean):
                    """Adapts average of 'arr' to 'target_mean' value without removing zeros/1.0s."""
                    # handle NaN target cfs
                    if np.isnan(target_mean):
                        _arr = np.empty(len(arr))
                        _arr[:] = np.nan
                        return _arr

                    FLH_in = np.sum(arr)
                    FLH_target = target_mean * len(arr)
                    FLH_diff = FLH_target - FLH_in
                    _break = False

                    if FLH_diff > 0:
                        while sum(arr) < FLH_target:
                            delta_max = 1 - arr[arr < 1].max()
                            _add = np.where(arr > 0.5, delta_max, arr * delta_max)
                            _add[arr == 1.0] = (
                                0  # set delta to zero for cf=1 to have a correct total FLH delta
                            )
                            # scale if needed
                            if _add.sum() > (FLH_target - arr.sum()):
                                _add = _add * (FLH_target - arr.sum()) / _add.sum()
                                _break = True
                            arr = np.where(arr < 1, arr + _add, arr)
                            if _break:
                                break

                    if FLH_diff < 0:
                        while sum(arr) > FLH_target:
                            delta_min = arr[arr > 0].min()
                            _ded = np.where(arr < 0.5, delta_min, (1 - arr) * delta_min)
                            _ded[arr == 0] = (
                                0  # set delta to zero for cf=0 to have a correct total FLH delta
                            )
                            # scale if needed
                            if _ded.sum() > (arr.sum() - FLH_target):
                                _ded = _ded * abs(arr.sum() - FLH_target) / _ded.sum()
                                _break = True
                            arr = np.where(arr > 0, arr - _ded, arr)
                            if _break:
                                break

                    return arr

                # iterate over locations with diverging cfs
                for i, _non_conv in enumerate(_non_convs):
                    if _non_conv:
                        gen_current[:, i] = correct_cf(
                            gen_current[:, i], _target_cfs[i]
                        )
                        # ensure that the forced avg adaptation achieved a deviation < tolerance
                        assert (
                            np.isnan(_target_cfs[i])
                            or abs(gen_current[:, i].mean() / _target_cfs[i] - 1)
                            < tolerance
                        ), f"Tolerance was not met after enforced adaptation of average cf."

                # calculate new current cf per location after convergence fix
                avg_gen_current = np.nanmean(gen_current, axis=0)
                # now calculate the latest deviation factors after convergence fix
                _deviations_current = avg_gen_current / _target_cfs

                # LAST STEP - RENAME FOR NEXT ITERATION

                # the "current" iteration becomes "last" for the next round
                gen_last = gen_current.copy()
                avg_gen_last = avg_gen_current.copy()
                _deviations_last = _deviations_current.copy()
                # delete the "current" variables, will be recalculated next iteration
                del gen_current, avg_gen_current, _deviations_current
                # increase iteration counter by 1
                _itercount += 1

            # when required tolerance is achieved, continue
            if verbose:
                print(
                    datetime.datetime.now(),
                    f"Required tolerance of {tolerance} reached after {_itercount} additional iteration(s). Maximum remaining rel. deviation: {round(max(abs(_deviations_last - 1)),4)}.",
                    flush=True,
                )

            _max_cfs = gen_last.max(axis=0)
            if (gen_last > 1).any():
                print(
                    datetime.datetime.now(),
                    f"Required target cf could not be reached for some locations, cf will be reduced by factor min/max. {np.nanmin(1/_max_cfs)}/{np.nanmax(1/_max_cfs)} in order to not exceed cf=1.0.",
                    flush=True,
                )
                _red = 1 / _max_cfs
                _red[_max_cfs <= 1] = 1
                gen_last = gen_last * _red

            if _batch == 0:
                tot_gen = gen_last
            else:
                tot_gen = np.concatenate([tot_gen, gen_last], axis=1)

        self.sim_data["capacity_factor"] = tot_gen

        return self

    def apply_availability_factor(self, availability_factor):
        """
        Applies a relative reduction factor to the energy output (capacity factor) time series
        to statistically account for non-availabilities.

        Parameters
        ----------
        availability_factor : float
            Factor that will be applied to the output time series.

        Return
        ------
            A reference to the invoking WindWorkflowManager
        """
        assert (
            availability_factor > 0 and availability_factor <= 1
        ), f"availability_factor must be between 0 and 1.0."

        self.sim_data["capacity_factor"] = (
            self.sim_data["capacity_factor"] * availability_factor
        )

        return self

    def apply_availability_factor(self, availability_factor):
        """
        Applies a relative reduction factor to the energy output (capacity factor) time series
        to statistically account for non-availabilities.

        Parameters
        ----------
        availability_factor : float
            Factor that will be applied to the output time series.

        Return
        ------
            A reference to the invoking WindWorkflowManager
        """
        assert (
            availability_factor > 0 and availability_factor <= 1
        ), f"availability_factor must be between 0 and 1.0."

        self.sim_data["capacity_factor"] = (
            self.sim_data["capacity_factor"] * availability_factor
        )

        return self

    def interpolate_raster_vals_to_hub_height(
        self, name: str, height_to_raster_dict: dict, **kwargs
    ):
        """Given several raster datasets which correspond to a desired value (e.g. average wind speed) at
        different altitudes, this function will read values for each placement location from each of these
        datasets, and will then linearly interpolate them to the hub height of each turbine

        Parameters
        ----------
        name : str
            The name of the variable to create (will be placed in the `self.placements` member)

        height_to_raster_dict : dict
            A dictionary which maps altitude values to raster datasets

        Return
        ------
            A reference to the invoking WindWorkflowManager
        """
        known_heights = sorted(height_to_raster_dict.keys())
        known_vals = []
        for h in known_heights:
            known_vals.append(
                self.extract_raster_values_at_placements(
                    height_to_raster_dict[h], **kwargs
                )
            )

        interpolated_vals = np.full_like(known_vals[0], np.nan)
        hh = self.placements["hub_height"].values
        for hi in range(len(known_heights) - 1):
            sel = np.logical_and(hh >= known_heights[hi], hh < known_heights[hi + 1])
            if sel.any():
                interpolated_vals[sel] = (hh[sel] - known_heights[hi]) / (
                    known_heights[hi + 1] - known_heights[hi]
                ) * (known_vals[hi + 1] - known_vals[hi]) + known_vals[hi]

        if np.isnan(interpolated_vals).any():
            raise RuntimeError("Could not determine interpolation for all hub heights")

        self.placements[name] = interpolated_vals
        return self

    def set_correction_factors(self, correction_factors):
        """
        Gets the correction factors if necessary and sets them as class attribute.

        Parameters
        ----------
        correction_factors : str, float
            correction factor as float or path to the correction factor raster file

        Return
        --------
            A reference to the invoking WindWorkflowManager
        """
        if isinstance(correction_factors, str):
            if not isfile(correction_factors):
                raise FileNotFoundError(
                    f"correction_factors was passed as str but is not an existing file: {correction_factors}"
                )
            correction_factors = gk.raster.interpolateValues(
                correction_factors, self.locs, mode="near"
            )
            assert not np.isnan(
                correction_factors
            ).any(), f"correction_factors extracted from raster must not be nan"
        elif not isinstance(correction_factors, (float, int)):
            raise TypeError(
                f"correction_factors must either be a str formatted raster filepath or a float value"
            )
        else:
            correction_factors = [correction_factors] * len(self.locs)

        # write to attribute
        self.correction_factors = np.array(correction_factors)

        return self
