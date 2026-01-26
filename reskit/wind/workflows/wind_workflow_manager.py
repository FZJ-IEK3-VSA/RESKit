import datetime
import time
import warnings
from collections import OrderedDict, namedtuple
from os import environ, mkdir
from os.path import isdir, isfile, join
from types import FunctionType

import geokit as gk
import numpy as np
import pandas as pd
import windpowerlib

from reskit.wind import core as rk_wind_core
from reskit.wind.core import turbine_library
from reskit.workflow_manager import WorkflowManager


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
        assert "capacity" in self.placements.columns, "Placement dataframe needs 'capacity' column"
        assert "hub_height" in self.placements.columns, "Placement dataframe needs 'hub_height' column"

        # Check for power curve. If not found, make it!
        self.powerCurveLibrary = dict()

        # Should we automatically generate synthetic power curves?
        def generate_missing_synthetic_power_curves(self):
            """
            Generates synthetic power curves for all placements that do not have a power curve defined.
            """
            placements_wo_pc = self.placements[
                self.placements.powerCurve.isna() | (self.placements.powerCurve == "nan")
            ]
            assert "rotor_diam" in placements_wo_pc.columns, "Placements needs 'rotor_diam' or 'powerCurve' specified"

            if len(placements_wo_pc) == 0:
                return

            specific_power = rk_wind_core.power_curve.compute_specific_power(
                placements_wo_pc["capacity"], placements_wo_pc["rotor_diam"]
            ).astype(float)

            if synthetic_power_curve_rounding is not None:
                specific_power = (
                    np.round(specific_power / synthetic_power_curve_rounding) * synthetic_power_curve_rounding
                )
                specific_power = specific_power.astype(int)

            power_curve = []
            for sppow in specific_power:
                pcid = "SPC:%d,%d" % (sppow, synthetic_power_curve_cut_out)
                power_curve.append(pcid)

            self.placements.loc[placements_wo_pc.index, "powerCurve"] = power_curve

        if not "powerCurve" in self.placements.columns:
            assert "rotor_diam" in self.placements.columns, (
                "Placement dataframe needs 'rotor_diam' or 'powerCurve' column"
            )
            self.placements["powerCurve"] = None
        generate_missing_synthetic_power_curves(self)

        # Put power curves into the power curve library
        for pc in self.placements.powerCurve.values:
            assert isinstance(pc, str), "Power curve value needs to be a string, not " + type(pc)

            if pc in self.powerCurveLibrary:
                continue

            if pc[:4] == "SPC:":
                sppow, cutout = pc.split(":")[1].split(",")
                self.powerCurveLibrary[pc] = rk_wind_core.PowerCurve.from_specific_power(
                    specific_power=float(sppow), cutout=float(cutout)
                )
            else:
                self.powerCurveLibrary[pc] = rk_wind_core.turbine_library.turbine_library().loc[pc].PowerCurve

    def project_windspeeds_to_hub_height(
        self,
        height_scaling_method,
        height_scaling_data,
        consider_boundary_layer_height=True,
        allow_extrapolation=True,
    ):
        """
        Projects wind speeds to hub heights for a given projection/scaling method.

        Parameters
        ----------
        height_scaling_method : tuple
            The method to project the windspeeds from the default height (here
            100m in ERA-5/GWA3) to hub height (possibly affected by the planetary
            boundary layer height). First tuple entry (str) describes the general
            approach (e.g. logarithmic scaling or based on long-run-average
            windspeeds). No height scaling will be applied when None. Options are:
            ("lra", [vertical method]) : Calculation based on the long-run average
                wind speeds (e.g. GWA) of the 2 nearest available height levels.
                [vertical method] (str) describes the form of interpolation, e.g.
                "linear".
            ("log", [landcover]) : Logarithmic height scaling based on surface
                roughness defined via a mapping of the land cover category.
                [landcover] (str) defines the landcover data used for roughness
                mapping. All landcover types accepted as land_cover_type in
                logarithmic_profile.roughness_from_land_cover_classification() are
                allowed, by default "cci" (ESA CCI raster).
        height_scaling_data : str, dict
            The data required for the selected height_scaling_method (see above).
            The expected data formats are, depending on height_scaling_method:
            ("log", [landcover]) : str
                Path to the respective "landcover" raster file.
            ("lra", [vertical method]) : {int : str}
                Dict with heights as keys and paths to the LRA-windspeeds at the
                respective heights as values. Must contain at least one higher and
                one lower height than the reference height of real_lra_ws_path.
        consider_boundary_layer_height : bool, optional
            Corrects the given hub heights for locations by planetary boundary
            layer (PBL) effects if True, see consider_boundary_height() for
            details. By default True.
        allow_extrapolation : bool, optional
            Takes effect only in interpolating methods, will then allow to
            extrapolate beyond the min/max height range, by default True.

        Return
        ------
            A reference to the invoking WindWorkflowManager
        """
        # check consider_boundary_layer_height arg
        if not isinstance(consider_boundary_layer_height, bool):
            raise TypeError("consider_boundary_layer_height must be boolean.")
        if not isinstance(height_scaling_method, tuple) and len(height_scaling_method) == 2:
            raise TypeError(f"height_scaling_method must be a tuple of length 2.")

        if height_scaling_method[0] == "log":
            # we have a logarithmic scaling approach, check landcover raster
            if not isinstance(height_scaling_data, str) and isfile(height_scaling_method):
                raise TypeError(
                    "height_scaling_method must be str formatted path if height_scaling_method==('log', [landcover])"
                )
            # first get surface roughness per location, then project
            self.estimate_roughness_from_land_cover(path=height_scaling_data, source_type=height_scaling_method[1])
            self.logarithmic_projection_of_wind_speeds_to_hub_height(
                consider_boundary_layer_height=consider_boundary_layer_height
            )

        elif height_scaling_method[0] == "lra":
            # we have an interpolation method based on other available LRA wind speed heights

            # if keys of height_scaling_data are str, try to convert to int or float
            if isinstance(height_scaling_data, dict) and all([isinstance(k, str) for k in height_scaling_data.keys()]):
                new_dict = dict()
                for k, v in height_scaling_data.items():
                    try:
                        new_k = int(k)
                    except ValueError:
                        try:
                            new_k = float(k)
                        except ValueError:
                            raise ValueError(
                                f"height_scaling_data keys must be int or float, not {k} of type {type(k)}"
                            )
                    new_dict[new_k] = v
                height_scaling_data = new_dict

            lra_funcs = {
                "linear": {
                    "func": self.wind_shear_projection_of_wind_speeds_to_hub_height,
                    "args": {
                        "alternative_wind_speed_rasters": height_scaling_data,
                        "consider_boundary_layer_height": consider_boundary_layer_height,
                        "allow_extrapolation": allow_extrapolation,
                    },
                },
            }
            if height_scaling_method[1] in lra_funcs:
                lra_funcs[height_scaling_method[1]]["func"](**lra_funcs[height_scaling_method[1]]["args"])
            else:
                raise ValueError(
                    f"2nd entry of height_scaling_method [vertical method] unknown. Select from: {', '.join(lra_funcs.keys())}"
                )

        else:
            raise ValueError(f"Unknown height_scaling_method '{height_scaling_method}'.")

        return self

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

        See Also
        --------
            roughness_from_land_cover_classification

        Return
        --------
            A reference to the invoking WindWorkflowManager
        """
        num = gk.raster.interpolateValues(path, self.locs, mode="near")
        self.placements["roughness"] = rk_wind_core.logarithmic_profile.roughness_from_land_cover_classification(
            num, source_type
        )
        return self

    def consider_boundary_height(self):
        """
        Corrects the given target heights for locations by planetary boundary
        layer (PBL) effects, by limiting the target height either to the PBL
        height when elevated (starting) height < PBL < target height or avoiding
        scaling altogether when PBL <= elevated (startig) height (by setting
        target height to elevated starting height).

        Return
        ------
        numpy array
            The adapted target heights.
        """
        assert hasattr(self, "elevated_wind_speed_height")
        assert "hub_height" in self.placements.columns

        # correct the target height acc. to one of these cases
        # 1) EH <= PBLH & HH <= PBLH -> TH = HH (no influence of PBL)
        # 2) EH <= PBLH & HH > PBLH -> TH = PBLH (set PBLH as upper target height limit)
        # 3) PBLH < EH & PBLH <= HH -> TH = EH (all heights are outside of planetary influence, use (constant) ws(EH))
        # 4) PBLH < EH & PBLH > HH -> TH = HH, EH -> PBLH (scaling relative to PBLH since ws(EH) == ws(PBLH)

        # Get the relevant variables
        pbl_height = self.sim_data["boundary_layer_height"]
        hub_height = self.placements["hub_height"].values
        elevated_height = self.elevated_wind_speed_height

        # Initialize target height array
        target_height = np.zeros_like(pbl_height, dtype=float)

        # Case 1: EH <= PBLH & HH <= PBLH -> TH = HH (no influence of PBL)
        case1 = (elevated_height <= pbl_height) & (hub_height <= pbl_height)
        target_height = np.where(case1, hub_height, target_height)

        # Case 2: EH <= PBLH & HH > PBLH -> TH = PBLH (set PBLH as upper target height limit)
        case2 = (elevated_height <= pbl_height) & (hub_height > pbl_height)
        target_height = np.where(case2, pbl_height, target_height)

        # Case 3: PBLH < EH & PBLH <= HH -> TH = EH (all heights are outside of planetary influence, use (constant) ws(EH))
        case3 = (pbl_height < elevated_height) & (pbl_height <= hub_height)
        target_height = np.where(case3, elevated_height, target_height)

        # Case 4: PBLH < EH & PBLH > HH -> TH = HH, EH -> PBLH (scaling relative to PBLH since ws(EH) == ws(PBLH))
        # Note: This case also requires adjusting the elevated_wind_speed_height, but that's handled in the calling function
        case4 = (pbl_height < elevated_height) & (pbl_height > hub_height)
        target_height = np.where(case4, hub_height, target_height)

        assert (np.array([case1, case2, case3, case4]).sum(axis=0) == 1).all()

        return target_height

    def logarithmic_projection_of_wind_speeds_to_hub_height(self, consider_boundary_layer_height=False):
        """
        Projects the wind speed values to the hub height.

        consider_boundary_layer_height : bool, optional
            If True, the wind speed will be scaled only to max. the
            boundary layer height. By default False.

        Return
        ------
            A reference to the invoking WindWorkflowManager
        """
        assert "roughness" in self.placements.columns
        assert hasattr(self, "elevated_wind_speed_height")

        if consider_boundary_layer_height:
            # only scale up to the maximum of boundary height or hub height
            target_height = self.consider_boundary_height()
        else:
            # else simply scale to hub height
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

    def wind_shear_projection_of_wind_speeds_to_hub_height(
        self,
        alternative_wind_speed_rasters,
        consider_boundary_layer_height=False,
        allow_extrapolation=True,
    ):
        """
        Projects the wind speed values to the hub height.

        consider_boundary_layer_height : bool, optional
            If True, the wind speed will be scaled only to max. the
            boundary layer height. By default False.
        allow_extrapolation . BOOL; OPTIONAL
            If False, target heights must be between minimum and maximum
            height keys provided in alternative_wind_speed_rasters. By
            default True.

        Return
        ------
            A reference to the invoking WindWorkflowManager
        """
        assert isinstance(allow_extrapolation, bool), f"allow_extrapolation must be boolean"
        assert isinstance(alternative_wind_speed_rasters, dict) and all(
            [isinstance(k, (int, float)) and k > 0] for k in alternative_wind_speed_rasters.keys()
        ), f"alternative_wind_speed_rasters is expected to be a dict with positive int or float as keys"
        assert all([isinstance(v, str) and isfile(v)] for v in alternative_wind_speed_rasters.values()), (
            f"alternative_wind_speed_rasters values must be paths to existing windspeed rasters."
        )
        assert hasattr(self, "elevated_wind_speed_height")

        # CONSIDER BOUNDARY LAYER HEIGHT - OR NOT

        if consider_boundary_layer_height:
            # only scale up to the maximum of boundary height or hub height
            target_height = self.consider_boundary_height()
        else:
            # else simply scale to hub height, but repeat columns for every timestep
            target_height = self.placements["hub_height"].values
            target_height = np.repeat(
                target_height[np.newaxis, :],
                self.sim_data["elevated_wind_speed"].shape[0],
                axis=0,
            )

        # GET AVERAGE WINDSPEEDS AT NEAREST GIVEN REFERENCE HEIGHTS

        # first get all available reference heights, including the default elevated ws height
        ref_heights = sorted(set(list(alternative_wind_speed_rasters.keys()) + [self.elevated_wind_speed_height]))
        # make sure that we have enough reference height sampling points
        if allow_extrapolation:
            # we should have at least one height level "in every (height) direction where data points are"
            if (target_height < self.elevated_wind_speed_height).any() and not min(
                ref_heights
            ) < self.elevated_wind_speed_height:
                raise KeyError(
                    f"data contains target heights but not reference heights below elevated wind speed height. alternative_wind_speed_rasters must contain key < {self.elevated_wind_speed_height}"
                )
            if (target_height > self.elevated_wind_speed_height).any() and not max(
                ref_heights
            ) > self.elevated_wind_speed_height:
                raise KeyError(
                    f"data contains target heights but not reference heights above elevated wind speed height. alternative_wind_speed_rasters must contain key > {self.elevated_wind_speed_height}"
                )
        else:
            # no data point may be outside the min-max height range
            if (target_height < min(ref_heights)).any():
                raise KeyError(
                    f"data contains target heights below elevated wind speed height and allow_extrapolation is False. alternative_wind_speed_rasters must contain key <= {target_height.min()}"
                )
            if (target_height > max(ref_heights)).any():
                raise KeyError(
                    f"data contains target heights above elevated wind speed height and allow_extrapolation is False. alternative_wind_speed_rasters must contain key >= {target_height.max()}"
                )

        # bin the target heights to reference spacing binds
        idx = np.searchsorted(ref_heights, target_height, side="left")
        # get first the indices of the respective lower and higher reference heights and then the values
        lower_idx = np.clip(idx - 1, 0, len(ref_heights) - 1)
        ref_height_lower = np.array(ref_heights)[lower_idx]
        upper_idx = np.clip(idx, 0, len(ref_heights) - 1)
        ref_height_upper = np.array(ref_heights)[upper_idx]
        # cover edge cases when both ref heights are the same
        both_lowest_idx = (ref_height_lower == ref_height_upper) & (ref_height_lower == min(ref_heights))
        ref_height_upper[both_lowest_idx] = ref_heights[1]  # set to second lowest value
        both_highest_idx = (ref_height_lower == ref_height_upper) & (ref_height_upper == max(ref_heights))
        ref_height_lower[both_highest_idx] = ref_heights[-2]  # set to 2nd highest value

        def _get_ws(arr):
            """Extracts windspeeds for given reference height arrays."""
            no_data = -9999
            ws = np.full(shape=arr.shape, fill_value=no_data)  # initialize as noData=-9999

            # first set the previously extracted elevated_wind_speed at default reference height
            sel = arr == self.elevated_wind_speed_height
            ws = np.where(sel, self.real_lra, ws)  # set default real_lra at all positions with default height

            # then get the values for all other reference heights in array
            for _height in np.unique(arr):
                if _height == self.elevated_wind_speed_height:
                    continue  # already handled default height

                # get and check filepath from alternative LRA ws height rasters
                fp = alternative_wind_speed_rasters[_height]
                if not (isinstance(fp, str) and isfile(fp)):
                    raise FileNotFoundError(
                        f"value of alternative_wind_speed_rasters[{_height}] must be a str-formatted path to an existing file: {fp}"
                    )

                # extract ws only for points (rows) with this height
                _sel_points = (arr == _height).any(axis=0)
                _ws = self.get_scalar_values_from_raster(
                    fp=fp,
                    spatial_interpolation="linear-spline",
                    points=list(
                        zip(
                            self.placements[_sel_points]["lon"],
                            self.placements[_sel_points]["lat"],
                        )
                    ),
                )
                # write into output array
                sel = arr == _height

                # iteratively replace values of interest in affected columns
                col_idx = np.where(_sel_points)[0]  # affected column indices
                for i, c in enumerate(col_idx):
                    ws[sel[:, c], c] = _ws[i]

            assert not (ws == no_data).any()  # make sure that values for all locs were extracted

            return ws

        # get the wind speeds at the respective lower and higher reference height
        ref_ws_lower = _get_ws(arr=ref_height_lower)
        ref_ws_upper = _get_ws(arr=ref_height_upper)

        # calculate the interpolated ws at target height
        delta = ref_height_upper - ref_height_lower
        fraction = np.divide(
            target_height - ref_height_lower,
            delta,
            out=np.zeros_like(delta, dtype=float),
            where=delta != 0,  # avoid division by zero
        )
        target_ws = ref_ws_lower + fraction * (ref_ws_upper - ref_ws_lower)

        # calculate scaling factor relative to default LRA height
        scale = target_ws / self.real_lra

        # scale hourly ws to the new target hub height and overwrite attr
        self.sim_data["elevated_wind_speed"] = scale * self.sim_data["elevated_wind_speed"]

        self.elevated_wind_speed_height = self.placements["hub_height"].values

        return self

    def apply_air_density_correction_to_wind_speeds(self):
        """
        Applies air density corrections to the wind speeds at the hub height.

        Return
        ------
            A reference to the invoking WindWorkflowManager


        """
        assert "surface_air_temperature" in self.sim_data, "surface_air_temperature has not been read from a source"
        assert "surface_pressure" in self.sim_data, "surface_pressure has not been read from a source"
        assert hasattr(self, "elevated_wind_speed_height")

        self.sim_data["elevated_wind_speed"] = rk_wind_core.air_density_adjustment.apply_air_density_adjustment(
            self.sim_data["elevated_wind_speed"],
            pressure=self.sim_data["surface_pressure"],
            temperature=self.sim_data["surface_air_temperature"],
            height=self.elevated_wind_speed_height,
        )

        return self

    def apply_wake_correction_of_wind_speeds(
        self,
        wake_curve="dena_mean",
    ):
        """
        Applies a wind-speed dependent reduction factor to the wind speeds at elevated height,
        based on

        Parameters
        ----------
        wake_curve : str, optional
            string value to describe the wake reduction method. None will
            cause no reduction. Location-sepcific values can also be
            given in a 'wake_curve' column of the placements dataframe,
            the latter will be overridden by the 'wake_curve' argument.
            By default "dena_mean". Choose from (see more information
            here under wind_efficiency_curve_name[1]):
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
        if wake_curve is None:
            if not "wake_curve" in self.placements.columns:
                # no wake effects to be applied
                return self
            else:
                # we have no wake wake_curve value
                # but use reduction curve name info in the placements df
                print(
                    f"Wake reduction curve names will be extracted from 'wake_curve' column in placements dataframe.",
                    flush=True,
                )
                wake_curves = np.array(self.placements["wake_curve"])
                if pd.isnull(wake_curves).all():
                    print(
                        f"All 'wake_curve' column entries are NaN, no wake effects willbe applied.",
                        flush=True,
                    )
                pass
        else:
            # use the given wake curve name
            if "wake_curve" in self.placements.columns:
                # prioritize wake_curve function arg over the wake_curve placements column
                print(
                    f"NOTE: 'wake_curve' column in placements not considered since 'wake_curve' workflow argument was given as: {wake_curve}",
                    flush=True,
                )
            wake_curves = np.array([wake_curve] * len(self.placements))

        # iterate over the possibly different wake reduction curves that are not NaN
        for wake_curve_i in set(wake_curves[np.array([not ((pd.isnull(x)) or (x == "None")) for x in wake_curves])]):
            self.sim_data["elevated_wind_speed"][:, wake_curves == wake_curve_i] = (
                windpowerlib.wake_losses.reduce_wind_speed(
                    self.sim_data["elevated_wind_speed"][:, wake_curves == wake_curve_i],
                    wind_efficiency_curve_name=wake_curve_i,
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
            self.powerCurveLibrary[key] = self.powerCurveLibrary[key].convolute_by_gaussian(
                scaling=scaling, base=base, **kwargs
            )

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
            The max. No. of simulation iterations allowed for iterative
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
                self.sim_data["elevated_wind_speed"][:, _batch * max_batch_size : (_batch + 1) * max_batch_size]
            )

            for pckey, pc in self.powerCurveLibrary.items():
                _sel = (
                    self.placements.iloc[_batch * max_batch_size : (_batch + 1) * max_batch_size, :].powerCurve == pckey
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
        _batches = np.ceil(self.sim_data["elevated_wind_speed"].shape[1] / max_batch_size)

        # get and set correction factor
        self.set_correction_factors(correction_factors=cf_correction_factor, verbose=verbose)

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
                    f"Based on max_batch_size={max_batch_size}, the total of {len(self.locs)} placements were split into {int(_batches)} sub batches. Proceeding with batch {_batch + 1}/{int(_batches)} (id={_batch}) with {len_locs} placements.",
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
                avg_gen_last * self.correction_factors[_batch * max_batch_size : (_batch + 1) * max_batch_size]
            )
            # make sure that the average (usually annual) target cs is realistic in all locs
            if isinstance(cf_correction_factor, (float, int)) and cf_correction_factor == 1:
                # we have no correction, but cfs may not be > 1
                if (_target_cfs > 1).any():
                    raise ValueError(f"The turbine parameters lead to average capacity factors greater 1.0.")
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
                    f"WARNING: {len(self.locs[_batch * max_batch_size : (_batch + 1) * max_batch_size][np.isnan(_target_cfs)])} NaNs detected in weather data LRA: {self.locs[_batch * max_batch_size : (_batch + 1) * max_batch_size][np.isnan(_target_cfs)]}"
                )

            # set the initial deviation based on initial, undistorted generation vs target generation
            _deviations_last = avg_gen_last / _target_cfs

            # calculate the min. required relative convergence per iteration to achieve tolerance after max. iterations
            min_convergence = 1 - (tolerance / abs(_deviations_last - 1)) ** (1 / max_iterations)

            # initialize the correction factors as 1.0 everywhere, will be adapted first thing if tolerance is not met by deviations
            _ws_corrs_current = np.array([1.0] * len(_deviations_last))

            # iterate until the target cf average is met, i.e. until absolute deltas of 1.0 and deviations are all less than tolerance
            _itercount = 0
            while (abs(_deviations_last - 1) > tolerance).any():
                # safety fallback - exit in case of infinite loops
                if _itercount > max_iterations:
                    raise TimeoutError(
                        f"{str(datetime.datetime.now())} The simulation did not reach the required tolerance of {tolerance} within the given max. {max_iterations} iterations. Remaining max. absolute deviation is {round(max(abs(_deviations_last - 1)), 4)}. Number of placements with deviation > {tolerance}: {sum(abs(_deviations_last - 1) > tolerance)}/{len(_deviations_last)}. Increase tolerance or max_iterations value."
                    )
                # print deviation status for the current iteration
                if verbose:
                    print(
                        datetime.datetime.now(),
                        f"Maximum rel. deviation after {'initial simulation' if _itercount == 0 else str(_itercount) + ' additional iteration(s)'} is {round(max(abs(_deviations_last - 1)), 4)}, Number/share of placements with deviation > tolerance ({tolerance}): {sum(abs(_deviations_last - 1) > tolerance)}/{len(_deviations_last)}. More iterations required.",
                    )

                # update the estimated correction factor for the wind speed for this iteration
                _ws_corrs_current = _ws_corrs_current * np.cbrt(1 / _deviations_last)  # power law

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
                    abs(_deviations_current - 1) > (1 - min_convergence) * abs(_deviations_last - 1)
                ) & (abs(_deviations_current - 1) > tolerance)
                if _non_convs.sum() > 0:
                    print(
                        f"{_non_convs.sum()}/{len(_deviations_current)} placements ({round(_non_convs.sum() / len(_deviations_current) * 100, 2)}%) did not converge (sufficiently). Average cf will be enforced.",
                        flush=True,
                    )

                if ((((gen_current == 0) | (gen_current == 1)).sum(axis=0) / 8760 < 0.15)[_non_convs]).any():
                    f"{((((gen_current == 0) | (gen_current == 1)).sum(axis=0) / 8760 < 0.15)[_non_convs]).sum()}non-converging placements with <15% cf=0 or cf=1.0 found: {(((gen_current == 0) | (gen_current == 1)).sum(axis=0) / 8760)[_non_convs]}"

                def correct_cf(arr, target_mean):
                    """Adapts average of 'arr' to 'target_mean' value without removing zeros/1.0s."""
                    # handle NaN target cfs
                    if np.isnan(target_mean):
                        _arr = np.empty(len(arr))
                        _arr[:] = np.nan
                        return _arr

                    flh_in = np.sum(arr)
                    flh_target = target_mean * len(arr)
                    flh_diff = flh_target - flh_in
                    _break = False

                    if flh_diff > 0:
                        while sum(arr) < flh_target:
                            delta_max = 1 - arr[arr < 1].max()
                            _add = np.where(arr > 0.5, delta_max, arr * delta_max)
                            _add[arr == 1.0] = 0  # set delta to zero for cf=1 to have a correct total FLH delta
                            # scale if needed
                            if _add.sum() > (flh_target - arr.sum()):
                                _add = _add * (flh_target - arr.sum()) / _add.sum()
                                _break = True
                            arr = np.where(arr < 1, arr + _add, arr)
                            if _break:
                                break

                    if flh_diff < 0:
                        while sum(arr) > flh_target:
                            delta_min = arr[arr > 0].min()
                            _ded = np.where(arr < 0.5, delta_min, (1 - arr) * delta_min)
                            _ded[arr == 0] = 0  # set delta to zero for cf=0 to have a correct total FLH delta
                            # scale if needed
                            if _ded.sum() > (arr.sum() - flh_target):
                                _ded = _ded * abs(arr.sum() - flh_target) / _ded.sum()
                                _break = True
                            arr = np.where(arr > 0, arr - _ded, arr)
                            if _break:
                                break

                    return arr

                # iterate over locations with diverging cfs
                for i, _non_conv in enumerate(_non_convs):
                    if _non_conv:
                        gen_current[:, i] = correct_cf(gen_current[:, i], _target_cfs[i])
                        # ensure that the forced avg adaptation achieved a deviation < tolerance
                        assert (
                            np.isnan(_target_cfs[i]) or abs(gen_current[:, i].mean() / _target_cfs[i] - 1) < tolerance
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
                    f"Required tolerance of {tolerance} reached after {_itercount} additional iteration(s). Maximum remaining rel. deviation: {round(max(abs(_deviations_last - 1)), 4)}.",
                    flush=True,
                )

            _max_cfs = gen_last.max(axis=0)
            if (gen_last > 1).any():
                print(
                    datetime.datetime.now(),
                    f"Required target cf could not be reached for some locations, cf will be reduced by factor min/max. {np.nanmin(1 / _max_cfs)}/{np.nanmax(1 / _max_cfs)} in order to not exceed cf=1.0.",
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
        assert availability_factor > 0 and availability_factor <= 1, f"availability_factor must be between 0 and 1.0."

        self.sim_data["capacity_factor"] = self.sim_data["capacity_factor"] * availability_factor

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
        assert availability_factor > 0 and availability_factor <= 1, f"availability_factor must be between 0 and 1.0."

        self.sim_data["capacity_factor"] = self.sim_data["capacity_factor"] * availability_factor

        return self

    def interpolate_raster_vals_to_hub_height(self, name: str, height_to_raster_dict: dict, **kwargs):
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
            known_vals.append(self.extract_raster_values_at_placements(height_to_raster_dict[h], **kwargs))

        interpolated_vals = np.full_like(known_vals[0], np.nan)
        hh = self.placements["hub_height"].values
        for hi in range(len(known_heights) - 1):
            sel = np.logical_and(hh >= known_heights[hi], hh < known_heights[hi + 1])
            if sel.any():
                interpolated_vals[sel] = (hh[sel] - known_heights[hi]) / (known_heights[hi + 1] - known_heights[hi]) * (
                    known_vals[hi + 1] - known_vals[hi]
                ) + known_vals[hi]

        if np.isnan(interpolated_vals).any():
            raise RuntimeError("Could not determine interpolation for all hub heights")

        self.placements[name] = interpolated_vals
        return self

    def set_correction_factors(self, correction_factors, verbose=True):
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
            if verbose:
                print(
                    datetime.datetime.now(),
                    f"Now extracting correction factors for a total of {len(self.locs)} placements from {correction_factors}:",
                )
            correction_factors = gk.raster.interpolateValues(correction_factors, self.locs, mode="near")
            assert not np.isnan(correction_factors).any(), f"correction_factors extracted from raster must not be nan"
        elif not isinstance(correction_factors, (float, int)):
            raise TypeError(f"correction_factors must either be a str formatted raster filepath or a float value")
        else:
            correction_factors = [correction_factors] * len(self.locs)

        # write to attribute
        self.correction_factors = np.array(correction_factors)

        return self
