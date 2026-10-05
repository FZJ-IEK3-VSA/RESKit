"""
pvfactors' full-mode engine with a block-structured linear solve.

pvlib.bifacial.pvfactors.pvfactors_timeseries() runs pvfactors' PVEngine.run_full_mode(), which
solves the radiosity system A q0 = irradiance of every time step by inverting A, a dense
(n_surfaces + 1) x (n_surfaces + 1) matrix (41 x 41 for 3 PV rows). That inversion is the
largest single cost of the bifacial workflow. A is far from dense, though:

    A = | D_G    -F_GP   -F_GS |    G: ground surfaces, P: PV row surfaces, S: the sky
        | -F_PG  A_PP    -F_PS |
        | 0      0       d_S   |

Flat ground does not see itself, so the ground block D_G is diagonal (the inverse
reflectivities), and the sky row is empty. Both follow from how pvfactors builds the view
factor matrix, not from the data. The sky radiosity is therefore a division, the ground
radiosities follow from the PV row ones, and only the Schur complement
A_PP - F_PG D_G^-1 F_GP of the PV row surfaces (12 x 12 for 3 rows) needs a dense solve.

Building the view factor matrix is the other large cost, mostly the view factors between the
PV row and ground surfaces. pvfactors computes them for the ground on both sides of every PV
row surface and then zeroes one side wherever the rows are tilted the other way; for a fixed
tilt that is the same side at every time step, so about half of that work is discarded
whole. The view factor calculator below skips it, and computes the rest for all ground surfaces
on one side of a PV row at once, sharing what depends only on the ground between the PV row
surfaces; the view factor matrix stays bit-identical to pvfactors'.

pvfactors_timeseries() below is pvlib's function of the same name with this engine and view
factor calculator plugged in; the results agree with pvlib's to floating point rounding.
"""

import functools

import numpy as np
import pandas as pd


@functools.cache
def _block_solve_engine():
    """The engine class, built on first use: solarfactors is imported only when needed."""
    from pvfactors.engine import PVEngine

    class BlockSolvePVEngine(PVEngine):
        """PVEngine whose run_full_mode() solves the radiosity system block-wise."""

        def run_full_mode(self, fn_build_report=None):
            if self.vf_calculator.vf_aoi_methods is not None:
                # AOI-dependent reflection losses build a different absorption matrix, which
                # this engine does not reproduce; pvlib's wrapper never configures them
                return super().run_full_mode(fn_build_report=fn_build_report)

            pvarray = self.pvarray
            # shape = n_surfaces + 1, n_timesteps
            irradiance_mat, rho_mat, invrho_mat, _ = self.irradiance.get_full_ts_modeling_vectors(pvarray)
            # shape = n_surfaces + 1, n_surfaces + 1, n_timesteps
            vf = self.vf_calculator.build_ts_vf_matrix(pvarray)
            pvarray.ts_vf_matrix = vf

            # pvarray.all_ts_surfaces lists the ground surfaces first, then the PV rows; the sky is last
            n_gnd = pvarray.ts_ground.n_ts_surfaces
            g, p, s = slice(0, n_gnd), slice(n_gnd, -1), -1
            if vf[g, g].any() or vf[s].any():
                # not the structure pvfactors builds: fall back rather than solve the wrong system
                return super().run_full_mode(fn_build_report=fn_build_report)

            # the sky radiosity: its row of A holds only the (dummy) inverse sky reflectivity
            q0_s = irradiance_mat[s] / invrho_mat[s]
            # right-hand sides with the sky term moved over
            r_g = irradiance_mat[g] + vf[g, s] * q0_s
            r_p = irradiance_mat[p] + vf[p, s] * q0_s
            # ground radiosities in terms of the PV row ones: q0_G = (r_G + F_GP q0_P) / invrho_G
            f_pg_scaled = vf[p, g] / invrho_mat[g][None]  # F_PG D_G^-1, (P, G, t)
            # the PV row system: (diag(invrho_P) - F_PP - F_PG D_G^-1 F_GP) q0_P = r_P + F_PG D_G^-1 r_G
            schur = -vf[p, p]
            # Each ground surface is seen by only a few PV row surfaces and sees only a few, so
            # F_PG D_G^-1 F_GP is summed over the non-zero pairs only, ground surface by ground
            # surface: about an eighth of the dense product for 3 rows, same summation order.
            f_gp = vf[g, p]
            seen_by, sees = f_pg_scaled.any(axis=2), f_gp.any(axis=2)  # (P, G), (G, P)
            for j in range(n_gnd):
                rows, cols = np.flatnonzero(seen_by[:, j]), np.flatnonzero(sees[j])
                if rows.size and cols.size:
                    schur[rows[:, None], cols[None, :]] -= f_pg_scaled[rows, j][:, None, :] * f_gp[j, cols][None, :, :]
            n_pv = schur.shape[0]
            schur[np.arange(n_pv), np.arange(n_pv)] += invrho_mat[p]
            rhs = r_p + np.einsum("pgt,gt->pt", f_pg_scaled, r_g)
            q0_p = np.linalg.solve(np.moveaxis(schur, -1, 0), rhs.T[..., None])[..., 0].T
            q0_g = (r_g + np.einsum("gpt,pt->gt", f_gp, q0_p)) / invrho_mat[g]

            q0 = np.empty_like(irradiance_mat)
            q0[g], q0[p], q0[s] = q0_g, q0_p, q0_s

            # the rest as in PVEngine.run_full_mode(), with its diagonal matrices as vectors
            qinc = invrho_mat * q0
            isotropic_mat = vf[:-1, -1, :] * irradiance_mat[-1, :]
            reflection_mat = qinc[:-1, :] - irradiance_mat[:-1, :] - isotropic_mat
            # without AOI methods the absorption matrix is (1 - rho_i) * vf_ik, row by row, and
            # F q0 needs no product: A q0 = irradiance with A = diag(invrho) - F gives F q0 = qinc - irradiance
            irradiance_abs_mat = self.irradiance.get_summed_components(pvarray, absorbed=True)
            qabs = (1.0 - rho_mat[:-1]) * (qinc[:-1] - irradiance_mat[:-1]) + irradiance_abs_mat
            # PVEngine also stores ts_vf_aoi_matrix, a third (n + 1)^2 x n_timesteps array that
            # nothing downstream of the report needs; it is left unset (None)

            for idx_surf, ts_surface in enumerate(pvarray.all_ts_surfaces):
                ts_surface.update_params(
                    {
                        "q0": q0[idx_surf, :],
                        "qinc": qinc[idx_surf, :],
                        "isotropic": isotropic_mat[idx_surf, :],
                        "reflection": reflection_mat[idx_surf, :],
                        "qabs": qabs[idx_surf, :],
                    }
                )
            return None if fn_build_report is None else fn_build_report(pvarray)

    return BlockSolvePVEngine


@functools.cache
def _vf_calculator():
    """The view factor calculator class, built on first use like the engine."""
    from pvfactors.viewfactors.calculator import VFCalculator
    from pvfactors.viewfactors.vfmethods import VFTsMethods

    from pvfactors.config import DISTANCE_TOLERANCE

    def _distance(pt_1, pt_2):
        # VFTsMethods._distance(), on arrays that broadcast
        return np.sqrt((pt_2[1] - pt_1[1]) ** 2 + (pt_2[0] - pt_1[0]) ** 2)

    def _angle_with_x_axis(pt_1, pt_2):
        # VFTsMethods._angle_with_x_axis()
        return np.arctan2(pt_2[1] - pt_1[1], pt_2[0] - pt_1[0])

    def _xy(ts_point):
        return np.asarray(ts_point.x, dtype=float), np.asarray(ts_point.y, dtype=float)

    class VectorisedVFTsMethods(VFTsMethods):
        """VFTsMethods with the PV row to ground view factors computed for all ground surfaces at once."""

        def vf_pvrow_gnd_surf(self, ts_pvrows, ts_ground, tilted_to_left, vf_matrix):
            # As VFTsMethods.vf_pvrow_gnd_surf() with vf_pvrow_surf_to_gnd_surf_obstruction_hottel(),
            # _vf_surface_to_surface() and _vf_hottel_gnd_surf(), with the same arithmetic in the
            # same order, so the matrix is bit-identical. pvfactors calls those once per pair of
            # PV row and ground surface; here the ground surfaces on one side of a PV row are
            # stacked into (n_ground, n_steps) arrays, and what depends only on the ground and
            # the obstructing row -- angles and distances to its lowest point -- is computed once
            # for all PV row surfaces instead of once per pair and string.
            #
            # pvfactors keeps the front's view to the left ground and the back's to the right only
            # where the rows are tilted to the left, and vice versa; a side kept at no time step
            # would only write zeros over the zeros already there and is skipped.
            n_pvrows = len(ts_pvrows)
            n_steps = len(tilted_to_left)
            any_left, any_right = bool(np.any(tilted_to_left)), bool(not np.all(tilted_to_left))
            for idx_pvrow, ts_pvrow in enumerate(ts_pvrows):
                for is_left, side_name in ((True, "left"), (False, "right")):
                    pvrow_sides = [
                        (is_back, pvrow_side)
                        for is_back, pvrow_side in ((False, ts_pvrow.front), (True, ts_pvrow.back))
                        # the front sees the left ground when tilted to the left, the back the right
                        if (any_left if is_left != is_back else any_right)
                    ]
                    gnd_surfaces = [
                        surf
                        for surf in ts_ground.ts_surfaces_side_of_cut_point(side_name, idx_pvrow)
                        if not surf.is_empty
                    ]
                    if not pvrow_sides or not gnd_surfaces:
                        continue
                    j = np.array([surf.index for surf in gnd_surfaces])
                    # the ground surfaces' boundary points as (n_ground, n_steps) arrays
                    gnd_b1, gnd_b2 = (
                        tuple(
                            np.stack([np.broadcast_to(c, (n_steps,)) for c in coords])
                            for coords in zip(*(_xy(getattr(surf.coords, b)) for surf in gnd_surfaces))
                        )
                        for b in ("b1", "b2")
                    )
                    gnd_length = _distance(gnd_b1, gnd_b2)  # TsLineCoords.length
                    no_obstruction = (is_left and idx_pvrow == 0) or (not is_left and idx_pvrow == n_pvrows - 1)
                    if not no_obstruction:
                        # the neighbouring row's lowest point may block the view; per ground point
                        idx_obstr = idx_pvrow - 1 if is_left else idx_pvrow + 1
                        obstr = _xy(ts_pvrows[idx_obstr].full_pvrow_coords.lowest_point)
                        alpha_obstr = {b: _angle_with_x_axis(pt, obstr) for b, pt in (("b1", gnd_b1), ("b2", gnd_b2))}
                        dist_gnd_obstr = {b: _distance(pt, obstr) for b, pt in (("b1", gnd_b1), ("b2", gnd_b2))}

                    for is_back, pvrow_side in pvrow_sides:
                        for pvrow_surf in pvrow_side.all_ts_surfaces:
                            if pvrow_surf.is_empty:
                                continue
                            i = pvrow_surf.index
                            width = pvrow_surf.length
                            if no_obstruction:
                                # _vf_surface_to_surface(pvrow_surf.coords, gnd_surf, width)
                                pv_b1, pv_b2 = _xy(pvrow_surf.coords.b1), _xy(pvrow_surf.coords.b2)
                                sum_1 = _distance(pv_b1, gnd_b1) + _distance(pv_b2, gnd_b2)
                                sum_2 = _distance(pv_b1, gnd_b2) + _distance(pv_b2, gnd_b1)
                                numerator = np.abs(sum_2 - sum_1)
                            else:
                                # _vf_hottel_gnd_surf(highest, lowest, gnd b1, gnd b2, obstr, width, is_left)
                                high, low = _xy(pvrow_surf.highest_point), _xy(pvrow_surf.lowest_point)
                                dist_obstr_pv = {"high": _distance(obstr, high), "low": _distance(obstr, low)}

                                def hottel_string_length(pv_name, pv, gnd_name, gnd):
                                    l_pv = _distance(pv, gnd)
                                    alpha_pv = _angle_with_x_axis(gnd, pv)
                                    if is_left:
                                        is_obstructing = alpha_pv > alpha_obstr[gnd_name]
                                    else:
                                        is_obstructing = alpha_pv < alpha_obstr[gnd_name]
                                    l_obstr = dist_gnd_obstr[gnd_name] + dist_obstr_pv[pv_name]
                                    return np.where(is_obstructing, l_obstr, l_pv)

                                near, far = (
                                    (("b1", gnd_b1), ("b2", gnd_b2)) if is_left else (("b2", gnd_b2), ("b1", gnd_b1))
                                )
                                l1 = hottel_string_length("high", high, *near)
                                l2 = hottel_string_length("low", low, *far)
                                d1 = hottel_string_length("high", high, *far)
                                d2 = hottel_string_length("low", low, *near)
                                numerator = d1 + d2 - l1 - l2
                            vf_pvrow_to_gnd = np.divide(
                                numerator,
                                2.0 * width,
                                where=width > DISTANCE_TOLERANCE,
                                out=np.zeros(numerator.shape),
                            )
                            if is_left != is_back:
                                vf_pvrow_to_gnd = np.where(tilted_to_left, vf_pvrow_to_gnd, 0.0)
                            else:
                                vf_pvrow_to_gnd = np.where(tilted_to_left, 0.0, vf_pvrow_to_gnd)
                            vf_gnd_to_pvrow = np.divide(
                                vf_pvrow_to_gnd * width,
                                gnd_length,
                                where=gnd_length > DISTANCE_TOLERANCE,
                                out=np.zeros_like(gnd_length),
                            )
                            vf_matrix[i, j, :] = vf_pvrow_to_gnd
                            vf_matrix[j, i, :] = vf_gnd_to_pvrow

    class VectorisedVFCalculator(VFCalculator):
        """VFCalculator with VectorisedVFTsMethods."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.vf_ts_methods = VectorisedVFTsMethods()

    return VectorisedVFCalculator


def pvfactors_timeseries(
    solar_azimuth,
    solar_zenith,
    surface_azimuth,
    surface_tilt,
    axis_azimuth,
    timestamps,
    dni,
    dhi,
    gcr,
    pvrow_height,
    pvrow_width,
    albedo,
    n_pvrows=3,
    index_observed_pvrow=1,
    rho_front_pvrow=0.03,
    rho_back_pvrow=0.05,
    horizon_band_angle=15.0,
):
    """
    pvlib.bifacial.pvfactors.pvfactors_timeseries() with the block-solving engine.

    Same arguments and return values as pvlib's function: the front and back incident
    irradiance and the front and back absorbed irradiance of the observed PV row, each as a
    pandas.Series indexed by `timestamps`.
    """
    from pvfactors.run import run_timeseries_engine

    solar_azimuth = np.array(solar_azimuth)
    solar_zenith = np.array(solar_zenith)
    dni = np.array(dni)
    dhi = np.array(dhi)
    surface_tilt = np.full_like(solar_zenith, surface_tilt)
    surface_azimuth = np.full_like(solar_zenith, surface_azimuth)

    pvarray_parameters = {
        "n_pvrows": n_pvrows,
        "axis_azimuth": axis_azimuth,
        "pvrow_height": pvrow_height,
        "pvrow_width": pvrow_width,
        "gcr": gcr,
    }
    irradiance_model_params = {
        "rho_front": rho_front_pvrow,
        "rho_back": rho_back_pvrow,
        "horizon_band_angle": horizon_band_angle,
    }

    def fn_build_report(pvarray):
        observed = pvarray.ts_pvrows[index_observed_pvrow]
        return {
            "total_inc_back": observed.back.get_param_weighted("qinc"),
            "total_inc_front": observed.front.get_param_weighted("qinc"),
            "total_abs_back": observed.back.get_param_weighted("qabs"),
            "total_abs_front": observed.front.get_param_weighted("qabs"),
        }

    report = run_timeseries_engine(
        fn_build_report,
        pvarray_parameters,
        timestamps,
        dni,
        dhi,
        solar_zenith,
        solar_azimuth,
        surface_tilt,
        surface_azimuth,
        albedo,
        cls_engine=_block_solve_engine(),
        cls_vf=_vf_calculator(),
        irradiance_model_params=irradiance_model_params,
    )
    df_report = pd.DataFrame(report, index=timestamps)
    return df_report.total_inc_front, df_report.total_inc_back, df_report.total_abs_front, df_report.total_abs_back
