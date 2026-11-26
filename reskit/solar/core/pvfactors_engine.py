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
whole. pvfactors also computes each pair of surfaces through some 40 numpy operations, each
making a temporary array over all time steps, and decides whether the neighbouring row blocks
a view by comparing two arctan2 angles. The view factor calculator below computes all pairs on
one side of a PV row in one compiled loop (numba), skips the values pvfactors zeroes, and makes
the blocking decision by the sign of a cross product. The view factor matrix agrees with
pvfactors' to floating point rounding; entries can differ in the last bits where a ground
point lies exactly on the line through a PV row edge and the neighbouring row's lowest point
(pvfactors places the shadow edges there): the two angles are then equal and the decision
depends on rounding, but both string lengths are the same up to rounding.

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
    from pvfactors.config import DISTANCE_TOLERANCE
    from pvfactors.viewfactors.calculator import VFCalculator
    from pvfactors.viewfactors.vfmethods import VFTsMethods

    def _stack(points, n_steps):
        # x and y of timeseries points as two contiguous (n_points, n_steps) arrays
        xy = [(np.asarray(p.x, dtype=float), np.asarray(p.y, dtype=float)) for p in points]
        return tuple(np.ascontiguousarray(np.stack([np.broadcast_to(c[k], (n_steps,)) for c in xy])) for k in (0, 1))

    class CompiledVFTsMethods(VFTsMethods):
        """VFTsMethods with the PV row to ground view factors computed by a compiled kernel."""

        def vf_pvrow_gnd_surf(self, ts_pvrows, ts_ground, tilted_to_left, vf_matrix):
            # As VFTsMethods.vf_pvrow_gnd_surf() with vf_pvrow_surf_to_gnd_surf_obstruction_hottel(),
            # _vf_surface_to_surface() and _vf_hottel_gnd_surf(). pvfactors calls those once per
            # pair of PV row and ground surface, on arrays over all time steps; here the PV row and
            # ground surfaces on one side of a PV row go to _pvrow_gnd_view_factors() together,
            # which computes each value in one pass, with the same arithmetic in the same order.
            #
            # pvfactors keeps the front's view to the left ground and the back's to the right only
            # where the rows are tilted to the left, and vice versa; a side kept at no time step
            # would only write zeros over the zeros already there and is skipped.
            n_pvrows = len(ts_pvrows)
            n_steps = len(tilted_to_left)
            tilted_to_left = np.ascontiguousarray(tilted_to_left, dtype=np.bool_)
            any_left, any_right = bool(np.any(tilted_to_left)), bool(not np.all(tilted_to_left))
            for idx_pvrow, ts_pvrow in enumerate(ts_pvrows):
                for is_left, side_name in ((True, "left"), (False, "right")):
                    pv_surfaces = [
                        (is_back, surf)
                        for is_back, pvrow_side in ((False, ts_pvrow.front), (True, ts_pvrow.back))
                        # the front sees the left ground when tilted to the left, the back the right
                        if (any_left if is_left != is_back else any_right)
                        for surf in pvrow_side.all_ts_surfaces
                        if not surf.is_empty
                    ]
                    gnd_surfaces = [
                        surf
                        for surf in ts_ground.ts_surfaces_side_of_cut_point(side_name, idx_pvrow)
                        if not surf.is_empty
                    ]
                    if not pv_surfaces or not gnd_surfaces:
                        continue
                    pvs = [surf for _, surf in pv_surfaces]
                    no_obstruction = (is_left and idx_pvrow == 0) or (not is_left and idx_pvrow == n_pvrows - 1)
                    if no_obstruction:
                        obstr = (np.zeros(n_steps), np.zeros(n_steps))
                    else:
                        # the neighbouring row's lowest point may block the view
                        idx_obstr = idx_pvrow - 1 if is_left else idx_pvrow + 1
                        ox, oy = _stack([ts_pvrows[idx_obstr].full_pvrow_coords.lowest_point], n_steps)
                        obstr = (ox[0], oy[0])
                    width = np.ascontiguousarray(
                        np.stack([np.broadcast_to(np.asarray(s.length, dtype=float), (n_steps,)) for s in pvs])
                    )
                    _compiled_kernels()(
                        vf_matrix,
                        np.array([surf.index for surf in pvs]),
                        np.array([surf.index for surf in gnd_surfaces]),
                        *_stack([s.coords.b1 for s in pvs], n_steps),
                        *_stack([s.coords.b2 for s in pvs], n_steps),
                        *_stack([s.highest_point for s in pvs], n_steps),
                        *_stack([s.lowest_point for s in pvs], n_steps),
                        width,
                        *_stack([s.coords.b1 for s in gnd_surfaces], n_steps),
                        *_stack([s.coords.b2 for s in gnd_surfaces], n_steps),
                        *obstr,
                        no_obstruction,
                        is_left,
                        # pvfactors keeps a view where the rows tilt towards this ground side
                        np.array([is_left != is_back for is_back, _ in pv_surfaces]),
                        tilted_to_left,
                        DISTANCE_TOLERANCE,
                    )

    class CompiledVFCalculator(VFCalculator):
        """VFCalculator with CompiledVFTsMethods."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.vf_ts_methods = CompiledVFTsMethods()

    return CompiledVFCalculator


@functools.cache
def _compiled_kernels():
    """The PV row to ground view factor kernel, compiled on first use (and cached on disk)."""
    import math

    import numba

    @numba.njit(cache=True, inline="always")
    def distance(x1, y1, x2, y2):
        # VFTsMethods._distance(pt_1, pt_2), with the same operations in the same order
        return math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

    @numba.njit(cache=True, inline="always")
    def string_length(px, py, gx, gy, ox, oy, dist_gnd_obstr, dist_obstr_pv, is_left):
        # The length of the string from ground point g to PV row point p in
        # VFTsMethods._hottel_string_length(): around the obstruction point o where the
        # neighbouring row blocks the straight line, else straight.
        # pvfactors decides that by comparing the angles of g -> p and g -> o with the x axis.
        # Where both points lie above the ground point, that comparison is the sign of the
        # cross product of the two vectors, which needs no arctan2; the two can only differ for
        # (nearly) collinear points, where both string lengths are the same. Otherwise (rows
        # reaching below the ground) the angles are compared as pvfactors does.
        l_pv = distance(px, py, gx, gy)
        if py > gy and oy > gy:
            cross = (ox - gx) * (py - gy) - (oy - gy) * (px - gx)
            is_obstructing = cross > 0.0 if is_left else cross < 0.0
        else:
            alpha_pv = math.atan2(py - gy, px - gx)
            alpha_obstr = math.atan2(oy - gy, ox - gx)
            is_obstructing = alpha_pv > alpha_obstr if is_left else alpha_pv < alpha_obstr
        return dist_gnd_obstr + dist_obstr_pv if is_obstructing else l_pv

    @numba.njit(cache=True)
    def pvrow_gnd_view_factors(
        vf_matrix,
        idx_pv,
        idx_gnd,
        b1x, b1y, b2x, b2y, hx, hy, lx, ly, width,
        g1x, g1y, g2x, g2y,
        ox, oy,
        no_obstruction, is_left, keep_if_left, tilted_to_left, tol,
    ):  # fmt: skip
        """Writes the view factors between the PV row and ground surfaces into vf_matrix.

        idx_pv, idx_gnd: matrix indices of the PV row and ground surfaces. b1, b2, h(ighest),
        l(owest): points of the PV row surfaces and width their lengths, (n_pv, n_steps);
        g1, g2: boundary points of the ground surfaces, (n_gnd, n_steps); o: lowest point of
        the obstructing row, (n_steps,); keep_if_left: per PV row surface, whether its view is
        kept where the rows tilt to the left (else where they tilt to the right).

        The arithmetic is that of _vf_surface_to_surface() (no obstruction) and
        _vf_hottel_gnd_surf() (Hottel's crossed strings around the obstructing row), in the
        same order.
        """
        n_pv, n_steps = width.shape
        n_gnd = g1x.shape[0]
        # what depends only on a ground surface, or only on a PV row surface, and the
        # obstructing row, computed once
        gnd_len = np.empty((n_gnd, n_steps))
        d1os, d2os = np.zeros((n_gnd, n_steps)), np.zeros((n_gnd, n_steps))
        for g in range(n_gnd):
            for t in range(n_steps):
                gnd_len[g, t] = distance(g1x[g, t], g1y[g, t], g2x[g, t], g2y[g, t])
                if not no_obstruction:
                    d1os[g, t] = distance(g1x[g, t], g1y[g, t], ox[t], oy[t])
                    d2os[g, t] = distance(g2x[g, t], g2y[g, t], ox[t], oy[t])
        dhs, dls = np.zeros((n_pv, n_steps)), np.zeros((n_pv, n_steps))
        if not no_obstruction:
            for s in range(n_pv):
                for t in range(n_steps):
                    dhs[s, t] = distance(ox[t], oy[t], hx[s, t], hy[s, t])
                    dls[s, t] = distance(ox[t], oy[t], lx[s, t], ly[s, t])
        for s in range(n_pv):
            i = idx_pv[s]
            for g in range(n_gnd):
                j = idx_gnd[g]
                for t in range(n_steps):
                    w = width[s, t]
                    if tilted_to_left[t] != keep_if_left[s] or not w > tol:
                        # pvfactors zeroes this side at this step, or the surface has no length
                        vf_matrix[i, j, t] = 0.0
                        vf_matrix[j, i, t] = 0.0
                        continue
                    if no_obstruction:
                        sum_1 = distance(b1x[s, t], b1y[s, t], g1x[g, t], g1y[g, t]) + distance(
                            b2x[s, t], b2y[s, t], g2x[g, t], g2y[g, t]
                        )
                        sum_2 = distance(b1x[s, t], b1y[s, t], g2x[g, t], g2y[g, t]) + distance(
                            b2x[s, t], b2y[s, t], g1x[g, t], g1y[g, t]
                        )
                        numerator = abs(sum_2 - sum_1)
                    else:
                        # near and far ground point: b1 and b2 on the left side, b2 and b1 on the right
                        if is_left:
                            nx, ny, d_near, fx, fy, d_far = (
                                g1x[g, t],
                                g1y[g, t],
                                d1os[g, t],
                                g2x[g, t],
                                g2y[g, t],
                                d2os[g, t],
                            )
                        else:
                            nx, ny, d_near, fx, fy, d_far = (
                                g2x[g, t],
                                g2y[g, t],
                                d2os[g, t],
                                g1x[g, t],
                                g1y[g, t],
                                d1os[g, t],
                            )
                        high_x, high_y, low_x, low_y = hx[s, t], hy[s, t], lx[s, t], ly[s, t]
                        l1 = string_length(high_x, high_y, nx, ny, ox[t], oy[t], d_near, dhs[s, t], is_left)
                        l2 = string_length(low_x, low_y, fx, fy, ox[t], oy[t], d_far, dls[s, t], is_left)
                        d1 = string_length(high_x, high_y, fx, fy, ox[t], oy[t], d_far, dhs[s, t], is_left)
                        d2 = string_length(low_x, low_y, nx, ny, ox[t], oy[t], d_near, dls[s, t], is_left)
                        numerator = d1 + d2 - l1 - l2
                    vf = numerator / (2.0 * w)
                    vf_matrix[i, j, t] = vf
                    gnd_length = gnd_len[g, t]
                    vf_matrix[j, i, t] = vf * w / gnd_length if gnd_length > tol else 0.0

    return pvrow_gnd_view_factors


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
