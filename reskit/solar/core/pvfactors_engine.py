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

pvfactors_timeseries() below is pvlib's function of the same name with this engine plugged in;
the results agree with pvlib's to floating point rounding.
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
        irradiance_model_params=irradiance_model_params,
    )
    df_report = pd.DataFrame(report, index=timestamps)
    return df_report.total_inc_front, df_report.total_inc_back, df_report.total_abs_front, df_report.total_abs_back
