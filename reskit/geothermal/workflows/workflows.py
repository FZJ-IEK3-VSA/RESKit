from distutils.log import warn
import numpy as np
import pandas as pd
import xarray as xr
import os
import geokit as gk
import time
import warnings
from datetime import datetime

from .egs_workflow_manager import EGSWorkflowManager

from ..data import path_temperatures
from ..data import path_heat_flow_sustainable_W_per_m2


def egs_workflow(
    placements: pd.DataFrame,
    sourceTemperature=path_temperatures,
    sourceSustainableHeatflow=path_heat_flow_sustainable_W_per_m2,
    savepath=None,
    configuration="doublette",
    manual_values={},
):
    """
    Executes the Enhanced Geothermal System (EGS) workflow for given placements.

    Parameters
    ----------
        placements (pd.DataFrame): Locations where the EGS workflow will be applied. Needs to have lat lon and geokit geoms.
        sourceTemperature (str or Path, optional): Path to the geothermal temperature data.
            Defaults to `path_temperatures`.
        sourceSustainableHeatflow (str or Path, optional): Path to the sustainable heat flow data.
            Defaults to `path_heat_flow_sustainable_W_per_m2`.
        savepath (str or Path, optional): Directory where results will be saved. Defaults to None which outputs the data.
        configuration (str, optional): Type of geothermal system configuration.
            Defaults to 'doublette'.
        manual_values (dict, optional): Dictionary of manually specified values for overriding defaults.

    Returns
    -------
        None or xarray.Dataset
            None if `savepath` is given (results are written to file instead).
            Otherwise an xarray.Dataset with dimension ``placements``.

        All columns of the input `placements` (e.g. lat, lon, geom) are passed
        through to the output unchanged. In addition, the dataset contains:

        Site data (read from the input files):
            surface_temperature : Surface temperature [degC].
            qdot_sust_W_per_m2 : Sustainable surface heat flow density [W/m^2].

        Three technical methods are evaluated, identified by a suffix {M}:
            VM : Volume method. Heat in place of a reservoir of
                 `reservoir_size_m3` cooled by `dT_drawdown` with
                 `recovery_factor`, extracted evenly over `lifetime_a`.
            GR : Gringarten method. Analytical heat extraction from parallel
                 fractures with a fixed total volume flow `Vdot_total_m3_per_s`.
            SU : Sustainable method. Only the heat replenished by the
                 sustainable heat flow through the horizontal reservoir area
                 (reservoir_size_m3 / depth resolution) is extracted.

        All depth-dependent values of method {M} are given at that method's own
        optimal depth `opt_depth_{M}_m`, so depths can differ between methods.
        Placements without an eligible depth are NaN. A value of -1 means
        "not defined for this method".

            opt_depth_{M}_m : Depth with minimal net LCOE, considering only
                depths <= maxDepth_m with rock temperature >=
                minRockTemperature_degC [m].
            temperature_{M}_degC : Undisturbed rock temperature at the optimal
                depth [degC].
            Qdot_out_{M}_MW : Thermal power extracted, averaged over the
                lifetime [MW_th].
            P_out_{M}_MW : Gross electric power averaged over the lifetime,
                Qdot_out * eta_plant(T). Nameplate capacity is P_out / CF [MW_el].
            P_Pump_{M}_MW : Pumping power of all production wells [MW_el].
            P_out_net_{M}_MW : Net electric power, P_out - P_Pump [MW_el].
            mdot_water_{M}_kg_per_s : Total produced water mass flow [kg/s].
            mdot_water_{M}_kg_per_s_per_well : Water mass flow per production
                well [kg/s].
            T_Water_out_{M}_degC : Production water temperature. VM: mean over
                the lifetime, T_rock - dT_drawdown / 2. SU: equals the rock
                temperature [degC].
            T_Rock_abandon_{M}_degC : Mean rock temperature at the end of the
                lifetime [degC].
            dT_active_res_{M}_K : Temperature drawdown of the actively flushed
                rock. VM: dT_drawdown, SU: 0, GR: -1 [K].
            dT_total_res_{M}_K : Mean temperature drawdown of the whole
                reservoir volume at the end of the lifetime. SU: 0 [K].
            recovery_fac_amb_{M}_1 : Extracted heat divided by the heat in place
                relative to the surface temperature. SU: -1 [-].
            resourceUseTime_{M}_a : Years until the rock reaches
                minRockTemperature_degC at the method's cooling rate. SU: inf [a].
            regeneration_time_{M}_a : Years the sustainable heat flow needs to
                replenish the heat extracted during the lifetime
                (SU: equals lifetime_a by construction) [a].
            TOTEX_MUSD_{M}_per_a : Annual total cost, annuitized CAPEX
                (WACC, lifetime_a) plus fixed OPEX [MUSD/a].
            LCOE_gross_{M}_EUR_per_kWh : LCOE based on P_out [EUR/kWh].
            LCOE_{M}_EUR_per_kWh : LCOE based on P_out_net [EUR/kWh].

        Volume method only:
            Total_thermal_energy_PJ : Heat in place of the reservoir relative to
                the surface temperature, rho_rock * cp_rock * V * (T - T_surface),
                at opt_depth_VM_m [PJ].

    Citation:
         Franzmann, D., Heinrichs, H. and Stolten, D. (2025), Global geothermal electricity
         potentials: A technical, economic, and thermal renewability assessment.
         Renewable Energy 250, 123199. https://doi.org/10.1016/j.renene.2025.123199
    """
    citation = """
    This workflow can be cited as:
    Franzmann, D., Heinrichs, H. and Stolten, D. (2025), Global geothermal
    electricity potentials: A technical, economic, and thermal renewability
    assessment. Renewable Energy 250, 123199.
    https://doi.org/10.1016/j.renene.2025.123199
    """

    print(citation)

    wfm = EGSWorkflowManager(placements=placements)

    ### data loading
    tic_data_loading = time.time()
    now = datetime.now()
    print("Starting loading data =", now, flush=True)

    wfm.loadDataAllDepths(
        vars=[
            "temperature",
        ],
        source=sourceTemperature,
    )
    wfm.loadData(vars=["surface_temperature"], source=sourceTemperature)
    wfm.loadData(
        vars=[
            "heat_flow_sustainable_W_per_m2",
        ],
        source=sourceSustainableHeatflow,
        newVarNamesDict={"heat_flow_sustainable_W_per_m2": "qdot_sust_W_per_m2"},
    )

    wfm.loadPlantData(
        configuration=configuration,
        manual_values=manual_values,
    )

    ### Calculations
    tic_calc = time.time()
    now = datetime.now()
    print("Starting calc =", now, flush=True)

    # own data
    wfm.VolumeMethod()
    wfm.GringartenMethodFixeVdot()
    wfm.SustainableHeat()

    ### Cost and selecting
    tic_cost = time.time()
    now = datetime.now()
    print("Starting cost calc =", now, flush=True)

    techMethods = wfm._getTechMethods()
    # loop all considered technological approaches
    for techMethod in techMethods:
        wfm.calculatePumpLosses(techMethod=techMethod)
        wfm.calculateCosts(techMethod=techMethod)
        wfm.calculateLCOE(techMethod=techMethod)
        wfm.getRegenerationTime(techMethod=techMethod)
        wfm.getOptDepth(techMethod=techMethod)
        wfm.getValuesAtOptDepth(techMethod=techMethod)

    output = wfm.saveOutput(savepath=savepath, deepsave=True)  # TODO: change to False

    tic_done = time.time()
    print("\nTime eval.:")
    print(f"Data loading finished in {str(int(tic_calc - tic_data_loading))}s.")
    print(f"Calculation finished in {str(int(tic_cost - tic_calc))}s.")
    print(f"Cost calculation finished in {str(int(tic_done - tic_cost))}s.")
    print(f"RESkit EGS done within {str(int(tic_done - tic_data_loading))}s for {len(placements)} points..")

    if savepath is None:
        return output


##########################
# DEPRECATED NAMES (#226) #
##########################
# The names below were renamed for PEP 8 in RESKit v0.6.0. Each old name stays
# available as a warning wrapper until v1.0.0. Do not add new code here.


def EGSworkflow(*args, **kwargs):
    """
    Deprecated alias of :func:`egs_workflow`.

    Kept for backward compatibility and scheduled for removal in RESKit v1.0.0.
    Use :func:`egs_workflow` instead. All arguments are passed through unchanged.
    """
    warnings.warn(
        "EGSworkflow() is deprecated and will be removed in RESKit v1.0.0. Use egs_workflow() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return egs_workflow(*args, **kwargs)


if __name__ == "__main__":
    print("\nThis is not an executable file. Pls run egs_workflow(args)\n")
