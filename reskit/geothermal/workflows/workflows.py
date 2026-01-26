import os
import time
from datetime import datetime
from distutils.log import warn

import geokit as gk
import numpy as np
import pandas as pd
import xarray as xr

from reskit.geothermal.data import path_heat_flow_sustainable_w_per_m2, path_temperatures
from reskit.geothermal.workflows.egs_workflow_manager import EGSWorkflowManager


def egs_workflow(
    placements: pd.DataFrame,
    source_temperature=path_temperatures,
    source_sustainable_heatflow=path_heat_flow_sustainable_w_per_m2,
    savepath=None,
    configuration="doublette",
    manual_values={},
):
    """
    Executes the Enhanced Geothermal System (EGS) workflow for given placements.

    Parameters
    ----------
        placements (pd.DataFrame): Locations where the EGS workflow will be applied. Needs to have lat lon and geokit geoms.
        source_temperature (str or Path, optional): Path to the geothermal temperature data.
            Defaults to `path_temperatures`.
        source_sustainable_heatflow (str or Path, optional): Path to the sustainable heat flow data.
            Defaults to `path_heat_flow_sustainable_W_per_m2`.
        savepath (str or Path, optional): Directory where results will be saved. Defaults to None which outputs the data.
        configuration (str, optional): Type of geothermal system configuration.
            Defaults to 'doublette'.
        manual_values (dict, optional): Dictionary of manually specified values for overriding defaults.

    Returns
    -------
        None or xarray object: Workflow results, optionally saved to `savepath`.

    Citation:
         Franzmann, David and Heinrichs, Heidi and Stolten, Detlef, Global Electricity Potentials
         from Geothermal Power Under Technical, Economic, Sustainable Evaluation. Available at SSRN:
         https://ssrn.com/abstract=5029989 or http://dx.doi.org/10.2139/ssrn.5029989
    """
    citation = """
    This workflow can be cited as:
    Franzmann, David and Heinrichs, Heidi
    and Stolten, Detlef, Global Electricity Potentials from Geothermal Power
    Under Technical, Economic, Sustainable Evaluation.
    Available at SSRN: https://ssrn.com/abstract=5029989
    or http://dx.doi.org/10.2139/ssrn.5029989
    """

    print(citation)

    wfm = EGSWorkflowManager(placements=placements)

    ### data loading
    tic_data_loading = time.time()
    now = datetime.now()
    print("Starting loading data =", now, flush=True)

    wfm.load_data_all_depths(
        vars=[
            "temperature",
        ],
        source=source_temperature,
    )
    wfm.load_data(vars=["surface_temperature"], source=source_temperature)
    wfm.load_data(
        vars=[
            "heat_flow_sustainable_W_per_m2",
        ],
        source=source_sustainable_heatflow,
        new_var_names_dict={"heat_flow_sustainable_W_per_m2": "qdot_sust_W_per_m2"},
    )

    wfm.load_plant_data(
        configuration=configuration,
        manual_values=manual_values,
    )

    ### Calculations
    tic_calc = time.time()
    now = datetime.now()
    print("Starting calc =", now, flush=True)

    # own data
    wfm.volume_method()
    wfm.gringarten_method_fixe_v_dot()
    wfm.sustainable_heat()

    ### Cost and selecting
    tic_cost = time.time()
    now = datetime.now()
    print("Starting cost calc =", now, flush=True)

    techMethods = wfm._get_tech_methods()
    # loop all considered technological approaches
    for techMethod in techMethods:
        wfm.calculate_pump_losses(tech_method=techMethod)
        wfm.calculate_costs(tech_method=techMethod)
        wfm.calculate_lcoe(tech_method=techMethod)
        wfm.get_regeneration_time(tech_method=techMethod)
        wfm.get_opt_depth(tech_method=techMethod)
        wfm.get_values_at_opt_depth(tech_method=techMethod)

    output = wfm.save_output(savepath=savepath, deepsave=True)  # TODO: change to False

    tic_done = time.time()
    print("\nTime eval.:")
    print(f"Data loading finished in {str(int(tic_calc - tic_data_loading))}s.")
    print(f"Calculation finished in {str(int(tic_cost - tic_calc))}s.")
    print(f"Cost calculation finished in {str(int(tic_done - tic_cost))}s.")
    print(f"RESkit EGS done within {str(int(tic_done - tic_data_loading))}s for {len(placements)} points..")

    if savepath is None:
        return output


if __name__ == "__main__":
    print("\nThis is not an executable file. Pls run EGSworkflow(args)\n")
