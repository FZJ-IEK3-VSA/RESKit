from distutils.log import warn
import numpy as np
import pandas as pd
import xarray as xr
import os
import geokit as gk
import time
from datetime import datetime

from .egs_workflow_manager import EGS_workflowmanager

from ..data import path_temperatures
from ..data import path_heat_flow_sustainable_W_per_m2


def EGSworkflow(
    placements: pd.DataFrame,
    sourceTemperature=path_temperatures,
    sourceSustainableHeatflow=path_heat_flow_sustainable_W_per_m2,
    savepath=None,
    configuration="doublette",
    manual_values={},
):
    """
    Executes the Enhanced Geothermal System (EGS) workflow for given placements.

    Parameters:
        placements (pd.DataFrame): Locations where the EGS workflow will be applied. Needs to have lat lon and geokit geoms.
        sourceTemperature (str or Path, optional): Path to the geothermal temperature data.
            Defaults to `path_temperatures`.
        sourceSustainableHeatflow (str or Path, optional): Path to the sustainable heat flow data.
            Defaults to `path_heat_flow_sustainable_W_per_m2`.
        savepath (str or Path, optional): Directory where results will be saved. Defaults to None which outputs the data.
        configuration (str, optional): Type of geothermal system configuration.
            Defaults to 'doublette'.
        manual_values (dict, optional): Dictionary of manually specified values for overriding defaults.

    Returns:
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
    Available at SSRN:
    https://ssrn.com/abstract=5029989
    or http://dx.doi.org/10.2139/ssrn.5029989
    """

    print(citation)

    wfm = EGS_workflowmanager(placements=placements)

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

    ### Calulations
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
    print(f"Data loading finished in {str(int(tic_calc-tic_data_loading))}s.")
    print(f"Calculation finished in {str(int(tic_cost-tic_calc))}s.")
    print(f"Cost calculation finished in {str(int(tic_done-tic_cost))}s.")
    print(
        f"RESkit EGS done within {str(int(tic_done-tic_data_loading))}s for {len(placements)} points.."
    )

    if savepath is None:
        return output


if __name__ == "__main__":
    print("\nThis is not an executable file. Pls run EGSworkflow(args)\n")
