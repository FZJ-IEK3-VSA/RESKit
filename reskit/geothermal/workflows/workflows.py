import os
import warnings

import pandas as pd

from .egs_workflow_manager import EGSWorkflowManager
from reskit.util.input_validation import declare_inputs
from reskit.util.provenance import record_provenance

from ..data import path_temperatures
from ..data import path_heat_flow_sustainable_W_per_m2


@declare_inputs(
    EGSWorkflowManager,
    files=("sourceTemperature", "sourceSustainableHeatflow"),
    outputs=("output_netcdf_path", "savepath"),
)
@record_provenance
def egs_workflow(
    placements: pd.DataFrame,
    sourceTemperature=path_temperatures,
    sourceSustainableHeatflow=path_heat_flow_sustainable_W_per_m2,
    savepath=None,
    configuration="doublette",
    manual_values=None,
    *,
    output_netcdf_path=None,
    output_variables=None,
):
    """
    Simulates Enhanced Geothermal System (EGS) plants at the given placements.

    For each placement and each of three methods, the workflow finds the drilling depth with
    the lowest LCOE and returns the plant's values at that depth: the volume method, the
    Gringarten method and the sustainable heat flow method. The results have no time series,
    so the dataset has only the dimension 'location'.

    Parameters
    ----------
    placements : pd.DataFrame
        The placements, with point geometries in the column 'geom', or the columns 'lon' and
        'lat' in EPSG:4326. A column 'location_id' becomes the coordinate of 'location'.
    sourceTemperature : str or Path, optional
        Path to the rock temperature per depth, by default the file shipped with RESKit
        (`reskit.geothermal.data.path_temperatures`)
    sourceSustainableHeatflow : str or Path, optional
        Path to the sustainable heat flow, by default the file shipped with RESKit
        (`reskit.geothermal.data.path_heat_flow_sustainable_W_per_m2`)
    savepath : str, optional
        Deprecated, use `output_netcdf_path`; removed in RESKit v1.0.0. A path ending in
        .nc or .nc4 is written as NetCDF, .csv as a table.
    configuration : str, optional
        The well configuration, 'doublette' (default) or 'triplette'
    manual_values : dict, optional
        Values which replace the plant parameters of the configuration, e.g. {"x_ED_1": 8}
    output_netcdf_path : str, optional
        If given, the results are written to this NetCDF file and its path is returned
    output_variables : list of str, optional
        The variables to include in the results, by default all

    Returns
    -------
    xarray.Dataset or str
        The results along the dimension 'location', or the path of the file written. The
        attributes of the dataset give the unit of each variable. Besides 'lon', 'lat',
        'surface_temperature' and 'sustainable_heat_flow', the variables are the values at
        the optimal depth of each method, named with the suffix '_volume', '_gringarten' or
        '_sustainable':

        - optimal_depth: the depth with the lowest LCOE; NaN, like all other values of the
          method, where no depth is eligible
        - lcoe, lcoe_gross: the LCOE from the net and from the gross power output
        - totex: the annuity of the capital cost plus the fixed operating cost
        - heat_output: the thermal output; the mean over the lifetime for the volume
          method, the output in the last year of the lifetime for the Gringarten method
        - power_output_gross, power_output_net: the electric output before and after
          subtracting pump_power_per_well
        - pump_power_per_well: the power of the pump of one production well
        - water_mass_flow, water_mass_flow_per_well: the water flow of all production wells
          and of one
        - rock_temperature: the initial rock temperature at the optimal depth
        - rock_temperature_abandonment: the mean rock temperature at the end of the lifetime
        - water_outlet_temperature: the temperature of the produced water; the initial
          rock temperature for the volume and the sustainable method
        - temperature_drawdown_reservoir: the mean temperature drop of the reservoir over
          the lifetime
        - temperature_drawdown_active: the temperature drop of the rock the water flows
          through, a parameter of the volume method; -1 for the Gringarten method, which
          does not calculate it
        - recovery_factor: the share of the heat in place which is extracted over the
          lifetime; -1 for the sustainable method, which does not calculate it
        - resource_use_time: the years until the rock (volume method) or the produced
          water (Gringarten method) cools to the minimum rock temperature of the
          configuration; infinite for the sustainable method
        - regeneration_time: the years the sustainable heat flow needs to replace the heat
          extracted over the lifetime

        The volume method also gives heat_in_place_volume, the heat in the reservoir above
        the surface temperature.

    Citation
    --------
    Franzmann, D., Heinrichs, H. and Stolten, D. (2025), Global geothermal electricity
    potentials: A technical, economic, and thermal renewability assessment.
    Renewable Energy 250, 123199. https://doi.org/10.1016/j.renene.2025.123199
    """
    if savepath is not None:
        warnings.warn(
            "The argument savepath of egs_workflow() is deprecated and will be removed in RESKit v1.0.0. "
            "Use output_netcdf_path instead.",
            DeprecationWarning,
            # past the wrappers of record_provenance and declare_inputs, to the caller
            stacklevel=4,
        )
        if output_netcdf_path is not None:
            raise ValueError("Give either savepath or output_netcdf_path, not both")

    wfm = EGSWorkflowManager(placements=placements)

    # data loading
    wfm.loadDataAllDepths(vars=["temperature"], source=sourceTemperature)
    wfm.loadData(vars=["surface_temperature"], source=sourceTemperature)
    wfm.loadData(
        vars=["heat_flow_sustainable_W_per_m2"],
        source=sourceSustainableHeatflow,
        newVarNamesDict={"heat_flow_sustainable_W_per_m2": "sustainable_heat_flow"},
    )
    wfm.loadPlantData(configuration=configuration, manual_values=manual_values or {})

    # thermal and electric output per depth
    wfm.VolumeMethod()
    wfm.GringartenMethodFixeVdot()
    wfm.SustainableHeat()

    # costs, and the values at the depth with the lowest LCOE
    for techMethod in wfm._getTechMethods():
        wfm.calculatePumpLosses(techMethod=techMethod)
        wfm.calculateCosts(techMethod=techMethod)
        wfm.calculateLCOE(techMethod=techMethod)
        wfm.getRegenerationTime(techMethod=techMethod)
        wfm.getOptDepth(techMethod=techMethod)
        wfm.getValuesAtOptDepth(techMethod=techMethod)

    if savepath is None:
        return wfm.to_xarray(
            output_netcdf_path=output_netcdf_path, output_variables=output_variables, custom_attributes=wfm.units
        )
    return _save_deprecated(wfm, str(savepath), output_variables)


def _save_deprecated(wfm, savepath, output_variables):
    """Writes the results to savepath as egs_workflow did before RESKit v0.6.0 and returns the path."""
    filetype = os.path.splitext(savepath)[1].lower()
    # .shp is no longer accepted: shapefile field names have at most 10 characters, so
    # writing the results as shapefile always failed
    if filetype not in (".nc", ".nc4", ".csv"):
        raise ValueError(f"savepath must end in .nc, .nc4 or .csv, not '{filetype}'")
    if filetype in (".nc", ".nc4"):
        return wfm.to_xarray(
            output_netcdf_path=savepath, output_variables=output_variables, custom_attributes=wfm.units
        )

    wfm.to_xarray(output_variables=output_variables).to_dataframe().to_csv(savepath)
    return savepath


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
