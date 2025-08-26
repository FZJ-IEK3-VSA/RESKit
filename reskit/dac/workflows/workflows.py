# import primary packages
# import othert modules
from .dac_workflow_manager import DACWorkflowManager
from typing import List
import pandas as pd
from ...util.relative_humidity import calculate_relative_humidity


def lt_dac_era5_wenzel2025(
    placements: pd.DataFrame,
    era5_path: str,
    output_netcdf_path: str = None,
    output_variables: List[str] = None,
    model: str = "LT_jajjawi",
    fillMethod: str = "nearest",
):
    """
    Simulation of LT-DAC plants based on ERA5 weather data.

    Parameters
    ----------
    output_netcdf_path: str
        Path to a file that you want to save your output NETCDF file at.
        Default is None

    output_variables: str
        Output variables of the simulation that you want to save into your NETCDF Outputfile.

    model: str
        DAC Model data to utilize

    fillMethod (str):
        method to use when the weather conditions are not inside the hull of the DAC model weather data.
        -nearest: use the nearest available datapoint
        -offTmin: cut off for temperature ranges, nearest for relative humidity
        default: "nearest"
    """

    wf = DACWorkflowManager(placements)

    wf.read(
        variables=["surface_air_temperature", "surface_dew_temperature"],
        source_type="ERA5",
        source=era5_path,
        set_time_index=True,
        verbose=False,
    )

    wf.sim_data["relative_humidity"] = calculate_relative_humidity(
        dewpoint_temperature=wf.sim_data["surface_dew_temperature"],
        air_temperature=wf.sim_data["surface_air_temperature"],
    )
    wf.load_lt_dac_model_data(model=model)
    wf.simulate_lt_dac_model(fillMethod=fillMethod)

    return wf.to_xarray(
        output_netcdf_path=output_netcdf_path,
        output_variables=output_variables,
        custom_attributes=wf.units,
    )


def ht_dac_era5_wenzel2025(
    placements: pd.DataFrame,
    era5_path: str,
    output_netcdf_path: str = None,
    output_variables: List[str] = None,
    model: str = "HT_okosun",
):
    """
    Simulation of HT-DAC plants based on ERA5 weather data.

    Parameters
    ----------
    output_netcdf_path: str
        Path to a file that you want to save your output NETCDF file at.
        Default is None

    output_variables: str
        Output variables of the simulation that you want to save into your NETCDF Outputfile.

    """
    assert model in ["HT_okosun"], f"Invalid model: {model}. You can chose 'HT_okosun'"

    wf = DACWorkflowManager(placements)

    wf.read(
        variables=["surface_air_temperature", "surface_dew_temperature"],
        source_type="ERA5",
        source=era5_path,
        set_time_index=True,
        verbose=False,
    )
    wf.sim_data["relative_humidity"] = calculate_relative_humidity(
        dewpoint_temperature=wf.sim_data["surface_dew_temperature"],
        air_temperature=wf.sim_data["surface_air_temperature"],
    )
    wf.simulate_ht_dac_model(model=model)

    return wf.to_xarray(
        output_netcdf_path=output_netcdf_path,
        output_variables=output_variables,
        custom_attributes=wf.units,
    )
