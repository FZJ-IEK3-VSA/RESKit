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
    Simulate LT-DAC plants using ERA5 weather data.

    This function runs a full simulation workflow for low-temperature direct air capture (LT-DAC)
    plants. It reads ERA5 weather data, calculates relative humidity, loads the specified LT-DAC
    model data, performs the simulation, and optionally saves results to a NetCDF file.

    Parameters
    ----------
    placements : pd.DataFrame
        DataFrame specifying the plant locations and capacities.
    era5_path : str
        Path to the ERA5 weather data source.
    output_netcdf_path : str, optional
        Path to save the output NetCDF file. If None, no file is saved. Default is None.
    output_variables : list of str, optional
        List of variables from the simulation to include in the output NetCDF file.
        If None, all available variables are included. Default is None.
    model : str, optional
        DAC model data to utilize. Default is "LT_jajjawi".
    fillMethod : str, optional
        Method for filling weather conditions outside the DAC model data hull:
        - "nearest" : use the nearest available datapoint (default)
        - "offTmin" : cut off for temperatures outside the model range, nearest for relative humidity

    Returns
    -------
    xarray.Dataset
        Simulation results, optionally limited to `output_variables` and including all plant locations.

    Notes
    -----
    The simulation includes calculation of:
    - relative humidity
    - DAC capacity factor
    - electricity, heat, and water conversion factors
    - CO2, water, electricity, and heat outputs per plant
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
    Simulate HT-DAC plants using ERA5 weather data.

    This function runs a full simulation workflow for high-temperature direct air capture (HT-DAC)
    plants. It reads ERA5 weather data, calculates relative humidity, runs the specified HT-DAC
    model simulation, and optionally saves results to a NetCDF file.

    Parameters
    ----------
    placements : pd.DataFrame
        DataFrame specifying the plant locations and capacities.
    era5_path : str
        Path to the ERA5 weather data source.
    output_netcdf_path : str, optional
        Path to save the output NetCDF file. If None, no file is saved. Default is None.
    output_variables : list of str, optional
        List of variables from the simulation to include in the output NetCDF file.
        If None, all available variables are included. Default is None.
    model : str, optional
        DAC model to use. Currently, only "HT_okosun" is implemented. Default is "HT_okosun".

    Returns
    -------
    xarray.Dataset
        Simulation results, optionally limited to `output_variables` and including all plant locations.

    Raises
    ------
    AssertionError
        If `model` is not "HT_okosun".

    Notes
    -----
    The simulation includes calculation of:
    - relative humidity
    - DAC capacity factor
    - electricity conversion factor
    - CO2 output per plant

    The simulation relies on the `DACWorkflowManager` and the specified HT-DAC model.
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
