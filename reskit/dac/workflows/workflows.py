# import primary packages
# import othert modules
from .dac_workflow_manager import DACWorkflowManager


def lt_dac_era5_wenzel2025(
    placements,
    era5_path,
    output_netcdf_path=None,
    output_variables=None,
    model="LT_jajjawi",
    fillMethod="nearest",
):
    """
    Simulation of LT-DAC plants based on ERA5 weather data.

    output_netcdf_path: str
            Path to a file that you want to save your output NETCDF file at.
            Default is None

    output_variables: str
            Output variables of the simulation that you want to save into your NETCDF Outputfile.

    model: str
            DAC Model data to utilize

    fillMethod (str): method to use when the weather conditions are not inside the hull of the DAC model weather data.
            -nearest: use the nearest available datapoint
            -offTmin: cut off for temperature ranges, nearest for relative humidity
            default: "nearest"

    """
    assert model in [
        "LT_jajjawi",
        "LT_sendi",
    ], f"Invalid model: {model}. You can chose between 'LT_jajjawi' or 'LT_sendi'"

    wf = DACWorkflowManager(placements)

    wf.read(
        variables=["surface_air_temperature", "surface_dew_temperature"],
        source_type="ERA5",
        source=era5_path,
        set_time_index=True,
        verbose=False,
    )
    wf.calculate_relative_humidity()
    wf.load_lt_dac_model_data(model)
    wf.simulate_lt_dac_model(fillMethod=fillMethod)

    return wf.to_xarray(
        output_netcdf_path=output_netcdf_path,
        output_variables=output_variables,
        custom_attributes=wf.units,
    )


def ht_dac_era5_wenzel2025(
    placements,
    era5_path,
    output_netcdf_path=None,
    output_variables=None,
    model="HT_okosun",
):
    """
    Simulation of HT-DAC plants based on ERA5 weather data.

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
    wf.calculate_relative_humidity()
    wf.simulate_ht_dac_model(model)

    return wf.to_xarray(
        output_netcdf_path=output_netcdf_path,
        output_variables=output_variables,
        custom_attributes=wf.units,
    )
