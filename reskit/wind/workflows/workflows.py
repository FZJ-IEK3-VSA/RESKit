from ... import weather as rk_weather
from ... import util as rk_util
from .wind_workflow_manager import WindWorkflowManager
import numpy as np


def onshore_wind_merra_ryberg2019_europe(
    placements,
    merra_path,
    gwa_50m_path,
    clc2012_path,
    output_netcdf_path=None,
    output_variables=None,
    max_batch_size=25000,
):
    # TODO: Add range limitation over Europe by checking placements
    """
    Simulates onshore wind generation in Europe using NASA's MERRA2 database [1].

    Parameters
    ----------
    placements : pandas Dataframe
        A Dataframe object with the parameters needed by the simulation.
    merra_path : str
        Path to the MERRA2 data.
    gwa_50m_path : str
        Path to the Global Wind Atlas at 50m [2] rater file.
    clc2012_path : str
        Path to the CLC 2012 raster file [3].
    output_netcdf_path : str, optional
        Path to a directory to put the output files, by default None
    output_variables : str, optional
        Restrict the output variables to these variables, by default None
    max_batch_size: int
        The maximum number of locations to be simulated simultaneously, else multiple batches will be simulated
        iteratively. Helps limiting RAM requirements but may affect runtime. By default 25 000.

    Returns
    -------
    xarray.Dataset
        A xarray dataset including all the output variables you defined as your output variables.

    Sources
    ------
    [1] NASA (National Aeronautics and Space Administration). (2019). Modern-Era Retrospective analysis for Research and Applications, Version 2. NASA Goddard Earth Sciences (GES) Data and Information Services Center (DISC). https://disc.gsfc.nasa.gov/datasets?keywords=%22MERRA-2%22&page=1&source=Models%2FAnalyses MERRA-2
    [2] DTU Wind Energy. (2019). Global Wind Atlas. https://globalwindatlas.info/
    [3] Copernicus (European Union’s Earth Observation Programme). (2012). Corine Land Cover 2012. Copernicus. https://land.copernicus.eu/pan-european/corine-land-cover/clc-2012

    """

    wf = WindWorkflowManager(placements)

    wf.read(
        variables=[
            "elevated_wind_speed",
            "surface_pressure",
            "surface_air_temperature",
        ],
        source_type="MERRA",
        source=merra_path,
        set_time_index=True,
        verbose=False,
    )

    wf.adjust_variable_to_long_run_average(
        variable="elevated_wind_speed",
        source_long_run_average=rk_weather.MerraSource.LONG_RUN_AVERAGE_WINDSPEED,
        real_long_run_average=gwa_50m_path,
    )

    wf.estimate_roughness_from_land_cover(path=clc2012_path, source_type="clc")

    wf.logarithmic_projection_of_wind_speeds_to_hub_height()

    wf.apply_air_density_correction_to_wind_speeds()

    wf.convolute_power_curves(scaling=0.06, base=0.1)

    wf.simulate(max_batch_size=max_batch_size)

    wf.apply_loss_factor(
        loss=lambda x: rk_util.low_generation_loss(x, base=0.0, sharpness=5.0)
    )

    return wf.to_xarray(
        output_netcdf_path=output_netcdf_path, output_variables=output_variables
    )


def offshore_wind_merra_caglayan2019(
    placements,
    merra_path,
    output_netcdf_path=None,
    output_variables=None,
    max_batch_size=25000,
):
    """
    Simulates offshore wind generation using NASA's MERRA2 database [1].

    Parameters
    ----------
    placements : pandas Dataframe
        A Dataframe object with the parameters needed by the simulation.
    merra_path : str
        Path to the MERRA2 data.
    output_netcdf_path : str, optional
        Path to a directory to put the output files, by default None
    output_variables : str, optional
        Restrict the output variables to these variables, by default None
    max_batch_size: int
        The maximum number of locations to be simulated simultaneously, else multiple batches will be simulated
        iteratively. Helps limiting RAM requirements but may affect runtime. By default 25 000.

    Returns
    -------
    xarray.Dataset
        A xarray dataset including all the output variables you defined as your output variables.

    Sources
    ------
    [1] National Aeronautics and Space Administration. (2019). Modern-Era Retrospective analysis for Research and Applications, Version 2. NASA Goddard Earth Sciences (GES) Data and Information Services Center (DISC). https://disc.gsfc.nasa.gov/datasets?keywords=%22MERRA-2%22&page=1&source=Models%2FAnalyses MERRA-2

    """

    wf = WindWorkflowManager(placements)

    wf.read(
        variables=[
            "elevated_wind_speed",
        ],
        source_type="MERRA",
        source=merra_path,
        set_time_index=True,
        verbose=False,
    )

    wf.set_roughness(0.0002)

    wf.logarithmic_projection_of_wind_speeds_to_hub_height()

    wf.convolute_power_curves(
        scaling=0.04,  # TODO: Check values with Dil
        base=0.5,  # TODO: Check values with Dil
    )

    wf.simulate(max_batch_size=max_batch_size)

    wf.apply_loss_factor(
        loss=lambda x: rk_util.low_generation_loss(
            x, base=0.1, sharpness=3.5
        )  # TODO: Check values with Dil
    )

    return wf.to_xarray(
        output_netcdf_path=output_netcdf_path, output_variables=output_variables
    )


def offshore_wind_era5(
    placements,
    era5_path,
    gwa_100m_path=None,
    output_netcdf_path=None,
    output_variables=None,
    max_batch_size=25000,
):
    """
    Simulates offshore wind generation using NASA's ERA5 database [1].

    Parameters
    ----------
    placements : pandas Dataframe
        A Dataframe object with the parameters needed by the simulation.
    era5_path : str
        Path to the ERA5 data.
    output_netcdf_path : str, optional
        Path to a directory to put the output files, by default None
    output_variables : str, optional
        Restrict the output variables to these variables, by default None
    max_batch_size: int
        The maximum number of locations to be simulated simultaneously, else multiple batches will be simulated
        iteratively. Helps limiting RAM requirements but may affect runtime. By default 25 000.

    Returns
    -------
    xarray.Dataset
        A xarray dataset including all the output variables you defined as your output variables.

    Sources
    ------
    [1] European Centre for Medium-Range Weather Forecasts. (2019). ERA5 dataset. https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5.

    """
    wf = WindWorkflowManager(placements)

    wf.read(
        # Why we dont read P, T or boundary_layer_height?
        variables=[
            "elevated_wind_speed",
        ],
        source_type="ERA5",
        source=era5_path,
        set_time_index=True,
        verbose=False,
    )

    # wf.adjust_variable_to_long_run_average(
    #     variable='elevated_wind_speed',
    #     source_long_run_average=rk_weather.Era5Source.LONG_RUN_AVERAGE_WINDSPEED,
    #     real_long_run_average=gwa_100m_path
    # )

    wf.set_roughness(0.0002)

    wf.logarithmic_projection_of_wind_speeds_to_hub_height()

    # gaussian convolution of the power curve to account for statistical events in wind speed
    wf.convolute_power_curves(
        scaling=0.01,  # standard deviation of gaussian equals scaling*v + base
        base=0.00,  # values are derived from validation with real wind turbine data
    )

    # Adjust wind speeds
    # elevated windspeds are corrected by a linear function by comparing to real wind turbine data.
    # corrected_speed = windspeed * wind_speed_scaling + wind_speed_offset [m/s]
    wind_speed_scaling = 0.95
    wind_speed_offset = 0.0
    wf.sim_data["elevated_wind_speed"] = np.maximum(
        wf.sim_data["elevated_wind_speed"] * wind_speed_scaling + wind_speed_offset, 0
    )

    wf.simulate(max_batch_size=max_batch_size)

    return wf.to_xarray(
        output_netcdf_path=output_netcdf_path, output_variables=output_variables
    )


def onshore_wind_era5(
    placements,
    era5_path,
    gwa_100m_path,
    esa_cci_path,
    output_netcdf_path=None,
    output_variables=None,
    nodata_fallback="nan",
    max_batch_size=25000,
):
    """
    Simulates onshore wind generation using ECMWF's ERA5 database [1].

    NOTE: Validation documentation is in progress...

    Parameters
    ----------
    placements : pandas Dataframe
        A Dataframe object with the parameters needed by the simulation.
    era5_path : str
        Path to the ERA5 data.
    gwa_100m_path : str
        Path to the Global Wind Atlas at 100m [2] rater file.
    esa_cci_path : str
        Path to the ESA CCI raster file [3].
    output_netcdf_path : str, optional
        Path to a directory to put the output files, by default None
    output_variables : str, optional
        Restrict the output variables to these variables, by default None
    nodata_fallback: str, optional
        If no GWA available, use: (1) 'source' for ERA5 raw for simulation, (2) 'nan' for nan output
        get flags for missing values:
        - f'missing_values_{os.path.basename(path_to_LRA_source)}

    Returns
    -------
    xarray.Dataset
        A xarray dataset including all the output variables you defined as your output variables.

    Sources
    ------
    [1] European Centre for Medium-Range Weather Forecasts. (2019). ERA5 dataset. https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5
    [2] DTU Wind Energy. (2019). Global Wind Atlas. https://globalwindatlas.info/
    [3] ESA. Land Cover CCI Product User Guide Version 2. Tech. Rep. (2017). Available at: maps.elie.ucl.ac.be/CCI/viewer/download/ESACCI-LC-Ph2-PUGv2_2.0.pdf
    """

    wf = WindWorkflowManager(placements)

    # limit the input placements longitude to range of -180...180
    assert wf.placements["lon"].between(-180, 180, inclusive=True).any()
    # limit the input placements latitude to range of -90...90
    assert wf.placements["lat"].between(-90, 90, inclusive=True).any()

    wf.read(
        variables=[
            "elevated_wind_speed",
            "surface_pressure",
            "surface_air_temperature",
            "boundary_layer_height",
        ],
        source_type="ERA5",
        source=era5_path,
        set_time_index=True,
        verbose=False,
    )

    wf.adjust_variable_to_long_run_average(
        variable="elevated_wind_speed",
        source_long_run_average=rk_weather.Era5Source.LONG_RUN_AVERAGE_WINDSPEED,
        real_long_run_average=gwa_100m_path,
        nodata_fallback=nodata_fallback,
    )

    wf.estimate_roughness_from_land_cover(path=esa_cci_path, source_type="cci")

    wf.logarithmic_projection_of_wind_speeds_to_hub_height(
        consider_boundary_layer_height=True
    )

    wf.apply_air_density_correction_to_wind_speeds()

    # gaussian convolution of the power curve to account for statistical events in wind speed
    wf.convolute_power_curves(
        scaling=0.01,  # standard deviation of gaussian equals scaling*v + base
        base=0.00,  # values are derived from validation with real wind turbine data
    )

    # Adjust wind speeds
    # elevated windspeds are corrected by a linear function by comparing to real wind turbine data.
    # corrected_speed = windspeed * 0.75 + 1.2 [m/s]
    # Empirically found to improve simulation accuracy
    wf.sim_data["elevated_wind_speed"] = np.maximum(
        wf.sim_data["elevated_wind_speed"] * 0.75 + 0.75, 0
    )

    # do simulation
    wf.simulate(max_batch_size=max_batch_size)

    return wf.to_xarray(
        output_netcdf_path=output_netcdf_path, output_variables=output_variables
    )


def wind_era5_2023(
    placements,
    era5_path,
    gwa_100m_path,
    esa_cci_path,
    output_netcdf_path=None,
    output_variables=None,
    nodata_fallback="nan",
    max_batch_size=25000,
):
    """
    Simulates onshore and offshore (200km from shoreline) wind generation using ECMWF's ERA5 database [1].

    NOTE: Validation documentation is in progress...

    Parameters
    ----------
    placements : pandas Dataframe
        A Dataframe object with the parameters needed by the simulation.
    era5_path : str
        Path to the ERA5 data.
    gwa_100m_path : str
        Path to the Global Wind Atlas at 100m [2] raster file.
    esa_cci_path : str
        Path to the ESA CCI raster file [3].
    output_netcdf_path : str, optional
        Path to a directory to put the output files, by default None
    output_variables : str, optional
        Restrict the output variables to these variables, by default None
    nodata_fallback: str, optional
        If no GWA available, use: (1) 'source' for ERA5 raw for simulation, (2) 'nan' for nan output
        get flags for missing values:
        - f'missing_values_{os.path.basename(path_to_LRA_source)}
    max_batch_size: int
        The maximum number of locations to be simulated simultaneously, else multiple batches will be simulated
        iteratively. Helps limiting RAM requirements but may affect runtime. By default 25 000.

    Returns
    -------
    xarray.Dataset
        A xarray dataset including all the output variables you defined as your output variables.

    Sources
    ------
    [1] European Centre for Medium-Range Weather Forecasts. (2019). ERA5 dataset. https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5
    [2] DTU Wind Energy. (2019). Global Wind Atlas. https://globalwindatlas.info/
    [3] ESA. Land Cover CCI Product User Guide Version 2. Tech. Rep. (2017). Available at: maps.elie.ucl.ac.be/CCI/viewer/download/ESACCI-LC-Ph2-PUGv2_2.0.pdf
    """

    wf = WindWorkflowManager(placements)

    # limit the input placements longitude to range of -180...180
    assert wf.placements["lon"].between(-180, 180, inclusive=True).any()
    # limit the input placements latitude to range of -90...90
    assert wf.placements["lat"].between(-90, 90, inclusive=True).any()

    wf.read(
        variables=[
            "elevated_wind_speed",
            "surface_pressure",
            "surface_air_temperature",
            "boundary_layer_height",
        ],
        source_type="ERA5",
        source=era5_path,
        set_time_index=True,
        verbose=False,
    )

    wf.adjust_variable_to_long_run_average(
        variable="elevated_wind_speed",
        source_long_run_average=rk_weather.Era5Source.LONG_RUN_AVERAGE_WINDSPEED,
        real_long_run_average=gwa_100m_path,
        nodata_fallback=nodata_fallback,
    )

    wf.estimate_roughness_from_land_cover(path=esa_cci_path, source_type="cci")

    wf.logarithmic_projection_of_wind_speeds_to_hub_height(
        consider_boundary_layer_height=True
    )

    # Adjust wind speeds with global correction factors
    x = 0.722764112282280
    b = 1.4715877632502439
    wf.sim_data["elevated_wind_speed"] = wf.sim_data["elevated_wind_speed"] * x + b

    wf.apply_air_density_correction_to_wind_speeds()

    # gaussian convolution of the power curve to account for statistical events in wind speed
    wf.convolute_power_curves(
        scaling=0.01,  # standard deviation of gaussian equals scaling*v + base
        base=0.00,  # values are derived from validation with real wind turbine data
    )

    # do simulation
    wf.simulate(max_batch_size=max_batch_size)

    return wf.to_xarray(
        output_netcdf_path=output_netcdf_path, output_variables=output_variables
    )


def mean_capacity_factor_from_sectoral_weibull(
    placements, a_rasters, k_rasters, f_rasters, output=None
):
    pass
