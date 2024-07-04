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
        iteratively. Helps limiting RAM requirements but may affect runtime. By default 25 000. Roughly 7GB RAM per 10k locations.

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
        iteratively. Helps limiting RAM requirements but may affect runtime. By default 25 000. Roughly 7GB RAM per 10k locations.

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
        iteratively. Helps limiting RAM requirements but may affect runtime. By default 25 000. Roughly 7GB RAM per 10k locations.

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
        variables=[
            "elevated_wind_speed",
        ],  # Why we dont read P, T or boundary_layer_height?
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
    assert wf.placements["lon"].between(-180, 180, inclusive="both").any()
    # limit the input placements latitude to range of -90...90
    assert wf.placements["lon"].between(-180, 180, inclusive="both").any()

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
    wf.sim_data["elevated_wind_speed"] = np.maximum(
        wf.sim_data["elevated_wind_speed"] * 0.75 + 0.75, 0
    )  # Empirically found to improve simulation accuracy

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
    correction_factor=1.0,
    max_batch_size=25000,
    wake_reduction_curve_name="dena_mean",
    availability_factor=0.98,
    era5_lra_path=None,
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
    correction_factor: str, float, optional
        The wind speeds will be adapted such that the average capacity factor output is
        scaled by the given factor. The factor may either be a float or a str formatted
        raster path containing local float correction factors.By default 1.0, i.e. no correction.
    max_batch_size: int
        The maximum number of locations to be simulated simultaneously, else multiple batches will be simulated
        iteratively. Helps limiting RAM requirements but may affect runtime. By default 25 000. Roughly 7GB RAM per 10k locations.
    wake_reduction_curve_name : str, optional
        string value to describe the wake reduction method. None will cause no reduction, by default
        "dena_mean". Choose from (see more information here under wind_efficiency_curve_name[1]): "dena_mean",
        "knorr_mean", "dena_extreme1", "dena_extreme2", "knorr_extreme1", "knorr_extreme2", "knorr_extreme3",
    availability_factor : float, otional
        This factor accounts for all downtimes and applies an average reduction to the output curve,
        assuming a statistical deviation of the downtime occurences and a large enough turbine fleet.
        By default 0.98 as suggested availability including technical availability of turbine and connector
        as well as outages for ecological reasons (e.g. bat protection). This does not include wake effects
        (see above) or curtailment/outage for economical reasons or transmission grid congestion.

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
    assert wf.placements["lon"].between(-180, 180, inclusive="both").any()
    # limit the input placements latitude to range of -90...90
    assert wf.placements["lon"].between(-180, 180, inclusive="both").any()

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

    if not era5_lra_path:
        era5_lra_path = rk_weather.Era5Source.LONG_RUN_AVERAGE_WINDSPEED
    wf.adjust_variable_to_long_run_average(
        variable="elevated_wind_speed",
        source_long_run_average=era5_lra_path,
        real_long_run_average=gwa_100m_path,
        nodata_fallback=nodata_fallback,
        spatial_interpolation="average",
    )

    wf.estimate_roughness_from_land_cover(path=esa_cci_path, source_type="cci")

    wf.logarithmic_projection_of_wind_speeds_to_hub_height(
        consider_boundary_layer_height=True
    )

    # Adjust wind speeds with global correction factors
    x = 0.7506109812177267  # Correction factors from wind validation paper
    b = 0.9064913929439484  # Correction factors from wind validation paper
    wf.sim_data["elevated_wind_speed"] = wf.sim_data["elevated_wind_speed"] * x + b

    wf.apply_air_density_correction_to_wind_speeds()

    # do wake reduction if applicable
    wf.apply_wake_correction_of_wind_speeds(
        wake_reduction_curve_name=wake_reduction_curve_name
    )

    # gaussian convolution of the power curve to account for statistical events in wind speed
    wf.convolute_power_curves(
        scaling=0.01,  # standard deviation of gaussian equals scaling*v + base
        base=0.00,  # values are derived from validation with real wind turbine data
    )

    # do simulation
    wf.simulate(cf_correction_factor=correction_factor, max_batch_size=max_batch_size)

    # apply availability factor
    wf.apply_availability_factor(availability_factor=availability_factor)

    return wf.to_xarray(
        output_netcdf_path=output_netcdf_path, output_variables=output_variables
    )


def mean_capacity_factor_from_sectoral_weibull(
    placements, a_rasters, k_rasters, f_rasters, output=None
):
    pass


def wind_config(
    placements,
    weather_path,
    weather_source_type,
    lra_ws_path, #1
    lra_ws_scaling,
    lra_ws_spatial_interpolation,
    landcover_path, 
    landcover_source_type,
    lra_ws_nodata_fallback,
    ws_correction_func,
    cf_correction_factor,
    wake_reduction_curve_name,
    availability_factor,
    weather_lra_path,
    consider_boundary_layer_height,
    power_curve_scaling,
    power_curve_base,
    convolute_power_curves_args={},
    output_variables=None,
    max_batch_size=25000,
    output_netcdf_path=None,
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
    correction_factor: str, float, optional
        The wind speeds will be adapted such that the average capacity factor output is
        scaled by the given factor. The factor may either be a float or a str formatted
        raster path containing local float correction factors.By default 1.0, i.e. no correction.
    max_batch_size: int
        The maximum number of locations to be simulated simultaneously, else multiple batches will be simulated
        iteratively. Helps limiting RAM requirements but may affect runtime. By default 25 000. Roughly 7GB RAM per 10k locations.
    wake_reduction_curve_name : str, optional
        string value to describe the wake reduction method. None will cause no reduction, by default
        "dena_mean". Choose from (see more information here under wind_efficiency_curve_name[1]): "dena_mean",
        "knorr_mean", "dena_extreme1", "dena_extreme2", "knorr_extreme1", "knorr_extreme2", "knorr_extreme3",
    availability_factor : float, otional
        This factor accounts for all downtimes and applies an average reduction to the output curve,
        assuming a statistical deviation of the downtime occurences and a large enough turbine fleet.
        By default 0.98 as suggested availability including technical availability of turbine and connector
        as well as outages for ecological reasons (e.g. bat protection). This does not include wake effects
        (see above) or curtailment/outage for economical reasons or transmission grid congestion.

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
    assert callable(ws_correction_func), \
        f"ws_correction_func must be an executable with a single argument that can be passed as np.array."

    wf = WindWorkflowManager(placements)

    # limit the input placements longitude to range of -180...180
    assert wf.placements["lon"].between(-180, 180, inclusive="both").any()
    # limit the input placements latitude to range of -90...90
    assert wf.placements["lon"].between(-180, 180, inclusive="both").any()

    wf.read(
        variables=[
            "elevated_wind_speed",
            "surface_pressure",
            "surface_air_temperature",
            "boundary_layer_height",
        ],
        source_type=weather_source_type,
        source=weather_path,
        set_time_index=True,
        verbose=False,
    )

    wf.adjust_variable_to_long_run_average(
        variable="elevated_wind_speed",
        source_long_run_average=weather_lra_path,
        real_long_run_average=lra_ws_path,
        nodata_fallback=lra_ws_nodata_fallback,
        spatial_interpolation=lra_ws_spatial_interpolation,
        real_lra_scaling=lra_ws_scaling,
    )

    if landcover_path:
        wf.estimate_roughness_from_land_cover(path=landcover_path, source_type=landcover_source_type)

        wf.logarithmic_projection_of_wind_speeds_to_hub_height(
            consider_boundary_layer_height=consider_boundary_layer_height
        )

    # correct wind speeds
    wf.sim_data["elevated_wind_speed"] = ws_correction_func(wf.sim_data["elevated_wind_speed"])


    wf.apply_air_density_correction_to_wind_speeds()

    # do wake reduction if applicable
    wf.apply_wake_correction_of_wind_speeds(
        wake_reduction_curve_name=wake_reduction_curve_name
    )

    # gaussian convolution of the power curve to account for statistical events in wind speed
    wf.convolute_power_curves(
        scaling=power_curve_scaling,  # standard deviation of gaussian equals scaling*v + base
        base=power_curve_base,  # values are derived from validation with real wind turbine data
        **convolute_power_curves_args,
    )

    # do simulation
    wf.simulate(cf_correction_factor=cf_correction_factor, max_batch_size=max_batch_size)

    # apply availability factor
    wf.apply_availability_factor(availability_factor=availability_factor)

    return wf.to_xarray(
        output_netcdf_path=output_netcdf_path, output_variables=output_variables
    )