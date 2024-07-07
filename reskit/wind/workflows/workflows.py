from ... import weather as rk_weather
from ... import util as rk_util
from .wind_workflow_manager import WindWorkflowManager
import numpy as np
import pandas as pd
import os
import yaml


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
    max_batch_size=15000,
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


def onshore_wind_iconlam_2023(
    placements,
    icon_lam_path,
    esa_cci_path,
    output_netcdf_path=None,
    output_variables=None,
    max_batch_size=25000,
):
    """
    Simulates onshore wind generation using high-resolution dynamically downscaled dataset ICON-LAM over southern Africa.

    Parameters
    ----------
    placements : pandas Dataframe
        A Dataframe object with the parameters needed by the simulation.
    icon_lam_path : str
        Path to the specified weather data.
        May contain '<X-TILE>' and <Y-TILE>' spacers, in that case,
        a zoom level value is expected and the correct tiles will be assigned for every
        loation individually.
    esa_cci_path : str
        Path to the ESA CCI raster file [1].
    output_netcdf_path : str, optional
        Path to a directory to put the output files, by default None
    output_variables : str, optional
        Restrict the output variables to these variables, by default None
    max_batch_size: int
        The maximum number of locations to be simulated simultaneously, else multiple batches will be simulated
        iteratively. Helps limiting RAM requirements but may affect runtime. By default 20 000.


    Returns
    -------
    xarray.Dataset
        A xarray dataset including all the output variables.

    Sources
    ------
    [1] ESA. Land Cover CCI Product User Guide Version 2. Tech. Rep. (2017). Available at: maps.elie.ucl.ac.be/CCI/viewer/download/ESACCI-LC-Ph2-PUGv2_2.0.pdf
    """

    wf = WindWorkflowManager(placements)

    # limit the input placements longitude to range of -180...180
    assert wf.placements["lon"].between(-180, 180, inclusive=True).any()
    # limit the input placements latitude to range of -90...90
    assert wf.placements["lat"].between(-90, 90, inclusive=True).any()

    # read data through wind workflow
    # WindWorkflowManager=rk.wind.WindWorkflowManager
    # wf = WindWorkflowManager(placements)
    # read the variables
    wf.read(
        variables=[
            "elevated_wind_speed",  # by default is a 100m!!
            "surface_pressure",
            "surface_air_temperature",
            "boundary_layer_height",
        ],
        source_type="ICON-LAM",
        source=icon_lam_path,
        set_time_index=True,
        spatial_interpolation_mode="near",
        verbose=False,
    )

    # derive roughness value from land cover type
    wf.estimate_roughness_from_land_cover(path=esa_cci_path, source_type="cci")

    # consider boundary layer height on wind speed value, assume wind speed is all the same in the height
    # that is larger than the height of the boundary layer and is equal to wind speed at the boundary layer height.
    wf.logarithmic_projection_of_wind_speeds_to_hub_height(
        consider_boundary_layer_height=False
    )  # you can change it to True

    # Apply density correction
    wf.apply_air_density_correction_to_wind_speeds()

    # Power curve convolution
    # Ryberg, 2019, Energy: scaling factor of 0.06 and base value of 0.1, by default
    wf.convolute_power_curves(scaling=0.01, base=0.00)

    # simulate wind power
    wf.simulate(max_batch_size=max_batch_size)

    return wf.to_xarray(
        output_netcdf_path=output_netcdf_path, output_variables=output_variables
    )


def onshore_wind_era5_pure_2023(
    placements,
    era5_path,
    esa_cci_path,
    output_netcdf_path=None,
    output_variables=None,
):
    """
    SChen: copied from
    Simulates onshore wind generation using pure ECMWF's ERA5 database [1].

    NOTE: Validation documentation is in progress...

    Parameters
    ----------
    placements : pandas Dataframe
        A Dataframe object with the parameters needed by the simulation.
    era5_path : str
        Path to the ERA5 data
    esa_cci_path : str
        Path to the ESA CCI raster file [2].
    output_netcdf_path : str, optional
        Path to a directory to put the output files, by default None
    output_variables : str, optional
        Restrict the output variables to these variables, by default None
    max_batch_size: int
        The maximum number of locations to be simulated simultaneously, else multiple batches will be simulated
        iteratively. Helps limiting RAM requirements but may affect runtime. By default 20 000.

    Returns
    -------
    xarray.Dataset
        A xarray dataset including all the output variables you defined as your output variables.

    Sources
    ------
    [1] European Centre for Medium-Range Weather Forecasts. (2019). ERA5 dataset. https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5
    [2] ESA. Land Cover CCI Product User Guide Version 2. Tech. Rep. (2017). Available at: maps.elie.ucl.ac.be/CCI/viewer/download/ESACCI-LC-Ph2-PUGv2_2.0.pdf
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

    wf.estimate_roughness_from_land_cover(path=esa_cci_path, source_type="cci")

    wf.logarithmic_projection_of_wind_speeds_to_hub_height(
        consider_boundary_layer_height=False
    )

    wf.apply_air_density_correction_to_wind_speeds()

    # gaussian convolution of the power curve to account for statistical events in wind speed
    wf.convolute_power_curves(
        scaling=0.01,  # standard deviation of gaussian equals scaling*v + base
        base=0.0,  # values are derived from validation with real wind turbine data
    )

    # do simulation
    wf.simulate()

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
    weather_lra_ws_path,
    real_lra_ws_path,
    real_lra_ws_scaling,
    real_lra_ws_spatial_interpolation,
    real_lra_ws_nodata_fallback,
    landcover_path,
    landcover_source_type,
    ws_correction_func,
    cf_correction_factor,
    wake_reduction_curve_name,
    availability_factor,
    consider_boundary_layer_height,
    power_curve_scaling,
    power_curve_base,
    convolute_power_curves_args={},
    output_variables=None,
    max_batch_size=25000,
    output_netcdf_path=None,
):
    """
    A generic configuration workflow for wind simulations that allows
    flexible calibration of all arguments used in the workflow.
    NOTE: Only for calibration/validation purposes!

    Parameters
    ----------
    placements : pandas Dataframe
        A Dataframe object with the parameters needed by the simulation.
    weather_path : str
        Path to the temporally resolved weather data, e.g. ERA-5 or MERRA-2 etc.
    weather_lra_ws_path : str
        The path to a raster with the corresponding long-run-average
        windspeeds of the actual weather data (will be corrected to the
        real lra if given, else weather_lra_path has no effect)
    real_lra_ws_path : str, float
        Either a float/int (1.0 means no scaling) or a path to a raster
        with real long-run-average wind speeds, e.g. the Global Wind Atlas
        at the same height as the weather data.
    real_lra_ws_scaling : float
        Accounts for unit differences, set to 1.0 if both weather data and
        real_lra_ws are in the same unit.
    real_lra_ws_spatial_interpolation : str
        The spatial interpolation how the real lra ws shall be extracted,
        e.g. 'near', 'average', 'linear_spline', 'cubic_spline'
    real_lra_ws_nodata_fallback : str, optional
        If no real lra available, use: (1) 'source' for weather data raw
        lra for simulation, (2) 'nan' for nan output
        get flags for missing values:
        - f'missing_values_{os.path.basename(real_lra_ws_path)}
    landcover_path : str
        The path to the categorical landcover raster file.
        Set to None if no hub height scaling at all shall be applied.
    landcover_source_type : str
        Determines the conversion of landcover categories into roughness
        factors, e.g. 'cci', 'clc-code', 'clc', 'globCover', 'modis'.
        Takes effect only if landcover_path is not None.
    ws_correction_func :float, callable, tuple, list
        An executable function that takes a numpy array as single input
        argument and returns an adapted windspeed. If 1.0 is passed, no
        windspeed corrrection will be applied. Can also be passed as tuple
        or list of length 2 with data_type (e.g. 'linear' or 'ws_bins')
        and data dict (dict or path to yaml) with parameters.
    cf_correction_factor : float, str
        The factor by which the output capacity factors will be corrected
        indirectly (via corresponding adaptation of the windspeeds). Can
        be str formatted path to a raster with spatially resolved correction
        factors, set to 1.0 to not apply any correction.
    wake_reduction_curve_name : str
        string value to describe the wake reduction method. None will
        cause no reduction. Else choose from (see more information here
        under wind_efficiency_curve_name[1]): "dena_mean","knorr_mean",
        "dena_extreme1", "dena_extreme2", "knorr_extreme1",
        "knorr_extreme2", "knorr_extreme3",
    availability_factor : float
        This factor accounts for all downtimes and applies an average reduction to the output curve,
        assuming a statistical deviation of the downtime occurences and a large enough turbine fleet.
    consider_boundary_layer_height : bool
        If True, boundary layer height will be considered.
    power_curve_scaling : float
        The scaling factor to smoothen the power curve, for details see:
        convolute_power_curves()
    power_curve_base : float
        The base factor to smoothen the power curve, for details see:
        convolute_power_curves()
    convolute_power_curves_args : dict, optional
        Further convolute_power_curve() arguments, for details see:
        convolute_power_curves(). By default {}.
    output_variables : str, optional
        Restrict the output variables to these variables, by default None
    max_batch_size: int
        The maximum number of locations to be simulated simultaneously, else multiple batches will be simulated
        iteratively. Helps limiting RAM requirements but may affect runtime. By default 25 000. Roughly 7GB RAM per 10k locations.
    output_netcdf_path : str, optional
        Path to a directory to put the output files, by default None

    Returns
    -------
    xarray.Dataset
        A xarray dataset including all the output variables you defined as your output variables.
    """
    if isinstance(ws_correction_func, (int, float)):
        import copy

        factor = copy.copy(ws_correction_func)

        def _dummy_corr(x):
            return factor * x

        ws_correction_func = _dummy_corr
    elif isinstance(ws_correction_func, (tuple, list)):
        assert len(ws_correction_func) == 2
        assert isinstance(ws_correction_func[0], str)
        assert isinstance(ws_correction_func[1], (dict, str))

        # helper function to generate the actual correction function
        def build_ws_correction_function(type, data_dict):
            """
            type: str
                type of correction function
            data_dict: dict, str
                dictionary or json file containing the data needed to
                build the correction function
            """
            if isinstance(data_dict, str):
                assert os.path.isfile(
                    data_dict
                ), f"data_dict is a str but not an existing file: {data_dict}"
                assert os.path.splitext(data_dict)[-1] in [
                    ".yaml",
                    ".yml",
                ], f"data_dict must be a yaml file if given as str path."
                with open(data_dict, "r") as f:
                    data_dict = yaml.load(f, Loader=yaml.FullLoader)
            if type == "polynomial":
                # convert tuple to dict first if needed
                if isinstance(data_dict, (list, tuple)):
                    # assume that the polynomial factors a_i*x^^i are sorted (a_n, ..., a_2, a_1, a_0)
                    data_dict = {i: v for i, v in enumerate(list(data_dict)[::-1])}
                assert isinstance(
                    data_dict, dict
                ), f"data_dict must be a dict if not given as a tuple of polynomial factors."
                assert all(
                    [x % 1 == 0 for x in data_dict.keys()]
                ), f"All data_dict keys must be integers i with values a_i, for all required polynomial factors a_i*x^^i."

                def correction_function(x):
                    _func = 0
                    for deg, fac in data_dict.items():
                        _func = _func + fac * x ** int(deg)
                    return _func

                return correction_function
            elif type == "ws_bins":
                assert (
                    "ws_bins" in data_dict.keys()
                ), "data_dict must contain key 'ws_bins' with a dict of ws bins and factors."
                if not all(
                    isinstance(ws_bin, pd.Interval)
                    for ws_bin in data_dict["ws_bins"].keys()
                ):
                    ws_bins_dict = {}
                    for range_str, factor in data_dict["ws_bins"].copy().items():
                        left, right = range_str.split("-")
                        left = float(left)
                        right = float(right) if right != "inf" else np.inf
                        ws_bins_dict[pd.Interval(left, right, closed="right")] = factor
                    data_dict["ws_bins"] = ws_bins_dict

                # check if all keys are of instance pd.Interval
                assert all(
                    isinstance(ws_bin, pd.Interval)
                    for ws_bin in data_dict["ws_bins"].keys()
                )
                ws_bins_correction = data_dict["ws_bins"]

                def correction_function(x):
                    # x is numpy array. modify x based on ws_bins
                    corrected_x = x.copy()
                    for ws_bin, factor in ws_bins_correction.items():
                        mask = (x >= ws_bin.left) & (x < ws_bin.right)
                        corrected_x[mask] = x[mask] * (1 - factor)
                    return corrected_x

                return correction_function
            else:
                raise ValueError(
                    f"Invalid ws_correction_func type: {type}. Select from: 'polynomial', 'ws_bins'."
                )

        # generate the actual ws corr func
        ws_correction_func = build_ws_correction_function(
            type=ws_correction_func[0],
            data_dict=ws_correction_func[1],
        )
    assert callable(
        ws_correction_func
    ), f"ws_correction_func must be an executable with a single argument that can be passed as np.array (if not 1)."

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
        source_long_run_average=weather_lra_ws_path,
        real_long_run_average=real_lra_ws_path,
        nodata_fallback=real_lra_ws_nodata_fallback,
        spatial_interpolation=real_lra_ws_spatial_interpolation,
        real_lra_scaling=real_lra_ws_scaling,
    )

    if landcover_path:
        wf.estimate_roughness_from_land_cover(
            path=landcover_path, source_type=landcover_source_type
        )

        wf.logarithmic_projection_of_wind_speeds_to_hub_height(
            consider_boundary_layer_height=consider_boundary_layer_height
        )

    # correct wind speeds
    wf.sim_data["elevated_wind_speed"] = ws_correction_func(
        wf.sim_data["elevated_wind_speed"]
    )

    wf.apply_air_density_correction_to_wind_speeds()

    # do wake reduction if applicable
    wf.apply_wake_correction_of_wind_speeds(
        wake_reduction_curve_name=wake_reduction_curve_name
    )

    # gaussian convolution of the power curve to account for statistical events in wind speed
    wf.convolute_power_curves(
        scaling=power_curve_scaling,
        base=power_curve_base,
        **convolute_power_curves_args,
    )

    # do simulation
    wf.simulate(
        cf_correction_factor=cf_correction_factor, max_batch_size=max_batch_size
    )

    # apply availability factor
    wf.apply_availability_factor(availability_factor=availability_factor)

    return wf.to_xarray(
        output_netcdf_path=output_netcdf_path, output_variables=output_variables
    )
