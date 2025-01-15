# import primary packages
import numpy as np
import os
import pandas as pd
import yaml
# import modules
import reskit.weather as rk_weather
import reskit.util as rk_util
from reskit.wind import DATAFOLDER
from reskit.wind.core.windspeed_correction import build_ws_correction_function
from reskit.wind.workflows.wind_workflow_manager import WindWorkflowManager


def wind_era5_PenaSanchezDunkelWinkler2025(
    placements,
    era5_path,
    gwa_100m_path,
    esa_cci_path,
    output_netcdf_path=None,
    output_variables=None,
    max_batch_size=15000,
    cf_correction=True,
    **simulate_kwargs
):
    """
    Simulates wind turbine locations onshore and offshore using ECMWF's 
    ERA5 database [1], with an optional correction loop to ensure that
    generated capacity factors for historic wind fleets meet reported 
    generation/capacity based on Renewables Market Report [2] by the 
    International Energy Agency (IEA).

    Please cite the following publication when using the workflow [3]: 
    Peña-Sánchez, Dunkel, Winkler et al. (2025): Towards high resolution, 
    validated and open global wind power assessments. 
    https://doi.org/10.48550/arXiv.2501.07937
    
    Parameters
    ----------
    placements : pandas Dataframe
        A Dataframe object with the parameters needed by the simulation.
    era5_path : str
        Path to the ERA5 data.
    gwa_100m_path : str
        Path to the Global Wind Atlas at 100m [4] raster file.
    esa_cci_path : str
        Path to the ESA CCI raster file [5].
    output_netcdf_path : str, optional
        Path to a directory to put the output files, by default None
    output_variables : str, optional
        Restrict the output variables to these variables, by default None
    max_batch_size: int
        The maximum number of locations to be simulated simultaneously, 
        else multiple batches will be simulated iteratively. Helps 
        limiting RAM requirements but may affect runtime. Should be 
        adapted to individual computation system (roughly 7GB RAM per 
        10k locations), by default 25 000.
    cf_correction : bool, optional
        If False, the capacity factors will be calculated based on a 
        calibrated physical model only, else an additional correction 
        step will be added to ensure that historic capacity factors based
        on [2] are met if historic wind fleets are simulated. By default 
        True.
    simulate_kwargs : optional
        Will be passed on to simulate().

    Returns
    -------
    xarray.Dataset
        A xarray dataset including all the output variables you defined as your output variables.

    Sources
    ------
    [1] European Centre for Medium-Range Weather Forecasts. (2019). ERA5 dataset. https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5
    [2] International Energy Agency. (2023). Renewables Market Report. https://www.iea.org/reports/renewables-2023 
    [3] Peña-Sánchez, Dunkel, Winkler et al. (2025): Towards high resolution, validated and open global wind power assessments. https://doi.org/10.48550/arXiv.2501.07937
    [4] DTU Wind Energy. (2019). Global Wind Atlas. https://globalwindatlas.info/
    [5] ESA. Land Cover CCI Product User Guide Version 2. Tech. Rep. (2017). Available at: maps.elie.ucl.ac.be/CCI/viewer/download/ESACCI-LC-Ph2-PUGv2_2.0.pdf
    """
    # default data used as per [3]
    ws_correction_function=(
        "ws_bins",
        os.path.join(DATAFOLDER, f"ws_correction_factors_PSDW2025.yaml")
    )
    cf_correction_factor=os.path.join(DATAFOLDER, f"cf_correction_factors_PSDW2025.tif")
    wake_curve="dena_mean"
    availability_factor=0.98
    nodata_fallback=np.nan
    era5_lra_path = rk_weather.Era5Source.LONG_RUN_AVERAGE_WINDSPEED_2008TO2017

    # initialize wf manager instance
    wf = WindWorkflowManager(placements)

    # read data
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

    # adjust hourly wind speeds based on ERA-5 LRA and GWA
    wf.adjust_variable_to_long_run_average(
        variable="elevated_wind_speed",
        source_long_run_average=era5_lra_path,
        real_long_run_average=gwa_100m_path,
        nodata_fallback=nodata_fallback,
        spatial_interpolation="average",
    )

    # estimate roughness and use for velocity correction based on hub height
    wf.estimate_roughness_from_land_cover(path=esa_cci_path, source_type="cci")
    wf.logarithmic_projection_of_wind_speeds_to_hub_height(
        consider_boundary_layer_height=True
    )

    # generate the actual ws corr func and correct wind speeds
    ws_correction_func = build_ws_correction_function(
        type=ws_correction_func[0],
        data_dict=ws_correction_func[1],
    )
    wf.sim_data["elevated_wind_speed"] = ws_correction_func(
        wf.sim_data["elevated_wind_speed"]
    )
    
    # apply air density correction
    wf.apply_air_density_correction_to_wind_speeds()
    # do wake reduction 
    wf.apply_wake_correction_of_wind_speeds(wake_curve=wake_curve)
    # gaussian convolution of the power curve to account for statistical events in wind speed
    wf.convolute_power_curves(
        scaling=0.01,  # standard deviation of gaussian equals scaling*v + base
        base=0.00,  # values are derived from validation with real wind turbine data
    )

    # do simulation
    if not cf_correction:
        # set cf correction factor to 1.0, i.e. do not correct
        cf_correction_factor = 1.0
    wf.simulate(
        cf_correction_factor=cf_correction_factor, 
        max_batch_size=max_batch_size,
        **simulate_kwargs
    )

    # apply availability factor
    wf.apply_availability_factor(availability_factor=availability_factor)

    return wf.to_xarray(
        output_netcdf_path=output_netcdf_path, 
        output_variables=output_variables
    )


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
    wake_curve,
    availability_factor,
    consider_boundary_layer_height,
    power_curve_scaling,
    power_curve_base,
    convolute_power_curves_args={},
    loss_factor_args={},
    output_variables=None,
    max_batch_size=25000,
    output_netcdf_path=None,
    elevated_wind_speed=None,
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
        If no GWA available, use for simulation: 
        (1) float value for a multiple of the 'weather_lra_ws_path' value 
            (ERA5 raw), i.e. 1.0 means weather_lra_ws_path value
        (2) np.nan for nan output
        (3) a callable function to be applied to the weather_lra_ws_path 
            (ERA-5) value in the format: 
            nodata_fallback(locs, weather_lra_ws_path_value)
        (4) a filepath to a raster file containing the fallback values
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
    wake_curve : str, optional
        string value to describe the wake reduction method. None will
        cause no reduction, by default "dena_mean". Choose from (see more
        information here under wind_efficiency_curve_name[1]): "dena_mean",
        "knorr_mean", "dena_extreme1", "dena_extreme2", "knorr_extreme1",
        "knorr_extreme2", "knorr_extreme3". Alternatively, the
        'wake_curve' str can also be provided per each location in a
        'wake_curve' column of the placements dataframe, 'wake_curve'
        argument must then be None.
    availability_factor : float, otional
        This factor accounts for all downtimes and applies an average reduction to the output curve,
        assuming a statistical deviation of the downtime occurences and a large enough turbine fleet.
        Suggested availability is 0.98 including technical availability of turbine and connector
        as well as outages for ecological reasons (e.g. bat protection). This does not include wake effects
        (see above) or curtailment/outage for economical reasons or transmission grid congestion.
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
    loss_factor_args : dict, optional
        Arguments that are passed to reskit.utils.low_generation_loss()
        besides the capacity factor. If empty dict ({}), no loss will be
        applied. For details see: reskit.utils.loss_factors.low_generation_loss()
        By default {}.
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
        assert isinstance(ws_correction_func[1], (dict, str, list, tuple))

        # generate the actual ws corr func
        ws_correction_func = build_ws_correction_function(
            type=ws_correction_func[0],
            data_dict=ws_correction_func[1],
        )
    assert callable(
        ws_correction_func
    ), f"ws_correction_func must be an executable with a single argument that can be passed as np.array (if not 1)."

    wf = WindWorkflowManager(placements)

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
    wf.apply_wake_correction_of_wind_speeds(wake_curve=wake_curve)

    if elevated_wind_speed is not None:
        print("Using provided elevated_wind_speed")
        wf.sim_data["elevated_wind_speed"] = elevated_wind_speed

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

    if loss_factor_args != {}:
        wf.apply_loss_factor(
            loss=lambda x: rk_util.low_generation_loss(x, **loss_factor_args)
        )

    # apply availability factor
    wf.apply_availability_factor(availability_factor=availability_factor)

    return wf.to_xarray(
        output_netcdf_path=output_netcdf_path, output_variables=output_variables
    )



########################
# DEPRECATED WORKFLOWS #
########################

# The following workflows are deprecated and can only be used by checking
# out the respective commit status of RESkit

def wind_era5_2023(**kwargs):
    """
    Simulates onshore and offshore (200km from shoreline) wind generation using ECMWF's ERA5 database [1].

    Sources
    ------
    [1] European Centre for Medium-Range Weather Forecasts. (2019). ERA5 dataset. https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5
    """
    # this is the commit hash with the latest workflow status
    commit_hash = "379645675cb1b2559ffa8d73c84be0dd0daef55e"
    raise rk_util.RESKitDeprecationError(commit_hash)

def onshore_wind_era5(**kwargs):
    """
    Simulates onshore wind generation using ECMWF's ERA5 database [1].

    Sources
    ------
    [1] European Centre for Medium-Range Weather Forecasts. (2019). ERA5 dataset. https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5
    """
    # this is the commit hash with the latest workflow status
    commit_hash = "379645675cb1b2559ffa8d73c84be0dd0daef55e"
    raise rk_util.RESKitDeprecationError(commit_hash)

def offshore_wind_era5(**kwargs):
    """
    Simulates offshore wind generation using NASA's ERA5 database [1].

    Sources
    ------
    [1] European Centre for Medium-Range Weather Forecasts. (2019). ERA5 dataset. https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5.

    """
    # this is the commit hash with the latest workflow status
    commit_hash = "379645675cb1b2559ffa8d73c84be0dd0daef55e"
    raise rk_util.RESKitDeprecationError(commit_hash)

def onshore_wind_era5_pure_2023(**kwargs):
    """
    Simulates onshore wind generation using pure ECMWF's ERA5 database [1]
    without further disaggregation or correction besides height projection
    and power curve convolution.

    Sources
    ------
    [1] European Centre for Medium-Range Weather Forecasts. (2019). ERA5 dataset. https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5
    """
    # this is the commit hash with the latest workflow status
    commit_hash = "379645675cb1b2559ffa8d73c84be0dd0daef55e"
    raise rk_util.RESKitDeprecationError(commit_hash)