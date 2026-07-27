# import primary packages
import warnings

from collections.abc import Callable, Iterable
import numpy as np
import pandas as pd

from reskit import util as rk_util

# import othert modules
from reskit import weather as rk_weather
from reskit.solar.workflows.solar_workflow_manager import SolarWorkflowManager


def openfield_pv_merra_ryberg2019(
    placements,
    merra_path,
    global_solar_atlas_ghi_path,
    module="WINAICO WSx-240P6",
    elev=300,
    tracking="fixed",
    inverter=None,
    inverter_kwargs={},
    tracking_args={},
    output_netcdf_path=None,
    output_variables=None,
    tech_year=2050,
):
    """

    openfield_pv_merra_ryberg2019(placements, merra_path, global_solar_atlas_ghi_path, module="WINAICO WSx-240P6", elev=300, tracking="fixed",
                                    inverter=None, inverter_kwargs={}, tracking_args={}, output_netcdf_path=None, output_variables=None)

    Simulation of an openfield  PV openfield system based on MERRA Data.

    Parameters
    ----------
    placements: Pandas Dataframe
        Locations where to perform simulations at.
        Columns need to be lat (latitudes), lon (longitudes), tilt and capacity.

    merra_path: str
        Path to the MERRA Data on your computer.
        Can be a single ".nc" file, or a directory containing many ".nc" files.

    global_solar_atlas_ghi_path: str
        Path to the global solar atlas ghi data on your computer.

    module: str
        Name of the module that you want to use for the simulation.
        Default is Winaico Wsx-240P6.
        See reskit.solar.SolarWorkflowManager.configure_cec_module for more usage information.

    elev: float
        Elevation that you want to model your PV system at.

    tracking: str
                Option 1 is 'fixed' meaning that the module does not have any tracking capabilities.
                Option 2 is 'singleaxis' meaning that the module has singleaxis tracking capabilities.


    inverter: str
        Determines whether or not you want to model your PV system with an inverter.
        Default is None, meaning no inverter is assumed
        See reskit.solar.SolarWorkflowManager.apply_inverter_losses for more usage information

    output_netcdf_path: str
        Path to a file that you want to save your output NETCDF file at.
        Default is None

    output_variables: str
        Output variables of the simulation that you want to save into your NETCDF Outputfile.

    tech_year : int, optional
                If given in combination with the projected module str names "WINAICO WSx-240P6" or
                "LG Electronics LG370Q1C-A5", the effifiency will be scaled linearly to the given
                year. Must then be between year of market introduction for that module and 2050.
                Will be ignored when non-projected existing module names or specific parameters
                are given, can then be None. By default 2050.

    Returns
    -------
    A xarray dataset including all the output variables you defined as your output_variables.

    """
    wf = SolarWorkflowManager(placements)
    wf.configure_cec_module(module=module, tech_year=tech_year, tracking=tracking, database="CECMod")
    # ensure the tracking parameter is correct
    assert tracking == "fixed", "tracking must be 'fixed' for this workflow"

    # estimates tilt, azimuth and elev
    wf.estimate_missing_params(
        elev, 
        ground_albedo=0.25, 
        gcr=2.0 / 7.0,
        fixed_module_tilt_convention="Ryberg2020",
        fixed_module_azimuth_convention="NorthSouth",        
        )

    wf.read(
        variables=[
            "surface_wind_speed",
            "surface_pressure",
            "surface_air_temperature",
            "surface_dew_temperature",
            "global_horizontal_irradiance",
        ],
        source_type="MERRA",
        source=merra_path,
        set_time_index=True,
        verbose=False,
    )

    wf.adjust_variable_to_long_run_average(
        variable="global_horizontal_irradiance",
        source_long_run_average=rk_weather.MerraSource.LONG_RUN_AVERAGE_GHI,
        real_long_run_average=global_solar_atlas_ghi_path,
        real_lra_scaling=1000 / 24,  # cast to hourly average kWh
    )

    wf.determine_solar_position()
    wf.filter_positive_solar_elevation()
    wf.determine_extra_terrestrial_irradiance(model="spencer", solar_constant=1370)
    wf.determine_air_mass(model="kastenyoung1989")
    wf.apply_DIRINT_model()
    wf.diffuse_horizontal_irradiance_from_trigonometry()

    if tracking == "singleaxis":
        wf.permit_single_axis_tracking(**tracking_args)

    wf.determine_angle_of_incidence()
    wf.estimate_plane_of_array_irradiances(transposition_model="perez")

    wf.apply_angle_of_incidence_losses_to_poa()

    wf.cell_temperature_from_sapm()

    wf.simulate_with_interpolated_single_diode_approximation()

    if inverter is not None:
        wf.apply_inverter_losses(inverter=inverter, **inverter_kwargs)

    variables = [_var for _var in ["capacity_factor", "total_system_generation"] if _var in wf.sim_data.keys()]
    wf.apply_loss_factor(0.20, variables=variables)

    return wf.to_xarray(output_netcdf_path=output_netcdf_path, output_variables=output_variables)


def openfield_pv_era5(
    placements,
    era5_path,
    global_solar_atlas_ghi_path,
    global_solar_atlas_dni_path,
    module="WINAICO WSx-240P6",
    elev=300,
    tracking="fixed",
    inverter=None,
    inverter_kwargs={},
    tracking_args={},
    DNI_nodata_fallback=1.0,
    DNI_nodata_fallback_scaling=1.0,
    GHI_nodata_fallback=1.0,
    GHI_nodata_fallback_scaling=1.0,
    output_netcdf_path=None,
    output_variables=None,
    gsa_nodata_fallback="source",
    tech_year=2050,
    time_slice=None,
):
    """
    Simulation of an openfield  PV openfield system based on ERA5 Data.

    Parameters
    ----------
    placements: Pandas Dataframe
            Locations that you want to do the simulations for.
            Columns need to be lat (latitudes), lon (longitudes), capacity.
            Tilt and azimuths can be provided as columns or will be extracted 
            based on conventions, see 'tracking' description for details.

    era5_path: str
            Path to the ERA5 Data on your computer.
            Can be a single ".nc" file, or a directory containing many ".nc" files.

    global_solar_atlas_ghi_path: str
            Path to the global solar atlas ghi data on your computer.

    global_solar_atlas_dni_path: str
            Path to the global solar atlas dni data on your computer.

    module: str
            Name of the module that you wanna use for the simulation.
            Default is Winaico Wsx-240P6

    elev: float
            Elevation that you want to model your PV system at. Will be taken 
            from 'elev' column if available.

    tracking: str
            Determines wether your PV system is fixed or not. Default is fixed.
            NOTE: Has been limited to 'fixed' for this workflow due to 
            inconsistencies when using 'singleaxis', argument is not removed 
            only for reasons of backward compatibility. For single-axis tracking
            calculations please use pv_era5_WinklerUnpublished() instead.
            
            NOTE: The tilt and azimuth definitions change with different tracking systems.
            For fixed tilt systems the following column names apply: 
            * module_tilt_col="modtilt"
            * module_azimuth_col="modazimuth"
            The column names for the tracker axis tilts and azimuth are instead:
            * axis_azimuth_col="axazimuth"
            * axis_tilt_col="axtilt"
            * crossaxis_tilt_col="caxtilt"
            Note that the use of 'tilt' and 'azimuth' columns is discouraged, 
            they will be interpreted as and renamed to the respective module or 
            axis column names depending depending on tracking.

    inverter: str
            Determines whether you want to model your PV system with an inverter or not.
            Default is None.
            See reskit.solar.SolarWorkflowManager.apply_inverter_losses for more usage information.

    DNI_nodata_fallback: str | float | Callable, optional
            When global_solar_atlas_dni_path has no data, one can decide between different fallback options, by default 1.0:
            - np.nan or None : return np.nan for missing values in global_solar_atlas_dni_path
            - float : Apply this float value as a scaling factor for all no-data locations only: source_long_run_average * DNI_nodata_fallback.
            NOTE: A value of 1.0 will return the source lra value in case of missing global_solar_atlas_dni_path values.
            - str : Will be interpreted as a filepath to a raster with alternative absolute global_solar_atlas_dni_path values
            - Callable : any callable method taking the arguments (all iterables): 'locs' and 'source_long_run_average_value'
            (the locations as gk.geom.point objects and original value from source data). The output values will be considered as
            the new real_long_run_average for missing locations only.
            NOTE: np.nan will also be returned in case that the nodata fallback does not yield values either.

    DNI_nodata_fallback_scaling: float, optional
            The scaling factor that will be applied to the DNI nodata fallback e.g. in case of different units compared to source data.
            By default 1.0, i.e. no effect.

    GHI_nodata_fallback: str, | float | Callable, optional
            When global_solar_atlas_ghi_path has no data, one can decide between different fallback options, by default 1.0:
            - np.nan or None : return np.nan for missing values in global_solar_atlas_ghi_path
            - float : Apply this float value as a scaling factor for all no-data locations only: source_long_run_average * GHI_nodata_fallback.
                NOTE: A value of 1.0 will return the source lra value in case of missing global_solar_atlas_ghi_path values.
            - str : Will be interpreted as a filepath to a raster with alternative absolute global_solar_atlas_ghi_path values
            - Callable : any callable method taking the arguments (all iterables): 'locs' and 'source_long_run_average_value'
                (the locations as gk.geom.point objects and original value from source data). The output values will be considered as
                the new real_long_run_average for missing locations only.
            NOTE: np.nan will also be returned in case that the nodata fallback does not yield values either

    GHI_nodata_fallback_scaling: float, optional
            The scaling factor that will be applied to the GHI nodata fallback e.g. in case of different units compared to source data.
            By default 1.0, i.e. no effect.

    output_netcdf_path: str
            Path to a file that you want to save your output NETCDF file at.
            Default is None

    output_variables: str
            Output variables of the simulation that you want to save into your NETCDF Outputfile.

    gsa_nodata_fallback: str, optional
            NOTE: DEPRECATED! Will be removed soon!
            When real_long_run_average has no data, it can be decided between fallback options:
            -'source': use source data (ERA5 raw simulation)
            -'nan': return np.nan for missing values
            get flags for missing values:
            - f'missing_values_{os.path.basename(path_to_LRA_source)}_nodata_fallback{nodata_fallback}'

    tech_year : int, optional
                If given in combination with the projected module str names "WINAICO WSx-240P6" or
                "LG Electronics LG370Q1C-A5", the effifiency will be scaled linearly to the given
                year. Must then be between year of market introduction for that module and 2050.
                Will be ignored when non-projected existing module names or specific parameters
                are given, can then be None. By default 2050.

    time_slice : slice, optional
            Limit the time span loaded from the ERA5 source. Only supported for
            Zarr-backed ERA5 sources, where it is strongly recommended to avoid
            loading whole multi-year cloud stores. Raises for netCDF4-backed ERA5
            sources; support for those is planned.

    Returns
    -------
    A xarray dataset including all the output variables you defined as your output_variables.
    """
    wf = SolarWorkflowManager(placements)
    wf.configure_cec_module(module=module, tech_year=tech_year, tracking=tracking, database="CECMod", bifaciality_factor=0) 

    # tilt and azimuth were ambiguous depending on tracking, rename to consistent attribute names throughout the wfm
    if not tracking == "fixed":
        raise rk_util.RESKitDeprecationError("tracking has been limited to 'fixed' due to inconsistencies, use pv_era5_WinklerUnpublished() for single-axis tracking.")
    if "tilt" in wf.placements:
        assert 'modtilt' not in wf.placements, f"'tilt' and 'modtilt' columns cannot exist both when tracking == 'fixed'."
        warnings.warn(f"'tilt' column will be interpreted as and renamed to 'modtilt'.")
        wf.placements.rename(columns={"tilt" : "modtilt"})
    if "azimuth" in wf.placements:
        assert "modazimuth" not in wf.placements, f"'azimuth' and 'modazimuth' columns cannot exist both when tracking == 'fixed'."
        warnings.warn(f"'azimuth' column will be interpreted as and renamed to 'modazimuth'.")
        wf.placements.rename(columns={"azimuth" : "modazimuth"})
    # estimates tilt, azimuth and elev
    wf.estimate_missing_params(
        elev=elev, 
        ground_albedo=0.25,
        gcr=2.0 / 7.0,
        fixed_module_tilt_convention="Ryberg2020", 
        fixed_module_azimuth_convention="NorthSouth",
    )

    wf.read(
        variables=[
            "global_horizontal_irradiance",
            "direct_horizontal_irradiance",
            "surface_wind_speed",
            "surface_pressure",
            "surface_air_temperature",
            "surface_dew_temperature",
            "snow_albedo",
            "snow_density",
            "snow_depth_water_equivalent",
            "snowfall_water_equivalent",
        ],
        source_type="ERA5",
        source=era5_path,
        set_time_index=True,
        time_index_from="direct_horizontal_irradiance",
        time_slice=time_slice,
        verbose=False,
    )
    # If there is a need to resimulate old data, this line must be inserted.
    # wf.sim_data['global_horizontal_irradiance'] = wf.sim_data['global_horizontal_irradiance_archive']

    wf.determine_solar_position()
    wf.filter_positive_solar_elevation()

    wf.direct_normal_irradiance_from_trigonometry()

    # wf.spatial_disaggregation(
    #     variable='global_horizontal_irradiance',
    #     source_high_resolution=global_solar_atlas_ghi_path,
    #     source_low_resolution=rk_weather.GSAmeanSource.GHI_with_ERA5_pixel,
    # )

    # TODO remove the following mid 2024, also remove gsa_nodata_fallback in workflow args
    if gsa_nodata_fallback != "source":
        warnings.warn(
            "'gsa_nodata_fallback' is deprecated and will be removed soon. Use 'GHI_nodata_fallback' and 'GHI_nodata_fallback' instead.",
            DeprecationWarning,
        )
        # deprecated gsa nodata fallback has been changed!
        if GHI_nodata_fallback != 1.0 or DNI_nodata_fallback == 1.0:
            # also, changes have been made to GHI and DNI fallbacks
            raise ValueError(
                "When GHI_nodata_fallback and DNI_nodata_fallback have been adapted, gsa_nodata_fallback must not be adapted (recommended to ignore, deprecated)"
            )
        else:
            # GHI and DNI fallbacks have not been changed, but 'source' has - adapt DNI and GHI fallbacks accordingly
            if gsa_nodata_fallback == "nan":
                GHI_nodata_fallback = np.nan
                DNI_nodata_fallback = np.nan
            else:
                raise ValueError(
                    "'gsa_nodata_fallback' (deprecated) must be 'nan' or 'source'. Better use 'GHI_nodata_fallback' and 'GHI_nodata_fallback' instead, however."
                )

    wf.adjust_variable_to_long_run_average(
        variable="global_horizontal_irradiance",
        source_long_run_average=rk_weather.Era5Source.LONG_RUN_AVERAGE_GHI_2020_03,
        real_long_run_average=global_solar_atlas_ghi_path,
        real_lra_scaling=1000 / 24,  # cast to hourly average kWh
        nodata_fallback=GHI_nodata_fallback,
        nodata_fallback_scaling=GHI_nodata_fallback_scaling,
    )

    wf.adjust_variable_to_long_run_average(
        variable="direct_normal_irradiance",
        source_long_run_average=rk_weather.Era5Source.LONG_RUN_AVERAGE_DNI_2020_03,
        real_long_run_average=global_solar_atlas_dni_path,
        real_lra_scaling=1000 / 24,  # cast to hourly average kWh
        nodata_fallback=DNI_nodata_fallback,
        nodata_fallback_scaling=DNI_nodata_fallback_scaling,
    )

    wf.determine_extra_terrestrial_irradiance(model="spencer", solar_constant=1370)
    wf.determine_air_mass(model="kastenyoung1989")

    wf.diffuse_horizontal_irradiance_from_trigonometry()

    if tracking == "singleaxis":
        wf.permit_single_axis_tracking(**tracking_args)

    wf.determine_angle_of_incidence()
    wf.estimate_plane_of_array_irradiances(transposition_model="perez")

    wf.apply_angle_of_incidence_losses_to_poa()

    wf.cell_temperature_from_sapm()

    wf.simulate_with_interpolated_single_diode_approximation()

    if inverter is not None:
        wf.apply_inverter_losses(inverter=inverter, **inverter_kwargs)

    loss_factor = 0.115  # validation by d.franzmann, 2022/01/13
    variables = [_var for _var in ["capacity_factor", "total_system_generation"] if _var in wf.sim_data.keys()]
    wf.apply_loss_factor(loss_factor, variables=variables)

    return wf.to_xarray(output_netcdf_path=output_netcdf_path, output_variables=output_variables)


def pv_era5_WinklerUnpublished(
    placements : pd.DataFrame,
    tracking : str,
    era5_path : str,
    global_solar_atlas_ghi_path : str,
    global_solar_atlas_dni_path : str,
    module_azimuth : int | float | str | Iterable | None = None,
    module_tilt : int | float | str | Iterable | None = None,
    singleaxis_azimuth : int | float | str | Iterable | None = None,
    singleaxis_tilt : int | float | str | Iterable | None = None,
    crossaxis_tilt : int | float | str | Iterable | None = None,
    elevation : int | float | str | Iterable | None = 840,
    north_slope : int | float | str | Iterable = 0,
    east_slope : int | float | str | Iterable = 0,
    gcr  : float | str | Iterable | None = None,
    ground_albedo : float | str | Iterable = 0.25,
    distant_horizon_profile : np.ndarray | str | None = None,
    consider_snow_effects : bool | Iterable = True,
    DNI_nodata_fallback : float | str | Callable = 1.0,
    DNI_nodata_fallback_scaling : float = 1.0,
    GHI_nodata_fallback : float | str | Callable = 1.0,
    GHI_nodata_fallback_scaling : float = 1.0,
    capacity : int | float | Iterable = None,
    module : str = 'Trina Solar Co.Ltd TSM-700NEG21C.20',
    bifaciality_factor : float | Iterable = 0.9,
    max_tracking_angle : int | Iterable = 60,
    backtracking : bool | Iterable = True,
    pvrow_height : int | float | Iterable | None = None,
    module_configuration : str = "2P", 
    inverter : str = None,
    inverter_kwargs : dict = {},
    tech_year : int = 2035, # was 2050
    output_netcdf_path : str =None,
    output_variables : list | None=None,
    new_style : bool =True, #TODO remove
):
    """
    Simulation of an openfield PV system based on ERA-5 Data, geospatially disaggregated based 
    on Global Solar Atlas long-run averages. Allows for the consideration of snow effects, hill
    slopes, module bifaciality, interrow shading, horizon shading.

    NOTE: None, np.nan, pd.NA or "null" or "" values in below parameters will generally be 
    considered as no-data entries and will take no effect.

    Parameters
    ----------
    placements: Pandas Dataframe
            Locations that you want to do the simulations for.
            Columns need to be lat (latitudes), lon (longitudes), capacity.
            Tilt and azimuths can be provided as columns or will be extracted 
            based on conventions, see 'tracking' description for details.
    tracking: str
            Determines wether your PV system is fixed or has tracking capability.
            * 'fixed' means fixed module tilt, no tracking ability.
            * 'singleaxis' stands for single-axis tracking capacbility, row axis
              can be horizontal or tilted (see singleaxis_tilt)
            NOTE: Depending on the 'tracking' setting, different input arguments 
            become mandatory, the respective others should then be None:
            * 'Fixed' tilt arguments: module_tilt, module_azimuth
            * 'Singleaxis' tracking args: singleaxis_azimuth, axis_tilt, crossaxis_tilt
    era5_path: str
            Path to the ERA5 Data on your computer. Can be a single ".nc" file, or a 
            directory containing many ".nc" files.
    global_solar_atlas_ghi_path: str
            Path to the global solar atlas ghi raster on your computer.
    global_solar_atlas_dni_path: str
            Path to the global solar atlas dni raster on your computer.
    module_azimuth : int | float | Iterable | str | None, optional
            The module azimuths in degrees clockwise from North = 0°. Module 
            azimuths can also be assigned optimally when no angles are provided,
            then a single or one string per placement is needed determining the 
            desired module tilt convention (e.g. 'NorthSouth' etc., 
            see reskit.solar.core.system_design.location_to_module_azimuth()).
            Can be provided as a scalar or an iterable per location.
            Note : None is expected only if tracking != 'fixed'. By default None.
    module_tilt : int | float | Iterable | str | None, optional
            The module tilt towards the module azimuth in degrees from flat ground.
            Module tilts can also be assigned optimally when no angles are provided,
            then a single or one string per placement is needed determining the 
            desired module tilt convention (e.g. 'Ryberg2019', 'Winkler2027' etc., 
            see reskit.solar.core.system_design.location_to_module_tilt()).
            Can be provided as a scalar or an iterable per location.
            Note : None is expected only if tracking != 'fixed'. By default None.
    singleaxis_azimuth : int | float | Iterable | str | None, optional
            The main tracking axis azimuth of a single-axis tracking system clockwise 
            from North = 0°. If a str is given, a known convention is expected
            (see reskit.solar.core.system_design.location_to_tracker_axis_azimuth()).
            Can be provided as a scalar or an iterable per location.
            None is expected only if tracking != 'singleaxis'. By default None.
    singleaxis_tilt : int | float | Iterable | str | None, optional
            The main tracking axis tilt (angle to horizontal) of a single-axis 
            tracking system descending towards above axis azimuth. Will be 
            calculated from axis azimuth, hill slopes and orientation assuming 
            constant ground distance if not given. A known convention is expected
            (see reskit.solar.core.system_design.location_to_tracker_axis_tilt()) 
            if a str is given. Can be provided as a scalar or an iterable per 
            location. None is expected if tracking != 'singleaxis'. By default None.
    crossaxis_tilt : int | float | Iterable | str | None, optional
            The cross-axis tilt perpendicular to the main axis vector (angle to 
            horizontal) of a single-axis tracking system. Will be 
            calculated from axis azimuth, hill slopes and orientation assuming 
            constant ground distance if not given. A known convention is expected
            (see reskit.solar.core.system_design.location_to_cross_axis_tilt()) 
            if a str is given. Can be provided as a scalar or an iterable per 
            location. None is expected if tracking != 'singleaxis'. By default None.
    elevation: int | float | str | Iterable, optional
            Elevation of the PV system over sea level in [m]. If a str is given, a
            filepath to a DEM raster is expected. Can be provided as a scalar or an 
            iterable per location. Can be provided as a scalar or an iterable per 
            location. Defaults to 840 (average global landmass elevation).
    north_slope : int | float | str | Iterable, optional
            The slope facing/descending towards North in degrees over horizontal.
            Can be provided as a scalar or an iterable per location. If a str is 
            given, a filepath to a slope raster is expected. Will affect both 
            local horizon shading and row/cross axis tilts. By default 0, 
            i.e. flat terrain in North-South orientation.
    east_slope : int | float | str | Iterable, optional
            The slope facing/descending towards East in degrees over horizontal.
            Can be provided as a scalar or an iterable per location. If a str is 
            given, a filepath to a slope raster is expected.  Will affect both 
            local horizon shading and row/cross axis tilts. By default 0, 
            i.e. flat terrain in North-South orientation.
    gcr  : float | str | Iterable | None, optional
            The ground coverage ratio, understood as a vertical projection (bird 
            view). Can be provided as a scalar or an iterable per location.
            If None is provided, the gcr convention will be assigned based on the
            tracking style (see reskit.solar.core.system_design.location_to_gcr):
            * singleaxis: "tonita_et_al_2023_5perc" convention
            * fixed: "winter_solstice_rule" convention
            By default None.
    ground_albedo : float | str | Iterable | tuple, optional
            The average base ground albedo without temporal snow effects. 
            Can be provided as a scalar or an iterable per location:
            * tuple: format (dataset name, dataset filepath) to point to a 
              landcover dataset. Albedo values will then be mapped to landcover 
              classes based on [2] for every single location.
            * float : The same albedo value to be set for all placements. 
            * Iterable : Iterable of float values per location.
            If snow effects are considered, ground albedo will be increased in 
            hours with ground covered by snow. Default value is 0.25 (based on pvlib)
    distant_horizon_profile : numpy.ndarray | str | None, optional
            The horizon profile in degrees from level horizon, positive for 
            mountains. If provided as np.ndarray, one row is expected per placement, 
            the columns are then the horizon angles clockwise starting from North. 
            The 360° full circle will be divided by the number of columns, i.e. 36 
            columns mean one sampling point every 10° azimuth rotation. If a single 
            string or an iterable of strings with one per placement is provided, 
            existing filepaths to a digital elevation model (DEM) raster file are 
            expected from which the horizon profile will be calculated. None means 
            no consideration of the horizon shading. The distant horizon will be 
            combined with a local horizon from hill slope if given. By default None.
    consider_snow_effects : bool | Iterable, optional
            Boolean as a a scalar or per location if snow effects shall be considered,
            then affects both ground albedo in times of snow-covered ground as well
            as shadowing of by snow covered modules. Can be provided as a scalar or 
            an iterable per location, by default True.
    DNI_nodata_fallback: float | str | Callable, optional
            When global_solar_atlas_dni_path has no data, one can decide between different 
            fallback options, by default 1.0:
            * np.nan or None : return np.nan for missing values in global_solar_atlas_dni_path
            * float : Apply this float value as a scaling factor for all no-data locations only: 
                source_long_run_average * DNI_nodata_fallback.
                NOTE: A value of 1.0 will return the source lra value in case of 
                missing global_solar_atlas_dni_path values.
            * str : Will be interpreted as a filepath to a raster with alternative absolute 
                global_solar_atlas_dni_path values
            * Callable : any callable method taking the arguments (all iterables): 'locs' and 
                'source_long_run_average_value' (the locations as gk.geom.point objects and 
                original value from source data). The output values will be considered as
                the new real_long_run_average for missing locations only.
            NOTE: np.nan will still be returned in case that the nodata fallback does not yield values either.
    DNI_nodata_fallback_scaling: float, optional
            The scaling factor that will be applied to the DNI nodata fallback e.g. in case of 
            different units compared to source data. By default 1.0, i.e. no effect.
    GHI_nodata_fallback: str | str | Callable, optional
            When global_solar_atlas_ghi_path has no data, one can decide between different 
            fallback options, by default 1.0:
            - np.nan or None : return np.nan for missing values in global_solar_atlas_ghi_path
            - float : Apply this float value as a scaling factor for all no-data locations only: 
                source_long_run_average * GHI_nodata_fallback.
                NOTE: A value of 1.0 will return the source lra value in case of missing 
                global_solar_atlas_ghi_path values.
            - str : Will be interpreted as a filepath to a raster with alternative absolute 
                global_solar_atlas_ghi_path values
            - callable : any callable method taking the arguments (all iterables): 'locs' and 
                'source_long_run_average_value' (the locations as gk.geom.point objects and 
                original value from source data). The output values will be considered as
                the new real_long_run_average for missing locations only.
            NOTE: np.nan will also be returned in case that the nodata fallback does not yield values either
    GHI_nodata_fallback_scaling: float, optional
            The scaling factor that will be applied to the GHI nodata fallback e.g. in case of 
            different units compared to source data. By default 1.0, i.e. no effect.
    capacity : int | float | Iterable, optional
            The capacity of the PV plant in kW, will then also return energy production (else 
            only capacity factors). Can be provided as a scalar or an iterable per location.
            By default None.
    module : str, optional
            The module whose technical parameters shall be assumed for the simulation. Must be
            in the CEC database, unless a custom database has been added. By default
            'Trina Solar Co.Ltd TSM-700NEG21C.20'
    bifaciality_factor : float | Iterable, optional
            The bifaciality factor that shall be assumed, will overwrite potential module
            information. Can be provided as a scalar or an iterable per location. Bifaciality
            factor of 0.0 means monofacial module. By default 0.9.
    max_tracking_angle : int | Iterable, optional
            The maximum allowed tracking angle in degrees around the single-axis tracking axis. 
            Can be provided as a scalar or an iterable per location. Will take effect only when 
            tracking = 'singleaxis'. By default 60°.
    backtracking : bool | Iterable, optional
            If backtracking is allowed to minimize self-shadowing of a single-axis tracking 
            system. Can be provided as a scalar or an iterable per location. Will take effect 
            only when tracking = 'singleaxis'. By default True.
    pvrow_height : int | float | None | Iterable, optional
            The row center axis height measured perpendicular to the ground, i.e. not 
            necessarily vertical for sloped hills. Can be provided as a scalar or an iterable 
            per location. If None is given, height will be calculated such that the lower 
            module edges can just not touch the ground in a maximally rotated (vertical) position.
            By default None.
    module_configuration : str, optional
            How many modules are stacked along the sloped axis of an array width, and if they are
            mounted in 'portrait' or 'landscape' orientation, e.g. "2P" or "3L". By default "2P". 
    inverter : str 
            The name of the inverted if used, else no inverted will be assumed. By default None.
            See reskit.solar.SolarWorkflowManager.apply_inverter_losses for more usage information.
    inverter_kwargs : dict, optional
            A dictionary with inverter arguments for solar_workflow_manager.apply_inverter_losses()
            if an inverter is given, by default empty {}.
    tech_year : int, optional
            The technological year to which the selected module shall be projected if such feature 
            is implemented for the selected module, by default 2035.
    output_netcdf_path : str, optional
            The path where the results shall be saved as netcdf file, by default None.
    output_variables : list | None
            The list of output variables which shall be added to the output dataset, by default 
            None, i.e. ALL eligible parameters will be returned.
    new_style : bool, optional #TODO remove
        Defaults to True 

    Returns
    -------
    A xarray dataset including all the output variables you defined as your output_variables.
    """
    # SET UP WORKFLOW MANAGER

    # initialize workflow manager and module/system
    wf = SolarWorkflowManager(placements)

    # LOAD WEATHER DATA

    # read weather variables
    vars = [
        "global_horizontal_irradiance",
        "direct_horizontal_irradiance",
        "surface_wind_speed",
        "surface_pressure",
        "surface_air_temperature",
        "surface_dew_temperature",
    ]
    if np.asarray(consider_snow_effects).any():
        # add snow variables to the vars to be loaded from ERA-5
        vars += [        
        "snowfall_water_equivalent",
        "snow_albedo",
        "snow_depth_water_equivalent",
        "snow_density",
    ]
    wf.read(
        variables=vars,
        source_type="ERA5",
        source=era5_path,
        set_time_index=True,
        time_index_from="direct_horizontal_irradiance",
        verbose=False,
    )

    # PREPROCESS PLANT INPUT DATA

    # configure the module #TODO this should be solved via plant specific parameters in the future
    wf.configure_cec_module(
        module=module, 
        tech_year=tech_year, 
        tracking=tracking, 
        database="CEC Modules.csv",
        module_configuration = module_configuration,
        )

    # preprocess the individual plant parameters #TODO use the plant-specific variables stored herein!
    wf.preprocess_bifaciality_factor(bifaciality_factor = bifaciality_factor)
    wf.preprocess_hill_slope_and_azimuth(
        north_slope = north_slope,
        east_slope = east_slope,
        )
    wf.preprocess_elevation(elevation = elevation)
    wf.preprocess_horizon_profile(
        distant_horizon_profile = distant_horizon_profile,
        azimuthal_stepsize = 3.0,
        min_sampling_points = 12,
        )
    wf.preprocess_ground_albedo(
        ground_albedo = ground_albedo, 
        consider_snow_albedo = consider_snow_effects,
        fallback = 0.25, # based on pvlib defaults and existing RK solar workflows
        )
    if tracking == "fixed":
        wf.preprocess_fixed_module_azimuth(module_azimuth = module_azimuth)
        wf.preprocess_fixed_module_tilt(module_tilt = module_tilt)
        # some arguments should explicitly be None to avoid the user expecting effects from these
        assert wf._is_none(singleaxis_azimuth).all(), "singleaxis_azimuth is expected to be None when tracking == 'fixed'"
        assert wf._is_none(singleaxis_tilt).all(), "singleaxis_tilt is expected to be None when tracking == 'fixed'"
        assert wf._is_none(crossaxis_tilt).all(), "crossaxis_tilt is expected to be None when tracking == 'fixed'"
    elif tracking == "singleaxis":
        wf.preprocess_singleaxis_and_crossaxis(
            singleaxis_azimuth = singleaxis_azimuth,
            singleaxis_tilt = singleaxis_tilt,
            crossaxis_tilt = crossaxis_tilt,
        )
        wf.preprocess_tracking_angle(max_tracking_angle=max_tracking_angle)
        wf.preprocess_backtracking(backtracking=backtracking)
        # some arguments should explicitly be None to avoid the user expecting effects from these
        assert wf._is_none(module_tilt).all(), "module_tilt is expected to be None when tracking == 'singlexis'"
        assert wf._is_none(module_azimuth).all(), "module_azimuth is expected to be None when tracking == 'singlexis'"
    else:
        raise ValueError(f"tracking may only be 'fixed' or 'singleaxis' but is '{tracking}' here.")
    if gcr is None:
        # if not provided explitly, define gcr as tracking-specific defaults for this workflow
        gcr = "winter_solstice_rule" if tracking == "fixed" else "tonita_et_al_2023_5perc" #TODO was 0.358
    wf.preprocess_pvrow_height(pvrow_height=pvrow_height)
    wf.preprocess_ground_coverage_ratio(gcr = gcr, min_gcr = 0.3)
    wf.preprocess_capacity(capacity = capacity)

    # PREPROCESS IRRADIATION

    # apply geometric operations to solar radiation angles
    wf.determine_solar_position()
    wf.filter_positive_solar_elevation()
    wf.direct_normal_irradiance_from_trigonometry()

    # disaggregate ERA-5 hourly variables based on high-res long-run average values
    wf.adjust_variable_to_long_run_average(
        variable="global_horizontal_irradiance",
        source_long_run_average=rk_weather.Era5Source.LONG_RUN_AVERAGE_GHI,
        real_long_run_average=global_solar_atlas_ghi_path,
        real_lra_scaling=1000 / 24,  # cast to hourly average kWh
        nodata_fallback=GHI_nodata_fallback,
        nodata_fallback_scaling=GHI_nodata_fallback_scaling,
    )
    wf.adjust_variable_to_long_run_average(
        variable="direct_normal_irradiance",
        source_long_run_average=rk_weather.Era5Source.LONG_RUN_AVERAGE_DNI,
        real_long_run_average=global_solar_atlas_dni_path,
        real_lra_scaling=1000 / 24,  # cast to hourly average kWh
        nodata_fallback=DNI_nodata_fallback,
        nodata_fallback_scaling=DNI_nodata_fallback_scaling,
    )
    # GSA already contains terrain shading, correct the shading-affected real LRA upwards to not double-count horizon shading
    # will take no effect if no/flat horizon is considered only
    wf.scale_to_unshaded_real_lra(
        max_scaling_factor=1/0.9 # GSA terrain losses limited to 10% acc. to GSA manual : https://documents1.worldbank.org/curated/en/529431592893043403/pdf/Global-Solar-Atlas-2-0-Technical-Report.pdf
    )

    # CALCULATE ABSORBED PLANE OF ARRAY IRRADIANCES

    # calculate diffuse horizontal irradiance from scaled GHI and DNI
    wf.diffuse_horizontal_irradiance_from_trigonometry()

    # determine angle of incidence and resulting insolation
    if wf.tracking == "singleaxis":
        wf.permit_single_axis_tracking()
    if new_style:
        wf.estimate_absorbed_plane_of_array_irradiances()
    else:
        wf.determine_angle_of_incidence()
        wf.estimate_plane_of_array_irradiances(transposition_model="perez")
        wf.apply_angle_of_incidence_losses_to_poa()

    # SIMULATE MODULE RESPONSE AND ELECTRICAL YIELD
    
    if np.any(np.asarray(consider_snow_effects)):
        wf.estimate_snow_coverage_loss(consider_snow_loss=consider_snow_effects)
    wf.cell_temperature_from_sapm()
    wf.simulate_with_interpolated_single_diode_approximation(consider_snow_cover=consider_snow_effects)

    # apply losses from inverter and general loss factor from calibration
    if inverter is not None:
        wf.apply_inverter_losses(inverter=inverter, **inverter_kwargs)
    loss_factor = 0.115  # assumed from openfield_pv_era5() by confirming practically the same output under same conditions (Winkler, 10/2025)
    variables = [
        _var
        for _var in ["capacity_factor", "total_system_generation"]
        if _var in wf.sim_data.keys()
    ]
    wf.apply_loss_factor(loss_factor, variables=variables)

    # SAVE AND RETURN

    ds = wf.to_xarray(
        output_netcdf_path=output_netcdf_path, output_variables=output_variables
    )

    return ds



def openfield_pv_sarah_unvalidated(
    placements,
    sarah_path,
    era5_path,
    module="WINAICO WSx-240P6",
    elev=300,
    tracking="fixed",
    inverter=None,
    inverter_kwargs={},
    tracking_args={},
    output_netcdf_path=None,
    output_variables=None,
    tech_year=2050,
):
    """

    openfield_pv_sarah_unvalidated(placements, sarah_path, era5_path, module="WINAICO WSx-240P6", elev=300, tracking="fixed", inverter=None, inverter_kwargs={}, tracking_args={}, output_netcdf_path=None, output_variables=None)

    Simulation of an openfield  PV openfield system based on Sarah and ERA5 Data.

    Parameters
    ----------
    placements: Pandas Dataframe
                    Locations that you want to do the simulations for.
                    Columns need to be lat (latitudes), lon (longitudes), tilt and capacity.

    sarah_path: str
                Path to the SARAH Data on your computer.
                Can be a single ".nc" file, or a directory containing many ".nc" files.

    era5_path: str
                Path to the ERA5 Data on your computer.
                Can be a single ".nc" file, or a directory containing many ".nc" files.


    module: str
            Name of the module that you wanna use for the simulation.
            Default is Winaico Wsx-240P6

    elev: float
            Elevation that you want to model your PV system at.

    tracking: str
                Determines whether your PV system is fixed or not.
                Default is fixed.
                Option 1 is 'fixed' meaning that the module does not have any tracking capabilities.
                Option 2 is 'singleaxis' meaning that the module has single-axis tracking capabilities.

    inverter: str
                Determines whether you want to model your PV system with an inverter or not.
                Default is None.
                See reskit.solar.SolarWorkflowManager.apply_inverter_losses for more usage information.

    output_netcdf_path: str
                        Path to a file that you want to save your output NETCDF file at.
                        Default is None

    output_variables: str
                        Output variables of the simulation that you want to save into your NETCDF Outputfile.

    tech_year : int, optional
                If given in combination with the projected module str names "WINAICO WSx-240P6" or
                "LG Electronics LG370Q1C-A5", the effifiency will be scaled linearly to the given
                year. Must then be between year of market introduction for that module and 2050.
                Will be ignored when non-projected existing module names or specific parameters
                are given, can then be None. By default 2050.

    Returns
    -------
    A xarray dataset including all the output variables you defined as your output_variables.

    """
    wf = SolarWorkflowManager(placements)
    wf.configure_cec_module(module=module, tech_year=tech_year, tracking=tracking, database="CECMod")
    # ensure the tracking parameter is correct
    assert tracking == "fixed", f"Only tracking = 'fixed' allowed in this workflow."

    if "modtilt" not in wf.placements.columns:
        wf.estimate_module_tilt_from_latitude(convention="Ryberg2020")
    if "modazimuth" not in wf.placements.columns:
        wf.estimate_module_azimuth_from_latitude(convention="NorthSouth")
    if "elev" not in wf.placements.columns:
        wf.assign_elevation(elev)
    if "grdalbedo" not in wf.placements.columns:
        wf.assign_ground_albedo(ground_albedo=0.25)

    wf.read(
        variables=["direct_normal_irradiance", "global_horizontal_irradiance"],
        source_type="SARAH",
        source=sarah_path,
        set_time_index=True,
        verbose=False,
    )

    wf.read(
        variables=[
            "surface_wind_speed",
            "surface_pressure",
            "surface_air_temperature",
            "surface_dew_temperature",
        ],
        source_type="ERA5",
        source=era5_path,
        set_time_index=False,
        time_index_from="direct_horizontal_irradiance",
        verbose=False,
    )

    wf.determine_solar_position()
    wf.filter_positive_solar_elevation()
    wf.determine_extra_terrestrial_irradiance(model="spencer", solar_constant=1370)
    wf.determine_air_mass(model="kastenyoung1989")

    wf.diffuse_horizontal_irradiance_from_trigonometry()

    if tracking == "singleaxis":
        wf.permit_single_axis_tracking(**tracking_args)

    wf.determine_angle_of_incidence()
    wf.estimate_plane_of_array_irradiances(transposition_model="perez")

    wf.apply_angle_of_incidence_losses_to_poa()

    wf.cell_temperature_from_sapm()

    wf.simulate_with_interpolated_single_diode_approximation()

    if inverter is not None:
        wf.apply_inverter_losses(inverter=inverter, **inverter_kwargs)

    variables = [_var for _var in ["capacity_factor", "total_system_generation"] if _var in wf.sim_data.keys()]
    wf.apply_loss_factor(0.20, variables=variables)

    return wf.to_xarray(output_netcdf_path=output_netcdf_path, output_variables=output_variables)


def openfield_pv_iconlam(
    placements,
    icon_lam_path,
    module="WINAICO WSx-240P6",
    elev=300,
    tracking="fixed",
    inverter=None,
    inverter_kwargs={},
    tracking_args={},
    output_netcdf_path=None,
    output_variables=None,
    tech_year=2050,
):
    """
    Simulation of an openfield  PV openfield system based on ICON-LAM Data.

    Parameters
    ----------
    placements: Pandas Dataframe
            Locations that you want to do the simulations for.
            Columns need to be lat (latitudes), lon (longitudes), tilt and capacity.

    icon_lam_path: str
            Path to the ICON-LAM Data on your computer.
            Can be a single ".nc" file, or a directory containing many ".nc" files.

    module: str
            Name of the module that you wanna use for the simulation.
            Default is Winaico Wsx-240P6

    elev: float
            Elevation that you want to model your PV system at.
            SChen: Or you can provide a string directory when a terrain raster can be found

    tracking: str
            Determines whether your PV system is fixed or not.
            Default is fixed.
            Option 1 is 'fixed' meaning that the module does not have any tracking capabilities.
            Option 2 is 'singleaxis' meaning that the module has single-axis tracking capabilities.

    inverter: str
            Determines whether you want to model your PV system with an inverter or not.
            Default is None.
            See reskit.solar.SolarWorkflowManager.apply_inverter_losses for more usage information.

    output_netcdf_path: str
            Path to a file that you want to save your output NETCDF file at.
            Default is None

    output_variables: str
            Output variables of the simulation that you want to save into your NETCDF Outputfile.

    tech_year : int, optional
                If given in combination with the projected module str names "WINAICO WSx-240P6" or
                "LG Electronics LG370Q1C-A5", the effifiency will be scaled linearly to the given
                year. Must then be between year of market introduction for that module and 2050.
                Will be ignored when non-projected existing module names or specific parameters
                are given, can then be None. By default 2050.

    Returns
    -------
    A xarray dataset including all the output variables you defined as your output_variables.
    """
    wf = SolarWorkflowManager(placements)
    wf.configure_cec_module(module=module, tech_year=tech_year, tracking=tracking, database="CECMod")

    # ensure the tracking parameter is correct
    if tracking in ["single-axis", "single_axis"]: 
        tracking = "singleaxis"
    assert tracking in [
        "fixed",
        "singleaxis",
    ], "tracking must be either 'fixed' or 'singleaxis'"

    # estimates tilt, azimuth and elev
    wf.estimate_missing_params(
        elev,
        ground_albedo=0.25, 
        gcr=2.0 / 7.0,
        fixed_module_tilt_convention="Ryberg2020",
        fixed_module_azimuth_convention="NorthSouth",    
        )

    wf.read(
        variables=[
            "global_horizontal_irradiance",
            "direct_horizontal_irradiance",
            "surface_wind_speed",
            "surface_pressure",
            "surface_air_temperature",
            "surface_dew_temperature",
        ],
        source_type="ICON-LAM",
        source=icon_lam_path,
        set_time_index=True,
        time_index_from="direct_horizontal_irradiance",
        spatial_interpolation_mode="near",
        verbose=False,
    )

    wf.determine_solar_position()
    wf.filter_positive_solar_elevation()

    wf.direct_normal_irradiance_from_trigonometry()

    wf.determine_extra_terrestrial_irradiance(model="spencer", solar_constant=1370)
    wf.determine_air_mass(model="kastenyoung1989")

    wf.diffuse_horizontal_irradiance_from_trigonometry()

    if tracking == "singleaxis":
        wf.permit_single_axis_tracking(**tracking_args)

    wf.determine_angle_of_incidence()
    wf.estimate_plane_of_array_irradiances(transposition_model="perez")

    wf.apply_angle_of_incidence_losses_to_poa()

    wf.cell_temperature_from_sapm()

    wf.simulate_with_interpolated_single_diode_approximation()

    if inverter is not None:
        wf.apply_inverter_losses(inverter=inverter, **inverter_kwargs)

    # this loss_factor was particularly tuned for ERA5GSA RESKit solar workflow
    # loss_factor = 0.115  # validation by d.franzmann, 2022/01/13
    # wf.apply_loss_factor(
    #     loss_factor, variables=["capacity_factor", "total_system_generation"]
    # )

    loss_factor = 0.107  # general loss_factor by s.chen, 2024/05/08
    wf.apply_loss_factor(loss_factor, variables=["capacity_factor", "total_system_generation"])

    return wf.to_xarray(output_netcdf_path=output_netcdf_path, output_variables=output_variables)


########################
# DEPRECATED WORKFLOWS #
########################

# The following workflows are deprecated and can only be used by checking
# out the respective commit status of RESkit


def openfield_pv_era5pure(**kwargs):
    """
    Simulation of an openfield  PV openfield system based on ERA5 original
    Data without further disaggregation.
    """
    # this is the github commit url with the latest workflow status
    commit_hash = "379645675cb1b2559ffa8d73c84be0dd0daef55e"
    raise rk_util.RESKitDeprecationError(commit_hash)


def openfield_pv_era5_unvalidated(**kwargs):
    """
    Simulation of an openfield  PV openfield system based on ERA5 Data,
    with unvalidated default loss factor of 0.107 based on literature.
    """
    # this is the github commit url with the latest workflow status
    commit_hash = "379645675cb1b2559ffa8d73c84be0dd0daef55e"
    raise rk_util.RESKitDeprecationError(commit_hash)
