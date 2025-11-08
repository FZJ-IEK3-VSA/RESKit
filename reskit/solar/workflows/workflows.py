# import primary packages
import numpy as np
import warnings

# import othert modules
from ... import weather as rk_weather
from .solar_workflow_manager import SolarWorkflowManager
from ... import util as rk_util


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
        Determines wether or not you want to model your PV system with an inverter.
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

    variables = [
        _var
        for _var in ["capacity_factor", "total_system_generation"]
        if _var in wf.sim_data.keys()
    ]
    wf.apply_loss_factor(0.20, variables=variables)

    return wf.to_xarray(
        output_netcdf_path=output_netcdf_path, output_variables=output_variables
    )


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
            Determines wether your PV system is fixed or not.
            Default is fixed.
            Option 1 is 'fixed' meaning that the module does not have any tracking capabilities.
            Option 2 is 'singleaxis' meaning that the module has single-axis tracking capabilities.
            
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
            Determines wether you want to model your PV system with an inverter or not.
            Default is None.
            See reskit.solar.SolarWorkflowManager.apply_inverter_losses for more usage information.

    DNI_nodata_fallback: str, optional
            When global_solar_atlas_dni_path has no data, one can decide between different fallback options, by default 1.0:
            - np.nan or None : return np.nan for missing values in global_solar_atlas_dni_path
            - float : Apply this float value as a scaling factor for all no-data locations only: source_long_run_average * DNI_nodata_fallback.
                NOTE: A value of 1.0 will return the source lra value in case of missing global_solar_atlas_dni_path values.
            - str : Will be interpreted as a filepath to a raster with alternative absolute global_solar_atlas_dni_path values
            - callable : any callable method taking the arguments (all iterables): 'locs' and 'source_long_run_average_value'
                (the locations as gk.geom.point objects and original value from source data). The output values will be considered as
                the new real_long_run_average for missing locations only.
            NOTE: np.nan will also be returned in case that the nodata fallback does not yield values either.

    DNI_nodata_fallback_scaling: float, optional
            The scaling factor that will be applied to the DNI nodata fallback e.g. in case of different units compared to source data.
            By default 1.0, i.e. no effect.

    GHI_nodata_fallback: str, optional
            When global_solar_atlas_ghi_path has no data, one can decide between different fallback options, by default 1.0:
            - np.nan or None : return np.nan for missing values in global_solar_atlas_ghi_path
            - float : Apply this float value as a scaling factor for all no-data locations only: source_long_run_average * GHI_nodata_fallback.
                NOTE: A value of 1.0 will return the source lra value in case of missing global_solar_atlas_ghi_path values.
            - str : Will be interpreted as a filepath to a raster with alternative absolute global_solar_atlas_ghi_path values
            - callable : any callable method taking the arguments (all iterables): 'locs' and 'source_long_run_average_value'
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

    Returns
    -------
    A xarray dataset including all the output variables you defined as your output_variables.
    """

    wf = SolarWorkflowManager(placements)
    wf.configure_cec_module(module=module, tech_year=tech_year, tracking=tracking, database="CECMod", bifaciality_factor=0) 

    # tilt and azimuth were ambiguous depending on tracking, rename to consistent attribute names throughout the wfm
    if "tilt" in wf.placements:
        _newtlt = {"fixed" : "modtilt", "singleaxis" : "axtilt"}
        assert _newtlt[tracking] not in wf.placements, f"'tilt' and '{_newtlt[tracking]}' columns cannot exist both when tracking = '{tracking}'."
        warnings.warn(f"'tilt' column will be interpreted as and renamed to '{_newtlt[tracking]}'.")
        wf.placements.rename(columns={"tilt" : _newtlt[tracking]})
    if "azimuth" in wf.placements:
        _newaz = {"fixed" : "modazimuth", "singleaxis" : "axazimuth"}
        assert _newaz[tracking] not in wf.placements, f"'azimuth' and '{_newaz[tracking]}' columns cannot exist both when tracking = '{tracking}'."
        warnings.warn(f"'azimuth' column will be interpreted as and renamed to '{_newaz[tracking]}'.")
        wf.placements.rename(columns={"azimuth" : _newaz[tracking]})
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
        ],
        source_type="ERA5",
        source=era5_path,
        set_time_index=True,
        time_index_from="direct_horizontal_irradiance",
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
    variables = [
        _var
        for _var in ["capacity_factor", "total_system_generation"]
        if _var in wf.sim_data.keys()
    ]
    wf.apply_loss_factor(loss_factor, variables=variables)

    return wf.to_xarray(
        output_netcdf_path=output_netcdf_path, output_variables=output_variables
    )


def pv_era5_WinklerUnpublished(
    placements,
    era5_path,
    global_solar_atlas_ghi_path,
    global_solar_atlas_dni_path,
    module="WINAICO WSx-240P6",
    elev=840,
    tracking="fixed",
    ground_albedo=0.22,
    inverter=None,
    inverter_kwargs={},
    tracking_args={},
    bifaciality_factor=0.75,
    DNI_nodata_fallback=1.0,
    DNI_nodata_fallback_scaling=1.0,
    GHI_nodata_fallback=1.0,
    GHI_nodata_fallback_scaling=1.0,
    output_netcdf_path=None,
    output_variables=None,
    tech_year=2050,
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
            from 'elev' column if available. Defaults to 840 [m], the average
            global landmass elevation.

    ground_albedo : float, tuple, optional
            Albedo of the ground surface below PV system.
            * tuple: format (dataset name, dataset filepath) to point to a 
              landcover dataset. Albedo values will then be mapped to landcover 
              classes based on [2] for every single location.
            * float : The same albedo value to be set for all placements. 
            Defaults to 0.22 Can alternatively be provided as 'grdalbedo' 
            column in the placements dataframe. 

    tracking: str
            Determines wether your PV system is fixed or not.
            Default is fixed.
            Option 1 is 'fixed' meaning that the module does not have any tracking capabilities.
            Option 2 is 'singleaxis' meaning that the module has single-axis tracking capabilities.
            
            NOTE: The tilt and azimuth definitions change with different tracking systems.
            For fixed tilt systems the following column names apply: 
            * module_tilt_col="modtilt"
            * module_azimuth_col="modazimuth"
            For tracking systems, any potential 'tilt' and 'azimuth' columns must be removed! 
            The column names for the tracker axis tilts and azimuth are instead:
            * axis_azimuth_col="axazimuth"
            * axis_tilt_col="axtilt"
            * crossaxis_tilt_col="caxtilt"

    inverter: str
            Determines wether you want to model your PV system with an inverter or not.
            Default is None.
            See reskit.solar.SolarWorkflowManager.apply_inverter_losses for more usage information.
    
    inverter_kwargs : #TODO
    
    tracking_args : #TODO

    DNI_nodata_fallback: str, optional
            When global_solar_atlas_dni_path has no data, one can decide between different fallback options, by default 1.0:
            - np.nan or None : return np.nan for missing values in global_solar_atlas_dni_path
            - float : Apply this float value as a scaling factor for all no-data locations only: source_long_run_average * DNI_nodata_fallback.
                NOTE: A value of 1.0 will return the source lra value in case of missing global_solar_atlas_dni_path values.
            - str : Will be interpreted as a filepath to a raster with alternative absolute global_solar_atlas_dni_path values
            - callable : any callable method taking the arguments (all iterables): 'locs' and 'source_long_run_average_value'
                (the locations as gk.geom.point objects and original value from source data). The output values will be considered as
                the new real_long_run_average for missing locations only.
            NOTE: np.nan will also be returned in case that the nodata fallback does not yield values either.

    DNI_nodata_fallback_scaling: float, optional
            The scaling factor that will be applied to the DNI nodata fallback e.g. in case of different units compared to source data.
            By default 1.0, i.e. no effect.

    GHI_nodata_fallback: str, optional
            When global_solar_atlas_ghi_path has no data, one can decide between different fallback options, by default 1.0:
            - np.nan or None : return np.nan for missing values in global_solar_atlas_ghi_path
            - float : Apply this float value as a scaling factor for all no-data locations only: source_long_run_average * GHI_nodata_fallback.
                NOTE: A value of 1.0 will return the source lra value in case of missing global_solar_atlas_ghi_path values.
            - str : Will be interpreted as a filepath to a raster with alternative absolute global_solar_atlas_ghi_path values
            - callable : any callable method taking the arguments (all iterables): 'locs' and 'source_long_run_average_value'
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
    # initialize workflow manager and module/system
    wf = SolarWorkflowManager(placements)
    wf.configure_cec_module(
        module=module, 
        tech_year=tech_year, 
        bifaciality_factor=bifaciality_factor, 
        database="CEC Modules.csv"
        )

    # estimate tilt, azimuth, albedo and elev
    wf.estimate_missing_params(
        elev=elev, 
        ground_albedo=ground_albedo,
        gcr="Winkler2026",
        fixed_module_tilt_convention="Ryberg2020", #TODO
        fixed_module_azimuth_convention="NorthSouth", #TODO
        singleaxis_tilt_convention="flat", #TODO
        singleaxis_azimuth_convention="North",
        crossaxis_tilt_convention="flat", #TODO
    )
    
    # read weather variables
    wf.read(
        variables=[
            "global_horizontal_irradiance",
            "direct_horizontal_irradiance",
            "surface_wind_speed",
            "surface_pressure",
            "surface_air_temperature",
            "surface_dew_temperature",
            # TODO add snow variables
        ],
        source_type="ERA5",
        source=era5_path,
        set_time_index=True,
        time_index_from="direct_horizontal_irradiance",
        verbose=False,
    )

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

    wf.determine_extra_terrestrial_irradiance(model="spencer", solar_constant=1370) #TODO needed?
    wf.determine_air_mass(model="kastenyoung1989") # TODO needed?

    wf.diffuse_horizontal_irradiance_from_trigonometry() #TODO needed?

    # determine angle of incidence and resulting insolation
    if wf.tracking == "singleaxis":
        wf.permit_single_axis_tracking(**tracking_args)
    wf.determine_angle_of_incidence()
    wf.estimate_plane_of_array_irradiances(transposition_model="perez")
    wf.apply_angle_of_incidence_losses_to_poa()

    # simulate module response and energy yield
    wf.cell_temperature_from_sapm()
    wf.simulate_with_interpolated_single_diode_approximation()

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

    return wf.to_xarray(
        output_netcdf_path=output_netcdf_path, output_variables=output_variables
    )


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
                Determines wether your PV system is fixed or not.
                Default is fixed.
                Option 1 is 'fixed' meaning that the module does not have any tracking capabilities.
                Option 2 is 'singleaxis' meaning that the module has single-axis tracking capabilities.

    inverter: str
                Determines wether you want to model your PV system with an inverter or not.
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

    variables = [
        _var
        for _var in ["capacity_factor", "total_system_generation"]
        if _var in wf.sim_data.keys()
    ]
    wf.apply_loss_factor(0.20, variables=variables)

    return wf.to_xarray(
        output_netcdf_path=output_netcdf_path, output_variables=output_variables
    )


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
            Determines wether your PV system is fixed or not.
            Default is fixed.
            Option 1 is 'fixed' meaning that the module does not have any tracking capabilities.
            Option 2 is 'singleaxis' meaning that the module has single-axis tracking capabilities.

    inverter: str
            Determines wether you want to model your PV system with an inverter or not.
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
    wf.apply_loss_factor(
        loss_factor, variables=["capacity_factor", "total_system_generation"]
    )

    return wf.to_xarray(
        output_netcdf_path=output_netcdf_path, output_variables=output_variables
    )


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
