from ... import weather as rk_weather
from .solar_workflow_manager import SolarWorkflowManager
import numpy as np
import warnings


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
                Option 2 is 'single_axis' meaning that the module has single_axis tracking capabilities.


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
    wf.configure_cec_module(module, tech_year)
    # ensure the tracking parameter is correct
    assert tracking in [
        "fixed",
        "single_axis",
    ], f"tracking must be either 'fixed' or 'single_axis'"

    # estimates tilt, azimuth and elev
    wf.generate_missing_params(elev)

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

    if tracking == "single_axis":
        wf.permit_single_axis_tracking(**tracking_args)

    wf.determine_angle_of_incidence()
    wf.estimate_plane_of_array_irradiances(transposition_model="perez")

    wf.apply_angle_of_incidence_losses_to_poa()

    wf.cell_temperature_from_sapm()

    wf.simulate_with_interpolated_single_diode_approximation(
        module=module, tech_year=tech_year
    )

    if inverter is not None:
        wf.apply_inverter_losses(inverter=inverter, **inverter_kwargs)

    wf.apply_loss_factor(0.20, variables=["capacity_factor", "total_system_generation"])

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
            Columns need to be lat (latitudes), lon (longitudes), tilt and capacity.

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
            Elevation that you want to model your PV system at.

    tracking: str
            Determines wether your PV system is fixed or not.
            Default is fixed.
            Option 1 is 'fixed' meaning that the module does not have any tracking capabilities.
            Option 2 is 'single_axis' meaning that the module has single_axis tracking capabilities.

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
            - f'missing_values_{os.path.basename(path_to_LRA_source)}

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
    wf.configure_cec_module(module, tech_year)

    # limit the input placements longitude to range of -180...180
    assert wf.placements["lon"].between(-180, 180, inclusive="both").any()
    # limit the input placements latitude to range of -90...90
    assert wf.placements["lat"].between(-90, 90, inclusive="both").any()
    # ensure the tracking parameter is correct
    assert tracking in [
        "fixed",
        "single_axis",
    ], f"tracking must be either 'fixed' or 'single_axis'"

    # estimates tilt, azimuth and elev
    wf.generate_missing_params(elev)

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
                f"When GHI_nodata_fallback and DNI_nodata_fallback have been adapted, gsa_nodata_fallback must not be adapted (recommended to ignore, deprecated)"
            )
        else:
            # GHI and DNI fallbacks have not been changed, but 'source' has - adapt DNI and GHI fallbacks accordingly
            if gsa_nodata_fallback == "nan":
                GHI_nodata_fallback = np.nan
                DNI_nodata_fallback = np.nan
            else:
                raise ValueError(
                    f"'gsa_nodata_fallback' (deprecated) must be 'nan' or 'source'. Better use 'GHI_nodata_fallback' and 'GHI_nodata_fallback' instead, however."
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

    if tracking == "single_axis":
        wf.permit_single_axis_tracking(**tracking_args)

    wf.determine_angle_of_incidence()
    wf.estimate_plane_of_array_irradiances(transposition_model="perez")

    wf.apply_angle_of_incidence_losses_to_poa()

    wf.cell_temperature_from_sapm()

    wf.simulate_with_interpolated_single_diode_approximation(
        module=module, tech_year=tech_year
    )

    if inverter is not None:
        wf.apply_inverter_losses(inverter=inverter, **inverter_kwargs)

    loss_factor = 0.115  # validation by d.franzmann, 2022/01/13
    wf.apply_loss_factor(
        loss_factor, variables=["capacity_factor", "total_system_generation"]
    )

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
                Option 2 is 'single_axis' meaning that the module has single_axis tracking capabilities.

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
    wf.configure_cec_module(module, tech_year)
    # ensure the tracking parameter is correct
    assert tracking in [
        "fixed",
        "single_axis",
    ], f"tracking must be either 'fixed' or 'single_axis'"

    if not "tilt" in wf.placements.columns:
        wf.estimate_tilt_from_latitude(convention="Ryberg2020")
    if not "azimuth" in wf.placements.columns:
        wf.estimate_azimuth_from_latitude()
    if not "elev" in wf.placements.columns:
        wf.apply_elevation(elev)

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

    if tracking == "single_axis":
        wf.permit_single_axis_tracking(**tracking_args)

    wf.determine_angle_of_incidence()
    wf.estimate_plane_of_array_irradiances(transposition_model="perez")

    wf.apply_angle_of_incidence_losses_to_poa()

    wf.cell_temperature_from_sapm()

    wf.simulate_with_interpolated_single_diode_approximation(
        module=module, tech_year=tech_year
    )

    if inverter is not None:
        wf.apply_inverter_losses(inverter=inverter, **inverter_kwargs)

    wf.apply_loss_factor(0.20, variables=["capacity_factor", "total_system_generation"])

    return wf.to_xarray(
        output_netcdf_path=output_netcdf_path, output_variables=output_variables
    )
