from ... import weather as rk_weather
from .solar_workflow_manager import SolarWorkflowManager


def openfield_pv_merra_ryberg2019(placements, merra_path, global_solar_atlas_ghi_path, module="WINAICO WSx-240P6", elev=300, tracking="fixed", inverter=None, inverter_kwargs={}, tracking_args={}, output_netcdf_path=None, output_variables=None):
        """
        ###EUPS: Usually the funtion is not normally repeated in the docstrings for Python since it is found in introspection when calling the 'help' function.
                                                                                                    # see https://numpydoc.readthedocs.io/en/latest/format.html 1. Short summary
                                                                                                    # So,  I did not include it again in the docstrings... 

        Simulation of an openfield  PV system based on NASA'S MERRA2  Data [1].
        
        Parameters
        ----------
        placements: Pandas Dataframe
                     Locations that you want to perform the simulations for.
                     Columns need to be lat (latitudes), lon (longitudes), tilt and capacity.
                     
        merra_path: str
                    Path to the MERRA Data on your computer.
                    
        global_solar_atlas_ghi_path: str
                                     Path to the global solar atlas ghi data on your computer.
                                     
        module: str
                Name of the module that you wanna use for the simulation.
                Default is Winaico Wsx-240P6
        
        elev: float
              Elevation that you want to model your PV system at.
              
        tracking: str
                  Determines wether your PV system is fixed or not.
                  Default is fixed.
                  
        inverter: str
                  Determines wether you want to model your PV system with an inverter or not.
                  Default is None.
                  
        output_netcdf_path: str
                            Path to a file that you want to save your output NETCDF file at.
                            Default is None
                            
        output_variables: str, optional
                          Output variables of the simulation that you want to save into your NETCDF Outputfile.
                  
        ###EUPS : I used only one tab to decribe the variable. For example:
                                                                        #Parameters
                                                                        #----------
                                                                        # 
                                                                        # placements : Pandas Dataframe
                                                                        #    Locations that you want to do the simulations for.
                                                                        #    Columns need to be lat (latitudes), lon (longitudes), tilt and capacity.
        # We can decide on how to do this

        ###EUPS : there must be a space between the variable, the colon, and the type. For example:
                                                                                                    Paramentes :
                                                                                                    ----------
                                                                                                    #x : type
        ###EUPS : Default values in Parameters are missing. For example:
                                                                        #Parameters
                                                                        #----------
                                                                        #output_netcdf_path : str, optional
                                                                        #    Path to a directory to put the output files, by default None
                                                                        #output_variables : str, optional
                                                                        #    Additional variables?, by default None

        Returns
        -------
        A xarray dataset including all the output variables you defined as your output_variables.


        References
        ------
        [1] National Aeronautics and Space Administration. (2019). Modern-Era Retrospective analysis for Research and Applications, Version 2. NASA Goddard Earth Sciences (GES) Data and Information Services Center (DISC). https://disc.gsfc.nasa.gov/datasets?keywords=%22MERRA-2%22&page=1&source=Models%2FAnalyses MERRA-2

                                                                                            
        """
    
    wf = SolarWorkflowManager(placements)
    wf.configure_cec_module(module)

    if not "tilt" in wf.placements.columns:
        wf.estimate_tilt_from_latitude(convention="Ryberg2020")
    if not "azimuth" in wf.placements.columns:
        wf.estimate_azimuth_from_latitude()
    if not "elev" in wf.placements.columns:
        wf.apply_elevation(elev)

    wf.read(
        variables=['surface_wind_speed',
                   "surface_pressure",
                   "surface_air_temperature",
                   "surface_dew_temperature",
                   "global_horizontal_irradiance"],
        source_type="MERRA",
        path=merra_path,
        set_time_index=True,
        verbose=False
    )

    wf.adjust_variable_to_long_run_average(
        variable='global_horizontal_irradiance',
        source_long_run_average=rk_weather.MerraSource.LONG_RUN_AVERAGE_GHI,
        real_long_run_average=global_solar_atlas_ghi_path,
        real_lra_scaling=1000 / 24,  # cast to hourly average kWh
    )

    wf.determine_solar_position()
    wf.filter_positive_solar_elevation()
    wf.determine_extra_terrestrial_irradiance(model="spencer", solar_constant=1370)
    wf.determine_air_mass(model='kastenyoung1989')
    wf.apply_DIRINT_model()
    wf.diffuse_horizontal_irradiance_from_trigonometry()

    if tracking == "single_axis":
        wf.permit_single_axis_tracking(**tracking_args)

    wf.determine_angle_of_incidence()
    wf.estimate_plane_of_array_irradiances(transposition_model="perez")

    wf.apply_angle_of_incidence_losses_to_poa()

    wf.cell_temperature_from_sapm()

    wf.simulate_with_interpolated_single_diode_approximation(module=module)

    if inverter is not None:
        wf.apply_inverter_losses(inverter=inverter, **inverter_kwargs)

    wf.apply_loss_factor(0.20, variables=['capacity_factor', 'total_system_generation'])

    return wf.to_xarray(output_netcdf_path=output_netcdf_path, output_variables=output_variables)


def openfield_pv_era5_unvalidated(placements, era5_path, global_solar_atlas_ghi_path, global_solar_atlas_dni_path, module="WINAICO WSx-240P6", elev=300, tracking="fixed", inverter=None, inverter_kwargs={}, tracking_args={}, output_netcdf_path=None, output_variables=None):
    
        """
        
        openfield_pv_era5_unvalidated(placements, era5_path, global_solar_atlas_ghi_path, global_solar_atlas_dni_path, module="WINAICO WSx-240P6", elev=300, tracking="fixed", inverter=None, inverter_kwargs={}, tracking_args={}, output_netcdf_path=None, output_variables=None)
        
        Simulation of an openfield  PV openfield system based on ERA5 Data.
        
        Parameters
        ----------
        placements: Pandas Dataframe
                     Locations that you want to do the simulations for.
                     Columns need to be lat (latitudes), lon (longitudes), tilt and capacity.
                     
        era5_path: str
                    Path to the ERA5 Data on your computer.
                    
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
                  
        inverter: str
                  Determines wether you want to model your PV system with an inverter or not.
                  Default is None.
                  
        output_netcdf_path: str
                            Path to a file that you want to save your output NETCDF file at.
                            Default is None
                            
        output_variables: str
                          Output variables of the simulation that you want to save into your NETCDF Outputfile.
                  
        Returns
        -------
        A xarray dataset including all the output variables you defined as your output_variables.
        
        """
    
    wf = SolarWorkflowManager(placements)
    wf.configure_cec_module(module)

    if not "tilt" in wf.placements.columns:
        wf.estimate_tilt_from_latitude(convention="Ryberg2020")
    if not "azimuth" in wf.placements.columns:
        wf.estimate_azimuth_from_latitude()
    if not "elev" in wf.placements.columns:
        wf.apply_elevation(elev)

    wf.read(
        variables=["global_horizontal_irradiance",
                   "direct_horizontal_irradiance",
                   "surface_wind_speed",
                   "surface_pressure",
                   "surface_air_temperature",
                   "surface_dew_temperature", ],
        source_type="ERA5",
        path=era5_path,
        set_time_index=True,
        verbose=False
    )

    wf.determine_solar_position()
    wf.filter_positive_solar_elevation()

    wf.direct_normal_irradiance_from_trigonometry()

    wf.adjust_variable_to_long_run_average(
        variable='global_horizontal_irradiance',
        source_long_run_average=rk_weather.Era5Source.LONG_RUN_AVERAGE_GHI,
        real_long_run_average=global_solar_atlas_ghi_path,
        real_lra_scaling=1000 / 24,  # cast to hourly average kWh
    )

    wf.adjust_variable_to_long_run_average(
        variable='direct_normal_irradiance',
        source_long_run_average=rk_weather.Era5Source.LONG_RUN_AVERAGE_DNI,
        real_long_run_average=global_solar_atlas_dni_path,
        real_lra_scaling=1000 / 24,  # cast to hourly average kWh
    )

    wf.determine_extra_terrestrial_irradiance(model="spencer", solar_constant=1370)
    wf.determine_air_mass(model='kastenyoung1989')

    wf.diffuse_horizontal_irradiance_from_trigonometry()

    if tracking == "single_axis":
        wf.permit_single_axis_tracking(**tracking_args)

    wf.determine_angle_of_incidence()
    wf.estimate_plane_of_array_irradiances(transposition_model="perez")

    wf.apply_angle_of_incidence_losses_to_poa()

    wf.cell_temperature_from_sapm()

    wf.simulate_with_interpolated_single_diode_approximation(module=module)

    if inverter is not None:
        wf.apply_inverter_losses(inverter=inverter, **inverter_kwargs)

    wf.apply_loss_factor(0.20, variables=['capacity_factor', 'total_system_generation'])

    return wf.to_xarray(output_netcdf_path=output_netcdf_path, output_variables=output_variables)


def openfield_pv_sarah_unvalidated(placements, sarah_path, era5_path, module="WINAICO WSx-240P6", elev=300, tracking="fixed", inverter=None, inverter_kwargs={}, tracking_args={}, output_netcdf_path=None, output_variables=None):
    
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
                     
        era5_path: str
                   Path to the ERA5 Data on your computer.
                    
                                     
        module: str
                Name of the module that you wanna use for the simulation.
                Default is Winaico Wsx-240P6
        
        elev: float
              Elevation that you want to model your PV system at.
              
        tracking: str
                  Determines wether your PV system is fixed or not.
                  Default is fixed.
                  
        inverter: str
                  Determines wether you want to model your PV system with an inverter or not.
                  Default is None.
                  
        output_netcdf_path: str
                            Path to a file that you want to save your output NETCDF file at.
                            Default is None
                            
        output_variables: str
                          Output variables of the simulation that you want to save into your NETCDF Outputfile.
                  
                   
        Returns
        -------
        A xarray dataset including all the output variables you defined as your output_variables.
        
        """
        
    wf = SolarWorkflowManager(placements)
    wf.configure_cec_module(module)

    if not "tilt" in wf.placements.columns:
        wf.estimate_tilt_from_latitude(convention="Ryberg2020")
    if not "azimuth" in wf.placements.columns:
        wf.estimate_azimuth_from_latitude()
    if not "elev" in wf.placements.columns:
        wf.apply_elevation(elev)

    wf.read(
        variables=["direct_normal_irradiance",
                   "global_horizontal_irradiance"],
        source_type="SARAH",
        path=sarah_path,
        set_time_index=True,
        verbose=False
    )

    wf.read(
        variables=["surface_wind_speed",
                   "surface_pressure",
                   "surface_air_temperature",
                   "surface_dew_temperature", ],
        source_type="ERA5",
        path=era5_path,
        set_time_index=False,
        verbose=False
    )

    wf.determine_solar_position()
    wf.filter_positive_solar_elevation()
    wf.determine_extra_terrestrial_irradiance(model="spencer", solar_constant=1370)
    wf.determine_air_mass(model='kastenyoung1989')

    wf.diffuse_horizontal_irradiance_from_trigonometry()

    if tracking == "single_axis":
        wf.permit_single_axis_tracking(**tracking_args)

    wf.determine_angle_of_incidence()
    wf.estimate_plane_of_array_irradiances(transposition_model="perez")

    wf.apply_angle_of_incidence_losses_to_poa()

    wf.cell_temperature_from_sapm()

    wf.simulate_with_interpolated_single_diode_approximation(module=module)

    if inverter is not None:
        wf.apply_inverter_losses(inverter=inverter, **inverter_kwargs)

    wf.apply_loss_factor(0.20, variables=['capacity_factor', 'total_system_generation'])

    return wf.to_xarray(output_netcdf_path=output_netcdf_path, output_variables=output_variables)
