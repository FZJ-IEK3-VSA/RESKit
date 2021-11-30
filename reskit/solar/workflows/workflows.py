from logging import warning

from reskit import workflow_manager
from ... import weather as rk_weather
from .solar_workflow_manager import SolarWorkflowManager
from .csp_workflow_manager import PTRWorkflowManager
from .CSP_data.database_loader import load_dataset
from numba import jit
import numpy as np
import time


def csp_ptr_V1(
    placements,
    era5_path,
    global_solar_atlas_dni_path,
    datasetname ='Validation 10',
    elev_path = None,
    output_netcdf_path=None,
    output_variables=None,
    return_self=True,
    JITaccelerate = False,
    verbose = False
    ):
    """ Calculates the heat output from the solar field based on parabolic trough technology. The workflow is not yet finally validated (but is still plausible).
        Date: 27.07.2021
        Author: David Franzmann IEK -3

    Args:
        placements ([type]): [description]
        era5_path ([type]): [description]
        output_netcdf_path ([type], optional): [description]. Defaults to None.
        output_variables ([type], optional): [description]. Defaults to None.
        return_self (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """

    # 1) Load input data    
    wf = PTRWorkflowManager(placements)
    
    ptr_data = wf.loadPTRdata(datasetname=datasetname)
    wf.determine_area()
    
    # 3) read in Input data
    if verbose:
        tic_start = time.time()
        print('Simulation started for {n_placements} placements.'.format(n_placements=len(wf.placements)))
        print('Reading in Weather data.')
    wf.read(
        variables=[#"global_horizontal_irradiance",
                   "direct_horizontal_irradiance",
                   "surface_wind_speed",
                   #"surface_pressure",
                   "surface_air_temperature",
        ],
        source_type="ERA5",
        source=era5_path,
        set_time_index=True,
        verbose=verbose)
    
    if verbose:
        tic_read = time.time()
        print('Data read in within {dt}s.'.format(dt = str(tic_read-tic_start)))
        print('Starting preanalysis.')
    # 4) get length of timesteps for later numpy sizing 

    wf.get_timesteps()
    
    # apply elevation
    wf.apply_elevation(elev_path)
    wf.apply_azimuth()
    # 5) calculate the solar position based on pvlib
    
    wf.calculateSolarPosition()  
     
    #calculate DNI from ERA5 to DNi convention
    #ERA5 DIN: Heat flux per horizontal plane
    #DNI convention: Heat flux per normal (to zenith) plane
    wf.direct_normal_irradiance_from_trigonometry()
    
    # do long run averaging for DNI
    if global_solar_atlas_dni_path == 'default_cluster':
        global_solar_atlas_dni_path = r"/storage/internal/data/gears/geography/irradiance/global_solar_atlas_v2.5/World_DNI_GISdata_LTAy_AvgDailyTotals_GlobalSolarAtlas-v2_GEOTIFF/DNI.tif"
    if global_solar_atlas_dni_path == 'default_local':
        global_solar_atlas_dni_path = r"R:\data\gears\geography\irradiance\global_solar_atlas_v2.5\World_DNI_GISdata_LTAy_AvgDailyTotals_GlobalSolarAtlas-v2_GEOTIFF\DNI.tif"


    #TODO: implement if working
    print('WARN WARN WARN: LRA skipped!')
    # if global_solar_atlas_dni_path != None:
    #     wf.adjust_variable_to_long_run_average(
    #         variable='direct_horizontal_irradiance',
    #         source_long_run_average=rk_weather.Era5Source.LONG_RUN_AVERAGE_DNI,
    #         real_long_run_average=global_solar_atlas_dni_path,
    #         real_lra_scaling=1000 / 24,  # cast to hourly average kWh
    # )

    # manipulationof input values for variation calculation
    wf._applyVariation()

    # 6) doing selfmade calulations until Heat to HTF
    #wf.calculateCosineLossesParabolicTrough(orientation=ptr_data['orientation']) shifted
    wf.calculateIAM(a1=ptr_data['a1'], a2=ptr_data['a2'], a3=ptr_data['a3'])
    wf.calculateShadowLosses(method='wagner2011', SF_density=ptr_data['SF_density_direct'])
    wf.calculateWindspeedLosses(max_windspeed_threshold=ptr_data['maxWindspeed'])
    wf.calculateDegradationLosses(efficencyDropPerYear=ptr_data['efficencyDropPerYear'], lifetime=ptr_data['lifetime'])
    wf.calculateHeattoHTF(eta_ptr_max=ptr_data['eta_ptr_max'], eta_cleaness=ptr_data['eta_cleaness'])

    wf.apply_capacity_sf()
    
    if verbose:
        tic_pre = time.time()
        print('Preanalysis within {dt}s.'.format(dt = str(tic_pre-tic_read)))   
        print('Starting core simulation of the solar field.')
    # 7) calculation heat to plant with loss model
    wf.applyHTFHeatLossModel(
        calculationmethod='dersch2018',
        params={'b': ptr_data['b'],
            'relTMplant': ptr_data['relTMplant'],
            'maxHTFTemperature': ptr_data['maxHTFTemperature'],
            'JITaccelerate': JITaccelerate, #TODO: from ptr manager
            'minHTFTemperature': ptr_data['minHTFTemperature'],
            'inletHTFTemperature': ptr_data['inletHTFTemperature'],
            'add_losses_coefficient': ptr_data['add_losses_coefficient'],
            'discretizationmethod': ptr_data['discretizationmethod']
            
            }
        )
    # wf.applyHTFHeatLossModel(calculationmethod='gafurov2013', params={'relHeatLosses': relHeatLosses, 'ratedFieldOutputHeat_W': ratedFieldOutputHeat_W})

    # 8) calculate Parasitic Losses of the plant
    wf.calculateParasitics(
        calculationmethod='gafurov2013',
        params={
            'I_DNI_nom': ptr_data['I_DNI_nom'],
            'PL_plant_fix': ptr_data['PL_plant_fix'],
            'PL_sf_track': ptr_data['PL_sf_track'],
            'PL_sf_pumping': ptr_data['PL_sf_pumping'],
            'PL_plant_pumping': ptr_data['PL_plant_pumping'],
            'PL_plant_other': ptr_data['PL_plant_other'],
        }
        )
    
    wf.calculateCapacityFactors()
    
    #9) calculate economics
    # Todo: adjust size of annual_heat... from 1D to 2D, or change the storage type
    wf.calculateEconomics_SolarField(WACC=ptr_data['WACC'],
                                     lifetime=ptr_data['lifetime'],
                                     calculationmethod='franzmann2021',
                                     params={
                                        'CAPEX_solar_field_USD_per_m^2_aperture': ptr_data['CAPEX_solar_field_USD_per_m^2_aperture'], 
                                        'CAPEX_land_USD_per_m^2_land': ptr_data['CAPEX_land_USD_per_m^2_land'],
                                        'CAPEX_indirect_cost_%_CAPEX': ptr_data['CAPEX_indirect_cost_%_CAPEX'],
                                        'electricity_price_USD_per_kWh': ptr_data['electricity_price_USD_per_kWh'],
                                        'OPEX_%_CAPEX': ptr_data['OPEX_%_CAPEX'],
                                     }
                                     )    

    if verbose:
        tic_sf_sim = time.time()
        print('Solar field simulation done in {dt}s.'.format(dt = str(tic_sf_sim-tic_pre)))
        print('Starting optimizing plant electric output.')
    
    wf.optimize_plant_size()
    
    # wf.optimize_heat_output_4D()
    # wf.calculateEconomics_Plant_Storage_4D()
    # wf.optimal_Plant_Configuration_4D()
    
    if verbose:
        tic_opt_plant = time.time()
        print('Optimal Sizing done in  {dt}s.'.format(dt = str(tic_opt_plant-tic_sf_sim)))
        
        
    wf.calculate_electrical_output()
    wf.calculate_LCOE()
        
        
    if verbose:
        tic_final = time.time()
        print('Total simulation done in {dt}s.'.format(dt = str(tic_final-tic_start)))

    if return_self == True:
        return wf
    else:
        return wf.to_xarray(output_netcdf_path=output_netcdf_path, output_variables=output_variables)


def openfield_pv_merra_ryberg2019(placements, merra_path, global_solar_atlas_ghi_path, module="WINAICO WSx-240P6", elev=300, tracking="fixed", inverter=None, inverter_kwargs={}, tracking_args={}, output_netcdf_path=None, output_variables=None):
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
                Option 2 is 'single-axis' meaning that the module has single-axis tracking capabilities.


    inverter: str
        Determines wether or not you want to model your PV system with an inverter.
        Default is None, meaning no inverter is assumed
        See reskit.solar.SolarWorkflowManager.apply_inverter_losses for more usage information

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
        variables=['surface_wind_speed',
                   "surface_pressure",
                   "surface_air_temperature",
                   "surface_dew_temperature",
                   "global_horizontal_irradiance"],
        source_type="MERRA",
        source=merra_path,
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


def openfield_pv_era5(placements, era5_path, global_solar_atlas_ghi_path, global_solar_atlas_dni_path, module="WINAICO WSx-240P6", elev=300, tracking="fixed", inverter=None, inverter_kwargs={}, tracking_args={}, output_netcdf_path=None, output_variables=None):
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
                Option 2 is 'single-axis' meaning that the module has single-axis tracking capabilities.

    inverter: str
                Determines wether you want to model your PV system with an inverter or not.
                Default is None.
                See reskit.solar.SolarWorkflowManager.apply_inverter_losses for more usage information.

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
        source=era5_path,
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

    #loss factor from preliminary validation. david franzmann, 29.06.2021
    wf.apply_loss_factor(0.215, variables=['capacity_factor', 'total_system_generation'])

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
                Option 2 is 'single-axis' meaning that the module has single-axis tracking capabilities.

    inverter: str
                Determines wether you want to model your PV system with an inverter or not.
                Default is None.
                See reskit.solar.SolarWorkflowManager.apply_inverter_losses for more usage information.

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
        source=sarah_path,
        set_time_index=True,
        verbose=False
    )

    wf.read(
        variables=["surface_wind_speed",
                   "surface_pressure",
                   "surface_air_temperature",
                   "surface_dew_temperature", ],
        source_type="ERA5",
        source=era5_path,
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
