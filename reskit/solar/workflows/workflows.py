from ... import weather as rk_weather
from .solar_workflow_manager import SolarWorkflowManager
from .csp_workflow_manager import PTRWorkflowManager
from .CSP_data.database_loader import load_dataset
import numpy as np
import time


def csp_ptr_V1(
    placements,
    era5_path,
    global_solar_atlas_dni_path,
    datasetname ='Validation 1',
    elev_path = None,
    output_netcdf_path=None,
    output_variables=None,
    return_self=True,
    JITaccelerate = False,
    verbose = False
    ):
    """ Calculates the heat output from the solar field based on parabolic trough technology. The workflow is not yet finally validated (but is still plausible).
        Status: 24.03.2021
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

    ptr_data = load_dataset(datasetname=datasetname)

    orientation = ptr_data['orientation']
    a1 = ptr_data['a1']                  #gafurov2013: 0.000884
    a2 = ptr_data['a2']                 #gafurov2013: 0.00005369
    a3 = ptr_data['a3']                          #gafurov2013
    SF_density = ptr_data['SF_density']              #gafurov2013
    eta_ptr_max = ptr_data['eta_ptr_max']             #gafurov2013: 0.742
    eta_cleaness = ptr_data['eta_cleaness']
    A_aperture_sf = ptr_data['A_aperture_sf']           #gafurov2013: 909060
    relHeatLosses = ptr_data['relHeatLosses']            #gafurov2013
    ratedFieldOutputHeat_W = 1000 * A_aperture_sf * eta_ptr_max #from nowhere
    maxWindspeed = ptr_data['maxWindspeed'] #m/s
    b = np.array([ptr_data['b0'],      #b0
                    ptr_data['b1'],    #b1
                    ptr_data['b2'],    #b2
                    ptr_data['b3'],    #b3
                    ptr_data['b4']])   #b4
    relTMplant = ptr_data['relTMplant'] #J/K m2
    maxHTFTemperature = ptr_data['maxHTFTemperature'] #°C
    minHTFTemperature = ptr_data['minHTFTemperature'] #°C
    inletHTFTemperature = ptr_data['inletHTFTemperature'] #°C
    add_losses_coefficient = ptr_data['add_losses_coefficient']
    discretizationmethod = ptr_data['discretizationmethod']
    efficencyDropPerYear = ptr_data['efficencyDropPerYear']
    lifetime = ptr_data['lifetime']

    #parasitic loss parameters from gafurov 2013
    params_PL_gafurov = {}
    params_PL_gafurov['I_DNI_nom'] = ptr_data['I_DNI_nom'] # w/m^2
    params_PL_gafurov['PL_plant_fix'] = ptr_data['PL_plant_fix']
    params_PL_gafurov['PL_sf_track'] = ptr_data['PL_sf_track']
    params_PL_gafurov['PL_sf_pumping'] = ptr_data['PL_sf_pumping']
    params_PL_gafurov['PL_plant_pumping'] = ptr_data['PL_plant_pumping']
    params_PL_gafurov['PL_plant_other'] = ptr_data['PL_plant_other']


    # 2) Create instance of PTR worflowmanager 

    wf = PTRWorkflowManager(placements)

    # 3) read in Input data from ERA5 
    if verbose:
        tic = time.time()
    wf.read(
        variables=["direct_horizontal_irradiance",
                    "surface_air_temperature",
                    "surface_wind_speed"],
        source_type="ERA5",
        source=era5_path,
        set_time_index=True,
        verbose=verbose)

    # do long run averaging for DNI
    if global_solar_atlas_dni_path == 'default_cluster':
        global_solar_atlas_dni_path = r"/storage/internal/data/gears/geography/irradiance/global_solar_atlas/World_DNI_GISdata_LTAy_DailySum_GlobalSolarAtlas_GEOTIFF/DNI.tif"
    if global_solar_atlas_dni_path == 'default_local':
        global_solar_atlas_dni_path = r"R:\data\gears\geography\irradiance\global_solar_atlas\World_DNI_GISdata_LTAy_DailySum_GlobalSolarAtlas_GEOTIFF\DNI.tif"
        


    if global_solar_atlas_dni_path != None:
        wf.adjust_variable_to_long_run_average(
            variable='direct_horizontal_irradiance',
            source_long_run_average=rk_weather.Era5Source.LONG_RUN_AVERAGE_DNI,
            real_long_run_average=global_solar_atlas_dni_path,
            real_lra_scaling=1000 / 24,  # cast to hourly average kWh
    )
    
    # apply elevation
    wf.apply_elevation(elev_path)

    wf.sim_data['ptr_data'] = ptr_data


    if verbose:
        toc = time.time()
        print('Data read in within {dt}s.'.format(dt = str(toc-tic)))

    # 4) get length of timesteps for later numpy sizing 

    wf.get_timesteps()

    # 5) calculate the solar position based on pvlib
    wf.calculateSolarPosition() 
    #wf.calculateSolarPositionfaster()

    # 6) doing selfmade calulations until Heat to HTF
    wf.calculateCosineLossesParabolicTrough(orientation=orientation)
    wf.calculateIAM(a1=a1, a2=a2, a3=a3)
    wf.calculateShadowLosses(method='wagner2011', SF_density=SF_density)
    wf.calculateWindspeedLosses(max_windspeed_threshold=maxWindspeed)
    wf.calculateDegradationLosses(efficencyDropPerYear=efficencyDropPerYear, lifetime=lifetime)
    wf.calculateHeattoHTF(A_aperture_sf=A_aperture_sf, eta_ptr_max=eta_ptr_max, eta_cleaness=eta_cleaness)

    # 7) calculation heat to plant with loss model
    wf.applyHTFHeatLossModel(
        calculationmethod='dersch2018',
        params={'b': b,
            'A': A_aperture_sf,
            'relTMplant': relTMplant,
            'maxHTFTemperature': maxHTFTemperature,
            'JITaccelerate': JITaccelerate,
            'minHTFTemperature': minHTFTemperature,
            'inletHTFTemperature': inletHTFTemperature,
            'add_losses_coefficient': add_losses_coefficient,
            'discretizationmethod': discretizationmethod
            
            }
        )
    # wf.applyHTFHeatLossModel(calculationmethod='gafurov2013', params={'relHeatLosses': relHeatLosses, 'ratedFieldOutputHeat_W': ratedFieldOutputHeat_W})

    # 8) calculate Parasitic Losses of the plant
    wf.calculateParasitics(
        calculationmethod='gafurov2013',
        params=params_PL_gafurov)
    
    # 9) calculate economics
    # Todo: adjust size of annual_heat... from 1D to 2D, or change the storage type
    # wf.calclateEconomics(
    #     params={'c_A_sf': 1, 'c_Land': 1, 'SF_density': SF_density} 
    #     )


    if verbose:
        toc = time.time()
        print('Total simulation done in in {dt}s.'.format(dt = str(toc-tic)))

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
