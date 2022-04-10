from logging import warning
from numpy.lib.arraysetops import isin

from reskit import workflow_manager
from ... import weather as rk_weather
from .csp_workflow_manager import PTRWorkflowManager
from .dataset_handler import dataset_handler
import numpy as np
import xarray as xr
import time

def CSP_PTR_ERA5(
    placements,
    era5_path,
    global_solar_atlas_dni_path,
    global_solar_atlas_tamb_path=None,
    datasets = None,
    cost_year = 2050,
    HTF_sel = ['Heliosol', 'SolarSalt', 'Therminol'],
    elev_path = None,
    output_netcdf_path=None,
    output_variables=None,
    return_self=True,
    JITaccelerate = False,
    verbose = False,
    debug_vars = False,
    onlynightuse=True,
    fullvariation=False,
    ):
    
    
    #handle inputs for datasets
    single_dataset = False
    if datasets==None: 
        #get datasets from HTF_sel and cost_year
        datasets = ['Dataset_' + htf + '_' + str(cost_year)for htf in HTF_sel]
        if len(datasets)==1:
            single_dataset=True
            datasets=datasets[0]
    elif isinstance(datasets, str):
        #datasets=datasets
        single_dataset=True
    elif isinstance(datasets, list):
        #datasets=datasets
        if len(datasets)==1:
            single_dataset=True
            datasets=datasets[0]
    else:
        raise TypeError(f'datasets got unkown datatype')
    
    if not single_dataset:
        assert isinstance(global_solar_atlas_tamb_path, str)
    
        
    
    if single_dataset: # only one dataset given 
        output = CSP_PTR_ERA5_specific_dataset(
            placements=placements,
            era5_path=era5_path,
            global_solar_atlas_dni_path=global_solar_atlas_dni_path,
            datasetname =datasets,
            elev_path = elev_path,
            output_netcdf_path=output_netcdf_path,
            output_variables=output_variables,
            return_self=return_self,
            JITaccelerate = JITaccelerate,
            verbose = verbose,
            debug_vars = debug_vars,
            onlynightuse = onlynightuse,
            fullvariation = fullvariation,
        )
        return output
    
    else: #multiple datasets found
        
        #1) split up placements for each htf
        d = dataset_handler(datasets)
        placements = d.split_placements(
            placements=placements,
            gsa_dni_path=global_solar_atlas_dni_path,
            gsa_tamb_path=global_solar_atlas_tamb_path,
            )
        del d
        
        #2) run each simulation
        ouputs = []
        for dataset in datasets:
            #select placements for current dataset
            placements_dataset = placements[placements['Dataset_opt'] == dataset]
            
            #skip empty batches
            if len(placements_dataset) == 0:
                continue
            
            #starting core simulation
            output_dataset = CSP_PTR_ERA5_specific_dataset(
                placements=placements_dataset,
                era5_path=era5_path,
                global_solar_atlas_dni_path=global_solar_atlas_dni_path,
                datasetname =dataset,
                elev_path = elev_path,
                output_netcdf_path=output_netcdf_path,
                output_variables=output_variables,
                return_self=False,
                JITaccelerate = JITaccelerate,
                verbose = verbose,
                debug_vars = debug_vars,
                onlynightuse = onlynightuse,
                fullvariation = fullvariation,
            )
            
            #add dataset to column (force output this!)
            if not 'datasetname' in output_dataset.variables:
                output_dataset['datasetname'] = (output_dataset['lon']*0).astype(str)
                output_dataset['datasetname'][:] = dataset
            if 'Dataset_opt' in output_dataset.variables:
                output_dataset = output_dataset.drop('Dataset_opt')

            #set index from placements
            output_dataset['location'] = placements_dataset.index
            
            #remember outputs for each dataset
            ouputs.append(output_dataset)
            del output_dataset
            
            
        #3) merge data
        #TODO: merge together
        output = xr.concat(ouputs, dim = 'location').sortby('location')
        return output
    
    

def CSP_PTR_ERA5_specific_dataset(
    placements,
    era5_path,
    global_solar_atlas_dni_path,
    datasetname ='Validation 10',
    elev_path = None,
    output_netcdf_path=None,
    output_variables=None,
    return_self=True,
    JITaccelerate = False,
    verbose = False,
    debug_vars = False,
    onlynightuse=True,
    fullvariation=False,
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
        print('Simulation started for {n_placements} placements.'.format(n_placements=len(wf.placements)), flush=True)
        print('Reading in Weather data.', flush=True)
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
        time_index_from = 'direct_horizontal_irradiance',
        verbose=False)
    
    if verbose:
        tic_read = time.time()
        print('Data read in within {dt}s.'.format(dt = str(tic_read-tic_start)), flush=True)
        print('Starting preanalysis.', flush=True)
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
    #TODO: remove
    if global_solar_atlas_dni_path == 'default_cluster':
        global_solar_atlas_dni_path = r"/storage/internal/data/gears/geography/irradiance/global_solar_atlas_v2.5/World_DNI_GISdata_LTAy_AvgDailyTotals_GlobalSolarAtlas-v2_GEOTIFF/DNI.tif"
    if global_solar_atlas_dni_path == 'default_local':
        global_solar_atlas_dni_path = r"R:\data\gears\geography\irradiance\global_solar_atlas_v2.5\World_DNI_GISdata_LTAy_AvgDailyTotals_GlobalSolarAtlas-v2_GEOTIFF\DNI.tif"


    wf.adjust_variable_to_long_run_average(
        variable='direct_normal_irradiance',
        source_long_run_average=rk_weather.Era5Source.LONG_RUN_AVERAGE_DNI,
        real_long_run_average=global_solar_atlas_dni_path,
        real_lra_scaling=1000 / 24,  # cast to hourly average kWh
    )

    # manipulationof input values for variation calculation
    wf._applyVariation() #only for developers, can be ignored otherwise

    # 6) doing selfmade calulations until Heat to HTF
    wf.calculateIAM(a1=ptr_data['a1'], a2=ptr_data['a2'], a3=ptr_data['a3'])
    wf.calculateShadowLosses(method='wagner2011', SF_density=ptr_data['SF_density_direct'])
    wf.calculateWindspeedLosses(max_windspeed_threshold=ptr_data['maxWindspeed'])
    wf.calculateDegradationLosses(efficencyDropPerYear=ptr_data['efficencyDropPerYear'], lifetime=ptr_data['lifetime'])
    wf.calculateHeattoHTF(eta_ptr_max=ptr_data['eta_ptr_max'], eta_cleaness=ptr_data['eta_cleaness'], eta_other = ptr_data['eta_other'])

    wf.apply_capacity_sf()
    if not debug_vars:
        del wf.sim_data['surface_wind_speed'], wf.sim_data['tracking_angle'], wf.sim_data['solar_altitude_angle_degree'], wf.sim_data['direct_horizontal_irradiance']
        del wf.sim_data['eta_wind'], wf.sim_data['eta_degradation'], wf.sim_data['eta_shdw']
        
    if verbose:
        tic_pre = time.time()
        print('Preanalysis within {dt}s.'.format(dt = str(tic_pre-tic_read)), flush=True)   
        print('Starting core simulation of the solar field.', flush=True)
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
    
    if not debug_vars:
        del wf.sim_data['surface_air_temperature'], wf.sim_data['HTF_mean_temperature_C'], wf.sim_data['Heat_Losses_W'], wf.sim_data['theta'], wf.sim_data['IAM']

    # 8) calculate Parasitic Losses of the plant
    wf.calculateParasitics(
        calculationmethod='dersch2018',#'gafurov2013',
        params={
            'PL_sf_fixed_W_per_m^2_ap': 1.486,
            'PL_sf_pumping_W_per_m^2_ap': 8.3,
            'PL_plant_fix': ptr_data['PL_plant_fix'],
            #'PL_sf_track': ptr_data['PL_sf_track'], #gafurov
            #'PL_sf_pumping': ptr_data['PL_sf_pumping'], #gafurov
            'PL_plant_pumping': ptr_data['PL_plant_pumping'],
            'PL_plant_other': ptr_data['PL_plant_other'],
        }
        )
    
    #9) calculate economics
    wf.calculateEconomics_SolarField(WACC=ptr_data['WACC'],
                                     lifetime=ptr_data['lifetime'],
                                     calculationmethod='franzmann2021',
                                     params={
                                        'CAPEX_solar_field_EUR_per_m^2_aperture': ptr_data['CAPEX_solar_field_EUR_per_m^2_aperture'], 
                                        'CAPEX_land_EUR_per_m^2_land': ptr_data['CAPEX_land_EUR_per_m^2_land'],
                                        'CAPEX_indirect_cost_perc_CAPEX': ptr_data['CAPEX_indirect_cost_perc_CAPEX'],
                                        'electricity_price_EUR_per_kWh': ptr_data['electricity_price_EUR_per_kWh'],
                                        'OPEX_perc_CAPEX': ptr_data['OPEX_perc_CAPEX'],
                                     }
                                     )    

    if verbose:
        tic_sf_sim = time.time()
        print('Solar field simulation done in {dt}s.'.format(dt = str(tic_sf_sim-tic_pre)), flush=True)
        print('Starting optimizing plant electric output.', flush=True)
    
    wf.optimize_plant_size(onlynightuse=onlynightuse, fullvariation=fullvariation, debug_vars=debug_vars)
    
    # wf.optimize_heat_output_4D()
    # wf.calculateEconomics_Plant_Storage_4D()
    # wf.optimal_Plant_Configuration_4D()
    
    if verbose:
        tic_opt_plant = time.time()
        print('Optimal Sizing done in  {dt}s.'.format(dt = str(tic_opt_plant-tic_sf_sim)), flush=True)
        
        
    wf.calculate_electrical_output(onlynightuse=onlynightuse,debug_vars=debug_vars)
    wf.calculate_LCOE()
    wf.calculateCapacityFactors()
        
    if verbose:
        tic_final = time.time()
        print('Total simulation done in {dt}s.'.format(dt = str(tic_final-tic_start)), flush=True)

    if return_self == True:
        return wf
    else:
        return wf.to_xarray(output_netcdf_path=output_netcdf_path, output_variables=output_variables)

