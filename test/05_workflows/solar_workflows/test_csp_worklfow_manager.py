import pandas as pd
import numpy as np
from reskit.solar.workflows.csp_workflow_manager import PTRWorkflowManager
import reskit as rk
import geokit as gk
import pytest

#%% Test Init

def test_PTRWorkflowManager__init__() -> PTRWorkflowManager:
    
    placements = pd.DataFrame()
    placements['lon'] = [ 6.083, 6.083, 5.583]     # Longitude
    placements['lat'] = [ 50.775, 51.475, 50.775,]    # Latitude

    wfm = PTRWorkflowManager(placements=placements)

    assert np.isclose(wfm.ext.xMin, 5.58300)
    assert np.isclose(wfm.ext.xMax, 6.083000)
    assert np.isclose(wfm.ext.yMin, 50.775000)
    assert np.isclose(wfm.ext.yMax, 51.475000)

    assert (wfm.placements['lon'] == placements['lon']).all()
    assert (wfm.placements['lat'] == placements['lat']).all()

    return wfm

@pytest.fixture
def pt_PTRWorkflowManager_initialized() -> PTRWorkflowManager:
    return test_PTRWorkflowManager__init__()


#%% test load elevation

def test_apply_elevation(pt_PTRWorkflowManager_initialized):
    wfm = pt_PTRWorkflowManager_initialized

    placements = pd.DataFrame()
    placements['lon'] = [ 6.083, 6.083, 5.583]     # Longitude
    placements['lat'] = [ 50.775, 51.475, 50.775,]    # Latitude

    wfm = PTRWorkflowManager(placements=placements)

    #load from file
    wfm.apply_elevation(rk.TEST_DATA['DEM-like.tif'])

    assert np.isclose(wfm.placements['elev'].tolist(), [185, 22, 118] ).all()

    #prevent reload 
    wfm.placements['elev'] = [1, 2, 3]
    wfm.apply_elevation(rk.TEST_DATA['DEM-like.tif'])

    assert np.isclose(wfm.placements['elev'].tolist(), [1, 2, 3] ).all()

    #elev from number
    wfm = pt_PTRWorkflowManager_initialized
    wfm.apply_elevation(11)
    assert np.isclose(wfm.placements['elev'].tolist(), [11, 11, 11] ).all()



#%% test read ERA5
def test_read_ERA5(pt_PTRWorkflowManager_initialized):
    wfm = pt_PTRWorkflowManager_initialized
    wfm.read(
        variables=["direct_horizontal_irradiance",
                    "surface_air_temperature",
                    "surface_wind_speed"],
        source_type="ERA5",
        source=rk.TEST_DATA['era5-like'],
        set_time_index=True,
        verbose=False
    )

    assert np.isclose(wfm.sim_data['direct_horizontal_irradiance'].mean(), 15.15181)
    assert np.isclose(wfm.sim_data['direct_horizontal_irradiance'].std(), 33.92488)
    assert np.isclose(wfm.sim_data['direct_horizontal_irradiance'].min(), 0)
    assert np.isclose(wfm.sim_data['direct_horizontal_irradiance'].max(), 162.14436)



@pytest.fixture
def pt_PTRWorkflowManager_loaded(pt_PTRWorkflowManager_initialized: PTRWorkflowManager) -> PTRWorkflowManager:
    wfm = pt_PTRWorkflowManager_initialized

    wfm.read(
        variables=["direct_horizontal_irradiance",
                    "surface_air_temperature",
                    "surface_wind_speed"],
        source_type="ERA5",
        source=rk.TEST_DATA['era5-like'],
        set_time_index=True,
        verbose=False
    )


    wfm.apply_elevation(rk.TEST_DATA['DEM-like.tif'])
    #wfm.get_timesteps()

    return wfm

#%% test Long rung averaging
def test_adjust_variable_to_long_run_average(pt_PTRWorkflowManager_loaded):
    wfm = pt_PTRWorkflowManager_loaded

    print('before LRA')
    print(wfm.sim_data['direct_horizontal_irradiance'].mean())
    print(wfm.sim_data['direct_horizontal_irradiance'].std())
    print(wfm.sim_data['direct_horizontal_irradiance'].min())
    print(wfm.sim_data['direct_horizontal_irradiance'].max())


    wfm.adjust_variable_to_long_run_average(
            variable='direct_horizontal_irradiance',
            source_long_run_average=rk.weather.Era5Source.LONG_RUN_AVERAGE_DNI,
            real_long_run_average=rk.TEST_DATA['gsa-dni-like.tif'],
            real_lra_scaling=1000 / 24,  # cast to hourly average kWh
    )

    print('after LRA')
    print(wfm.sim_data['direct_horizontal_irradiance'].mean())
    print(wfm.sim_data['direct_horizontal_irradiance'].std())
    print(wfm.sim_data['direct_horizontal_irradiance'].min())
    print(wfm.sim_data['direct_horizontal_irradiance'].max())

    assert np.isclose(wfm.sim_data['direct_horizontal_irradiance'].mean(), 16.06680060520131)
    assert np.isclose(wfm.sim_data['direct_horizontal_irradiance'].std(), 45.92803832175536)
    assert np.isclose(wfm.sim_data['direct_horizontal_irradiance'].min(), 0.0)
    assert np.isclose(wfm.sim_data['direct_horizontal_irradiance'].max(), 257.65673305845195)
    


#%% test get_timesteps
def test_get_timesteps(pt_PTRWorkflowManager_loaded):
    wfm = pt_PTRWorkflowManager_loaded

    wfm.get_timesteps()

    assert wfm._numtimesteps == 140
    assert wfm._numlocations == 3
    
@pytest.fixture
def pt_PTRWorkflowManager_timesteps(pt_PTRWorkflowManager_loaded: PTRWorkflowManager) -> PTRWorkflowManager:
    wfm = pt_PTRWorkflowManager_loaded

    wfm.get_timesteps()

    return wfm

#%% test calcualte solar position
def test_calculateSolarPosition(pt_PTRWorkflowManager_timesteps):
    wfm = pt_PTRWorkflowManager_timesteps

    wfm.calculateSolarPosition()

    assert wfm.sim_data['solar_zenith_degree'].shape == (140, 3)

    assert np.isclose(wfm.sim_data['solar_zenith_degree'].mean(), 108.83433, rtol=0.1)
    assert np.isclose(wfm.sim_data['solar_zenith_degree'].std(), 26.46376)
    assert np.isclose(wfm.sim_data['solar_zenith_degree'].min(), 73.31861)
    assert np.isclose(wfm.sim_data['solar_zenith_degree'].max(), 152.12749)

    assert np.isclose(wfm.sim_data['solar_altitude_angle_degree'].mean(),-18.834325)
    assert np.isclose(wfm.sim_data['solar_altitude_angle_degree'].std(), 26.463765)
    assert np.isclose(wfm.sim_data['solar_altitude_angle_degree'].min(), -62.12750)
    assert np.isclose(wfm.sim_data['solar_altitude_angle_degree'].max(), 16.681382)

    assert np.isclose(wfm.sim_data['aoi_northsouth'].mean(),20.04813)
    assert np.isclose(wfm.sim_data['aoi_northsouth'].std(), 28.55629)
    assert np.isclose(wfm.sim_data['aoi_northsouth'].min(), 0)
    assert np.isclose(wfm.sim_data['aoi_northsouth'].max(), 74.30066)
    
    assert np.isclose(wfm.sim_data['aoi_eastwest'].mean(), 9.30592)
    assert np.isclose(wfm.sim_data['aoi_eastwest'].std(), 15.76567)
    assert np.isclose(wfm.sim_data['aoi_eastwest'].min(), 0)
    assert np.isclose(wfm.sim_data['aoi_eastwest'].max(), 51.09474)

@pytest.fixture
def pt_PTRWorkflowManager_solarpos(pt_PTRWorkflowManager_timesteps: PTRWorkflowManager) -> PTRWorkflowManager:
    wfm = pt_PTRWorkflowManager_timesteps

    wfm.calculateSolarPosition()

    return wfm


#%% test cosine losses

def test_calculateCosineLossesParabolicTrough(pt_PTRWorkflowManager_solarpos):
    wfm = pt_PTRWorkflowManager_solarpos

    wfm.calculateCosineLossesParabolicTrough(orientation='northsouth')

    assert np.isclose(wfm.sim_data['theta'].mean(), 20.04813)
    assert np.isclose(wfm.sim_data['theta'].std(), 28.55629)
    assert np.isclose(wfm.sim_data['theta'].min(), 0)
    assert np.isclose(wfm.sim_data['theta'].max(), 74.30066)


    wfm = pt_PTRWorkflowManager_solarpos

    wfm.calculateCosineLossesParabolicTrough(orientation='eastwest')
    assert np.isclose(wfm.sim_data['theta'].mean(), 9.30592)
    assert np.isclose(wfm.sim_data['theta'].std(), 15.76566)
    assert np.isclose(wfm.sim_data['theta'].min(), 0)
    assert np.isclose(wfm.sim_data['theta'].max(), 51.09474)

    wfm = pt_PTRWorkflowManager_solarpos

    wfm.calculateCosineLossesParabolicTrough(orientation='song2013')
    assert np.isclose(wfm.sim_data['theta'].mean(), 9.30592)
    assert np.isclose(wfm.sim_data['theta'].std(), 15.76566)
    assert np.isclose(wfm.sim_data['theta'].min(), 0)
    assert np.isclose(wfm.sim_data['theta'].max(), 51.09474)

@pytest.fixture
def pt_PTRWorkflowManager_cos_losses(pt_PTRWorkflowManager_solarpos: PTRWorkflowManager) -> PTRWorkflowManager:
    wfm = pt_PTRWorkflowManager_solarpos

    wfm.calculateCosineLossesParabolicTrough(orientation='northsouth')

    return wfm  

#%% test calculate IAM

def test_calculateIAM(pt_PTRWorkflowManager_cos_losses):
    wfm = pt_PTRWorkflowManager_cos_losses

    wfm.calculateIAM()

    assert np.isclose(wfm.sim_data['IAM'].mean(), 0.798079)
    assert np.isclose(wfm.sim_data['IAM'].std(), 0.354184)
    assert np.isclose(wfm.sim_data['IAM'].min(), -0.338124)
    assert np.isclose(wfm.sim_data['IAM'].max(), 1)


#%% test shadow losses

def test_calculateShadowLosses(pt_PTRWorkflowManager_cos_losses):
    wfm = pt_PTRWorkflowManager_cos_losses

    wfm.calculateShadowLosses(method='wagner2011', SF_density=0.43)

    assert np.isclose(wfm.sim_data['eta_shdw'].mean(), 0.678173)
    assert np.isclose(wfm.sim_data['eta_shdw'].std(), 0.329811)
    assert np.isclose(wfm.sim_data['eta_shdw'].min(), 0.00169546)
    assert np.isclose(wfm.sim_data['eta_shdw'].max(), 1)


#%% test windspeed losses


def test_calculateWindspeedLosses(pt_PTRWorkflowManager_cos_losses):
    wfm = pt_PTRWorkflowManager_cos_losses

    wfm.calculateWindspeedLosses(max_windspeed_threshold=9)

    assert np.isclose(wfm.sim_data['eta_wind'].mean(), 0.988095)
    assert np.isclose(wfm.sim_data['eta_wind'].std(), 0.108458)
    assert np.isclose(wfm.sim_data['eta_wind'].min(), 0)
    assert np.isclose(wfm.sim_data['eta_wind'].max(), 1)


#test calculateDegradationLosses

def test_calculateWindspeedLosses(pt_PTRWorkflowManager_cos_losses):
    wfm = pt_PTRWorkflowManager_cos_losses

    wfm.calculateDegradationLosses(efficencyDropPerYear=0.02, lifetime=20)

    assert np.isclose(wfm.sim_data['eta_degradation'], 0.8643604692000185)

    wfm.calculateDegradationLosses(efficencyDropPerYear=0, lifetime=20)

    assert np.isclose(wfm.sim_data['eta_degradation'], 1)


@pytest.fixture
def pt_PTRWorkflowManager_all_losses(pt_PTRWorkflowManager_cos_losses: PTRWorkflowManager) -> PTRWorkflowManager:
    wfm = pt_PTRWorkflowManager_cos_losses

    wfm.calculateIAM()
    wfm.calculateShadowLosses(method='wagner2011', SF_density=0.43)
    wfm.calculateWindspeedLosses(max_windspeed_threshold=9)
    wfm.calculateDegradationLosses(efficencyDropPerYear=0.02, lifetime=20)

    return wfm

def test_calculateHeattoHTF(pt_PTRWorkflowManager_all_losses):
    wfm = pt_PTRWorkflowManager_all_losses
    
    wfm.calculateHeattoHTF(A_aperture_sf=900000, eta_ptr_max=0.8, eta_cleaness=0.99)

    assert np.isclose(wfm.sim_data['HeattoHTF_W'].mean(), 476140.22531221766)
    assert np.isclose(wfm.sim_data['HeattoHTF_W'].std(), 1820765.586866662)
    assert np.isclose(wfm.sim_data['HeattoHTF_W'].min(), -3873281.9942555986)
    assert np.isclose(wfm.sim_data['HeattoHTF_W'].max(), 11149489.438679732)


@pytest.fixture
def pt_PTRWorkflowManager_heat_to_HTF(pt_PTRWorkflowManager_all_losses: PTRWorkflowManager) -> PTRWorkflowManager:
    wfm = pt_PTRWorkflowManager_all_losses

    wfm.calculateHeattoHTF(A_aperture_sf=900000, eta_ptr_max=0.8, eta_cleaness=0.99)

    return wfm

#%% test applyHTFHeatLossModel

def test_applyHTFHeatLossModel(pt_PTRWorkflowManager_heat_to_HTF):
    wfm = pt_PTRWorkflowManager_heat_to_HTF
    

    # without Jit
    wfm.applyHTFHeatLossModel(
        calculationmethod='dersch2018',
        params={'b': np.array([0, 0.02421, 2.46E-05, 1.56E-07, 1.17E-09]),
            'A': 900000,
            'relTMplant': 3688,
            'maxHTFTemperature': 385,
            'JITaccelerate': False,
            'minHTFTemperature': 60,
            'inletHTFTemperature': 295,
            'add_losses_coefficient': 0.108,
            'discretizationmethod': 'euler explicit'
            
            }
        )

    assert np.isclose(wfm.sim_data['HeattoPlant_W'].mean(), 0)
    assert np.isclose(wfm.sim_data['HeattoPlant_W'].std(), 0)
    assert np.isclose(wfm.sim_data['HeattoPlant_W'].min(), 0)
    assert np.isclose(wfm.sim_data['HeattoPlant_W'].max(), 0)

    assert np.isclose(wfm.sim_data['HTF_mean_temperature_C'].mean(), 60.717487604450795)
    assert np.isclose(wfm.sim_data['HTF_mean_temperature_C'].std(), 4.281117691774051)
    assert np.isclose(wfm.sim_data['HTF_mean_temperature_C'].min(), 60.0)
    assert np.isclose(wfm.sim_data['HTF_mean_temperature_C'].max(), 100.67782189397873)
    
    assert np.isclose(wfm.sim_data['Heat_Losses_W'].mean(), 7043595.44904212)
    assert np.isclose(wfm.sim_data['Heat_Losses_W'].std(), 863436.8328967596)
    assert np.isclose(wfm.sim_data['Heat_Losses_W'].min(), 0)
    assert np.isclose(wfm.sim_data['Heat_Losses_W'].max(), 12366000.0)

    assert np.isclose(wfm.sim_data['P_heating_W'].mean(), 6328075.957449375)
    assert np.isclose(wfm.sim_data['P_heating_W'].std(), 2043059.346128178)
    assert np.isclose(wfm.sim_data['P_heating_W'].min(), 0)
    assert np.isclose(wfm.sim_data['P_heating_W'].max(), 10651478.248763243)


    # with Jit
    wfm.applyHTFHeatLossModel(
        calculationmethod='dersch2018',
        params={'b': np.array([0, 0.02421, 2.46E-05, 1.56E-07, 1.17E-09]),
            'A': 900000,
            'relTMplant': 3688,
            'maxHTFTemperature': 385,
            'JITaccelerate': True,
            'minHTFTemperature': 60,
            'inletHTFTemperature': 295,
            'add_losses_coefficient': 0.108,
            'discretizationmethod': 'euler explicit'
            
            }
        )

    assert np.isclose(wfm.sim_data['HeattoPlant_W'].mean(), 0)
    assert np.isclose(wfm.sim_data['HeattoPlant_W'].std(), 0)
    assert np.isclose(wfm.sim_data['HeattoPlant_W'].min(), 0)
    assert np.isclose(wfm.sim_data['HeattoPlant_W'].max(), 0)

    assert np.isclose(wfm.sim_data['HTF_mean_temperature_C'].mean(), 60.717487604450795)
    assert np.isclose(wfm.sim_data['HTF_mean_temperature_C'].std(), 4.281117691774051)
    assert np.isclose(wfm.sim_data['HTF_mean_temperature_C'].min(), 60.0)
    assert np.isclose(wfm.sim_data['HTF_mean_temperature_C'].max(), 100.67782189397873)
    
    assert np.isclose(wfm.sim_data['Heat_Losses_W'].mean(), 7043595.44904212)
    assert np.isclose(wfm.sim_data['Heat_Losses_W'].std(), 863436.8328967596)
    assert np.isclose(wfm.sim_data['Heat_Losses_W'].min(), 0)
    assert np.isclose(wfm.sim_data['Heat_Losses_W'].max(), 12366000.0)



#TODO:
# - calculate parastitics
# - calculate economics



### Code to generate above:

# from reskit.solar.workflows.csp_workflow_manager import PTRWorkflowManager
# import reskit as rk
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# placements = pd.DataFrame()
# placements['lon'] = [ 6.083, 6.083, 5.583]     # Longitude
# placements['lat'] = [ 50.775, 51.475, 50.775,]    # Latitude


# wfm = PTRWorkflowManager(placements)
# wfm.read(
#     variables=["direct_horizontal_irradiance",
#                 "surface_air_temperature",
#                 "surface_wind_speed"],
#     source_type="ERA5",
#     source=rk.TEST_DATA['era5-like'],
#     set_time_index=True,
#     verbose=False)
# wfm.get_timesteps()
# wfm.apply_elevation(rk.TEST_DATA['DEM-like.tif'])
# wfm.calculateSolarPosition()
# wfm.calculateCosineLossesParabolicTrough(orientation='northsouth')
# wfm.calculateIAM()
# wfm.calculateShadowLosses(method='wagner2011', SF_density=0.43)
# wfm.calculateWindspeedLosses(max_windspeed_threshold=9)
# wfm.calculateDegradationLosses(efficencyDropPerYear=0.02, lifetime=20)
# wfm.calculateHeattoHTF(A_aperture_sf=900000, eta_ptr_max=0.8, eta_cleaness=0.99)
# wfm.applyHTFHeatLossModel(
#     calculationmethod='dersch2018',
#     params={'b': np.array([0, 0.02421, 2.46E-05, 1.56E-07, 1.17E-09]),
#         'A': 900000,
#         'relTMplant': 3688,
#         'maxHTFTemperature': 385,
#         'JITaccelerate': False,
#         'minHTFTemperature': 60,
#         'inletHTFTemperature': 295,
#         'add_losses_coefficient': 0.108,
#         'discretizationmethod': 'euler explicit'
        
#         }
#     )


# END OF CODE


### Useful function for generating test result values:

# def print_testresults(variable):
#     print('mean: ', variable[0:140,:].mean())
#     print('std: ', variable[0:140,:].std())
#     print('min: ', variable[0:140,:].min())
#     print('max: ', variable[0:140,:].max())