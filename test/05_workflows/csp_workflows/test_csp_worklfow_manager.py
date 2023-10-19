import pandas as pd
import numpy as np
from reskit.csp.workflows.csp_workflow_manager import PTRWorkflowManager
import reskit as rk
import pytest

# %% Test Init


def funct():

    placements = pd.DataFrame()
    placements['lon'] = [6.083, 6.083, 5.583]     # Longitude
    placements['lat'] = [50.775, 51.475, 50.775, ]    # Latitude
    placements['area'] = [1E6, 3E6, 6E6]

    datasetname = 'Initial'
    verbose = False
    era5_path = rk.TEST_DATA['era5-like']
    elev_path = rk.TEST_DATA['DEM-like.tif']
    global_solar_atlas_dni_path = rk.TEST_DATA['gsa-dni-like.tif']
    JITaccelerate = False

    wf = PTRWorkflowManager(placements)

    ptr_data = wf.loadPTRdata(datasetname=datasetname)
    wf.determine_area()

    # 3) read in Input data
    wf.read(
        variables=[  # "global_horizontal_irradiance",
            "direct_horizontal_irradiance",
            "surface_wind_speed",
            # "surface_pressure",
            "surface_air_temperature",
        ],
        source_type="ERA5",
        source=era5_path,
        set_time_index=True,
        verbose=verbose)

    # 4) get length of timesteps for later numpy sizing

    wf.get_timesteps()

    # apply elevation
    wf.apply_elevation(elev_path)
    wf.apply_azimuth()
    # 5) calculate the solar position based on pvlib

    wf.calculateSolarPosition()

    # calculate DNI from ERA5 to DNi convention
    # ERA5 DIN: Heat flux per horizontal plane
    # DNI convention: Heat flux per normal (to zenith) plane
    wf.direct_normal_irradiance_from_trigonometry()

    # do long run averaging for DNI
    if global_solar_atlas_dni_path == 'default_cluster':
        global_solar_atlas_dni_path = r"/storage/internal/data/gears/geography/irradiance/global_solar_atlas_v2.5/World_DNI_GISdata_LTAy_AvgDailyTotals_GlobalSolarAtlas-v2_GEOTIFF/DNI.tif"
    if global_solar_atlas_dni_path == 'default_local':
        global_solar_atlas_dni_path = r"R:\data\gears\geography\irradiance\global_solar_atlas_v2.5\World_DNI_GISdata_LTAy_AvgDailyTotals_GlobalSolarAtlas-v2_GEOTIFF\DNI.tif"

    # TODO: implement if working
    # if global_solar_atlas_dni_path != None:
    #     wf.adjust_variable_to_long_run_average(
    #         variable='direct_horizontal_irradiance',
    #         source_long_run_average=rk_weather.Era5Source.LONG_RUN_AVERAGE_DNI,
    #         real_long_run_average=global_solar_atlas_dni_path,
    #         real_lra_scaling=1000 / 24,  # cast to hourly average kWh
    # )

    # 6) doing selfmade calulations until Heat to HTF
    # wf.calculateCosineLossesParabolicTrough(orientation=ptr_data['orientation']) shifted
    wf.calculateIAM(a1=ptr_data['a1'], a2=ptr_data['a2'], a3=ptr_data['a3'])
    wf.calculateShadowLosses(
        method='wagner2011', SF_density=ptr_data['SF_density_direct'])
    wf.calculateWindspeedLosses(
        max_windspeed_threshold=ptr_data['maxWindspeed'])
    wf.calculateDegradationLosses(
        efficencyDropPerYear=ptr_data['efficencyDropPerYear'], lifetime=ptr_data['lifetime'])
    wf.calculateHeattoHTF(
        eta_ptr_max=ptr_data['eta_ptr_max'], eta_cleaness=ptr_data['eta_cleaness'])

    # wf.apply_capacity()

    # 7) calculation heat to plant with loss model
    wf.applyHTFHeatLossModel(
        calculationmethod='dersch2018',
        params={'b': ptr_data['b'],
                'relTMplant': ptr_data['relTMplant'],
                'maxHTFTemperature': ptr_data['maxHTFTemperature'],
                'JITaccelerate': JITaccelerate,
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

    # 9) calculate economics
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


def print_testresults(variable):
    print('mean: ', variable.mean())
    print('std: ', variable.std())
    print('min: ', variable.min())
    print('max: ', variable.max())


####################################
#####       TEST Init         ######
####################################
def test_PTRWorkflowManager__init__() -> PTRWorkflowManager:

    placements = pd.DataFrame()
    placements['lon'] = [6.083, 6.083, 5.583]     # Longitude
    placements['lat'] = [50.775, 51.475, 50.775, ]    # Latitude
    placements['area'] = [1E6, 3E6, 6E6]

    wfm = PTRWorkflowManager(placements=placements)

    assert np.isclose(wfm.ext.xMin, 5.58300)
    assert np.isclose(wfm.ext.xMax, 6.083000)
    assert np.isclose(wfm.ext.yMin, 50.775000)
    assert np.isclose(wfm.ext.yMax, 51.475000)

    assert (wfm.placements['lon'] == placements['lon']).all()
    assert (wfm.placements['lat'] == placements['lat']).all()

    return wfm


####################################
#####  TEST data loading      ######
####################################
@pytest.fixture
def pt_PTRWorkflowManager_initialized() -> PTRWorkflowManager:
    return test_PTRWorkflowManager__init__()


# load ptr data
def test_loadPTRdata(pt_PTRWorkflowManager_initialized):
    wfm = pt_PTRWorkflowManager_initialized

    ptr_data = wfm.loadPTRdata(datasetname='Initial')

    assert hasattr(wfm, 'ptr_data')
    # assert ptr_data.shape  == (45,)
    assert ptr_data['orientation'] == 'song2013'
    assert ptr_data['a1'] == 0.000884
    assert ptr_data['discretizationmethod'] == 'euler explicit'
    assert ptr_data['SF_density_total'] == 0.383

# determine area


def test_determine_area(pt_PTRWorkflowManager_initialized):
    wfm = pt_PTRWorkflowManager_initialized

    ptr_data = wfm.loadPTRdata(datasetname='Initial')
    assert ptr_data['SF_density_total'] == 0.383

    wfm.determine_area()

    assert 'aperture_area_m2' in wfm.placements.columns
    assert 'land_area_m2' in wfm.placements.columns

    assert np.isclose(
        wfm.placements['land_area_m2'].mean(), 3333333.3333333335)
    assert np.isclose(wfm.placements['land_area_m2'].std(), 2516611.4784235833)
    assert np.isclose(
        wfm.placements['aperture_area_m2'].mean(), 1276666.6666666667)
    assert np.isclose(
        wfm.placements['aperture_area_m2'].std(), 963862.1962362323)

# @pytest.fixture
# def pt_PTRWorkflowManager_loaded() -> PTRWorkflowManager:

#     return test_determine_area()
# elevation


def test_apply_elevation(pt_PTRWorkflowManager_initialized):
    wfm = pt_PTRWorkflowManager_initialized

    # load from file
    wfm.apply_elevation(rk.TEST_DATA['DEM-like.tif'])

    assert np.isclose(wfm.placements['elev'].tolist(), [185, 22, 118]).all()

    # prevent reload
    wfm.placements['elev'] = [1, 2, 4]
    wfm.apply_elevation(rk.TEST_DATA['DEM-like.tif'])

    assert np.isclose(wfm.placements['elev'].tolist(), [1, 2, 4]).all()

    # elev from number
    wfm = pt_PTRWorkflowManager_initialized
    wfm.placements.drop(columns=['elev'], inplace=True)
    wfm.apply_elevation(11)
    print(wfm.placements)
    assert np.isclose(wfm.placements['elev'].tolist(), [11, 11, 11]).all()


# test apply azimuth
def test_apply_azimuth(pt_PTRWorkflowManager_initialized):
    wfm = pt_PTRWorkflowManager_initialized

    wfm.ptr_data = pd.Series()
    wfm.ptr_data['orientation'] = 'northsouth'
    wfm.apply_azimuth()
    assert np.isclose(wfm.placements['azimuth'].mean(), 180)
    assert np.isclose(wfm.placements['azimuth'].std(), 0)
    assert np.isclose(wfm.placements['azimuth'].min(), 180)
    assert np.isclose(wfm.placements['azimuth'].max(), 180)

    wfm.ptr_data['orientation'] = 'eastwest'
    wfm.apply_azimuth()
    assert np.isclose(wfm.placements['azimuth'].mean(), 90)
    assert np.isclose(wfm.placements['azimuth'].std(), 0)
    assert np.isclose(wfm.placements['azimuth'].min(), 90)
    assert np.isclose(wfm.placements['azimuth'].max(), 90)

    wfm.ptr_data['orientation'] = 'song2013'
    wfm.placements.loc[0, 'lat'] = 30
    wfm.apply_azimuth()
    assert np.isclose(wfm.placements['azimuth'].mean(), 120)
    assert np.isclose(wfm.placements['azimuth'].std(), 51.96152422706632)
    assert np.isclose(wfm.placements['azimuth'].min(), 90)
    assert np.isclose(wfm.placements['azimuth'].max(), 180)

# test read ERA5


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

    assert np.isclose(
        wfm.sim_data['direct_horizontal_irradiance'].mean(), 15.15181)
    assert np.isclose(
        wfm.sim_data['direct_horizontal_irradiance'].std(), 33.92488)
    assert np.isclose(wfm.sim_data['direct_horizontal_irradiance'].min(), 0)
    assert np.isclose(
        wfm.sim_data['direct_horizontal_irradiance'].max(), 162.14436)


####################################
#####  TEST determine time steps####
####################################

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
    return wfm


def test_get_timesteps(pt_PTRWorkflowManager_loaded):
    wfm = pt_PTRWorkflowManager_loaded

    wfm.get_timesteps()

    assert wfm._numtimesteps == 140
    assert wfm._numlocations == 3


####################################
#####  TEST solar position    ######
####################################

@pytest.fixture
def pt_PTRWorkflowManager_solarpos() -> PTRWorkflowManager:
    wfm = test_PTRWorkflowManager__init__()
    wfm.placements['azimuth'] = [90, 180, 180]
    wfm.placements['elev'] = [90, 180, 180]
    wfm.time_index = pd.date_range(
        "2014-12-31 23:30:00", periods=100, freq="H")
    wfm.get_timesteps()
    wfm.ptr_data = pd.Series()
    wfm.ptr_data['SF_density_direct'] = 0.383
    return wfm

# test calcualte solar position


def test_calculateSolarPosition(pt_PTRWorkflowManager_solarpos):
    wfm = pt_PTRWorkflowManager_solarpos
    wfm.calculateSolarPosition()

    assert wfm.sim_data['solar_zenith_degree'].shape == (100, 3)

    print_testresults(wfm.sim_data['solar_zenith_degree'])
    print_testresults(wfm.sim_data['solar_altitude_angle_degree'])
    print_testresults(wfm.sim_data['theta'])
    print_testresults(wfm.sim_data['tracking_angle'])

    assert np.isclose(
        wfm.sim_data['solar_zenith_degree'].mean(), 111.28957246057364)
    assert np.isclose(
        wfm.sim_data['solar_zenith_degree'].std(), 27.195085020417622)
    assert np.isclose(
        wfm.sim_data['solar_zenith_degree'].min(), 73.4811943930343)
    assert np.isclose(
        wfm.sim_data['solar_zenith_degree'].max(), 152.2150485902235)

    assert np.isclose(
        wfm.sim_data['solar_altitude_angle_degree'].mean(), -21.28957246057363)
    assert np.isclose(
        wfm.sim_data['solar_altitude_angle_degree'].std(), 27.195085020417622)
    assert np.isclose(
        wfm.sim_data['solar_altitude_angle_degree'].min(), -62.21504859022348)
    assert np.isclose(
        wfm.sim_data['solar_altitude_angle_degree'].max(), 16.518805606965696)

    assert np.isclose(wfm.sim_data['theta'].mean(), 15.392752844414987)
    assert np.isclose(wfm.sim_data['theta'].std(), 25.06289216984148)
    assert np.isclose(wfm.sim_data['theta'].min(), 0)
    assert np.isclose(wfm.sim_data['theta'].max(), 74.30174498418555)

    assert np.isclose(wfm.sim_data['tracking_angle'].mean(), 9.898689058532645)
    assert np.isclose(wfm.sim_data['tracking_angle'].std(), 38.089414877200646)
    assert np.isclose(wfm.sim_data['tracking_angle'].min(), -82.74319730966418)
    assert np.isclose(wfm.sim_data['tracking_angle'].max(), 89.34864532943253)


@pytest.mark.skip(reason="Function not used atm")
def test_calculateSolarPositionfaster(pt_PTRWorkflowManager_solarpos):
    wfm = pt_PTRWorkflowManager_solarpos

    wfm.calculateSolarPositionfaster()
    solar_zenith_degree_fast = wfm.sim_data['solar_zenith_degree']
    solar_altitude_angle_degree_fast = wfm.sim_data['solar_altitude_angle_degree']
    theta_fast = wfm.sim_data['theta']
    stracking_angle_fast = wfm.sim_data['tracking_angle']
    wfm.calculateSolarPosition()
    assert np.isclose(solar_zenith_degree_fast,
                      wfm.sim_data['solar_zenith_degree'])
    assert np.isclose(solar_altitude_angle_degree_fast,
                      wfm.sim_data['solar_altitude_angle_degree'])
    assert np.isclose(theta_fast, wfm.sim_data['theta'])
    assert np.isclose(stracking_angle_fast, wfm.sim_data['tracking_angle'])
    assert wfm.sim_data['solar_zenith_degree_fast'].shape == (140, 3)


####################################
#####  TEST dni_correction    ######
####################################

@pytest.fixture
def pt_PTRWorkflowManager_dni_from_trig(pt_PTRWorkflowManager_loaded: PTRWorkflowManager) -> PTRWorkflowManager:
    wfm = pt_PTRWorkflowManager_loaded

    wfm.ptr_data = [90, 180, 180]
    wfm.ptr_data = [90, 180, 180]
    wfm.get_timesteps()
    wfm.ptr_data = pd.Series()
    wfm.ptr_data['SF_density_direct'] = 0.383
    wfm.placements['elev'] = 0
    wfm.placements['azimuth'] = 180

    # include this heres, as this is a mainly pv lib function
    wfm.calculateSolarPosition()
    return wfm

# test dni calculation


def test_direct_normal_irradiance_from_trigonometry(pt_PTRWorkflowManager_dni_from_trig):
    wfm = pt_PTRWorkflowManager_dni_from_trig

    wfm.direct_normal_irradiance_from_trigonometry()

    assert 'direct_normal_irradiance' in wfm.sim_data.keys()
    assert wfm.sim_data['direct_normal_irradiance'].shape == (140, 3)
    print_testresults(wfm.sim_data['direct_normal_irradiance'])
    assert np.isclose(
        wfm.sim_data['direct_normal_irradiance'].mean(), 67.92523701209932)
    assert np.isclose(
        wfm.sim_data['direct_normal_irradiance'].std(), 140.0979306213717)
    assert np.isclose(wfm.sim_data['direct_normal_irradiance'].min(), 0.0)
    assert np.isclose(
        wfm.sim_data['direct_normal_irradiance'].max(), 583.4938777937742)


####################################
#####  TEST adjust_variable_to_long_run_average    ######
####################################

@pytest.fixture
def pt_PTRWorkflowManager_LRA(pt_PTRWorkflowManager_loaded: PTRWorkflowManager) -> PTRWorkflowManager:
    wfm = pt_PTRWorkflowManager_loaded

    return wfm

# test Long rung averaging


@pytest.mark.skip(reason="Function not used atm")
def test_adjust_variable_to_long_run_average(pt_PTRWorkflowManager_solarpos):
    wfm = pt_PTRWorkflowManager_solarpos

    print('before LRA')
    assert np.isclose(
        wfm.sim_data['direct_horizontal_irradiance'].mean(), 15.151816929634485)
    assert np.isclose(
        wfm.sim_data['direct_horizontal_irradiance'].std(), 33.92488851386296)
    assert np.isclose(wfm.sim_data['direct_horizontal_irradiance'].min(), 0.0)
    assert np.isclose(
        wfm.sim_data['direct_horizontal_irradiance'].max(), 162.14436231574447)

    wfm.adjust_variable_to_long_run_average(
        variable='direct_horizontal_irradiance',
        source_long_run_average=rk.weather.Era5Source.LONG_RUN_AVERAGE_DNI,
        real_long_run_average=rk.TEST_DATA['gsa-dni-like.tif'],
        real_lra_scaling=1000 / 24,  # cast to hourly average kWh
    )

    print('after LRA')
    print(wfm.sim_data['direct_normal_irradiance'].mean())
    print(wfm.sim_data['direct_normal_irradiance'].std())
    print(wfm.sim_data['direct_normal_irradiance'].min())
    print(wfm.sim_data['direct_normal_irradiance'].max())

    assert np.isclose(
        wfm.sim_data['direct_normal_irradiance'].mean(), 16.06680060520131)
    assert np.isclose(
        wfm.sim_data['direct_normal_irradiance'].std(), 45.902557468831624)
    assert np.isclose(wfm.sim_data['direct_normal_irradiance'].min(), 0.0)
    assert np.isclose(
        wfm.sim_data['direct_normal_irradiance'].max(), 257.65673305845195)


@pytest.mark.skip(reason="Function not used atm")
def test_calculateCosineLossesParabolicTrough(pt_PTRWorkflowManager_DNI):
    wfm = pt_PTRWorkflowManager_DNI

    wfm.calculateCosineLossesParabolicTrough(orientation='northsouth')

    assert np.isclose(wfm.sim_data['theta'].mean(), 20.048007412410186)
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


# %% test calculate IAM

####################################
#####  TEST IAM_estimate    ######
####################################

@pytest.fixture
def pt_PTRWorkflowManager_IAM() -> PTRWorkflowManager:
    wfm = test_PTRWorkflowManager__init__()
    wfm.placements['azimuth'] = [90, 180, 180]
    wfm.placements['elev'] = [90, 180, 180]
    wfm.time_index = pd.date_range(
        "2014-12-31 23:30:00", periods=100, freq="H")
    wfm.get_timesteps()
    wfm.sim_data['ptr_data'] = pd.Series()
    wfm.sim_data['ptr_data']['SF_density_direct'] = 0.383
    wfm.sim_data['theta'] = theta_test
    return wfm


def test_calculateIAM(pt_PTRWorkflowManager_IAM):
    wfm = pt_PTRWorkflowManager_IAM
    print(wfm.sim_data.keys())
    wfm.calculateIAM(a1=0.000884, a2=0.00005369, a3=0)
    print_testresults(wfm.sim_data['IAM'])
    assert np.isclose(wfm.sim_data['IAM'].mean(), 0.8697242301082351)
    assert np.isclose(wfm.sim_data['IAM'].std(), 0.27154198202026364)
    assert np.isclose(wfm.sim_data['IAM'].min(), 0)
    assert np.isclose(wfm.sim_data['IAM'].max(), 1)


####################################
#####  TEST shadow_losses     ######
####################################

@pytest.fixture
def pt_PTRWorkflowManager_Shadow() -> PTRWorkflowManager:
    wfm = test_PTRWorkflowManager__init__()
    wfm.sim_data['tracking_angle'] = tracking_angle_test
    wfm.sim_data['solar_zenith_degree'] = zenith_test
    return wfm


def test_calculateShadowLosses(pt_PTRWorkflowManager_Shadow):
    wfm = pt_PTRWorkflowManager_Shadow

    wfm.calculateShadowLosses(method='wagner2011', SF_density=0.43)
    print_testresults(wfm.sim_data['eta_shdw'])
    assert np.isclose(wfm.sim_data['eta_shdw'].mean(), 0.20020204625315727)
    assert np.isclose(wfm.sim_data['eta_shdw'].std(), 0.3461275222093702)
    assert np.isclose(wfm.sim_data['eta_shdw'].min(), 0.0)
    assert np.isclose(wfm.sim_data['eta_shdw'].max(), 1)


####################################
#####  TEST windspeed_losses  ######
####################################

def test_calculateWindspeedLosses(pt_PTRWorkflowManager_loaded):
    wfm = pt_PTRWorkflowManager_loaded

    wfm.calculateWindspeedLosses(max_windspeed_threshold=9)

    print_testresults(wfm.sim_data['eta_wind'])
    assert np.isclose(wfm.sim_data['eta_wind'].mean(), 0.988095)
    assert np.isclose(wfm.sim_data['eta_wind'].std(), 0.108458)
    assert np.isclose(wfm.sim_data['eta_wind'].min(), 0)
    assert np.isclose(wfm.sim_data['eta_wind'].max(), 1)


####################################
#####  TEST degradation_losses######
####################################

def test_calculateDegradationLosses(pt_PTRWorkflowManager_initialized):
    wfm = pt_PTRWorkflowManager_initialized

    wfm.calculateDegradationLosses(efficencyDropPerYear=0.02, lifetime=20)

    assert np.isclose(wfm.sim_data['eta_degradation'], 0.8643604692000185)

    wfm.calculateDegradationLosses(efficencyDropPerYear=0, lifetime=20)

    assert np.isclose(wfm.sim_data['eta_degradation'], 1)


####################################
#####  TEST Heat to HTF       ######
####################################
@pytest.fixture
def pt_PTRWorkflowManager_HeattoHTF(pt_PTRWorkflowManager_initialized) -> PTRWorkflowManager:
    wfm = pt_PTRWorkflowManager_initialized

    wfm.eta_ptr_max = 0.827
    wfm.eta_cleaness = 0.9
    wfm.sim_data['theta'] = theta_test
    wfm.sim_data['IAM'] = 0.8*np.ones_like(theta_test)
    wfm.sim_data['eta_shdw'] = 0.5*np.ones_like(theta_test)
    wfm.sim_data['eta_wind'] = 0.7*np.ones_like(theta_test)
    wfm.sim_data['eta_degradation'] = 0.99*np.ones_like(theta_test)
    wfm.placements['aperture_area_m2'] = [1E5, 1E6, 1E7]
    wfm.sim_data['direct_normal_irradiance'] = dni_test[:100, :]
    return wfm


def test_calculateHeattoHTF(pt_PTRWorkflowManager_HeattoHTF):
    wfm = pt_PTRWorkflowManager_HeattoHTF

    wfm.calculateHeattoHTF(eta_ptr_max=0.8, eta_cleaness=0.99, eta_other=1)

    print_testresults(wfm.sim_data['HeattoHTF_W'])
    assert np.isclose(wfm.sim_data['HeattoHTF_W'].mean(), 17292906.612040244)
    assert np.isclose(wfm.sim_data['HeattoHTF_W'].std(), 65028464.614948)
    assert np.isclose(wfm.sim_data['HeattoHTF_W'].min(), 0.0)
    assert np.isclose(wfm.sim_data['HeattoHTF_W'].max(), 449926470.13876075)


####################################
#####  TEST Heat_loss_model   ######
####################################
@pytest.fixture
def pt_PTRWorkflowManager_heat_loss() -> PTRWorkflowManager:

    placements = pd.DataFrame()
    placements['lon'] = [6.083, 6.083, 5.583]     # Longitude
    placements['lat'] = [50.775, 51.475, 50.775, ]    # Latitude
    placements['area'] = [1E6, 3E6, 6E6]

    datasetname = 'Initial'
    verbose = False
    era5_path = rk.TEST_DATA['era5-like']
    elev_path = rk.TEST_DATA['DEM-like.tif']

    wf = PTRWorkflowManager(placements)

    ptr_data = wf.loadPTRdata(datasetname=datasetname)
    wf.determine_area()

    wf.read(
        variables=[  # "global_horizontal_irradiance",
            "direct_horizontal_irradiance",
            "surface_wind_speed",
            # "surface_pressure",
            "surface_air_temperature",
        ],
        source_type="ERA5",
        source=era5_path,
        set_time_index=True,
        verbose=verbose)

    wf.get_timesteps()
    wf.apply_elevation(elev_path)
    wf.apply_azimuth()
    wf.calculateSolarPosition()
    wf.direct_normal_irradiance_from_trigonometry()

    wf.calculateIAM(a1=ptr_data['a1'], a2=ptr_data['a2'], a3=ptr_data['a3'])
    wf.calculateShadowLosses(
        method='wagner2011', SF_density=ptr_data['SF_density_direct'])
    wf.calculateWindspeedLosses(
        max_windspeed_threshold=ptr_data['maxWindspeed'])
    wf.calculateDegradationLosses(
        efficencyDropPerYear=ptr_data['efficencyDropPerYear'], lifetime=ptr_data['lifetime'])
    wf.calculateHeattoHTF(
        eta_ptr_max=ptr_data['eta_ptr_max'], eta_cleaness=ptr_data['eta_cleaness'], eta_other=ptr_data['eta_other'])

    # wf.apply_capacity()

    return wf

# test applyHTFHeatLossModel


def test_applyHTFHeatLossModel(pt_PTRWorkflowManager_heat_loss):
    wfm = pt_PTRWorkflowManager_heat_loss

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

    print_testresults(wfm.sim_data['HeattoPlant_W'])
    print_testresults(wfm.sim_data['HTF_mean_temperature_C'])
    print_testresults(wfm.sim_data['Heat_Losses_W'])
    print_testresults(wfm.sim_data['P_heating_W'])

    assert np.isclose(wfm.sim_data['HeattoPlant_W'].mean(), 10515869.951152233)
    assert np.isclose(wfm.sim_data['HeattoPlant_W'].std(), 51227401.07501093)
    assert np.isclose(wfm.sim_data['HeattoPlant_W'].min(), 0)
    assert np.isclose(wfm.sim_data['HeattoPlant_W'].max(), 535099281.5349062)

    assert np.isclose(
        wfm.sim_data['HTF_mean_temperature_C'].mean(), 133.1105765337709)
    assert np.isclose(
        wfm.sim_data['HTF_mean_temperature_C'].std(), 97.85535073399551)
    assert np.isclose(wfm.sim_data['HTF_mean_temperature_C'].min(), 60.0)
    assert np.isclose(wfm.sim_data['HTF_mean_temperature_C'].max(), 340.0)

    assert np.isclose(wfm.sim_data['Heat_Losses_W'].mean(), 27364332.885397065)
    assert np.isclose(wfm.sim_data['Heat_Losses_W'].std(), 34882077.637652025)
    assert np.isclose(wfm.sim_data['Heat_Losses_W'].min(), 0)
    assert np.isclose(wfm.sim_data['Heat_Losses_W'].max(), 157783484.47209877)

    assert np.isclose(wfm.sim_data['P_heating_W'].mean(), 4145809.1053530034)
    assert np.isclose(wfm.sim_data['P_heating_W'].std(), 6311097.60766845)
    assert np.isclose(wfm.sim_data['P_heating_W'].min(), 0)
    assert np.isclose(wfm.sim_data['P_heating_W'].max(), 19115325.849862307)

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

    assert np.isclose(wfm.sim_data['HeattoPlant_W'].mean(), 10515869.951152233)
    assert np.isclose(wfm.sim_data['HeattoPlant_W'].std(), 51227401.07501093)
    assert np.isclose(wfm.sim_data['HeattoPlant_W'].min(), 0)
    assert np.isclose(wfm.sim_data['HeattoPlant_W'].max(), 535099281.5349062)

    assert np.isclose(
        wfm.sim_data['HTF_mean_temperature_C'].mean(), 133.1105765337709)
    assert np.isclose(
        wfm.sim_data['HTF_mean_temperature_C'].std(), 97.85535073399551)
    assert np.isclose(wfm.sim_data['HTF_mean_temperature_C'].min(), 60.0)
    assert np.isclose(wfm.sim_data['HTF_mean_temperature_C'].max(), 340.0)

    assert np.isclose(wfm.sim_data['Heat_Losses_W'].mean(), 27364332.885397065)
    assert np.isclose(wfm.sim_data['Heat_Losses_W'].std(), 34882077.637652025)
    assert np.isclose(wfm.sim_data['Heat_Losses_W'].min(), 0)
    assert np.isclose(wfm.sim_data['Heat_Losses_W'].max(), 157783484.47209877)

    assert np.isclose(wfm.sim_data['P_heating_W'].mean(), 4145809.1053530034)
    assert np.isclose(wfm.sim_data['P_heating_W'].std(), 6311097.60766845)
    assert np.isclose(wfm.sim_data['P_heating_W'].min(), 0)
    assert np.isclose(wfm.sim_data['P_heating_W'].max(), 19115325.849862307)


####################################
#####  TEST Economics   ######
####################################

# test applyHTFHeatLossModel
def test_get_capex(pt_PTRWorkflowManager_initialized):
    wfm = pt_PTRWorkflowManager_initialized

    CAPEX_total_EUR = wfm._get_capex(
        A_aperture_m2=3E5,
        A_land_m2=1E6,
        Qdot_field_des_W=3E5*0.8*900,
        eta_des_power_plant=0.4,
        sm=2,
        tes=12,
        c_field_per_aperture_area_EUR_per_m2=165.44,
        c_land_per_land_area_EUR_per_m2=0.88,
        c_storage_EUR_per_kWh_th=25.52,
        c_plant_EUR_per_kW_el=1003.2,
        c_indirect_cost_perc_per_direct_Capex=11,
    )
    assert np.isclose(CAPEX_total_EUR, 140885817.6)


def test_get_opex(pt_PTRWorkflowManager_initialized):
    wfm = pt_PTRWorkflowManager_initialized

    OPEX_EUR_per_a = wfm._get_opex(
        CAPEX_total_EUR=140885817.6,
        OPEX_fix_perc_CAPEX_per_a=2,
        auxilary_power_Wh_per_a=0,
        electricity_price_EUR_per_kWh=0.05
    )
    assert np.isclose(OPEX_EUR_per_a, 2.817716352E6)

    OPEX_EUR_per_a = wfm._get_opex(
        CAPEX_total_EUR=140885817.6,
        OPEX_fix_perc_CAPEX_per_a=2,
        auxilary_power_Wh_per_a=4.830819e+10,
        electricity_price_EUR_per_kWh=0.05
    )
    assert np.isclose(OPEX_EUR_per_a, 5233125.852)


def test_get_totex(pt_PTRWorkflowManager_initialized):
    wfm = pt_PTRWorkflowManager_initialized

    TOTEX_EUR_per_a = wfm._get_totex(
        CAPEX_total_EUR_per_a=10,
        OPEX_EUR_per_a=2,
    )
    assert np.isclose(TOTEX_EUR_per_a, 12)


@pytest.fixture
def pt_PTRWorkflowManager_economics() -> PTRWorkflowManager:
    wfm = test_PTRWorkflowManager__init__()

    wfm.placements['aperture_area_m2'] = [3E5, 6E5, 3E5]
    wfm.placements['land_area_m2'] = [1E6, 2E6, 1E6]
    wfm.placements['capacity_sf_W_th'] = [
        3E5*0.8*900, 6E5*0.8*900, 3E5*0.8*900]
    wfm.placements['sm_opt'] = [2, 2, 2]
    wfm.placements['tes_opt'] = [12, 12, 12]

    if not hasattr(wfm, 'sim_data'):
        wfm.sim_data = {}

    wfm.sim_data['annuity'] = 0.093678779
    a = 4.830819e+10

    if not hasattr(wfm, 'sim_data_daily'):
        wfm.sim_data_daily = {}
    wfm.sim_data_daily['P_backup_heating_daily_Wh_el'] = np.array(
        [[a, a*2, a], [0, 0, 0]])

    if not hasattr(wfm, 'ptr_data'):
        wfm.ptr_data = {}
    wfm.ptr_data['eta_powerplant_1'] = 0.4
    wfm.ptr_data['CAPEX_solar_field_EUR_per_m^2_aperture'] = 165.44
    wfm.ptr_data['CAPEX_land_EUR_per_m^2_land'] = 0.88
    wfm.ptr_data['CAPEX_storage_cost_EUR_per_kWh'] = 25.52
    wfm.ptr_data['CAPEX_plant_cost_EUR_per_kW'] = 1003.2
    wfm.ptr_data['CAPEX_indirect_cost_perc_CAPEX'] = 11
    wfm.ptr_data['OPEX_perc_CAPEX'] = 2
    wfm.ptr_data['electricity_price_EUR_per_kWh'] = 0.05

    return wfm


def test_get_totex_from_self(pt_PTRWorkflowManager_economics):
    wfm = pt_PTRWorkflowManager_economics

    TOTEX_EUR_per_a = wfm._get_totex_from_self()
    # use those values, if 'Parasitics_solarfield_W_el' are bought as varOPEX
    assert np.isclose(TOTEX_EUR_per_a.values, [
                      18431137.22318471, 36862274.44636942, 18431137.22318471]).all()
    # assert np.isclose(TOTEX_EUR_per_a.values, [16.01572773E6, 2*16.01572773E6, 16.01572773E6]).all() # use those values, if 'Parasitics_solarfield_W_el' arent bought as varOPEX

    TOTEX_EUR_per_a = wfm._get_totex_from_self(
        sm_manipulation=1, tes_manipulation=10)
    # use those values, if 'Parasitics_solarfield_W_el' are bought as varOPEX
    assert np.isclose(TOTEX_EUR_per_a.values, [
                      26681959.73652098, 53363919.47304196, 26681959.73652098]).all()
    # assert np.isclose(TOTEX_EUR_per_a.values, [24266550.23652098, 48533100.47304196, 24266550.23652098]).all() # use those values, if 'Parasitics_solarfield_W_el' arent bought as varOPEX

    # Test from Excel sheet. cannot calculate parasitic losses
    wfm.sim_data_daily['P_backup_heating_daily_Wh_el'] = np.array(
        [[0, 0, 0], [0, 0, 0]])
    TOTEX_EUR_per_a = wfm._get_totex_from_self()
    assert np.isclose(TOTEX_EUR_per_a.values, [
                      16.01572773E6, 2*16.01572773E6, 16.01572773E6]).all()


####################################
#####  TEST Parasitics    ######
####################################

@pytest.fixture
def pt_PTRWorkflowManager_parasitics() -> PTRWorkflowManager:
    wfm = test_PTRWorkflowManager__init__()
    wfm.ptr_data = {}
    wfm.ptr_data['eta_powerplant_1'] = 0.5
    wfm.placements['capacity_sf_W_th'] = 58E6
    wfm.placements['aperture_area_m2'] = 1E5
    wfm.sim_data['HeattoPlant_W'] = dni_test * 1E5 * 0.7
    wfm.sim_data['HeattoHTF_W'] = wfm.sim_data['HeattoPlant_W'] * \
        1E9  # do this big, as only the +- sign should be relevant
    return wfm


def test_calculateParasitics(pt_PTRWorkflowManager_parasitics):

    wfm = pt_PTRWorkflowManager_parasitics

    params_gafurov = {
        'PL_plant_fix': 0.0055,
        'PL_sf_track': 0.0026,
        'PL_sf_pumping': 0.05,
        'PL_plant_pumping': 0.003,
        'PL_plant_other': 0.019,
    }

    params_dersch = {
        'PL_sf_fixed_W_per_m^2_ap': 1.486,
        'PL_sf_pumping_W_per_m^2_ap': 8.3,
        'PL_plant_fix': 0.0055,
        'PL_plant_pumping': 0.003,
        'PL_plant_other': 0.019,
    }

    wfm.calculateParasitics(
        calculationmethod='dersch2018',
        params=params_dersch,
    )

    assert wfm.sim_data['Parasitics_W_el'].shape == (140, 3)
    assert np.isclose(
        wfm.sim_data['Parasitics_W_el'].mean(), 142859.24061607895)
    assert np.isclose(wfm.sim_data['Parasitics_W_el'].std(), 274684.3071016811)
    assert np.isclose(wfm.sim_data['Parasitics_W_el'].min(), 0)
    assert np.isclose(wfm.sim_data['Parasitics_W_el'].max(), 1183094.33567266)

    assert np.isclose(
        wfm.sim_data['Parasitics_solarfield_W_el'].mean(), 70349.0500996118)
    assert np.isclose(
        wfm.sim_data['Parasitics_solarfield_W_el'].std(), 127515.52803736902)
    assert np.isclose(wfm.sim_data['Parasitics_solarfield_W_el'].min(), 0)
    assert np.isclose(
        wfm.sim_data['Parasitics_solarfield_W_el'].max(), 560214.6209076601)

    a = np.array([9690569.61684967, 10162240.40550065,  9693791.01948664])
    assert np.isclose(
        wfm.placements['Parasitics_solarfield_Wh_el_per_a'].values, a).all()

    b = np.array([10113416.57878985, 10692104.222701,  9648759.21542535])
    assert np.isclose(
        wfm.placements['Parasitics_plant_Wh_el_per_a'].values, b).all()

    # assert wfm.sim_data['Parasitics_W_el'] = wfm.sim_data['Parasitics_solarfield_W_el'] + wfm.sim_data['Parasitics_plant_W_el']
    # wfm.placements['Parasitics_solarfield_Wh_el_per_a']
    # wfm.placements['Parasitics_plant_Wh_el_per_a']


####################################
#####  TEST calculateEconomics_SF  ######
####################################

@pytest.fixture
def pt_PTRWorkflowManager_economicsSF() -> PTRWorkflowManager:
    wfm = test_PTRWorkflowManager__init__()
    wfm.ptr_data = {}
    wfm.sim_data['HeattoPlant_W'] = dni_test * 1E5 * 0.7
    wfm.sim_data['Parasitics_solarfield_W_el'] = dni_test * 1E5 * 0.76 * 0.1
    wfm.placements['aperture_area_m2'] = 1E5
    wfm.placements['land_area_m2'] = 1E5 / 0.3

    wfm._time_index_ = pd.date_range(
        "2014-12-31 23:30:00", periods=100, freq="H")

    return wfm


def test_calculateEconomics_SolarField(pt_PTRWorkflowManager_economicsSF):
    wfm = pt_PTRWorkflowManager_economicsSF

    params = {
        'CAPEX_solar_field_EUR_per_m^2_aperture': 100,
        'CAPEX_land_EUR_per_m^2_land': 1,
        'CAPEX_indirect_cost_perc_CAPEX': 0.11,
        'electricity_price_EUR_per_kWh': 0.05,
        'OPEX_perc_CAPEX': 0.03,
    }

    wfm.calculateEconomics_SolarField(WACC=0.08,
                                      lifetime=30,
                                      calculationmethod='franzmann2021',
                                      params=params
                                      )

    assert 'annualHeatfromSF_Wh' in wfm.placements.columns
    assert 'CAPEX_SF_EUR' in wfm.placements.columns
    assert 'Totex_SF_EUR_per_a' in wfm.placements.columns
    assert 'LCO_Heat_SF_EURct_per_kWh' in wfm.placements.columns
    a = np.array([197.37246424, 186.73164379, 206.84040309])
    assert np.isclose(
        wfm.placements['LCO_Heat_SF_EURct_per_kWh'].values, a).all()


####################################
#####  TEST def test_optimize_plant_size  ######
####################################

@pytest.fixture
def pt_PTRWorkflowManager_optplant(pt_PTRWorkflowManager_economics) -> PTRWorkflowManager:
    wfm = pt_PTRWorkflowManager_economics

    wfm.placements['capacity_sf_W_th'] = 58E6
    dni = np.tile(dni_test, [63, 1])[0:8760, :]
    wfm.sim_data['HeattoPlant_W'] = dni * 1E5 * 0.7
    wfm.sim_data['P_heating_W'] = (dni == 0) * 1E5
    wfm.sim_data['solar_zenith_degree'] = np.tile(
        zenith_test, [88, 1])[0:8760, :]
    wfm.sim_data['Parasitics_W_el'] = dni * 1E5 * 0.7 * 0.03
    wfm.ptr_data['storage_efficiency_1'] = 0.99
    wfm.ptr_data['eta_powerplant_1'] = 0.4
    # wfm.sim_data['Parasitics_solarfield_W_el'] = dni_test * 1E5 * 0.76 * 0.1
    # wfm.placements['aperture_area_m2'] = 1E5
    # wfm.placements['land_area_m2'] = 1E5 / 0.3

    wfm.time_index = pd.date_range(
        "2015-01-01 00:30:00", periods=8760, freq="H")

    return wfm


def test_optimize_plant_size(pt_PTRWorkflowManager_optplant):
    wfm = pt_PTRWorkflowManager_optplant

    debug_vars = False

    # case 1:
    onlynightuse = True
    fullvariation = False
    wfm.optimize_plant_size(onlynightuse=onlynightuse,
                            fullvariation=fullvariation, debug_vars=debug_vars)

    a = np.array([3.5, 3.5, 3.5])
    b = np.array([15, 12, 15])
    assert (wfm.placements['sm_opt'].values == a).all()
    assert (wfm.placements['tes_opt'].values == b).all()

    # case 2:
    onlynightuse = False
    fullvariation = False
    wfm.optimize_plant_size(onlynightuse=onlynightuse,
                            fullvariation=fullvariation, debug_vars=debug_vars)

    a = np.array([4.5, 4.5, 4.5])
    b = np.array([12, 9, 9])
    assert (wfm.placements['sm_opt'].values == a).all()
    assert (wfm.placements['tes_opt'].values == b).all()


####################################
#####  TEST calculate_electrical_output  ######
####################################

@pytest.fixture
def pt_PTRWorkflowManager_calcElecOut() -> PTRWorkflowManager:
    wfm = test_PTRWorkflowManager__init__()

    dni = np.tile(dni_test, [63, 1])[0:8760, :]
    wfm.sim_data['HeattoPlant_W'] = dni * 1E5 * 0.7
    wfm.sim_data['P_heating_W'] = (dni == 0) * 1E5
    wfm.sim_data['Parasitics_W_el'] = dni * 1E5 * 0.7 * 0.03
    wfm.sim_data['solar_zenith_degree'] = np.tile(
        zenith_test, [88, 1])[0:8760, :]
    wfm.sim_data['direct_normal_irradiance'] = dni
    wfm.placements['power_plant_capacity_W_el'] = 5E6
    wfm.placements['storage_capacity_kWh_th'] = 5E6*9/1000/0.4

    wfm.ptr_data = {}
    wfm.ptr_data['storage_efficiency_1'] = 0.99
    wfm.ptr_data['eta_powerplant_1'] = 0.4

    wfm._time_index_ = pd.date_range(
        "2014-12-31 23:30:00", periods=8760, freq="H")

    return wfm


def test_calculate_electrical_output(pt_PTRWorkflowManager_calcElecOut):
    wfm = pt_PTRWorkflowManager_calcElecOut

    debug_vars = False

    # case 1:
    onlynightuse = True
    wfm.calculate_electrical_output(
        onlynightuse=onlynightuse, debug_vars=debug_vars)

    assert wfm.sim_data_daily['Power_net_total_per_day_Wh'].shape == (365, 3)
    assert np.isclose(
        wfm.sim_data_daily['Power_net_total_per_day_Wh'].mean(), 26524777.959221434)
    assert np.isclose(
        wfm.sim_data_daily['Power_net_total_per_day_Wh'].std(), 12725454.904320534)
    assert np.isclose(
        wfm.sim_data_daily['Power_net_total_per_day_Wh'].min(), 0.0)
    assert np.isclose(
        wfm.sim_data_daily['Power_net_total_per_day_Wh'].max(), 36665967.17670143)

    assert wfm.sim_data_daily['Power_net_bound_per_day_Wh'].shape == (365, 3)
    assert np.isclose(
        wfm.sim_data_daily['Power_net_bound_per_day_Wh'].mean(), 0)
    assert np.isclose(
        wfm.sim_data_daily['Power_net_bound_per_day_Wh'].std(), 0)
    assert np.isclose(
        wfm.sim_data_daily['Power_net_bound_per_day_Wh'].min(), 0)
    assert np.isclose(
        wfm.sim_data_daily['Power_net_bound_per_day_Wh'].max(), 0)

    # case 2:
    onlynightuse = False
    wfm.calculate_electrical_output(
        onlynightuse=onlynightuse, debug_vars=debug_vars)

    assert wfm.sim_data_daily['Power_net_total_per_day_Wh'].shape == (365, 3)
    assert np.isclose(
        wfm.sim_data_daily['Power_net_total_per_day_Wh'].mean(), 35885138.24915207)
    assert np.isclose(
        wfm.sim_data_daily['Power_net_total_per_day_Wh'].std(), 23331667.68341452)
    assert np.isclose(
        wfm.sim_data_daily['Power_net_total_per_day_Wh'].min(), 0.0)
    assert np.isclose(
        wfm.sim_data_daily['Power_net_total_per_day_Wh'].max(), 80307702.4923343)

    assert wfm.sim_data_daily['Power_net_bound_per_day_Wh'].shape == (365, 3)
    assert np.isclose(
        wfm.sim_data_daily['Power_net_bound_per_day_Wh'].mean(), 9364395.790161656)
    assert np.isclose(
        wfm.sim_data_daily['Power_net_bound_per_day_Wh'].std(), 12716518.611837856)
    assert np.isclose(
        wfm.sim_data_daily['Power_net_bound_per_day_Wh'].min(), 0.0)
    assert np.isclose(
        wfm.sim_data_daily['Power_net_bound_per_day_Wh'].max(), 41235946.71834937)


####################################
#####  TEST calculate_calculate_LCOE  ######
####################################

@pytest.fixture
def pt_PTRWorkflowManager_calcLCOE(pt_PTRWorkflowManager_economics) -> PTRWorkflowManager:
    wfm = pt_PTRWorkflowManager_economics

    wfm.placements['Power_net_total_Wh_per_a'] = 2E11

    return wfm


def test_calculate_LCOE(pt_PTRWorkflowManager_calcLCOE):
    wfm = pt_PTRWorkflowManager_calcLCOE

    wfm.calculate_LCOE()

    a = np.array([9.21556861, 18.43113722,  9.21556861])
    assert np.isclose(wfm.placements['lcoe_EURct_per_kWh_el'].values, a).all()


####################################
#####  TEST calculateCapacityFactors  ######
####################################

@pytest.fixture
def pt_PTRWorkflowManager_calcCFs() -> PTRWorkflowManager:
    wfm = test_PTRWorkflowManager__init__()
    wfm.placements['capacity_sf_W_th'] = 58E6
    wfm.sim_data['HeattoPlant_W'] = dni_test * 1E5 * 0.7
    wfm.placements['power_plant_capacity_W_el'] = 58E6 / 2 * 0.4
    wfm.sim_data_daily['Power_net_total_per_day_Wh'] = dni_test * \
        1E5 * 0.7 * 0.99 * 0.4 * 0.9
    wfm.sim_data['P_heating_W'] = dni_test * 1E5 * 0.7 / 10
    # wfm.ptr_data = {}
    # wfm.sim_data['HeattoPlant_W'] = dni_test * 1E5 * 0.7
    # wfm.sim_data['Parasitics_solarfield_W_el'] = dni_test * 1E5 * 0.76 * 0.1
    # wfm.placements['aperture_area_m2'] = 1E5
    # wfm.placements['land_area_m2'] = 1E5 / 0.3

    # wfm._time_index_ = pd.date_range("2014-12-31 23:30:00", periods=100, freq="H")

    return wfm


def test_calculateCapacityFactors(pt_PTRWorkflowManager_calcCFs):
    wfm = pt_PTRWorkflowManager_calcCFs

    wfm.calculateCapacityFactors()

    assert wfm.sim_data['capacity_factor_sf'].shape == (140, 3)
    assert np.isclose(
        wfm.sim_data['capacity_factor_sf'].mean(), 0.08197873433178876)
    assert np.isclose(
        wfm.sim_data['capacity_factor_sf'].std(), 0.16908370937865658)
    assert np.isclose(wfm.sim_data['capacity_factor_sf'].min(), 0)
    assert np.isclose(
        wfm.sim_data['capacity_factor_sf'].max(), 0.7042167493103447)

    assert wfm.sim_data['capacity_factor_heat_FP_sf'].shape == (140, 3)
    assert np.isclose(
        wfm.sim_data['capacity_factor_heat_FP_sf'].min(), -0.07042167493103447)
    assert np.isclose(wfm.sim_data['capacity_factor_heat_FP_sf'].max(), 0)
    assert np.isclose(
        wfm.sim_data['capacity_factor_heat_FP_sf'].mean(), -0.008197873433178876)

    assert wfm.sim_data_daily['capacity_factor_plant'].shape == (140, 3)
    assert np.isclose(
        wfm.sim_data_daily['capacity_factor_plant'].mean(), 0.0060869210241353165)
    assert np.isclose(
        wfm.sim_data_daily['capacity_factor_plant'].std(), 0.012554465421365252)
    assert np.isclose(wfm.sim_data_daily['capacity_factor_plant'].min(), 0)
    assert np.isclose(
        wfm.sim_data_daily['capacity_factor_plant'].max(), 0.0522880936362931)


# Code to generate above:
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
# Useful function for generating test result values:
# def print_testresults(variable):
#     print('mean: ', variable[0:140,:].mean())
#     print('std: ', variable[0:140,:].std())
#     print('min: ', variable[0:140,:].min())
#     print('max: ', variable[0:140,:].max())#
theta_test = np.array(
    [
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [42.51498122, 47.05851153, 46.58451516],
        [29.41795914, 58.44752395, 57.84025516],
        [15.85027076, 68.565341, 67.8106125],
        [2.08953066, 74.30050715, 73.55369042],
        [11.69473521, 71.00770932, 70.68909905],
        [25.34474454, 61.7151318, 61.75590592],
        [38.63048282, 50.55558918, 50.80213438],
        [51.09139926, 38.90049472, 39.28042113],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [42.64730849, 46.92202648, 46.44783034],
        [29.5430145, 58.31032786, 57.70302608],
        [15.96757101, 68.43114614, 67.67676851],
        [2.19860794, 74.20312081, 73.45458208],
        [11.59497373, 70.99160592, 70.66767325],
        [25.25657853, 61.74238843, 61.7791655],
        [38.55850562, 50.59771548, 50.84149873],
        [51.0445576, 38.94529278, 39.32337028],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [42.78142161, 46.7825082, 46.30799984],
        [29.66865574, 58.16957029, 57.56212115],
        [16.08454198, 68.29190792, 67.53780772],
        [2.30657903, 74.0978771, 73.34769939],
        [11.49705148, 70.96751262, 70.63817908],
        [25.17101755, 61.76326438, 61.79589371],
        [38.4900164, 50.63429098, 50.87518811],
        [51.00234561, 38.98484592, 39.3609706],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [42.91721006, 46.6400431, 46.16511086],
        [29.79477517, 58.0253386, 57.4176289],
        [16.20108448, 68.1477298, 67.39383356],
        [2.4133561, 73.98487486, 73.23314101],
        [11.40104227, 70.93536352, 70.60056211],
        [25.08811782, 61.77765253, 61.8059883],
        [38.42504821, 50.66521809, 50.90310698],
        [50.96476749, 39.01907359, 39.39314127],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0, ],
        [0.0,          0.0,          0.0],
        [0.0,          0.0,          0.0],
    ]
)

tracking_angle_test = np.array(
    [
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [82.60771918, -82.73805605, -82.37636036],
        [77.34841735, -69.82644699, -69.30883898],
        [74.6840039, -48.36439604, -48.02708526],
        [73.73926137,  -7.74391779, -9.03989049],
        [74.23238202,  38.52376257, 36.10267348],
        [76.3090463,  64.60380697, 62.81077634],
        [80.63129659,  79.30700036, 78.07891156],
        [88.56898069,  89.3357273, 88.55823194],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [82.5591758, -82.71990288, -82.36146309],
        [77.28350739, -69.82204939, -69.30885247],
        [74.60967834, -48.44426468, -48.10866242],
        [73.65553301,  -8.10133877, -9.37541819],
        [74.13572654,  38.1043191, 35.69207715],
        [76.19244008,  64.31952237, 62.52441955],
        [80.48279867,  79.10274477, 77.87178567],
        [88.38656152,  89.19084991, 88.40620347],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [82.5005448, -82.69190073, -82.33683823],
        [77.20975345, -69.80502983, -69.29653005],
        [74.52737417, -48.50777933, -48.17455279],
        [73.5642729,  -8.44661875, -9.69910316],
        [74.03162985,  37.67690717, 35.27466706],
        [76.06816415,  64.02437659, 62.22762091],
        [80.32612546,  78.88941901, 77.65573007],
        [88.19498696,  89.03832135, 88.24651077],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [82.43175552, -82.65410547, -82.30253834],
        [77.12714109, -69.77550334, -69.27197692],
        [74.43710774, -48.55508737, -48.22488098],
        [73.46551774,  -8.7792228, -10.0104682],
        [73.92014523,  37.24234057, 34.8512387],
        [75.93628708,  63.71870535, 61.92072982],
        [80.16135501,  78.66719414, 77.43092514],
        [87.9942296,  88.87811595, 88.07920289],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
        [0.0,   0.0,  0.0],
    ]
)

zenith_test = np.array(
    [
        [152.21504857, 151.51706344, 152.18825374],
        [150.53864452, 149.90019811, 150.66726954],
        [144.88743615, 144.41766634, 145.1219116],
        [136.91095693, 136.62446784, 137.19961047],
        [127.84048865, 127.71951521, 128.15195573],
        [118.39140454, 118.41781264, 118.70735626],
        [109.01199489, 109.17305578, 109.31977052],
        [100.04841718, 100.33546591, 100.33707289],
        [91.83008161,  92.23564283, 92.08836689],
        [84.55807097,  85.05990883, 84.76857674],
        [79.00162995,  79.60340547, 79.15974843],
        [75.27977889,  75.94844263, 75.36991951],
        [73.75037322,  74.44732479, 73.76365545],
        [74.56794058,  75.24960104, 74.50297885],
        [77.64879437,  78.27439402, 77.512139],
        [82.69406479,  83.23005048, 82.49869681],
        [89.10126216,  89.48304249, 88.88403714],
        [97.43893041,  97.76247642, 97.15912712],
        [106.20842613, 106.40812288, 105.90582812],
        [115.4924509, 115.56032199, 115.17795783],
        [124.95581115, 124.88060774, 124.64121011],
        [134.19301946, 133.95819793, 133.89413924],
        [142.59244275, 142.17861773, 142.33523771],
        [149.10250165, 148.50865493, 148.93175398],
        [152.12749649, 151.42971896, 152.09948571],
        [150.49048191, 149.85076306, 150.61780344],
        [144.87189582, 144.40001535, 145.10549696],
        [136.91446324, 136.6257586, 137.20266476],
        [127.85252834, 127.72948509, 128.16383039],
        [118.40492396, 118.42944433, 118.72090671],
        [109.02190857, 109.18125241, 109.32986567],
        [100.05055587, 100.33605407, 100.33952389],
        [91.82065363,  92.22485191, 92.07937365],
        [84.53407557,  85.034901, 84.74514821],
        [78.95923494,  79.56023581, 79.11799039],
        [75.21730498,  75.88555732, 75.30811517],
        [73.66790198,  74.36482336, 73.68181553],
        [74.46773415,  75.14967608, 74.40329703],
        [77.53477547,  78.16085636, 77.39849738],
        [82.5711936,  83.107862, 82.37600943],
        [88.98568676,  89.37069462, 88.76713188],
        [97.30426591,  97.62817171, 97.02454648],
        [106.07178661, 106.27177803, 105.76922762],
        [115.35489709, 115.42302806, 115.04041559],
        [124.81806865, 124.74314908, 124.50345549],
        [134.05606768, 133.8216528, 133.75712709],
        [142.45894945, 142.04578098, 142.20153792],
        [148.98021134, 148.38721011, 148.80886616],
        [152.03213409, 151.33456976, 152.00293426],
        [150.43425449, 149.79327055, 150.56026446],
        [144.84880837, 144.37478419, 145.08151389],
        [136.91100164, 136.62002428, 137.19873493],
        [127.85795733, 127.73278156, 128.16908626],
        [118.41196556, 118.43453866, 118.72797792],
        [109.02530519, 109.18287924, 109.33344725],
        [100.04601553, 100.32991912, 100.33530273],
        [91.80430678,  92.20710803, 92.06347029],
        [84.50304811,  85.00286112, 84.71470553],
        [78.90946914,  79.50968789, 79.06886929],
        [75.14731642,  75.81515534, 75.2387981],
        [73.57796667,  74.27485735, 73.59250643],
        [74.36032965,  75.0425471, 74.29640507],
        [77.41400657,  78.04055158, 77.27808873],
        [82.44212308,  82.97944307, 82.24710395],
        [88.86423941,  89.25248917, 88.64434729],
        [97.16439567,  97.48860059, 96.88474573],
        [105.9303082, 106.13052263, 105.62777843],
        [115.21271289, 115.28102236, 114.89823916],
        [124.6756971, 124.60097178, 124.36107591],
        [133.91420961, 133.68010737, 133.61522351],
        [142.31989645, 141.90729784, 142.06230736],
        [148.85129148, 148.25908412, 148.67938938],
        [151.92902591, 151.23167942, 151.89866554],
        [150.36996823, 149.72772859, 150.4946605],
        [144.81813682, 144.34193908, 145.04992633],
        [136.9005201, 136.60721553, 137.18776914],
        [127.85672316, 127.72935419, 128.16767073],
        [118.41248269, 118.43305057, 118.72852299],
        [109.02214695, 109.17789972, 109.33047713],
        [100.03476905, 100.31703494, 100.32438191],
        [91.78102595,  92.18239686, 92.04064125],
        [84.46498239,  84.96378255, 84.67724182],
        [78.85234217,  79.45177154, 79.01239429],
        [75.06983648,  75.73726009, 75.16199112],
        [73.48060369,  74.17746319, 73.4957641],
        [74.2457761,  74.92826293, 74.18235158],
        [77.28654788,  77.9135394, 77.15097294],
        [82.30691974,  82.84485886, 82.1120469],
        [88.7369252,  89.1283895, 88.51570166],
        [97.01940547,  97.34384849, 96.73981054],
        [105.78407979, 105.98444528, 105.48156938],
        [115.06598872, 115.13439514, 114.75151893],
        [124.52878781, 124.45416703, 124.21416264],
        [133.76753776, 133.5336541, 133.46852095],
        [142.17537852, 141.76326275, 141.91764072],
        [148.71583567, 148.12436935, 148.5434175],
        [151.81824009, 151.12111512, 151.78674919],
        [150.29763437, 149.65415034, 150.42100476],
        [144.77984873, 144.3014508, 145.0107028],
        [136.88297024, 136.58728675, 137.16971923],
    ]
)

dni_test = np.array(
    [[0.00000000, 0.00000000, 0.00000000],
     [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [8.69906774E1, 1.60912250E2, 3.34106420E1],
        [1.82313437E2, 1.77583573E2, 9.62712637E1],
        [3.91094109E2, 2.56625692E2, 2.50307607E2],
        [4.92002620E2, 3.02899978E2, 3.60137786E2],
        [5.19355996E2, 3.24141288E2, 4.41956859E2],
        [4.27372201E2, 2.81207942E2, 4.33063828E2],
        [2.89423160E2, 1.27240794E2, 2.53166778E2],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 2.83516289, 9.05815011e-01],
        [5.47988859, 2.02231817E2, 9.53868224],
        [6.47256547E1, 4.29181035E2, 1.35448828E2],
        [8.61686460E1, 5.01259441E2, 2.00995454E2],
        [3.87728467E2, 4.99982525E2, 4.20229392E2],
        [2.17462252E2, 3.34885491E2, 2.63046733E2],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [1.48972108e-01, 1.63828618e-02, 5.41756011e-01],
        [0.00000000, 0.00000000, 7.52546240e-02],
        [0.00000000, 0.00000000, 0.00000000],
        [5.04773006e-03, 8.09264558e-01, 7.56299413e-03],
        [0.00000000, 2.31754696e-01, 5.27244895e-02],
        [3.43458798, 6.45777349e-01, 2.31405973],
        [3.63243410E1, 8.86940920, 2.71195708E1],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [1.69787833E2, 2.55515322E2, 1.64501935E2],
        [2.13007295E2, 3.88384508E2, 2.06283624E2],
        [2.83288260E2, 4.84241113E2, 2.39813520E2],
        [3.79913510E2, 5.00268224E2, 2.92895324E2],
        [2.68188406E2, 4.61937688E2, 2.24264482E2],
        [2.66258129E2, 3.88669821E2, 2.28255334E2],
        [2.66637477E2, 2.95184484E2, 2.94263370E2],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [2.59640826E2, 2.41878933E2, 2.56943620E2],
        [3.34297639E2, 3.24260260E2, 3.45735624E2],
        [5.32996889E2, 4.75994472E2, 5.28633914E2],
        [5.66755544E2, 5.62199860E2, 5.45646000E2],
        [5.83493878E2, 3.73927104E2, 5.40872199E2],
        [4.97935573E2, 3.54690308E2, 4.88111820E2],
        [3.42317267E2, 1.67857211E2, 3.59082782E2],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [7.86422690E1, 5.93229375E1, 6.99395733E1],
        [1.69179152E2, 1.17984872E2, 1.36298907E2],
        [1.93364844E2, 1.88729654E2, 1.99152862E2],
        [2.78638748E2, 2.53955026E2, 3.01000985E2],
        [2.58395286E2, 1.98491387E2, 2.88397239E2],
        [2.32139130E2, 1.65521875E2, 2.51428889E2],
        [1.09018525E2, 1.45448054E2, 1.48537719E2],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000]
     ])
