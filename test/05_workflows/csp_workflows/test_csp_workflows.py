import reskit as rk
import pandas as pd
import numpy as np
from reskit import TEST_DATA
import pytest

#%%
@pytest.fixture
def pt_pv_placements() -> pd.DataFrame:
    placements = pd.DataFrame()
    #noor 2 ptr plant, morocco
    placements['lon'] = [ -6.8, -6.8, -6.8]     # Longitude  # [ -6.8, -6.8, -6.8]
    placements['lat'] = [ 31.0, 31.4, 31.0,]   # Latitude  # [ 31.0, 31.4, 31.0,]
    placements['area_m2'] = [1E6, 5E6, 6E6]
    repeats = int(3/3)
    placements = placements.loc[placements.index.repeat(repeats)].reset_index(drop=True)
    return placements
# Make a placements dataframe

#%%
@pytest.mark.skip(reason='Not working on calamari')
def test_CSP_PTR_ERA5(pt_pv_placements):

    #local
    era5_path = r"R:\data\gears\weather\ERA5\processed\4\7\6\2015"
    global_solar_atlas_dni_path = r'R:\data\gears\geography\irradiance\global_solar_atlas_v2.5\World_DNI_GISdata_LTAy_AvgDailyTotals_GlobalSolarAtlas-v2_GEOTIFF\DNI.tif'
    global_solar_atlas_tamb_path = r"R:\data\gears\geography\irradiance\global_solar_atlas_v2.5\World_TEMP_GISdata_LTAy_GlobalSolarAtlas_GEOTIFF\TEMP.tif"
    
    #cluster
    era5_path = r'/storage/internal/data/gears/weather/ERA5/processed/4/7/6/2015/'
    global_solar_atlas_dni_path = r'/storage/internal/data/gears/geography/irradiance/global_solar_atlas_v2.5/World_DNI_GISdata_LTAy_AvgDailyTotals_GlobalSolarAtlas-v2_GEOTIFF/DNI.tif'
    global_solar_atlas_tamb_path = r"/storage/internal/data/gears/geography/irradiance/global_solar_atlas_v2.5/World_TEMP_GISdata_LTAy_GlobalSolarAtlas_GEOTIFF/TEMP.tif"


    out = rk.csp.CSP_PTR_ERA5(
        placements=pt_pv_placements, 
        era5_path= era5_path,
        global_solar_atlas_dni_path = global_solar_atlas_dni_path,
        global_solar_atlas_tamb_path = global_solar_atlas_tamb_path,
        verbose = True,
        cost_year=2030,
        JITaccelerate=False,
        return_self=False,
        debug_vars=True,
        onlynightuse=True,
    )

    print('Simulation done')

    #datasets
    a = np.array(['Dataset_SolarSalt_2030', 'Dataset_Therminol_2030',
       'Dataset_SolarSalt_2030'])
    assert (out['datasetname'].values==a).all()

    #HeattoHTF_W
    assert out['HeattoHTF_W'].values.shape == (8760,3)
    assert np.isclose(out['HeattoHTF_W'].values.mean(), 209593585.6439108)
    assert np.isclose(out['HeattoHTF_W'].values.std(), 360960162.2924555)
    assert np.isclose(out['HeattoHTF_W'].values.min(), 0.0)
    assert np.isclose(out['HeattoHTF_W'].values.max(), 1712568733.8678896)

    #HeattoPlant_W
    assert out['HeattoPlant_W'].values.shape == (8760,3)
    assert np.isclose(out['HeattoPlant_W'].values.mean(), 161141657.54576105)
    assert np.isclose(out['HeattoPlant_W'].values.std(), 303656816.1385911)
    assert np.isclose(out['HeattoPlant_W'].values.min(), 0.0)
    assert np.isclose(out['HeattoPlant_W'].values.max(), 1563863764.9382672)

    #P_heating_W
    assert out['P_heating_W'].values.shape == (8760,3)
    assert np.isclose(out['P_heating_W'].values.mean(), 10034275.990073396)
    assert np.isclose(out['P_heating_W'].values.std(), 21524641.01787983)
    assert np.isclose(out['P_heating_W'].values.min(), 0.0)
    assert np.isclose(out['P_heating_W'].values.max(), 70676641.74470554)

    #sm_opt
    a = np.array([2., 2., 2.])
    assert np.isclose(out['sm_opt'].values, a).all()
    #tes_opt
    a = np.array([12, 12, 12])
    assert np.isclose(out['tes_opt'].values, a).all()

    #Power_net_total_per_day_Wh
    assert out['Power_net_total_per_day_Wh'].values.shape == (365,3)
    assert np.isclose(out['Power_net_total_per_day_Wh'].values.mean(), 1072113430.6607083)
    assert np.isclose(out['Power_net_total_per_day_Wh'].values.std(), 791714368.3297758)
    assert np.isclose(out['Power_net_total_per_day_Wh'].values.min(), 0.0)
    assert np.isclose(out['Power_net_total_per_day_Wh'].values.max(), 2548116684.0465016)

    #P_backup_heating_daily_Wh_el
    assert out['P_backup_heating_daily_Wh_el'].values.shape == (365,3)
    assert np.isclose(out['P_backup_heating_daily_Wh_el'].values.mean(), 7068648.321407319)
    assert np.isclose(out['P_backup_heating_daily_Wh_el'].values.std(), 66722840.521766596)
    assert np.isclose(out['P_backup_heating_daily_Wh_el'].values.min(), 0.0)
    assert np.isclose(out['P_backup_heating_daily_Wh_el'].values.max(), 1041705751.1049646)

    #lcoe_EURct_per_kWh_el
    a = np.array([15.04214996, 17.24473617, 15.04214996])
    assert np.isclose(out['lcoe_EURct_per_kWh_el'].values, a).all()
