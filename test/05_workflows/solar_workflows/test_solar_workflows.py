from reskit.solar.workflows.workflows import (
    openfield_pv_era5,
    openfield_pv_merra_ryberg2019,
    openfield_pv_sarah_unvalidated)
from reskit import TEST_DATA
import pytest
import numpy as np
import geokit as gk
import pandas as pd


@pytest.fixture
def pt_pv_placements() -> pd.DataFrame:
    df = gk.vector.extractFeatures(TEST_DATA['turbinePlacements.shp'])
    df['capacity'] = 2000
    return df


def test_openfield_pv_era5(pt_pv_placements):
    gen = openfield_pv_era5(
        placements=pt_pv_placements,
        era5_path=TEST_DATA['era5-like'],
        global_solar_atlas_ghi_path=TEST_DATA['gsa-ghi-like.tif'],
        global_solar_atlas_dni_path=TEST_DATA['gsa-dni-like.tif'],
        module='WINAICO WSx-240P6',
        elev=300,
        tracking='fixed',
        inverter=None,
        inverter_kwargs={},
        tracking_args={},
        output_netcdf_path=None,
        output_variables=None,
    )

    assert (gen['location'].shape == (560,))
    assert (gen['capacity'].shape == (560,))
    assert (gen['lon'].shape == (560,))
    assert (gen['lat'].shape == (560,))
    assert (gen['tilt'].shape == (560,))
    assert (gen['azimuth'].shape == (560,))
    assert (gen['elev'].shape == (560,))
    assert (gen['time'].shape == (140,))
    assert (gen['global_horizontal_irradiance'].shape == (140, 560))
    assert (gen['direct_horizontal_irradiance'].shape == (140, 560))
    assert (gen['surface_wind_speed'].shape == (140, 560))
    assert (gen['surface_pressure'].shape == (140, 560))
    assert (gen['surface_air_temperature'].shape == (140, 560))
    assert (gen['surface_dew_temperature'].shape == (140, 560))
    assert (gen['solar_azimuth'].shape == (140, 560))
    assert (gen['apparent_solar_zenith'].shape == (140, 560))
    assert (gen['direct_normal_irradiance'].shape == (140, 560))
    assert (gen['extra_terrestrial_irradiance'].shape == (140, 560))
    assert (gen['air_mass'].shape == (140, 560))
    assert (gen['diffuse_horizontal_irradiance'].shape == (140, 560))
    assert (gen['angle_of_incidence'].shape == (140, 560))
    assert (gen['poa_global'].shape == (140, 560))
    assert (gen['poa_direct'].shape == (140, 560))
    assert (gen['poa_diffuse'].shape == (140, 560))
    assert (gen['poa_sky_diffuse'].shape == (140, 560))
    assert (gen['poa_ground_diffuse'].shape == (140, 560))
    assert (gen['cell_temperature'].shape == (140, 560))
    assert (gen['module_dc_power_at_mpp'].shape == (140, 560))
    assert (gen['module_dc_voltage_at_mpp'].shape == (140, 560))
    assert (gen['capacity_factor'].shape == (140, 560))
    assert (gen['total_system_generation'].shape == (140, 560))

    assert np.isclose(float(gen['location'].fillna(0).mean()), 279.5)
    assert np.isclose(float(gen['capacity'].fillna(0).mean()), 2000.0)
    assert np.isclose(float(gen['lon'].fillna(0).mean()), 6.16945196229404)
    assert np.isclose(float(gen['lat'].fillna(0).mean()), 50.80320853112445)
    assert np.isclose(float(gen['tilt'].fillna(0).mean()), 39.19976325987092)
    assert np.isclose(float(gen['azimuth'].fillna(0).mean()), 180.0)
    assert np.isclose(float(gen['elev'].fillna(0).mean()), 300.0)
    assert np.isclose(float(gen['global_horizontal_irradiance'].fillna(0).mean()), 32.515431915117816)
    assert np.isclose(float(gen['direct_horizontal_irradiance'].fillna(0).mean()), 15.41170374927935)
    assert np.isclose(float(gen['surface_wind_speed'].fillna(0).mean()), 1.696705522977915)
    assert np.isclose(float(gen['surface_pressure'].fillna(0).mean()), 38651.440408516704)
    assert np.isclose(float(gen['surface_air_temperature'].fillna(0).mean()), 0.9659668062938231)
    assert np.isclose(float(gen['surface_dew_temperature'].fillna(0).mean()), -0.029744251184882078)
    assert np.isclose(float(gen['solar_azimuth'].fillna(0).mean()), 68.6008947378997)
    assert np.isclose(float(gen['apparent_solar_zenith'].fillna(0).mean()), 31.143327387439044)
    assert np.isclose(float(gen['direct_normal_irradiance'].fillna(0).mean()), 122.17685778949215)
    assert np.isclose(float(gen['extra_terrestrial_irradiance'].fillna(0).mean()), 546.9559617849145)
    assert np.isclose(float(gen['air_mass'].fillna(0).mean()), 3.9100941154150948)
    assert np.isclose(float(gen['diffuse_horizontal_irradiance'].fillna(0).mean()), 9.96274731477096)
    assert np.isclose(float(gen['angle_of_incidence'].fillna(0).mean()), 19.148848622048433)
    assert np.isclose(float(gen['poa_global'].fillna(0).mean()), 100.57019788221743)
    assert np.isclose(float(gen['poa_direct'].fillna(0).mean()), 81.2793750926284)
    assert np.isclose(float(gen['poa_diffuse'].fillna(0).mean()), 19.290822789589022)
    assert np.isclose(float(gen['poa_sky_diffuse'].fillna(0).mean()), 18.52340070915684)
    assert np.isclose(float(gen['poa_ground_diffuse'].fillna(0).mean()), 0.7674220804321872)
    assert np.isclose(float(gen['cell_temperature'].fillna(0).mean()), 3.747658757320868)
    assert np.isclose(float(gen['module_dc_power_at_mpp'].fillna(0).mean()), 32.172430838525734)
    assert np.isclose(float(gen['module_dc_voltage_at_mpp'].fillna(0).mean()), 12.800509481262262)
    assert np.isclose(float(gen['capacity_factor'].fillna(0).mean()), 0.10505906273188251)
    assert np.isclose(float(gen['total_system_generation'].fillna(0).mean()), 210.118125463765)


def test_openfield_pv_merra_ryberg2019(pt_pv_placements):
    gen = openfield_pv_merra_ryberg2019(
        placements=pt_pv_placements,
        merra_path=TEST_DATA['merra-like'],
        global_solar_atlas_ghi_path=TEST_DATA["gsa-ghi-like.tif"],
        module='WINAICO WSx-240P6',
        elev=300,
        tracking='fixed',
        inverter=None,
        inverter_kwargs={},
        tracking_args={},
        output_netcdf_path=None,
        output_variables=None,
    )

    assert (gen['location'].shape == (560,))
    assert (gen['capacity'].shape == (560,))
    assert (gen['lon'].shape == (560,))
    assert (gen['lat'].shape == (560,))
    assert (gen['tilt'].shape == (560,))
    assert (gen['azimuth'].shape == (560,))
    assert (gen['elev'].shape == (560,))
    assert (gen['time'].shape == (71,))
    assert (gen['surface_wind_speed'].shape == (71, 560))
    assert (gen['surface_pressure'].shape == (71, 560))
    assert (gen['surface_air_temperature'].shape == (71, 560))
    assert (gen['surface_dew_temperature'].shape == (71, 560))
    assert (gen['global_horizontal_irradiance'].shape == (71, 560))
    assert (gen['solar_azimuth'].shape == (71, 560))
    assert (gen['apparent_solar_zenith'].shape == (71, 560))
    assert (gen['extra_terrestrial_irradiance'].shape == (71, 560))
    assert (gen['air_mass'].shape == (71, 560))
    assert (gen['direct_normal_irradiance'].shape == (71, 560))
    assert (gen['diffuse_horizontal_irradiance'].shape == (71, 560))
    assert (gen['angle_of_incidence'].shape == (71, 560))
    assert (gen['poa_global'].shape == (71, 560))
    assert (gen['poa_direct'].shape == (71, 560))
    assert (gen['poa_diffuse'].shape == (71, 560))
    assert (gen['poa_sky_diffuse'].shape == (71, 560))
    assert (gen['poa_ground_diffuse'].shape == (71, 560))
    assert (gen['cell_temperature'].shape == (71, 560))
    assert (gen['module_dc_power_at_mpp'].shape == (71, 560))
    assert (gen['module_dc_voltage_at_mpp'].shape == (71, 560))
    assert (gen['capacity_factor'].shape == (71, 560))
    assert (gen['total_system_generation'].shape == (71, 560))

    print(float(gen['location'].fillna(0).mean()))
    assert np.isclose(float(gen['location'].fillna(0).mean()), 279.5)
    print(float(gen['capacity'].fillna(0).mean()))
    assert np.isclose(float(gen['capacity'].fillna(0).mean()), 2000.0)
    print(float(gen['lon'].fillna(0).mean()))
    assert np.isclose(float(gen['lon'].fillna(0).mean()), 6.16945196229404)
    print(float(gen['lat'].fillna(0).mean()))
    assert np.isclose(float(gen['lat'].fillna(0).mean()), 50.80320853112445)
    print(float(gen['tilt'].fillna(0).mean()))
    assert np.isclose(float(gen['tilt'].fillna(0).mean()), 39.19976325987092)
    print(float(gen['azimuth'].fillna(0).mean()))
    assert np.isclose(float(gen['azimuth'].fillna(0).mean()), 180.0)
    print(float(gen['elev'].fillna(0).mean()))
    assert np.isclose(float(gen['elev'].fillna(0).mean()), 300.0)
    print(float(gen['surface_wind_speed'].fillna(0).mean()))
    assert np.isclose(float(gen['surface_wind_speed'].fillna(0).mean()), 1.5502203948117972)
    print(float(gen['surface_pressure'].fillna(0).mean()))
    assert np.isclose(float(gen['surface_pressure'].fillna(0).mean()), 38110.883667100796)
    print(float(gen['surface_air_temperature'].fillna(0).mean()))
    assert np.isclose(float(gen['surface_air_temperature'].fillna(0).mean()), 0.6923904404714382)
    print(float(gen['surface_dew_temperature'].fillna(0).mean()))
    assert np.isclose(float(gen['surface_dew_temperature'].fillna(0).mean()), 0.2735079282721086)
    print(float(gen['global_horizontal_irradiance'].fillna(0).mean()))
    assert np.isclose(float(gen['global_horizontal_irradiance'].fillna(0).mean()), 24.425654064650278)
    print(float(gen['solar_azimuth'].fillna(0).mean()))
    assert np.isclose(float(gen['solar_azimuth'].fillna(0).mean()), 67.69226199649943)
    print(float(gen['apparent_solar_zenith'].fillna(0).mean()))
    assert np.isclose(float(gen['apparent_solar_zenith'].fillna(0).mean()), 30.755085032338677)
    print(float(gen['extra_terrestrial_irradiance'].fillna(0).mean()))
    assert np.isclose(float(gen['extra_terrestrial_irradiance'].fillna(0).mean()), 539.2545578051567)
    print(float(gen['air_mass'].fillna(0).mean()))
    assert np.isclose(float(gen['air_mass'].fillna(0).mean()), 3.9380521870930165)
    print(float(gen['direct_normal_irradiance'].fillna(0).mean()))
    assert np.isclose(float(gen['direct_normal_irradiance'].fillna(0).mean()), 20.907640632171837)
    print(float(gen['diffuse_horizontal_irradiance'].fillna(0).mean()))
    assert np.isclose(float(gen['diffuse_horizontal_irradiance'].fillna(0).mean()), 19.589003059825554)
    print(float(gen['angle_of_incidence'].fillna(0).mean()))
    assert np.isclose(float(gen['angle_of_incidence'].fillna(0).mean()), 18.915157478877003)
    assert np.isclose(float(gen['poa_global'].fillna(0).mean()), 38.581645380369565)
    assert np.isclose(float(gen['poa_direct'].fillna(0).mean()), 15.491403834596987)
    assert np.isclose(float(gen['poa_diffuse'].fillna(0).mean()), 23.090241545772578)
    assert np.isclose(float(gen['poa_sky_diffuse'].fillna(0).mean()), 22.5137080215302)
    assert np.isclose(float(gen['poa_ground_diffuse'].fillna(0).mean()), 0.5765335242423779)
    assert np.isclose(float(gen['cell_temperature'].fillna(0).mean()), 1.757604391183729)
    assert np.isclose(float(gen['module_dc_power_at_mpp'].fillna(0).mean()), 12.625206716607272)
    assert np.isclose(float(gen['module_dc_voltage_at_mpp'].fillna(0).mean()), 14.060960010572025)
    assert np.isclose(float(gen['capacity_factor'].fillna(0).mean()), 0.04201539723986579)
    assert np.isclose(float(gen['total_system_generation'].fillna(0).mean()), 84.0307944797316)


def test_openfield_pv_sarah_unvalidated(pt_pv_placements):
    gen = openfield_pv_sarah_unvalidated(
        placements=pt_pv_placements,
        sarah_path=TEST_DATA['sarah-like'],
        era5_path=TEST_DATA['era5-like'],
        module='WINAICO WSx-240P6',
        elev=300,
        tracking='fixed',
        inverter=None,
        inverter_kwargs={},
        tracking_args={},
        output_netcdf_path=None,
        output_variables=None,
    )

    assert (gen['location'].shape == (560,))
    assert (gen['capacity'].shape == (560,))
    assert (gen['lon'].shape == (560,))
    assert (gen['lat'].shape == (560,))
    assert (gen['tilt'].shape == (560,))
    assert (gen['azimuth'].shape == (560,))
    assert (gen['elev'].shape == (560,))
    assert (gen['time'].shape == (48,))
    assert (gen['direct_normal_irradiance'].shape == (48, 560))
    assert (gen['global_horizontal_irradiance'].shape == (48, 560))
    assert (gen['surface_wind_speed'].shape == (48, 560))
    assert (gen['surface_pressure'].shape == (48, 560))
    assert (gen['surface_air_temperature'].shape == (48, 560))
    assert (gen['surface_dew_temperature'].shape == (48, 560))
    assert (gen['solar_azimuth'].shape == (48, 560))
    assert (gen['apparent_solar_zenith'].shape == (48, 560))
    assert (gen['extra_terrestrial_irradiance'].shape == (48, 560))
    assert (gen['air_mass'].shape == (48, 560))
    assert (gen['diffuse_horizontal_irradiance'].shape == (48, 560))
    assert (gen['angle_of_incidence'].shape == (48, 560))
    assert (gen['poa_global'].shape == (48, 560))
    assert (gen['poa_direct'].shape == (48, 560))
    assert (gen['poa_diffuse'].shape == (48, 560))
    assert (gen['poa_sky_diffuse'].shape == (48, 560))
    assert (gen['poa_ground_diffuse'].shape == (48, 560))
    assert (gen['cell_temperature'].shape == (48, 560))
    assert (gen['module_dc_power_at_mpp'].shape == (48, 560))
    assert (gen['module_dc_voltage_at_mpp'].shape == (48, 560))
    assert (gen['capacity_factor'].shape == (48, 560))
    assert (gen['total_system_generation'].shape == (48, 560))

    # assert np.isclose( float(gen['location'].fillna(0).mean() ), 279.5)
    assert np.isclose(float(gen['location'].fillna(0).mean()), 279.5)
    assert np.isclose(float(gen['capacity'].fillna(0).mean()), 2000.0)
    assert np.isclose(float(gen['lon'].fillna(0).mean()), 6.16945196229404)
    assert np.isclose(float(gen['lat'].fillna(0).mean()), 50.80320853112445)
    assert np.isclose(float(gen['tilt'].fillna(0).mean()), 39.19976325987092)
    assert np.isclose(float(gen['azimuth'].fillna(0).mean()), 180.0)
    assert np.isclose(float(gen['elev'].fillna(0).mean()), 300.0)
    assert np.isclose(float(gen['direct_normal_irradiance'].fillna(0).mean()), 155.98687394203432)
    assert np.isclose(float(gen['global_horizontal_irradiance'].fillna(0).mean()), 50.013295986799676)
    assert np.isclose(float(gen['surface_wind_speed'].fillna(0).mean()), 1.7378792235544152)
    assert np.isclose(float(gen['surface_pressure'].fillna(0).mean()), 37785.8373532626)
    assert np.isclose(float(gen['surface_air_temperature'].fillna(0).mean()), 0.8728509898681798)
    assert np.isclose(float(gen['surface_dew_temperature'].fillna(0).mean()), -0.35190959673979627)
    assert np.isclose(float(gen['solar_azimuth'].fillna(0).mean()), 68.01052582121625)
    assert np.isclose(float(gen['apparent_solar_zenith'].fillna(0).mean()), 30.37642178304629)
    assert np.isclose(float(gen['extra_terrestrial_irradiance'].fillna(0).mean()), 531.7569374999998)
    assert np.isclose(float(gen['air_mass'].fillna(0).mean()), 3.7819914016528204)
    assert np.isclose(float(gen['diffuse_horizontal_irradiance'].fillna(0).mean()), 16.019148937401813)
    assert np.isclose(float(gen['angle_of_incidence'].fillna(0).mean()), 18.703030123719145)
    assert np.isclose(float(gen['poa_global'].fillna(0).mean()), 140.86892453530422)
    assert np.isclose(float(gen['poa_direct'].fillna(0).mean()), 112.10522916272164)
    assert np.isclose(float(gen['poa_diffuse'].fillna(0).mean()), 28.762842046403616)
    assert np.isclose(float(gen['poa_sky_diffuse'].fillna(0).mean()), 27.582403760328745)
    assert np.isclose(float(gen['poa_ground_diffuse'].fillna(0).mean()), 1.180438286074874)
    assert np.isclose(float(gen['cell_temperature'].fillna(0).mean()), 4.639081351598623)
    assert np.isclose(float(gen['module_dc_power_at_mpp'].fillna(0).mean()), 45.26504648633102)
    assert np.isclose(float(gen['module_dc_voltage_at_mpp'].fillna(0).mean()), 11.95638277934221)
    assert np.isclose(float(gen['capacity_factor'].fillna(0).mean()), 0.150637447124134)
    assert np.isclose(float(gen['total_system_generation'].fillna(0).mean()), 301.274894248268)
