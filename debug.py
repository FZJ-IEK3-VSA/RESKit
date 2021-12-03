from reskit.solar.workflows.workflows import (
    openfield_pv_era5,
    openfield_pv_merra_ryberg2019,
    openfield_pv_sarah_unvalidated)
from reskit import TEST_DATA
import pytest
import numpy as np
import geokit as gk
import pandas as pd

df = gk.vector.extractFeatures(TEST_DATA['turbinePlacements.shp'])
df['capacity'] = 2000

gen = openfield_pv_merra_ryberg2019(
    placements=df,
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
print(float(gen['poa_global'].fillna(0).mean()))
assert np.isclose(float(gen['poa_global'].fillna(0).mean()), 39.017197520055454)
assert np.isclose(float(gen['poa_direct'].fillna(0).mean()), 15.491403834596987)
assert np.isclose(float(gen['poa_diffuse'].fillna(0).mean()), 23.525793685458467)
assert np.isclose(float(gen['poa_sky_diffuse'].fillna(0).mean()), 22.949260161216085)
assert np.isclose(float(gen['poa_ground_diffuse'].fillna(0).mean()), 0.5765335242423779)
assert np.isclose(float(gen['cell_temperature'].fillna(0).mean()), 1.7694856257094103)
assert np.isclose(float(gen['module_dc_power_at_mpp'].fillna(0).mean()), 12.768905171405205)
assert np.isclose(float(gen['module_dc_voltage_at_mpp'].fillna(0).mean()), 14.079408128765307)
assert np.isclose(float(gen['capacity_factor'].fillna(0).mean()), 0.04249361100670641)
assert np.isclose(float(gen['total_system_generation'].fillna(0).mean()), 84.9872220134128)