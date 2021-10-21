#%%
import reskit as rk
import pandas as pd
import numpy as np


#%%
columns = ['lat', 'lon', 'capacity']
data = [
    [50.1, 6.1, 100.1],
    # [51.1, 10.1, 10.1],
    [0.1, 0.1, 10.1],
    # [-24.82, 18.01, 10.1],
    # [56.34, -4.3, 100.1],
    [19.1758, -155.5876, 10.1],
    # [19.46, -155.588, 10.1],
    # [28.022, 92.6765, 10.1],
    # [28.0017, 92.8818, 10.1],
    # [-24.89, -67.89, 10.1],
    # [-24.827, -67.793, 10.1],
    [59.429167193, -146.554226190, 10.1],
    [70, 70, 10.1],
]

# data = [
#         [50.475, 6.1, 100.1], #middle
#         [50.001, 6.1, 100.1], #corner
#         [40, 6.1, 100.1], #outside
#     ]

placements = pd.DataFrame(data, columns=columns)

era5_path = '/storage/internal/data/gears/weather/ERA5/processed/4/8/3/2011/'

wf = rk.solar.SolarWorkflowManager(placements)
# wf.read(
#     variables=["global_horizontal_irradiance",
#                 "direct_horizontal_irradiance",
#                 ],
#     source_type="ERA5",
#     source=era5_path,
#     set_time_index=True,
#     verbose=True
# )

wf.sim_data['spatial_disaggregation_ghi'] = np.ones(shape=(1,placements.shape[0]))
wf.sim_data['spatial_disaggregation_dni'] = np.ones(shape=(1,placements.shape[0]))
wf.sim_data['spatial_disaggregation_gwa'] = np.ones(shape=(1,placements.shape[0]))

wf.sim_data['LRA_ghi'] = np.ones(shape=(1,placements.shape[0]))
wf.sim_data['LRA_dni'] = np.ones(shape=(1,placements.shape[0]))
wf.sim_data['LRA_gwa'] = np.ones(shape=(1,placements.shape[0]))

ghi_path = r'/storage/internal/data/gears/geography/irradiance/global_solar_atlas_v2.5/World_GHI_GISdata_LTAy_AvgDailyTotals_GlobalSolarAtlas-v2_GEOTIFF/GHI.tif'
dni_path = r'/storage/internal/data/gears/geography/irradiance/global_solar_atlas_v2.5/World_DNI_GISdata_LTAy_AvgDailyTotals_GlobalSolarAtlas-v2_GEOTIFF/DNI.tif'
gwa_path = r'/storage/internal/data/gears/geography/wind/global_wind_atlas/GWA_3.0/gwa3_250_wind-speed_100m.tif'

#spatial disaggregation
# wf.spatial_disaggregation(
#     variable='spatial_disaggregation_ghi',
#     source_high_resolution=ghi_path,
#     source_low_resolution=rk.weather.GSAmeanSource.GHI_with_ERA5_pixel,
#     )
# wf.spatial_disaggregation(
#     variable='spatial_disaggregation_dni',
#     source_high_resolution=dni_path,
#     source_low_resolution=rk.weather.GSAmeanSource.DNI_with_ERA5_pixel,
#     )
# wf.spatial_disaggregation(
#     variable='spatial_disaggregation_gwa',
#     source_high_resolution=gwa_path,
#     source_low_resolution=rk.weather.GWAmeanSource.GWA_with_ERA5_pixel,
#     )
# #mean value correction from ERA5 to GSA/GWA
# wf.spatial_disaggregation(
#     variable='spatial_disaggregation_ghi',
#     source_high_resolution=rk.weather.GSAmeanSource.GHI_with_ERA5_pixel,
#     source_low_resolution=rk.weather.Era5Source.LONG_RUN_AVERAGE_GHI,
#     real_lra_scaling=1000 / 24,
#     )
# wf.spatial_disaggregation(
#     variable='spatial_disaggregation_dni',
#     source_high_resolution=rk.weather.GSAmeanSource.DNI_with_ERA5_pixel,
#     source_low_resolution=rk.weather.Era5Source.LONG_RUN_AVERAGE_DNI,
#     real_lra_scaling=1000 / 24,
#     )
# wf.spatial_disaggregation(
#     variable='spatial_disaggregation_gwa',
#     source_high_resolution=rk.weather.GWAmeanSource.GWA_with_ERA5_pixel,
#     source_low_resolution=rk.weather.Era5Source.LONG_RUN_AVERAGE_WINDSPEED,
#     )

wf.adjust_variable_to_long_run_average(
        variable='LRA_ghi',
        source_long_run_average=rk.weather.Era5Source.LONG_RUN_AVERAGE_GHI,
        real_long_run_average=rk.TEST_DATA['gsa-ghi-like.tif'],#ghi_path,
        real_lra_scaling=1000 / 24,  # cast to hourly average kWh
        nodata_fallback='source'
    )
wf.adjust_variable_to_long_run_average(
        variable='LRA_dni',
        source_long_run_average=rk.weather.Era5Source.LONG_RUN_AVERAGE_GHI,
        real_long_run_average=rk.TEST_DATA['gsa-ghi-like.tif'],#dni_path,
        real_lra_scaling=1000 / 24,  # cast to hourly average kWh
        nodata_fallback='nan'
    )
wf.adjust_variable_to_long_run_average(
        variable='LRA_gwa',
        source_long_run_average=rk.weather.Era5Source.LONG_RUN_AVERAGE_WINDSPEED,
        real_long_run_average=gwa_path,
    )

#save results to df
out = pd.DataFrame(wf.placements['lat'], columns=['lat'])
out['lon'] = wf.placements['lon']
# out['spatial_disaggregation_ghi'] = wf.sim_data['spatial_disaggregation_ghi'].squeeze().tolist()
# out['spatial_disaggregation_dni'] = wf.sim_data['spatial_disaggregation_dni'].squeeze().tolist()
# out['spatial_disaggregation_gwa'] = wf.sim_data['spatial_disaggregation_gwa'].squeeze().tolist()

out['LRA_ghi'] = wf.sim_data['LRA_ghi'].squeeze().tolist()
out['LRA_dni'] = wf.sim_data['LRA_dni'].squeeze().tolist()
out['LRA_gwa'] = wf.sim_data['LRA_gwa'].squeeze().tolist()

# out['relchange_SpatDis/LRA_ghi'] = out['spatial_disaggregation_ghi'] / out['LRA_ghi']
# out['relchange_SpatDis/LRA_dni'] = out['spatial_disaggregation_dni'] / out['LRA_dni']
# out['relchange_SpatDis/LRA_gwa'] = out['spatial_disaggregation_gwa'] / out['LRA_gwa']


pass
# %%

columns = ['lat', 'lon', 'capacity']
data = [
        [-6, -38, 10.1], #middle
        [-5, -32.1, 10.1], #outside
    ]
data = [
        [70.0, 0.0, 10.1], #middle
        [72.0, 0.0, 10.1], #outside
    ]

era5_path = '/storage/internal/data/gears/weather/ERA5/processed/4/8/3/2011/'
ghi_path = r'/storage/internal/data/gears/geography/irradiance/global_solar_atlas_v2.5/World_GHI_GISdata_LTAy_AvgDailyTotals_GlobalSolarAtlas-v2_GEOTIFF/GHI.tif'
dni_path = r'/storage/internal/data/gears/geography/irradiance/global_solar_atlas_v2.5/World_DNI_GISdata_LTAy_AvgDailyTotals_GlobalSolarAtlas-v2_GEOTIFF/DNI.tif'


placements = pd.DataFrame(data, columns=columns)
out = rk.solar.openfield_pv_era5(
    placements=placements,
    era5_path=era5_path,
    global_solar_atlas_dni_path=dni_path,
    global_solar_atlas_ghi_path=ghi_path,
    gsa_nodata_fallback='source',

)
# %%
