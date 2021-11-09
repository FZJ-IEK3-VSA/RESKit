#%%
import matplotlib
import reskit as rk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%

# Make a placements dataframe
placements = pd.DataFrame()
#noor 2 ptr plant, morocco
placements['lon'] = [ -6.8, -6.8, -6.8]     # Longitude [ 6.083, 6.083, 5.583] #
placements['lat'] = [ 31.0, 31.4, 31.0,]   # Latitude [ 50.775, 50.775, 50.775,] #
placements['area'] = [1E6, 5E6, 6E6]
repeats = int(3/3)
placements = placements.loc[placements.index.repeat(repeats)].reset_index(drop=True)
    
#%%
out = rk.solar.workflows.workflows.csp_ptr_V1(
    placements=placements, 
    era5_path=r'/storage/internal/data/gears/weather/ERA5/processed/4/7/6/2015/', #r'C:\Users\d.franzmann\data\ERA5\7\6',
    global_solar_atlas_dni_path = 'default_local',
    datasetname='Dataset_SolarSalt_2030',
    verbose = True,
    JITaccelerate=False)

print('Simulation done')

#%%
#check dni

#%matplotlib
plt.ion
start = 4800
end = 4848
fig, axs = plt.subplots(2)
axs[0].plot(out.sim_data['direct_normal_irradiance'][start:end,0], label='dni')
axs[0].plot(out.sim_data['direct_horizontal_irradiance'][start:end,0], label='dhi')
axs[0].legend()

axs[1].plot(out.sim_data['Parasitics_solarfield_W'][start:end,0], label='T_HTF')

# %%
#plot sm / tes optimization:
#%matplotlib
plt.ion()

data_key = 'LCOE_USD_per_Wh'
#'est_LCOE_USD_per_kWh'
#'LCO_Heat_USD_per_Wh'
#'plant_storage_cost_per_heat_USD_per_Wh_3D'
# #'annualHeatStored_Wh_3D'
# #'speccosts_USD_per_kW_sf_2D'
# #'annualHeat_Wh_3D'


data = out.opt_data[data_key][0,:,:]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

sm = out.sm#out.placements['sm']
tes = out.tes#placements['tes']

sm_2D = np.tile(sm[:], (len(tes), 1)).T
tes_2D = np.tile(tes, (len(sm), 1))

ax.plot_surface(sm_2D, tes_2D, data)
#ax.plot_surface(sm_2D, tes_2D, out.sim_data['TOTEX_CSP_USD_per_a_3D'][0,:,:])
#ax.plot_surface(sm_2D, tes_2D, out.sim_data['TOTEX_SF_USD_per_a_3D'][0,:,:])
#ax.plot_surface(sm_2D, tes_2D, out.sim_data['TOTEX_Plant_storage_USD_per_a_3D'][0,:,:])

ax.set_xlabel('SM')
ax.set_ylabel('TES')
ax.set_zlabel(data_key)



# %%

# %%
