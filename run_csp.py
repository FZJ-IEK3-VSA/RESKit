#%%
import reskit as rk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm

#%%

# Make a placements dataframe
placements = pd.DataFrame()
#noor 2 ptr plant, morocco
placements['lon'] = [ -6.8, -6.8, -6.8]     # Longitude [ 6.083, 6.083, 5.583] #
placements['lat'] = [ 31.0, 31.4, 31.0,]   # Latitude [ 50.775, 50.775, 50.775,] #
placements['area_m2'] = [1E6, 5E6, 6E6]
repeats = int(3/3)
placements = placements.loc[placements.index.repeat(repeats)].reset_index(drop=True)
    
#%%
out = rk.csp.CSP_PTR_ERA5(
    placements=placements, 
    era5_path= r'C:\Users\d.franzmann\data\ERA5\7\6', #r'/storage/internal/data/gears/weather/ERA5/processed/4/7/6/2015/', #r'C:\Users\d.franzmann\data\ERA5\7\6',
    global_solar_atlas_dni_path = 'R:\data\gears\geography\irradiance\global_solar_atlas_v2.5\World_DNI_GISdata_LTAy_AvgDailyTotals_GlobalSolarAtlas-v2_GEOTIFF\DNI.tif',
    global_solar_atlas_tamb_path = "R:\data\gears\geography\irradiance\global_solar_atlas_v2.5\World_TEMP_GISdata_LTAy_GlobalSolarAtlas_GEOTIFF\TEMP.tif",
    verbose = True,
    cost_year=2030,
    JITaccelerate=False,
    return_self=True,
    debug_vars=True,
    onlynightuse=False,
)

print('Simulation done')


# %%
#plot sm / tes optimization:
#%matplotlib
plt.ion()

#data_key = 'TOTEX_Plant_storage_USD_per_a_3D'
#data_key = 'annualPowerOutput_Wh_3D'
data_key = 'LCOE_USD_per_Wh'
#data_key = 'annualHeat_Wh_3D'


if data_key == 'LCOE_USD_per_Wh':
    factor = 1E5 / 1.21
    showname = 'LCOE EURct/kWh'
else:
    factor=1
    showname = data_key

data = out.opt_data[data_key][0,:,:]*factor


fig = plt.figure(dpi = 600)
ax = fig.add_subplot(111, projection='3d')

sm = out.sm#out.placements['sm']
tes = out.tes#placements['tes']

sm_2D = np.tile(sm[:], (len(tes), 1)).T
tes_2D = np.tile(tes, (len(sm), 1))

ax.plot_surface(sm_2D, tes_2D, data)

ax.set_xlabel('SM')
ax.set_ylabel('TES')
ax.set_zlabel(showname)
ax.set_title(showname)
#ax.set_zlim(3.5, 5)

# %%

#plot sm / tes optimization:
#%matplotlib
plt.ion()

#data_key = 'TOTEX_CSP_USD_per_a_3D'
#data_key = 'annualPowerOutput_Wh_3D'
data_key = 'LCOE_USD_per_Wh'
#data_key = 'annualHeat_Wh_3D'

if data_key == 'LCOE_USD_per_Wh':
    factor = 1E5 / 1.21
    showname = 'LCOE EURct/kWh'
else:
    factor=1
    showname = data_key

data = out.opt_data[data_key][0,:,:] * factor


fig, ax = plt.subplots(dpi=1000)

sm = out.sm
tes = out.tes

cmap = cm.get_cmap('Blues', 100)

cmap = LinearSegmentedColormap.from_list(
    'peter',
    [cmap(20), cmap(100)],
    N=len(tes),
    )

for i, tes_i in enumerate(tes):
    
    
    data_i = data[:,i]
    
    ax.plot(
        sm,
        data_i,
        label = f'TES: {tes_i}h',
        color = cmap(i)      
    )

ax.set_xlabel('SM')
ax.set_ylabel(showname)
ax.set_title(showname)
ax.grid('on')
ax.legend(loc='upper right')
#ax.set_ylim((8, 20))

# %%

#curtailment
location_id = 2
cf = out.sim_data['capacity_factor'][:,location_id].copy()
cf_curtailed = cf.copy()
cf_curtailed[cf_curtailed>1]=1

start = 2250
stop = 2400

fig,axs = plt.subplots(2)
axs[0].plot(cf[start:stop], label='cf')
axs[0].plot(cf_curtailed[start:stop], label='cf_curt')
axs[1].plot(out.sim_data['direct_normal_irradiance'][:,location_id][start:stop])
plt.legend()
# %%
