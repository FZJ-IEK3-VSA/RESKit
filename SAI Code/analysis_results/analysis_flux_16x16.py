import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt


from analysis_funcitons import find_region
import os

cwd = os.getcwd()

path = cwd + '\SAI Code\calc_pv_pot/data/processed/sim_data_16x16/SAI_10Mt_year1.nc4'
ds= xr.open_dataset(path)

lat = 5 #51
lon = 120 #10

region_name, region_idx = find_region(ds, lat, lon)

# extract specific region
var = 'delta_DIR'                                           ### ANPASSEN
var_region = ds[var].sel(region=region_idx)*-1

time_index = pd.to_datetime(ds['time_stamps'].values)

# Erstelle ein neues DataArray mit Zeitstempel als Index
var_region = xr.DataArray(var_region.values, dims=['time'], coords={'time': time_index})

# PLOT gesamte Zeitreihe
plt.figure(figsize=(10, 6))
var_region.plot(label=f'{var}', color='b')
plt.title(f'{var} über die Zeit für Region {region_name}')
plt.xlabel('Time')
plt.ylabel(f'{var}')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()