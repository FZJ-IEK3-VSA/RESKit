from reskit import reskit as rk
import pandas as pd
import xarray as xr
import os

cwd = os.getcwd()
placements = pd.read_csv(rk.TEST_DATA["module_placements.csv"])
 
xds = rk.solar.openfield_pv_era5(
    placements=placements,
    era5_path=rk.TEST_DATA["era5-like"],
    global_solar_atlas_ghi_path=rk.TEST_DATA["gsa-ghi-like.tif"],
    global_solar_atlas_dni_path=rk.TEST_DATA["gsa-dni-like.tif"],
)
print('done')
xds

# Convert Xarray Dataset to Pandas DataFrame
df = xds.to_dataframe().reset_index()

# Save the DataFrame to a CSV file
df.to_csv(cwd + '\SAI Code\calc_pv_pot/data/test/Review_Reskit/Debug_Reskit_16x16_3.csv', index=False)  # index=False avoids writing row numbers
print('saved as csv')