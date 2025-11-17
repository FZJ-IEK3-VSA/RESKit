import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the Excel file
file_path = 'C:/Kamp/Code/CE_PV_Potential/calc_pv_pot/data/processed/flux_calc/delta_flux/delta_flux_out_v13.xlsx'
df = pd.read_excel(file_path)

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Plotting
regions = df['region'].unique()
fig, axs = plt.subplots(len(regions), 1, figsize=(15, 30), sharex=True)

for i, region in enumerate(regions):
    region_data = df[df['region'] == region]
    axs[i].plot(region_data['timestamp'], region_data['0.55_delta_dir'], label='Direct Radiation')
    axs[i].plot(region_data['timestamp'], region_data['0.55_delta_diff'], label='Diffuse Radiation')
    #axs[i].plot(region_data['timestamp'], region_data['0.55_delta_dn'], label='Total Radiation')
    axs[i].set_title(region)
    axs[i].legend()

plt.xlabel('Time')
plt.ylabel('Radiation')
plt.tight_layout()
plt.show()

save_path = 'C:/Kamp/Code/CE_PV_Potential/calc_pv_pot/data/results/plots/plot_delta_timeframe.png'
plt.savefig(save_path)
