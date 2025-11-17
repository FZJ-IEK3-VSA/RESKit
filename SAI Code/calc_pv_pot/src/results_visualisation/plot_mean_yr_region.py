import pandas as pd
import matplotlib.pyplot as plt

# Set paths
file_path = 'C:/Kamp/Code/CE_PV_Potential/calc_pv_pot/data/processed/flux_calc/delta_flux/delta_flux_out_Sc-V_175.xlsx'  # Replace with your file path
save_path = 'C:/Kamp/Code/CE_PV_Potential/calc_pv_pot/data/results/plots/'

data = pd.read_excel(file_path)

# Extract the year from the timestamp
data['year'] = data['timestamp'].dt.year

# Group by both 'region' and 'year', then calculate the mean for each group
yearly_avg = data.groupby(['region', 'year']).mean(numeric_only=True)

# Reset index to make it easier to work with
yearly_avg = yearly_avg.reset_index()


###################### MULTIPLE YEARS #################################


# Define the desired order of regions
region_order = ['NH_pol', 'NH_extrop_1', 'NH_extrop_2', 'NH_extrop_3', 'NH_extrop_4',
                'NH_trop_1', 'NH_trop_2', 'NH_trop_3', 'SH_trop_1', 'SH_trop_2',
                'SH_trop_3', 'SH_extrop_1', 'SH_extrop_2', 'SH_extrop_3', 'SH_extrop_4',
                'SH_pol']

# Get a list of unique years in the dataset
unique_years = yearly_avg['year'].unique()

# Create subplots with the number of unique years
fig, ax = plt.subplots(len(unique_years), 1, figsize=(12, 6 * len(unique_years)), sharex=True)

# Loop through each year and create a bar plot for that year
for i, year in enumerate(unique_years):

    plot_name = 'mean_by_region' + year

    # Filter the data for the current year
    yearly_data = yearly_avg[yearly_avg['year'] == year]
    
    # Aggregate the mean values for each region
    mean_values = yearly_data.groupby('region')[['0.55_delta_dir', '0.55_delta_diff', '0.55_delta_dn']].mean()
    
    # Reindex mean_values to the desired order
    mean_values = mean_values.reindex(region_order)
    
    # Plot the mean values for each region
    mean_values.plot(kind='bar', ax=ax[i])
    
    # Set the title and labels for each subplot
    ax[i].set_title(f'Mean Change in Solar Flux by Region - Year: {year}')
    ax[i].set_ylabel('Mean Flux Change (W/m²)')
    ax[i].set_xlabel('Region')
    ax[i].legend(["DNI", "DHI", "GHI"], loc='upper right')
    
    # Set the x-axis labels to be readable
    ax[i].set_xticks(range(len(region_order)))
    ax[i].set_xticklabels(region_order, rotation=45, ha='right')

    plt.savefig(save_path + 'plot_name' + '.png')

