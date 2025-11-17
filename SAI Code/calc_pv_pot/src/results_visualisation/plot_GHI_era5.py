from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import os

# List of years you want to process
years = list(range(1988, 1995))

# Directory where the data files are located
data_dir = 'C:/Kamp/Data/ERA5_download/'

# Initialize dictionaries to store monthly averages
monthly_averages = {}

# Loop through each year and process the data
for year in years:
    # Build file path
    file_path = os.path.join(data_dir, f'PHL_{year}_total_12h.nc')
    
    # Open the dataset
    dataset = Dataset(file_path, 'r')
    
    # Extract the variables
    fdir = dataset.variables['fdir'][:]  # fdir[time, latitude, longitude]
    
    # Aggregate: Calculate monthly averages
    monthly_avg = np.mean(fdir, axis=(1, 2))  # Mean over latitude and longitude
    
    # Convert J/m² to W/m²
    seconds_per_period = 60 * 60  # 3 hours * 60 minutes * 60 seconds
    monthly_avg_w = monthly_avg / seconds_per_period
    
    # Store the results in the dictionary
    monthly_averages[year] = monthly_avg_w
    
    # Close the dataset
    dataset.close()

# Create a plot
plt.figure(figsize=(12, 6))

# Define colors for plotting
colors = plt.cm.viridis(np.linspace(0, 1, len(years)))

# Loop through each year and plot the data
for i, year in enumerate(years):
    plt.plot(np.arange(1, 13), monthly_averages[year], marker='o', linestyle='-', color=colors[i], label=str(year))

# Set title and axis labels
plt.title('Average GHI on the Philippines')
plt.xlabel('Month')
plt.ylabel('Average Radiation (W/m²)')

# Set x-axis tick labels to month names
plt.xticks(np.arange(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

# Highlight June with a shaded region
plt.axvspan(6.5, 8, color='yellow', alpha=0.3, label='June Eruption')  # Shading June (month index 6)

# Add a legend to the plot
plt.legend()

# Set y-axis limits
plt.ylim(200, 320)

# Add grid lines to the plot
plt.grid(True)

# Save the plot
plt.savefig('C:/Kamp/Code/CE_PV_Potential/calc_pv_pot/data/results/plots/mean_GHI_88to94.png')

# Display the plot
plt.show()
