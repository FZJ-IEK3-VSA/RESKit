import numpy as np
import matplotlib.pyplot as plt

from calc.radiation_flux_SAI.flux_delta_eddinton import flux_delta_eddington
from calc.radiation_flux_SAI.flux_delta_eddinton import calc_delta
from calc.radiation_flux_SAI.delta_eddington import delta_eddington_param

import numpy as np
import matplotlib.pyplot as plt

# Constants
F_0 = 1361  # Solar constant [W/m^2]
w = 0.98    # Single scattering albedo
g = 0.5     # Asymmetry parameter
f = 0.65    # Forward scattering fraction
µ_0 = np.cos(np.radians(30))  # Example cosine of solar zenith angle (30 degrees zenith angle)

# Define a range of optical depth (tau) to simulate different AOD levels
tau_values = np.linspace(0.01, 1.2, 100)  # Range of tau (AOD) values from low to high

# Initialize arrays to store the results
F_dir_values = []
F_diff_values = []
F_dn_values = []

# Loop through each tau value and calculate the corresponding fluxes
for tau in tau_values:
    # Calculate the transmissivities and reflectivities using the delta-Eddington parameterization
    R_dir, T_dir, R_diff, T_diff, tau_star = delta_eddington_param(tau, w, g, µ_0, f)
    
    # Calculate the transmitted direct and diffuse fluxes
    F_dir, F_diff, F_dn = flux_delta_eddington(µ_0, tau_star, F_0, T_dir)
    
    # Store the results
    F_dir_values.append(F_dir)
    F_diff_values.append(F_diff)
    F_dn_values.append(F_dn)

# Convert results to numpy arrays for plotting
F_dir_values = np.array(F_dir_values)
F_diff_values = np.array(F_diff_values)
F_dn_values = np.array(F_dn_values)


# 1. Plot the irradiance components
plt.figure(figsize=(10, 6))
plt.plot(tau_values, F_dir_values, label='DNI (Direct)', color='blue')
plt.plot(tau_values, F_diff_values, label='DHI (Diffuse)', color='green')
plt.plot(tau_values, F_dn_values, label='GHI (Global)', color='orange')
plt.xlabel('Optical Depth (AOD)')
plt.ylabel('Irradiance [W/m²]')
plt.title('Irradiance Components as a Function of AOD')
plt.legend()
plt.grid(True)
plt.savefig('C:/Kamp/Code/CE_PV_Potential/calc_pv_pot/data/results/plots/Validierung/Validierung_AOD_Diff_Components.png')
#plt.show()

# Calculate the slope of the fluxes with respect to AOD
dF_dir = np.gradient(F_dir_values, tau_values)
dF_diff = np.gradient(F_diff_values, tau_values)

# # 2. Plot the slopes
# plt.figure(figsize=(10, 6))
# plt.plot(tau_values, dF_dir, label='DNI Slope', color='blue')
# plt.plot(tau_values, dF_diff, label='DHI Slope', color='green')
# plt.xlabel('Optical Depth (AOD)')
# plt.ylabel('Slope [W/m² per AOD]')
# plt.title('Slope of Irradiance Components as a Function of AOD')
# plt.legend()
# plt.grid(True)
# plt.axhline(0, color='black', linewidth=0.5, linestyle='--')  # Add a horizontal line at y=0
# plt.show()

# # 3. Calculate the contribution of DHI to GHI
# contribution_diff = (F_diff_values / F_dn_values) * 100

# # Plotting the contribution
# plt.figure(figsize=(10, 6))
# plt.plot(tau_values, contribution_diff, label='DHI Contribution to GHI', color='purple')
# plt.xlabel('Optical Depth (AOD)')
# plt.ylabel('Contribution [%]')
# plt.title('Contribution of Diffuse Irradiance to Global Irradiance')
# plt.legend()
# plt.grid(True)
# plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
# plt.show()

# # Basispunkt (z.B. bei niedrigem AOD)
# base_DHI = F_diff_values[tau_values < 0.1].mean()  # Mittelwert der DHI bei AOD < 0.1

# # Relative Änderung der DHI
# relative_change_DHI = (F_diff_values - base_DHI) / base_DHI * 100

# # Plotting der relativen Änderung
# plt.figure(figsize=(10, 6))
# plt.plot(tau_values, relative_change_DHI, label='Relative Change in DHI', color='purple')
# plt.xlabel('Optical Depth (AOD)')
# plt.ylabel('Relative Change [%]')
# plt.title('Relative Change of Diffuse Irradiance with AOD')
# plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
# plt.legend()
# plt.grid(True)
# plt.show()


# Berechne die Ableitungen für DHI, DNI und GHI
dF_diff = np.gradient(F_diff_values, tau_values)
dF_dir = np.gradient(F_dir_values, tau_values)
dF_dn = np.gradient(F_dn_values, tau_values)



# Plot der Ableitungen
plt.figure(figsize=(10, 6))
plt.plot(tau_values, dF_diff, label='Ableitung DHI (Diffuse)', color='green', linestyle='-')
plt.plot(tau_values, dF_dir, label='Ableitung DNI (Direct)', color='blue', linestyle='-')
plt.plot(tau_values, dF_dn, label='Ableitung GHI (Global)', color='orange', linestyle='-')

plt.xlabel('Optical Depth (AOD)')
plt.ylabel('Ableitung [W/m²]')
plt.title('Ableitungen der Irradiance-Komponenten als Funktion der AOD')

plt.legend()
plt.grid(True)
plt.savefig('C:/Kamp/Code/CE_PV_Potential/calc_pv_pot/data/results/plots/Validierung/Validierung_AOD_Diff_Ableitung.png')
#plt.show()

print('plot done')

# Definiere AOD-Intervalle
aod_intervals = [(0.0, 0.4), (0.4, 0.8), (0.8, 1.2)]
interval_labels = ['0.0 - 0.4', '0.4 - 0.8', '0.8 - 1.2']

# Initialisiere Listen für die absolute Änderung
absolute_changes_diff = []
absolute_changes_dir = []
absolute_changes_dn = []

# Berechne die absolute Änderung für jedes Intervall
for aod_range in aod_intervals:
    start_tau = aod_range[0]
    end_tau = aod_range[1]

    # Filter die DHI-Werte für das aktuelle Intervall
    mask = (tau_values >= start_tau) & (tau_values < end_tau)
    
    if np.any(mask):
        start_value_diff = F_diff_values[mask][0]  # Erster Wert im Intervall
        end_value_diff = F_diff_values[mask][-1]   # Letzter Wert im Intervall
        absolute_change_diff = end_value_diff - start_value_diff
        absolute_changes_diff.append(absolute_change_diff)

        # Füge ähnliche Berechnungen für DNI und GHI hinzu
        start_value_dir = F_dir_values[mask][0]
        end_value_dir = F_dir_values[mask][-1]
        absolute_change_dir = end_value_dir - start_value_dir
        absolute_changes_dir.append(absolute_change_dir)

        start_value_dn = F_dn_values[mask][0]
        end_value_dn = F_dn_values[mask][-1]
        absolute_change_dn = end_value_dn - start_value_dn
        absolute_changes_dn.append(absolute_change_dn)




# Plot DHI
plt.figure(figsize=(10, 6))
plt.plot(tau_values, F_diff_values, label='DHI (Diffuse)', color='green')
plt.xlabel('Optical Depth (AOD)')
plt.ylabel('DHI [W/m²]')
plt.title('DHI as a Function of AOD')
y_position_diff = max(F_diff_values) * 0.95
for i, aod_range in enumerate(aod_intervals):
    plt.axvline(x=aod_range[0], linestyle='--')
    plt.axvline(x=aod_range[1], linestyle='--')
    plt.text((aod_range[0] + aod_range[1]) / 2, y_position_diff, 
             f'Change: {absolute_changes_diff[i]:.2f} W/m²', ha='center', color='black')
plt.grid(True)
plt.savefig('C:/Kamp/Code/CE_PV_Potential/calc_pv_pot/data/results/plots/Validierung/Validierung_AOD_DHI.png')
plt.close()

# Plot DNI
plt.figure(figsize=(10, 6))
plt.plot(tau_values, F_dir_values, label='DNI (Direct)', color='blue')
plt.xlabel('Optical Depth (AOD)')
plt.ylabel('DNI [W/m²]')
plt.title('DNI as a Function of AOD')
y_position_dir = max(F_dir_values) * 0.95
for i, aod_range in enumerate(aod_intervals):
    plt.axvline(x=aod_range[0], linestyle='--')
    plt.axvline(x=aod_range[1], linestyle='--')
    plt.text((aod_range[0] + aod_range[1]) / 2, y_position_dir, 
             f'Change: {absolute_changes_dir[i]:.2f} W/m²', ha='center', color='black')
plt.grid(True)
plt.savefig('C:/Kamp/Code/CE_PV_Potential/calc_pv_pot/data/results/plots/Validierung/Validierung_AOD_DNI.png')
plt.close()

# Plot GHI
plt.figure(figsize=(10, 6))
plt.plot(tau_values, F_dn_values, label='GHI (Global)', color='orange')
plt.xlabel('Optical Depth (AOD)')
plt.ylabel('GHI [W/m²]')
plt.title('GHI as a Function of AOD')
y_position_dn = max(F_dn_values) * 0.95
for i, aod_range in enumerate(aod_intervals):
    plt.axvline(x=aod_range[0], linestyle='--')
    plt.axvline(x=aod_range[1], linestyle='--')
    plt.text((aod_range[0] + aod_range[1]) / 2, y_position_dn, 
             f'Change: {absolute_changes_dn[i]:.2f} W/m²', ha='center', color='black')
plt.grid(True)
plt.savefig('C:/Kamp/Code/CE_PV_Potential/calc_pv_pot/data/results/plots/Validierung/Validierung_AOD_GHI.png')
plt.close()