# PLOT der capfacs des RESKIT-OUTPUTS

from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

BEL_v = Dataset('R:/my_data/pv_potential_ma_luisa/BEL_test/OFPV_fixed/Base/volcanov1_eruption/BEL/100m/2018/cluster_vars/cluster_vars__OFPV_fixed__BEL.3_1__SG0732-0158__2018__100res__Base.nc4', 'r')
BEL_nov = Dataset('R:/my_data/pv_potential_ma_luisa/BEL_test/OFPV_fixed/Base/volcanov1_noeruption/BEL/100m/2018/cluster_vars/cluster_vars__OFPV_fixed__BEL.3_1__SG0732-0158__2018__100res__Base.nc4', 'r')

# CAPFACS
ts_capacity_factor_v = BEL_v.variables['ts_capacity_factor'][:]
ts_capacity_factor_no = BEL_nov.variables['ts_capacity_factor'][:]
capacity_difference = ts_capacity_factor_v[0, :] - ts_capacity_factor_no[0, :]


# TIME
time_v = BEL_v.variables['time'][:]
time_no = BEL_nov.variables['time'][:]

# specific day
start_hour = 6168
end_hour= start_hour + 24  # 24 hours in the day
time = np.arange(24)  # 24 hours in the day

# capfac-diff for specific day
capacity_difference_sept_15 = ts_capacity_factor_v[0, start_hour:end_hour] - ts_capacity_factor_no[0, start_hour:end_hour]



## PLOT 1: ## ts_capacity_factor over time for both datasets

plt.figure(figsize=(10, 6))
plt.plot(time_v, ts_capacity_factor_v[0, :], label='BEL_v Capacity Factor')  # [0,:] since LCOE_clstr has size 1
plt.plot(time_no, ts_capacity_factor_no[0, :], label='BEL_nov Capacity Factor')

# Add labels and title
plt.xlabel('Time (hours)')
plt.ylabel('Capacity Factor')
plt.title('PV Capacity Factor Over Time')
plt.legend()

# Show plot
plt.show()


## PLOT 2: ## diff ts_capacity_factor over time

plt.figure(figsize=(10, 6))
plt.plot(time_v, capacity_difference, label='Capacity Factor Difference (BEL_v - BEL_nov)', color='b')

# Add labels and title
plt.xlabel('Time (hours)')
plt.ylabel('Capacity Factor Difference')
plt.title('Difference in PV Capacity Factor Over Time')
plt.legend()

# Show plot
plt.show()

## PLOT 3: ## ts_capacity_factor of one day

plt.figure(figsize=(10, 6))
plt.plot(time, capacity_difference_sept_15, label='Capacity Factor Difference (BEL_v - BEL_nov)', color='b')

# Add labels and title
plt.xlabel('Time (hours)')
plt.ylabel('Capacity Factor Difference')
plt.title('Difference in PV Capacity Factor on 15th September 2018')
plt.legend()

# Show plot
plt.show()