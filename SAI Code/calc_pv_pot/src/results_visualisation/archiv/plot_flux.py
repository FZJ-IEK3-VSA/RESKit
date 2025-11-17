import matplotlib.pyplot as plt
import pandas as pd

# Import data
df_flux_out = pd.read_excel('C:/Kamp/Code/CE_PV_Potential/calc_pv_pot/data/processed/flux_deltaedd_output6_wvbis4µm.xlsx')

# Filter (jede 4. Zeile = erstes tau jeden Jahres)
df_filtered = df_flux_out.iloc[::4, :]
#df_filtered = df_filtered[(df_filtered['year'] >= 1900) & (df_filtered['year'] <= 1920)] #Filter für bestimmten Zeitraum
df_filtered = df_flux_out.iloc[::4, :]

columns_dir = ['0.25_F_dir', '0.3_F_dir', '0.35_F_dir', '0.4_F_dir', '0.45_F_dir']
columns_diff = ['0.25_F_diff', '0.3_F_diff', '0.35_F_diff', '0.4_F_diff', '0.45_F_diff']
columns_dn = ['0.25_F_dn', '0.3_F_dn', '0.35_F_dn', '0.4_F_dn', '0.45_F_dn']

# plot dir
# fig_dir = plt.figure(figsize=(12, 8))
# for column in columns_dir:
#     plt.plot(df_filtered['year'], df_filtered[column], marker='o', label=column)

# plt.xlabel('Year')
# plt.ylabel('flux_dir [W/m²]')
# plt.title('F_dir')
# plt.legend()
# plt.grid(True)
# plt.show()

# save_path = 'C:/Kamp/Code/CE_PV_Potential/calc_pv_pot/data/results/plots/plot_flux_dir_1900to1920.png'
#fig_dir.savefig(save_path)

# plot diff
# fig_diff = plt.figure(figsize=(12, 8))
# for column in columns_diff:
#     plt.plot(df_filtered['year'], df_filtered[column], marker='o', label=column)

# plt.xlabel('Year')
# plt.ylabel('flux_diff [W/m²]')
# plt.title('F_diff')
# plt.legend()
# plt.grid(True)
# plt.show()

# save_path = 'C:/Kamp/Code/CE_PV_Potential/calc_pv_pot/data/results/plots/plot_flux_diff_1900to1920.png'
#fig_diff.savefig(save_path)

# plot dn (total)
fig_dn = plt.figure(figsize=(12, 8))
for column in columns_dn:
    plt.plot(df_filtered['year'], df_filtered[column], marker='o', label=column)

plt.xlabel('Year')
plt.ylabel('flux_diff [W/m²]')
plt.title('F_diff')
plt.legend()
plt.grid(True)
plt.show()

save_path = 'C:/Kamp/Code/CE_PV_Potential/calc_pv_pot/data/results/plots/plot_flux_dn.png'
fig_dn.savefig(save_path)
