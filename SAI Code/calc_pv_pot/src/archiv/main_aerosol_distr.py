import pandas as pd
import numpy as np

from calc.simulation_aerosol_distribution.exchange_matrix_new import create_transilient_matrix_new
from calc.simulation_aerosol_distribution.initial_distribution import initial_distribution
from calc.simulation_aerosol_distribution.simulate_distribution import simulate_distribution
from calc.simulation_aerosol_distribution.get_region import get_region
from calc.simulation_aerosol_distribution.sim_linear_buildup import linear_buildup

from calc.AOD_RF.AOD import def_AOD
from calc.AOD_RF.AOD import easy_AOD
from calc.AOD_RF.AOD import easy_RF

# create exchange matrix for each season
matrix_w, matrix_sp, matrix_s, matrix_f = create_transilient_matrix_new()
transport_matrices = { 'winter': matrix_w,
                       'spring': matrix_sp,
                       'summer': matrix_s,
                       'fall': matrix_f    }

# Details of injection (one injection each hemisphere)
injection_NH = 15.5                                   # [Mt]
injection_SH = 15.5                                    # [Mt]
injection_global = injection_NH + injection_SH      # [Mt]

injection_lat_NH = 15.1425                              # latitude [°]
injection_lat_SH = -15.1425                              # latitude [°]
injection_date = 6                                      # assumption: injection at the beginning of the month
                                                        # [1=JAN, 2=FEB, 3=MAR, 4=APR, 5=MAY, 6=JUN, 7=JUL, 8=AUG, 9=SEP, 10=OCT, 11=NOV, 12=DEC]

# Timeframe of the simulation
timeframe = 12                                          # [months]

# get region from latitude input [°]
injection_reg_NH, injection_index_NH = get_region(injection_lat_NH)
injection_reg_SH, injection_index_SH = get_region(injection_lat_SH)

# initial distribution after injection (16 belts)
zero_distr = np.zeros(16)
init_distr_NH = initial_distribution(injection_reg_NH, injection_NH)
init_distr_SH = initial_distribution(injection_reg_SH, injection_SH)
init_distr_total = np.add(init_distr_NH, init_distr_SH) #überflüssig, weil innit distr in distr-4month enthalten ist

# simulate linear buildup
distr_4month_NH = linear_buildup(injection_NH, zero_distr, injection_index_NH)
distr_4month_SH = linear_buildup(injection_SH, zero_distr, injection_index_SH)

distr_4month = np.array([NH + SH for NH, SH in zip(distr_4month_NH, distr_4month_SH)]) # je die ersten 4 monats-verteilung von NH und SH addieren -> 4 gesamte monatsverteilungen

# simulation transport mechanisms and sedimentation: final distribution of aerosols in every belt
df_distribution_aerosol_belts = simulate_distribution(distr_4month, timeframe, injection_date, transport_matrices)

# store final distribution after time period in excel
df_distribution_aerosol_belts.to_excel('C:/Kamp/Code/CE_PV_Potential/calc_pv_pot/data/processed/aerosol_distribution/aerosol_distribution_v4.xlsx')


# calculation tau from aerosol distribution & store to excel
df_distribution_tau = df_distribution_aerosol_belts.apply(lambda aerosol: aerosol.map(easy_AOD))
df_distribution_tau.to_excel('C:/Kamp/Code/CE_PV_Potential/calc_pv_pot/data/processed/tau_distr/tau_distribution_v4.xlsx')


# calculation tau from aerosol distribution & store to excel
df_distribution_RF = df_distribution_tau.apply(lambda tau: tau.map(easy_RF))
df_distribution_RF.to_excel('C:/Kamp/Code/CE_PV_Potential/calc_pv_pot/data/processed/RF_distr/RF_distribution_v4.xlsx')


