import pandas as pd
import numpy as np
from netCDF4 import Dataset
from calc.Gen_16x16.workflow_16x16_aerosol_distr import create_aerosol_distr_16x16
from calc.Gen_16x16.workflow_16x16_simulation import create_sim_data_16x16
from calc.Gen_16x16.workflow_16x16_save import to_xarray_16x16
from calc.Gen_16x16.workflow_16x16_save import save_nc4_16x16
import os

cwd = os.getcwd()

# (1) CALCULATION OF AEROSOL DISTRIBUTION

version = 'Volc_MtPinatubo_15061992'
nc_filepath = cwd + f'/SAI Code/calc_pv_pot/data/processed/sim_data_16x16/{version}.nc4'


injection_NH = 17.5                                     # [Mt]
injection_SH = 0                                        # [Mt]
injection_global = injection_NH + injection_SH          # [Mt]

injection_lat_NH = 15                                   # latitude [°]
injection_lat_SH = 0                                    # latitude [°]

injection_year = 1991
injection_month = 6                                     # assumption: injection at the beginning of the month [1=JAN, 2=FEB, 3=MAR, 4=APR, 5=MAY, 6=JUN, 7=JUL, 8=AUG, 9=SEP, 10=OCT, 11=NOV, 12=DEC]
injection_day = 15

timeframe = 20                                          # Simulation of aerosol distribution [months]

flux_year = 1992                                        # simulation fluxes only for one year

df_distribution_tau_converted, times_flux_year_total, times_flux_year = create_aerosol_distr_16x16(version, 
                                                                                                    injection_NH, 
                                                                                                    injection_SH, 
                                                                                                    injection_lat_NH,
                                                                                                    injection_lat_SH,
                                                                                                    injection_year, 
                                                                                                    injection_month, 
                                                                                                    injection_day, 
                                                                                                    timeframe, flux_year)



sim_data_MA = create_sim_data_16x16(times_flux_year_total, df_distribution_tau_converted)

ds_16x16 = to_xarray_16x16(sim_data_MA)

#sim_data_MA = fill_year(ds_16x16, flux_year, times_flux_year)


save_nc4_16x16(ds_16x16, nc_filepath)