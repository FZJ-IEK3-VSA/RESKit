import pandas as pd
import numpy as np
from calc.Gen_16x16.workflow_16x16_aerosol_distr import create_aerosol_distr_16x16_SAI
from calc.Gen_16x16.workflow_16x16_simulation import create_sim_data_16x16
from calc.Gen_16x16.workflow_16x16_save import to_xarray_16x16
from calc.Gen_16x16.workflow_16x16_save import save_nc4_16x16
import os

cwd = os.getcwd()

# (1) CALCULATION OF AEROSOL DISTRIBUTION

version = 'SAI_10Mt_I12'
nc_filepath = cwd + f'/SAI Code/calc_pv_pot/data/processed/sim_data_16x16/{version}.nc4'

injection_NH = 10                                           # [Mt]
injection_SH = 10                                           # [Mt]
injection_global = injection_NH + injection_SH              # [Mt]

injection_lat_NH = 5                                        # latitude [°]
injection_lat_SH = -5                                       # latitude [°]

injection_year = 2008
injection_month = 12                                         # assumption: injection at the beginning of the month [1=JAN, 2=FEB, 3=MAR, 4=APR, 5=MAY, 6=JUN, 7=JUL, 8=AUG, 9=SEP, 10=OCT, 11=NOV, 12=DEC]
injection_day = 1

timeframe_years = 11
timeframe = 12 * timeframe_years                            # Simulation of aerosol distribution [months]

flux_year = 2018                                            # simulation fluxes only for one year

df_distribution_tau_converted, times = create_aerosol_distr_16x16_SAI(version, 
                                                                        injection_NH, 
                                                                        injection_SH, 
                                                                        injection_lat_NH,
                                                                        injection_lat_SH,
                                                                        injection_year, 
                                                                        injection_month, 
                                                                        injection_day, 
                                                                        timeframe, flux_year)

times_flux_year = times[times.year == flux_year]

sim_data_MA = create_sim_data_16x16(times_flux_year, df_distribution_tau_converted)

ds_16x16 = to_xarray_16x16(sim_data_MA)

save_nc4_16x16(ds_16x16, nc_filepath)