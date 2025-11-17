import pandas as pd
import numpy as np

# (1) CALCULATION OF AEROSOL DISTRIBUTION

from calc.simulation_aerosol_distribution.exchange_matrix_new import create_transilient_matrix_new
from calc.simulation_aerosol_distribution.initial_distribution import initial_distribution
from calc.simulation_aerosol_distribution.simulate_distribution import simulate_distribution
from calc.simulation_aerosol_distribution.get_region import get_region
from calc.simulation_aerosol_distribution.sim_linear_buildup import linear_buildup
from calc.simulation_aerosol_distribution.convert_tau_file import convert_tau_file

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
injection_NH = 15.5                                     # [Mt]
injection_SH = 15.5                                     # [Mt]
injection_global = injection_NH + injection_SH          # [Mt]

injection_lat_NH = 5                                    # latitude [°]
injection_lat_SH = -5                                   # latitude [°]

injection_year = 1991
injection_month = 6                                     # assumption: injection at the beginning of the month [1=JAN, 2=FEB, 3=MAR, 4=APR, 5=MAY, 6=JUN, 7=JUL, 8=AUG, 9=SEP, 10=OCT, 11=NOV, 12=DEC]
injection_day = 15

# Timeframe of the simulation
timeframe = 16                                          # [months]

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
df_distribution_aerosol_belts = simulate_distribution(distr_4month, timeframe, injection_month, transport_matrices)

# store final distribution after time period in excel
df_distribution_aerosol_belts.to_excel('C:/Kamp/Code/CE_PV_Potential/calc_pv_pot/data/processed/aerosol_distribution/aerosol_distribution_v7.xlsx')


# calculation tau from aerosol distribution & store to excel
df_distribution_tau = df_distribution_aerosol_belts.apply(lambda aerosol: aerosol.map(easy_AOD))
df_distribution_tau.to_excel('C:/Kamp/Code/CE_PV_Potential/calc_pv_pot/data/processed/tau_distr/tau_distribution_v7.xlsx')


# calculation tau from aerosol distribution & store to excel
df_distribution_RF = df_distribution_tau.apply(lambda tau: tau.map(easy_RF))
df_distribution_RF.to_excel('C:/Kamp/Code/CE_PV_Potential/calc_pv_pot/data/processed/RF_distr/RF_distribution_v7.xlsx')

# convert df (tau) in needed form (mirrored)
df_distribution_tau_converted = convert_tau_file(df_distribution_tau)                               # INPUT FOR (2)
df_distribution_tau_converted.to_excel('C:/Kamp/Code/CE_PV_Potential/calc_pv_pot/data/processed/tau_distr/transposed/tau_distr_transposed_v7.xlsx')


# (2) CALCULATION OF RADIATIVE FLUX

from calc.radiation_flux_SAI.preprocessing.data_optical_prop import data_optical_prop
from calc.radiation_flux_SAI.preprocessing.data_tau_volcanic import data_tau_volcanic

from calc.radiation_flux_SAI.flux_delta_eddinton import flux_delta_eddington
from calc.radiation_flux_SAI.delta_eddington import delta_eddington_param
from calc.radiation_flux_SAI.fsf_phasefunc import fsf_calc
from calc.radiation_flux_SAI.generation.header_gen import header_gen


# Initial conditions at incident radiation into the stratosphere
incident_solar_flux = 1361                                                                          # Solar constanr TOA [W/m²]
S_dir_init = incident_solar_flux
S_diff_init = 0
µ_0 = 0.5                 # Cosinus des Sonnenzenitwinkels

# Calculation of optical properties

# import optical properties of sulfate aerosols (humidity 0.0% -> stratsophere is dry)
df_opt_prop = pd.read_excel('C:/Kamp/Code/CE_PV_Potential/calc_pv_pot/data/raw/opt_data_S00.xlsx')
# import phasefunction of sulfate aerosols
df_fsf = pd.read_excel('C:/Kamp/Code/CE_PV_Potential/calc_pv_pot/data/raw/phasefunction_S00.xlsx')
# calculate forward scattering fraction
fsf = fsf_calc(df_fsf)

# Create output-Excel headers
func_rtrt_out = ['R_dir', 'T_dir', 'R_diff', 'T_diff']
func_flux_deltaedd_out = ['F_dir', 'F_diff', 'F_dn']
wavelengths = df_opt_prop['wavelength'][:5].tolist()                                                # list of relevant wavelengths
header_rtrt = header_gen(wavelengths, func_rtrt_out)                                                # header for output Reflectivity, Transmissivity
header_flux_deltaedd = header_gen(wavelengths, func_flux_deltaedd_out)                              # header for output Flux

# Create Dataframes 
df_rtrt_out = pd.DataFrame(columns = ['date', 'tau_bg', 'region'] + header_rtrt)                    # amount = 61 wavelengths * 4 (RRTT) + year, tau = 246 prüfen: passt
df_flux_deltaedd_out = pd.DataFrame(columns = ['date', 'tau_bg', 'region'] + header_flux_deltaedd)


# Schleife 1
for _, row in df_distribution_tau_converted.iterrows(): # _ weil Index nicht benötigt wird
    date = row['Date']
    
# Schleife 2
    for idx, tau in enumerate(row.iloc[1:]):
        header_reg = df_distribution_tau_converted.columns[idx+1][0]                                 # header = Regionen | col with region name as str, which was header of df
        l_rtrt_tau = [date, tau, header_reg]    
        l_two_stream = [date, tau, header_reg]  
        l_flux_deltaedd = [date, tau, header_reg]  
        
# Schleife 3
        for index, wv in enumerate(wavelengths): #wavelength[0]	ext.coef[1]	sca.coef[2]	abs.coef[3]	si.sc.alb=w[4] asym.par=g[5]	ext.nor	ref.real[6]	ref.imag[7]
            w_wv =  float(df_opt_prop['si.sc.alb'][index]) #scingle scattering albdo (w) je wellenlänge (also am index der gerade betrachteten Wellenlänge)
            g_wv = float(df_opt_prop['asym.par'][index])
            f_wv = float(fsf[index])
            
            #Delta-Eddington Approx.
            R_dir, T_dir, R_diff, T_diff = delta_eddington_param(tau, w_wv, g_wv, µ_0, f_wv) #tau aus Schleife 2, w&g aus Varibalen def., f aus liste mit berechneten fsf
            l_rtrt_tau.extend([R_dir, T_dir, R_diff, T_diff])

            #Flux-Calc. with Delta-Edd. (V1)
            F_dir, F_diff, F_dn = flux_delta_eddington(µ_0, tau, S_dir_init, T_dir)
            l_flux_deltaedd.extend([F_dir, F_diff, F_dn])

            
        df_rtrt_out.loc[len(df_rtrt_out)] = l_rtrt_tau
        df_flux_deltaedd_out.loc[len(df_flux_deltaedd_out)] = l_flux_deltaedd


# Generierung der Output-Excel
#df_rtrt_out.to_excel('C:/Kamp/Code/CE_PV_Potential/calc_pv_pot/data/processed/delta_edd_param/rtrt_output4.xlsx')
df_flux_deltaedd_out.to_excel('C:/Kamp/Code/CE_PV_Potential/calc_pv_pot/data/processed/flux_calc/flux_out_v7.xlsx')
