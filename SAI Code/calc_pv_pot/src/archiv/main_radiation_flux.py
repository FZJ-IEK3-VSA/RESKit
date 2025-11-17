import pandas as pd
import numpy as np

from calc_pv_pot.src.calc.radiation_flux_SAI.preprocessing.data_optical_prop import data_optical_prop
from calc_pv_pot.src.calc.radiation_flux_SAI.preprocessing.data_tau_volcanic import data_tau_volcanic

from calc_pv_pot.src.calc.radiation_flux_SAI.flux_delta_eddinton import flux_delta_eddington
from calc_pv_pot.src.calc.radiation_flux_SAI.delta_eddington import delta_eddington_param
from calc_pv_pot.src.calc.radiation_flux_SAI.fsf_phasefunc import fsf_calc
from calc_pv_pot.src.calc.radiation_flux_SAI.generation.header_gen import header_gen


# Calculation optical properties
# f bestimmen für jede Wellenlänge

incident_solar_flux = 1361 # Solar constanr TOA (W/m²)

# Initial conditions at incident radiation into the stratosphere
S_dir_init = incident_solar_flux
S_diff_init = 0
µ_0 = 0

# Import data
df_tau_volcanic = pd.read_excel('C:/Kamp/Code/CE_PV_Potential/calc_pv_pot/data/processed/aod_div1000.xlsx')
df_opt_prop = pd.read_excel('C:/Kamp/Code/CE_PV_Potential/calc_pv_pot/data/raw/opt_data_S00.xlsx')
df_fsf = pd.read_excel('C:/Kamp/Code/CE_PV_Potential/calc_pv_pot/data/raw/phasefunction_S00.xlsx')

# Calculation of fsf                   
fsf = fsf_calc(df_fsf) # OUT = Liste von FSF je Wellenlänge

# Erstellen der Kopfzeilen für die Excel
func_rtrt_out = ['R_dir', 'T_dir', 'R_diff', 'T_diff']
func_flux_deltaedd_out = ['F_dir', 'F_diff', 'F_dn']

wavelengths = df_opt_prop['wavelength'][:5].tolist()
header_rtrt = header_gen(wavelengths, func_rtrt_out)
header_flux_deltaedd = header_gen(wavelengths, func_flux_deltaedd_out)


# Dataframes erstellen
df_rtrt_out = pd.DataFrame(columns = ['year', 'tau_bg'] + header_rtrt) # Anzahl = 61 wellenlängen * 4 (RRTT) + year, tau = 246 prüfen: passt
df_flux_deltaedd_out = pd.DataFrame(columns = ['year', 'tau_bg'] + header_flux_deltaedd)


# Schleife 1
for _, row in df_tau_volcanic.iterrows(): # _ weil Index nicht benötigt wird
    year = row['Year']
    
# Schleife 2
    for tau in row.iloc[1:]:
        l_rtrt_tau = [year, tau]    
        l_two_stream = [year, tau]  
        l_flux_deltaedd = [year, tau]  
        
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
df_flux_deltaedd_out.to_excel('C:/Kamp/Code/CE_PV_Potential/calc_pv_pot/data/processed/flux_calc/flux_deltaedd_output6_wvbis4µm_µ=0.xlsx')



# Next steps:
'''
- Delta zwischen 1361 und berechneter F_dn_dir bestimmen
- Sonnenstand mit einbeziehen(µ?)
- 
'''







