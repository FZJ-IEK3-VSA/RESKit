import numpy as np
import pandas as pd
#from scipy.integrate import simps
 

# scattering angles in Spalte 1 mit -> 0 | 0,1 | 0,2 | ... | 0,9 | 1 | 1,1 | 1,2 | ... | 1,9 | 2 | 3 | 4 | ... | 10 | 12 | 13 | 14 | ... | 180 (Summe 112 Winkel)
# sca_angles = ESRTE SPALTE DER EXCEL

def fsf_calc(df):

    '''
    Funktion berechnet die FSF für jede Wellenlänge
    
    FSF = vorwärts-streuender Teil / total

    Args:
    - df: dataframe der Phasenwinkel

    Return: 
    - forward_scattering_fraction
    '''

    wavelengths = df.columns[1:]

    forward_scattering_fraction = []

    for wavelength in wavelengths:

        vpf_values = df[wavelength] #Betrachtung jede spalte einzeln (jede wv nacheinander)

        sc_angles = df['sc_ang']

        # Umwandlung der Winkel in Rad für die Integration
        sc_angles_rad = np.deg2rad(sc_angles)
        
        I_forward = simps(vpf_values[:91] * np.sin(sc_angles_rad[:91]), sc_angles_rad[:91])
        I_total = simps(vpf_values * np.sin(sc_angles_rad), sc_angles_rad)
        
        FSF = I_forward / I_total

        forward_scattering_fraction.append(FSF)

    return forward_scattering_fraction
