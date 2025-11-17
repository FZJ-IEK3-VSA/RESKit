import numpy as np
import pandas as pd

'''
- Calculation of aerosol disribution according to Grieser 1999 and Gao 2008
- further referred to as "matrix-approach"

Steps:
0. Exchange coeff.
1. Function for initial distribution
2. Function for aerosol lifetime (linear buildup and exp. decrease)
3. Function for exchange-coeff.-matrix (16x16x4)
4. Function for aerosol-transport simulation (depending on exchange-coeff-matrix and temporal resolution (aerosol lifetime))
5. In MAIN: create matrix -> Output final distribution?
'''

# Exchange coeff. according to Gao 2008 as dict.

'''
trop = tropics
extrop = extratropics
w = winter
s = summer
sp = spring
f = fall
'''

EXCHANGE_COEFF = {                 # in (%) per month
    
    'trop_to_trop': 0.91,
    'trop_to_extrop': 0.50,
    'extrop_to_trop': 0.07,
    'extrop_to_extrop_w_spr': 0.90,
    'extrop_to_extrop_s_f': 0.70,
    'extrop_to_polar_w': 0.10,
    'extrop_to_polar_s': 0.70,
    'extrop_to_polar_sp_f': 0.04,
    'pol_to_extrop': 0.04
}


# Exchange-Coeff.-Matrix
def create_transilient_matrix():

    '''
    Creation of 16x16x4 matrix for exchange coefficients between each 16 belts for every 4 seasons

    Args: 
    - no arguments
    
    Return:
    - matrix = exchange-matrix as array
    - df_exchange_matrix_w = matrix for winter as df
    - df_exchange_matrix_s = matrix for summer as df
    - df_exchange_matrix_sp = matrix for spring as df
    - df_exchange_matrix_f = matrix for fall as df
    '''

    header_regions = ['NH_pol', 'NH_extrop_1', 'NH_extrop_2', 'NH_extrop_3', 'NH_extrop_4', 'NH_trop_1', 'NH_trop_2', 'NH_trop_3', 'SH_trop_1', 'SH_trop_2', 'SH_trop_3', 'SH_extrop_1', 'SH_extrop_2', 'SH_extrop_3', 'SH_extrop_4', 'SH_pol']

    num_rows = 16
    num_cols = 16
    num_z = 4

    matrix = np.zeros((num_rows, num_cols, num_z)) # x-dimension: from region | y-dimension: to region | z-dimension: season
    
    for season in range(num_z): # seasons: winter [0], spring[1], summer[3], fall[4]
        
        for col in range(num_cols):

        # from TROPICS (independent from season)
            if 5 <= col <= 10:
                for row in range(num_rows):
                    if row == col:
                        matrix[row, col, season] = 0 # diagonal in matrix
                    elif row == 0 or col == 15:
                        matrix[row, col, season] = 0 # no exchange trop to pol
                    elif 5 <= row <= 10:
                        matrix[row, col, season] = EXCHANGE_COEFF['trop_to_trop']
                    else:
                        matrix[row, col, season] = EXCHANGE_COEFF['trop_to_extrop']
            
        # from POLAR REGIONS (independent from season)
            elif col == 0 or col == 15:
                for row in range(num_rows):
                    if row == col:
                        matrix[row, col, season] = 0 # diagonal in matrix
                    elif row == 0 or col == 15:
                        matrix[row, col, season] = 0 # no exchange pol to pol
                    elif 5 <= row <= 10:
                        matrix[row, col, season] = 0 # no exchange from pol to trop
                    else:
                        matrix[row, col, season] = EXCHANGE_COEFF['pol_to_extrop'] 

        # from EXTRATROPICS
            else:  
                for row in range(num_rows):
                    if row == col:
                        matrix[row, col, season] = 0 # diagonal in matrix
                    elif row == 0 or row == 15:
                        if season == 0: #winter
                            matrix[row, col, season] = EXCHANGE_COEFF['extrop_to_polar_w']
                        elif season == 2: #summer
                            matrix[row, col, season] = EXCHANGE_COEFF['extrop_to_polar_s']
                        else: #spring/fall
                            matrix[row, col, season] = EXCHANGE_COEFF['extrop_to_polar_sp_f']
                    elif 5 <= row <= 10:
                        matrix[row, col, season] = EXCHANGE_COEFF['extrop_to_trop']
                    else:
                        if season in [0,1]: #winter/spring
                            matrix[row, col, season] = EXCHANGE_COEFF['extrop_to_extrop_w_spr']
                        else: #summer/fall
                            matrix[row, col, season] = EXCHANGE_COEFF['extrop_to_extrop_s_f']
        
        # store matrix to df
        df_exchange_matrix_w = pd.DataFrame(matrix[:, :, 0], index=header_regions, columns=header_regions)
        df_exchange_matrix_s = pd.DataFrame(matrix[:, :, 1], index=header_regions, columns=header_regions)
        df_exchange_matrix_sp = pd.DataFrame(matrix[:, :, 2], index=header_regions, columns=header_regions)
        df_exchange_matrix_f = pd.DataFrame(matrix[:, :, 3], index=header_regions, columns=header_regions)
        
        # store df to excel
        df_exchange_matrix_w.to_excel('C:/Kamp/Code/CE_PV_Potential/calc_pv_pot/data/processed/exchange_matrix/exchange_matrix_winter.xlsx')
        df_exchange_matrix_s.to_excel('C:/Kamp/Code/CE_PV_Potential/calc_pv_pot/data/processed/exchange_matrix/exchange_matrix_summer.xlsx')
        df_exchange_matrix_sp.to_excel('C:/Kamp/Code/CE_PV_Potential/calc_pv_pot/data/processed/exchange_matrix/exchange_matrix_spring.xlsx')
        df_exchange_matrix_f.to_excel('C:/Kamp/Code/CE_PV_Potential/calc_pv_pot/data/processed/exchange_matrix/exchange_matrix_fall.xlsx')
        
        # output matrix as array
        matrix_w = matrix[:, :, 0]
        matrix_sp = matrix[:, :, 1]
        matrix_s = matrix[:, :, 2]
        matrix_f = matrix[:, :, 3]

    return matrix_w, matrix_sp, matrix_s, matrix_f


