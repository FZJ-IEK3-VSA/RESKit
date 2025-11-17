import numpy as np
import pandas as pd
import os

cwd = os.getcwd()

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
    
    'trop_to_trop': 0.91,               # 
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
def create_transilient_matrix_new():

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

    # assumption: transport from tropics polewards (s. stratospheric transport)
    # assumption: no transport between NH ans SH (over equator)

    matrix = np.zeros((num_rows, num_cols, num_z)) # x-dimension: from region | y-dimension: to region | z-dimension: season
    
    for season in range(num_z): # seasons: winter [0], spring[1], summer[3], fall[4]
        
        for col in range(num_cols):

        # from TROPICS (independent from season)
        # NH-trop col 5-7
            if col == 5:                                                    
                matrix[4, col, season] = EXCHANGE_COEFF['trop_to_extrop']
                matrix[col, col, season] = 1 - EXCHANGE_COEFF['trop_to_extrop']
            elif col == 6: 
                matrix[5, col, season] = EXCHANGE_COEFF['trop_to_trop']
                matrix[col, col, season] = 1 - EXCHANGE_COEFF['trop_to_trop']
            elif col == 7: 
                matrix[6, col, season] = EXCHANGE_COEFF['trop_to_trop']
                matrix[col, col, season] = 1 - EXCHANGE_COEFF['trop_to_trop']
            
        # SH-trop col 8-10    
            elif col == 8:  
                matrix[9, col, season] = EXCHANGE_COEFF['trop_to_trop']
                matrix[col, col, season] = 1 - EXCHANGE_COEFF['trop_to_trop']
            elif col == 9: 
                matrix[10, col, season] = EXCHANGE_COEFF['trop_to_trop']
                matrix[col, col, season] = 1 - EXCHANGE_COEFF['trop_to_trop']
            elif col == 10: 
                matrix[11, col, season] = EXCHANGE_COEFF['trop_to_extrop']
                matrix[col, col, season] = 1 - EXCHANGE_COEFF['trop_to_extrop']

            
        # from POLAR REGIONS (independent from season)
        # NH-pol
            elif col == 0:
                matrix[1, col, season] = EXCHANGE_COEFF['pol_to_extrop']
                matrix[col, col, season] = 1 - EXCHANGE_COEFF['pol_to_extrop'] 

        # SH-pol    
            elif col == 15: 
                matrix[14, col, season] = EXCHANGE_COEFF['pol_to_extrop']
                matrix[col, col, season] = 1 - EXCHANGE_COEFF['pol_to_extrop'] 


        # from EXTRATROPICS (seasonal dependency)
        # NH-extrop col 1-4
            elif col == 1:
                if season == 0: #winter
                    matrix[0, col, season] = EXCHANGE_COEFF['extrop_to_polar_w']
                    matrix[col, col, season] = 1 - EXCHANGE_COEFF['extrop_to_polar_w'] 
                elif season == 2: #summer
                    matrix[0, col, season] = EXCHANGE_COEFF['extrop_to_polar_s']
                    matrix[col, col, season] = 1 - EXCHANGE_COEFF['extrop_to_polar_s'] 
                else: #spring/fall
                    matrix[0, col, season] = EXCHANGE_COEFF['extrop_to_polar_sp_f']
                    matrix[col, col, season] = 1 - EXCHANGE_COEFF['extrop_to_polar_sp_f'] 

            elif col == 2:
                if season in [0,1]: #winter/spring
                    matrix[1, col, season] = EXCHANGE_COEFF['extrop_to_extrop_w_spr']
                    matrix[col, col, season] = 1 - EXCHANGE_COEFF['extrop_to_extrop_w_spr'] 
                else: #summer/fall
                    matrix[1, col, season] = EXCHANGE_COEFF['extrop_to_extrop_s_f']
                    matrix[col, col, season] = 1 - EXCHANGE_COEFF['extrop_to_extrop_s_f'] 
                    
            
            elif col == 3:
                if season in [0,1]: #winter/spring
                    matrix[2, col, season] = EXCHANGE_COEFF['extrop_to_extrop_w_spr']
                    matrix[col, col, season] = 1 - EXCHANGE_COEFF['extrop_to_extrop_w_spr'] 
                else: #summer/fall
                    matrix[2, col, season] = EXCHANGE_COEFF['extrop_to_extrop_s_f']
                    matrix[col, col, season] = 1 - EXCHANGE_COEFF['extrop_to_extrop_s_f'] 

            elif col == 4:
                if season in [0,1]: #winter/spring
                    matrix[3, col, season] = EXCHANGE_COEFF['extrop_to_extrop_w_spr']
                    matrix[5, col, season] = EXCHANGE_COEFF['extrop_to_trop']
                    matrix[col, col, season] = 1 - EXCHANGE_COEFF['extrop_to_extrop_w_spr'] - EXCHANGE_COEFF['extrop_to_trop']
                else: #summer/fall
                    matrix[3, col, season] = EXCHANGE_COEFF['extrop_to_extrop_s_f']
                    matrix[5, col, season] = EXCHANGE_COEFF['extrop_to_trop']
                    matrix[col, col, season] = 1 - EXCHANGE_COEFF['extrop_to_extrop_s_f'] - EXCHANGE_COEFF['extrop_to_trop']

        # SH-extrop col 11-14    
            elif col == 11:
                if season in [0,1]: #winter/spring
                    matrix[12, col, season] = EXCHANGE_COEFF['extrop_to_extrop_w_spr']
                    matrix[10, col, season] = EXCHANGE_COEFF['extrop_to_trop']
                    matrix[col, col, season] = 1 - EXCHANGE_COEFF['extrop_to_extrop_w_spr'] - EXCHANGE_COEFF['extrop_to_trop']
                else: #summer/fall
                    matrix[12, col, season] = EXCHANGE_COEFF['extrop_to_extrop_s_f']
                    matrix[10, col, season] = EXCHANGE_COEFF['extrop_to_trop']
                    matrix[col, col, season] = 1 - EXCHANGE_COEFF['extrop_to_extrop_s_f'] - EXCHANGE_COEFF['extrop_to_trop']

            elif col == 12:
                if season in [0,1]: #winter/spring
                    matrix[13, col, season] = EXCHANGE_COEFF['extrop_to_extrop_w_spr']
                    matrix[col, col, season] = 1 - EXCHANGE_COEFF['extrop_to_extrop_w_spr'] 
                else: #summer/fall
                    matrix[13, col, season] = EXCHANGE_COEFF['extrop_to_extrop_s_f']
                    matrix[col, col, season] = 1 - EXCHANGE_COEFF['extrop_to_extrop_s_f']

            elif col == 13:
                if season in [0,1]: #winter/spring
                    matrix[14, col, season] = EXCHANGE_COEFF['extrop_to_extrop_w_spr']
                    matrix[col, col, season] = 1 - EXCHANGE_COEFF['extrop_to_extrop_w_spr']
                else: #summer/fall
                    matrix[14, col, season] = EXCHANGE_COEFF['extrop_to_extrop_s_f']
                    matrix[col, col, season] = 1 - EXCHANGE_COEFF['extrop_to_extrop_s_f']

            elif col == 14:
                if season == 0: #winter
                    matrix[15, col, season] = EXCHANGE_COEFF['extrop_to_polar_w']
                    matrix[col, col, season] = 1 - EXCHANGE_COEFF['extrop_to_polar_w']
                elif season == 2: #summer
                    matrix[15, col, season] = EXCHANGE_COEFF['extrop_to_polar_s']
                    matrix[col, col, season] = 1 - EXCHANGE_COEFF['extrop_to_polar_s']
                else: #spring/fall
                    matrix[15, col, season] = EXCHANGE_COEFF['extrop_to_polar_sp_f']
                    matrix[col, col, season] = 1 - EXCHANGE_COEFF['extrop_to_polar_sp_f']

        
        # store matrix to df
        df_exchange_matrix_w = pd.DataFrame(matrix[:, :, 0], index=header_regions, columns=header_regions)
        df_exchange_matrix_sp = pd.DataFrame(matrix[:, :, 1], index=header_regions, columns=header_regions)
        df_exchange_matrix_s = pd.DataFrame(matrix[:, :, 2], index=header_regions, columns=header_regions)
        df_exchange_matrix_f = pd.DataFrame(matrix[:, :, 3], index=header_regions, columns=header_regions)
        
        # store df to excel cwd + f'/data/processed/sim_data_16x16/{version}.nc4'
        df_exchange_matrix_w.to_excel(cwd + '\SAI Code\calc_pv_pot/data/processed/exchange_matrix/ex_w_2.xlsx')
        df_exchange_matrix_sp.to_excel(cwd + '\SAI Code\calc_pv_pot/data/processed/exchange_matrix/ex_sp_2.xlsx')
        df_exchange_matrix_s.to_excel(cwd + '\SAI Code\calc_pv_pot/data/processed/exchange_matrix/ex_s_2.xlsx')
        df_exchange_matrix_f.to_excel(cwd + '\SAI Code\calc_pv_pot/data/processed/exchange_matrix/ex_f_2.xlsx')
        
        # output matrix as array
        matrix_w = matrix[:, :, 0]
        matrix_sp = matrix[:, :, 1]
        matrix_s = matrix[:, :, 2]
        matrix_f = matrix[:, :, 3]

        print('matrix created')

    return matrix_w, matrix_sp, matrix_s, matrix_f

############################################################################################

# Exchange-Coeff.-Matrix
def create_transilient_matrix_new_new():

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

    # assumption: transport from tropics polewards (s. stratospheric transport)
    # assumption: no transport between NH ans SH (over equator)

    matrix = np.zeros((num_rows, num_cols, num_z)) # x-dimension: from region | y-dimension: to region | z-dimension: season
    
    for season in range(num_z): # seasons: winter [0], spring[1], summer[2], fall[3]
        
        for col in range(num_cols):

        # from TROPICS (independent from season)
        # NH-trop col 5-7
            if col == 5:                                                    
                matrix[4, col, season] = EXCHANGE_COEFF['trop_to_extrop']
                matrix[col, col, season] = 1 - EXCHANGE_COEFF['trop_to_extrop']
            elif col == 6: 
                matrix[5, col, season] = EXCHANGE_COEFF['trop_to_trop']
                matrix[col, col, season] = 1 - EXCHANGE_COEFF['trop_to_trop']
            elif col == 7: 
                matrix[6, col, season] = EXCHANGE_COEFF['trop_to_trop']
                matrix[col, col, season] = 1 - EXCHANGE_COEFF['trop_to_trop']
            
        # SH-trop col 8-10    
            elif col == 8:  
                matrix[9, col, season] = EXCHANGE_COEFF['trop_to_trop']
                matrix[col, col, season] = 1 - EXCHANGE_COEFF['trop_to_trop']
            elif col == 9: 
                matrix[10, col, season] = EXCHANGE_COEFF['trop_to_trop']
                matrix[col, col, season] = 1 - EXCHANGE_COEFF['trop_to_trop']
            elif col == 10: 
                matrix[11, col, season] = EXCHANGE_COEFF['trop_to_extrop']
                matrix[col, col, season] = 1 - EXCHANGE_COEFF['trop_to_extrop']

            
        # from POLAR REGIONS (independent from season)
        # NH-pol
            elif col == 0:
                matrix[1, col, season] = EXCHANGE_COEFF['pol_to_extrop']
                matrix[col, col, season] = 1 - EXCHANGE_COEFF['pol_to_extrop'] 

        # SH-pol    
            elif col == 15: 
                matrix[14, col, season] = EXCHANGE_COEFF['pol_to_extrop']
                matrix[col, col, season] = 1 - EXCHANGE_COEFF['pol_to_extrop'] 


        # from EXTRATROPICS (seasonal dependency)
        # NH-extrop col 1-4
            elif col == 1:
                if season == 0: #winter
                    matrix[0, col, season] = EXCHANGE_COEFF['extrop_to_polar_w']
                    matrix[col, col, season] = 1 - EXCHANGE_COEFF['extrop_to_polar_w'] 
                elif season == 2: #summer
                    matrix[0, col, season] = EXCHANGE_COEFF['extrop_to_polar_s']
                    matrix[col, col, season] = 1 - EXCHANGE_COEFF['extrop_to_polar_s'] 
                else: #spring/fall
                    matrix[0, col, season] = EXCHANGE_COEFF['extrop_to_polar_sp_f']
                    matrix[col, col, season] = 1 - EXCHANGE_COEFF['extrop_to_polar_sp_f'] 

            elif col == 2:
                if season in [0,1]: #winter/spring
                    matrix[1, col, season] = EXCHANGE_COEFF['extrop_to_extrop_w_spr']
                    matrix[col, col, season] = 1 - EXCHANGE_COEFF['extrop_to_extrop_w_spr'] 
                else: #summer/fall
                    matrix[1, col, season] = EXCHANGE_COEFF['extrop_to_extrop_s_f']
                    matrix[col, col, season] = 1 - EXCHANGE_COEFF['extrop_to_extrop_s_f'] 
                    
            
            elif col == 3:
                if season in [0,1]: #winter/spring
                    matrix[2, col, season] = EXCHANGE_COEFF['extrop_to_extrop_w_spr']
                    matrix[col, col, season] = 1 - EXCHANGE_COEFF['extrop_to_extrop_w_spr'] 
                else: #summer/fall
                    matrix[2, col, season] = EXCHANGE_COEFF['extrop_to_extrop_s_f']
                    matrix[col, col, season] = 1 - EXCHANGE_COEFF['extrop_to_extrop_s_f'] 

            elif col == 4:
                if season in [0,1]: #winter/spring
                    matrix[3, col, season] = EXCHANGE_COEFF['extrop_to_extrop_w_spr']
                    matrix[5, col, season] = EXCHANGE_COEFF['extrop_to_trop']
                    matrix[col, col, season] = 1 - EXCHANGE_COEFF['extrop_to_extrop_w_spr'] - EXCHANGE_COEFF['extrop_to_trop']
                else: #summer/fall
                    matrix[3, col, season] = EXCHANGE_COEFF['extrop_to_extrop_s_f']
                    matrix[5, col, season] = EXCHANGE_COEFF['extrop_to_trop']
                    matrix[col, col, season] = 1 - EXCHANGE_COEFF['extrop_to_extrop_s_f'] - EXCHANGE_COEFF['extrop_to_trop']

        # SH-extrop col 11-14    
            elif col == 11:
                if season in [0,1]: #winter/spring CHANGE TO summer/fall
                    matrix[12, col, season] = EXCHANGE_COEFF['extrop_to_extrop_s_f']
                    matrix[10, col, season] = EXCHANGE_COEFF['extrop_to_trop']
                    matrix[col, col, season] = 1 - EXCHANGE_COEFF['extrop_to_extrop_s_f'] - EXCHANGE_COEFF['extrop_to_trop']
                else: #summer/fall CHANGE TO winter/spring
                    matrix[12, col, season] = EXCHANGE_COEFF['extrop_to_extrop_w_spr']
                    matrix[10, col, season] = EXCHANGE_COEFF['extrop_to_trop']
                    matrix[col, col, season] = 1 - EXCHANGE_COEFF['extrop_to_extrop_w_spr'] - EXCHANGE_COEFF['extrop_to_trop']

            elif col == 12:
                if season in [0,1]: #winter/spring
                    matrix[13, col, season] = EXCHANGE_COEFF['extrop_to_extrop_s_f']
                    matrix[col, col, season] = 1 - EXCHANGE_COEFF['extrop_to_extrop_s_f'] 
                else: #summer/fall
                    matrix[13, col, season] = EXCHANGE_COEFF['extrop_to_extrop_w_spr']
                    matrix[col, col, season] = 1 - EXCHANGE_COEFF['extrop_to_extrop_w_spr']

            elif col == 13:
                if season in [0,1]: #winter/spring
                    matrix[14, col, season] = EXCHANGE_COEFF['extrop_to_extrop_s_f']
                    matrix[col, col, season] = 1 - EXCHANGE_COEFF['extrop_to_extrop_s_f']
                else: #summer/fall
                    matrix[14, col, season] = EXCHANGE_COEFF['extrop_to_extrop_w_spr']
                    matrix[col, col, season] = 1 - EXCHANGE_COEFF['extrop_to_extrop_w_spr']

            elif col == 14:
                if season == 0: #winter
                    matrix[15, col, season] = EXCHANGE_COEFF['extrop_to_polar_s']
                    matrix[col, col, season] = 1 - EXCHANGE_COEFF['extrop_to_polar_s']
                elif season == 2: #summer
                    matrix[15, col, season] = EXCHANGE_COEFF['extrop_to_polar_w']
                    matrix[col, col, season] = 1 - EXCHANGE_COEFF['extrop_to_polar_w']
                else: #spring/fall
                    matrix[15, col, season] = EXCHANGE_COEFF['extrop_to_polar_sp_f']
                    matrix[col, col, season] = 1 - EXCHANGE_COEFF['extrop_to_polar_sp_f']

        
        # store matrix to df
        df_exchange_matrix_w = pd.DataFrame(matrix[:, :, 0], index=header_regions, columns=header_regions)
        df_exchange_matrix_sp = pd.DataFrame(matrix[:, :, 1], index=header_regions, columns=header_regions)
        df_exchange_matrix_s = pd.DataFrame(matrix[:, :, 2], index=header_regions, columns=header_regions)
        df_exchange_matrix_f = pd.DataFrame(matrix[:, :, 3], index=header_regions, columns=header_regions)
        
        # store df to excel
        df_exchange_matrix_w.to_excel(cwd + '\SAI Code\calc_pv_pot/data/processed/exchange_matrix/ex_w_new.xlsx')
        df_exchange_matrix_sp.to_excel(cwd + '\SAI Code\calc_pv_pot/data/processed/exchange_matrix/ex_sp_new.xlsx')
        df_exchange_matrix_s.to_excel(cwd + '\SAI Code\calc_pv_pot/data/processed/exchange_matrix/ex_s_new.xlsx')
        df_exchange_matrix_f.to_excel(cwd + '\SAI Code\calc_pv_pot/data/processed/exchange_matrix/ex_f_new.xlsx')
        
        # output matrix as array
        matrix_w = matrix[:, :, 0]
        matrix_sp = matrix[:, :, 1]
        matrix_s = matrix[:, :, 2]
        matrix_f = matrix[:, :, 3]

        print('matrix created')

    return matrix_w, matrix_sp, matrix_s, matrix_f
