import numpy as np
import pandas as pd

from calc.simulation_aerosol_distribution.get_season import get_season
from calc.simulation_aerosol_distribution.get_month_str import get_month_str
from calc.simulation_aerosol_distribution.get_sim_year import get_sim_year


LIST_REGIONS = ['NH_pol', 'NH_extrop_1', 'NH_extrop_2', 'NH_extrop_3', 'NH_extrop_4', 'NH_trop_1', 'NH_trop_2', 'NH_trop_3', 'SH_trop_1', 'SH_trop_2', 'SH_trop_3', 'SH_extrop_1', 'SH_extrop_2', 'SH_extrop_3', 'SH_extrop_4', 'SH_pol']

E_FOLDING_LIFETIME = {
    'NH_pol': 6,
    'NH_extrop_1': 12,
    'NH_extrop_2': 12,
    'NH_extrop_3': 12,
    'NH_extrop_4': 12,
    'NH_trop_1': 36,
    'NH_trop_2': 36,
    'NH_trop_3': 36,
  
    'SH_trop_1': 36,
    'SH_trop_2': 36,
    'SH_trop_3': 36,
    'SH_extrop_1': 12,
    'SH_extrop_2': 12,
    'SH_extrop_3': 12,
    'SH_extrop_4': 12,
    'SH_pol': 6 # FALL WINTER =3 später einfügen
}


def e_folding_vector():

    e_folding_array = np.zeros(len(LIST_REGIONS))           # 0er array mit der Länge der Regionen als initial

    for idx, region in enumerate(LIST_REGIONS):

        e_folding_time = E_FOLDING_LIFETIME[region]         # e-folding time jeder region aus dict holen
        decrease_rate = np.exp(- 1 / e_folding_time)        # exponential decrease aus e-folding time bestimmen
        e_folding_array[idx] = decrease_rate                # exp-abnahmerate in array hinzufügen

    return e_folding_array





def simulate_distribution(distr_4month, timeframe, injection_date, transport_matrices):

    '''
    calculate final distribution after applying transport mechanism and sedimentation
    Args:
    -
    - timeframe = time period of interest
    - injection_date = month of injection
    - transport_matrices = transport matrices for each seasons
    
    Return:
    - 
    '''

    #distr = np.copy(init_distr) # copy of initial distr.

    df_distr_months = pd.DataFrame()                            # dataframe to save the monthly distribution of the 16 belts
    distr_months = []                                           # list to save the monthly distribution of the 16 belts
    year_of_sim = get_sim_year(timeframe)                       # list of years as strings (year of simulation) for every month
    decrease_array = e_folding_vector()                         # vector of exp. decrease rates, depending on the region


    for idx, month in enumerate(range(timeframe)):              # for every month in timeframe that is looked at
        
        current_month = (injection_date + month - 1) % 12 + 1   # current month in simulation (0 in timesteps but 4 in reality (APR) due to month of injection)
        season = get_season(current_month)                      # season of the current month -> important for seasonal exchange-matrix
        current_month_str = get_month_str(current_month)        # current month as str | current month for calculation (in 1-12)


        # first distr. = initial distr.
        if month==0:
            distr_new = distr_4month[0]
            df_distr_months[current_month_str + '' + year_of_sim[idx]] = distr_new        # erstes array der erste-4-monate-distribution
            distr_months.append(distr_new)

        if month==1:
            distr_new = distr_4month[1]
            df_distr_months[current_month_str + '' + year_of_sim[idx]] = distr_new       # zweites array der erste-4-monate-distribution
            distr_months.append(distr_new)

        if month==2:
            distr_new = distr_4month[2]
            df_distr_months[current_month_str + '' + year_of_sim[idx]] = distr_new       # drittes array der erste-4-monate-distribution
            distr_months.append(distr_new)
        
        if month==3:
            distr_new = distr_4month[3]
            df_distr_months[current_month_str + '' + year_of_sim[idx]] = distr_new      # viertes array der erste-4-monate-distribution
            distr_months.append(distr_new)

        # from 5th month until end of time period
        if month > 3:
            distr_new = np.dot(transport_matrices[season], distr_months[month-1])       # calc of aerosol conc. after application of the transport matrix (vector matrix multiplication)
           
            distr_final = distr_new * decrease_array                                    # exopnentielle abnahme auf jeden Ort anwenden (und das jeden monat)
           
            df_distr_months[current_month_str + '' + year_of_sim[idx]] = distr_final
            distr_months.append(distr_final)
           
            #df_distr_months[current_month_str + '' + year_of_sim[idx]] = distr_new      # TEST: without exp. decrease
            #distr_months.append(distr_new)                                              # TEST: without exp. decrease
                
    df_distr_months.index = [LIST_REGIONS]

    print('distribution simulated') 

    return df_distr_months
