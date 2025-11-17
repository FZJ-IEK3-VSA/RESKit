import numpy as np
import pandas as pd


# Aerosol lifetime (linear buildup after eruption, exp. decrease with e-folding lifetime)

'''
- e-folding lifetime simulates sedimentation (Gao 2008)
- e_folding_time = months (time until ~38% of initial value reached)
- gloabl mean 12.2 months
'''

E_FOLDING_LIFETIME = {
    'trop': 36,
    'extrop': 12,
    'pol_w': 3,
    'pol_rest': 6
}

def aerosol_lifetime(injection, conc, time_steps, region):
    '''
    Calculation of aerosol-distribution with linear buildup for 4 months and exp. decrease from 5th month after eruption

    Args:
    - injection = amount of aerosols from eruption/ injection (maximal)
    - conc = concentration aerosol (vector)
    - time_steps = steps in months (timeframe of interest after eruption)

    Return:
    - conc = Aerosol-concentration after time_steps (months)
    '''
    
    # assign regions to general region
    if region in ['NH_pol', 'SH_pol']:
        general_region = 'pol_rest'
    elif region in ['NH_trop_1', 'NH_trop_2', 'NH_trop_3', 'SH_trop_1', 'SH_trop_2', 'SH_trop_3']:
        general_region = 'trop'
    else:
        general_region = 'extrop'


    # get e-folding lifetime for specific genral region
    e_folding_time = E_FOLDING_LIFETIME[general_region]


    for step in range(time_steps):
        if step <= 4:
            # linear buildup
            buildup_rate = injection / 4
            conc = conc + buildup_rate
        else:
            # exponential decrease
            decrease_rate = np.exp(-step / e_folding_time)
            conc = conc * decrease_rate

    return conc
