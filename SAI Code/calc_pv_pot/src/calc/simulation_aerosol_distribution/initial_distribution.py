import numpy as np

def initial_distribution(reg_injection, amount_injection):

    '''
    - default distr. = 0
    - function adds injection of aerosols to initial distribution

    Args:
    - lat_injection = latitude [°] of injection/ eruption
    - amount_injection = aerosol conc. [Mt]

    Return:
    - initial_distribution = updatet distribution of the 16 belts
    - key_region = region of lat. input 
    '''


    # initial distribution: default 0

    initial_distribution = np.zeros(16)

    list_regions = ['NH_pol', 'NH_extrop_1', 'NH_extrop_2', 'NH_extrop_3', 'NH_extrop_4', 'NH_trop_1', 'NH_trop_2', 'NH_trop_3', 'SH_trop_1', 'SH_trop_2', 'SH_trop_3', 'SH_extrop_1', 'SH_extrop_2', 'SH_extrop_3', 'SH_extrop_4', 'SH_pol']

    # linear buildup for 4 month -> first injection is 1/4 of max. amount
    t0_injection = amount_injection / 4

    # replace value of initial distribution with amount of injection at the right region

    for index, region in enumerate(list_regions):
        if region == reg_injection:
            initial_distribution[index] = t0_injection

    return initial_distribution