import math

def get_sim_year(timeframe):
    '''
    Year of simulation (1st, 2nd, 3rd ...)
    Args:
    - timeframe = time period of interest in month
    Return:
    - sim_year = year of simulation period (1st, 2nd, 3rd ...)
    '''
    l_timeframe = range(timeframe)
    real_timeframe = []
    
    for elem in l_timeframe:
        elem += 1
        real_timeframe.append(elem)

    
    sim_years = []  
    
    for month_step in real_timeframe: # für jeden Monat des betrachteten Zeitraums
        
        year = math.ceil(month_step/12)
            
        sim_year = str(year)
            
        sim_years.append(sim_year)

    return sim_years