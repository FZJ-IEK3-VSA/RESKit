import pvlib as pv
import pandas as pd
import numpy as np

REGIONS_LAT = { 'NH_pol': (90, 61),
                    
                    'NH_extrop_1': (61, 50.49),
                    'NH_extrop_2': (50.49, 39.79),
                    'NH_extrop_3': (39.79, 30.14),
                    'NH_extrop_4': (30.14, 22),

                    'NH_trop_1': (22, 14.84),
                    'NH_trop_2': (14.84, 7.48),
                    'NH_trop_3': (7.48, 0), 

                    'SH_trop_1': (0, -7.48), 
                    'SH_trop_2': (-7.48, -14.84), 
                    'SH_trop_3': (-14.84, -22), 

                    'SH_extrop_1': (-22, -30.14), 
                    'SH_extrop_2': (-30.14, -39.79), 
                    'SH_extrop_3': (-39.79, -50.49), 
                    'SH_extrop_4': (-50.49, -61), 

                    'SH_pol': (-61, -90)    }

LIST_REGIONS = ['NH_pol', 'NH_extrop_1', 'NH_extrop_2', 'NH_extrop_3', 'NH_extrop_4', 'NH_trop_1', 'NH_trop_2', 'NH_trop_3', 'SH_trop_1', 'SH_trop_2', 'SH_trop_3', 'SH_extrop_1', 'SH_extrop_2', 'SH_extrop_3', 'SH_extrop_4', 'SH_pol']


# Betrachtung gesamten Tag mit wandernder Sonne


# hourly resolution for a day

def get_time_period(timeframe, injection_month, injection_year, injection_day):

    '''
    Gets time period in hourly steps from injection date
    Args:
    - timeframe = period of month 
    - injection_month = month of injection as int
    - injection_year = month of injection as int
    - injection_day =month of injection as int
    Return:
    - times = date times from start to stop as ['1991-06-15 00:00:00']
    '''
    # Startzeitpunkt
    start = str(injection_year) + '-' + str(injection_month) + '-' + str(injection_day) + ' ' + '00:30'

    # Stopzeitpunkt
    years = round(timeframe/12) # Anzahl Jahre
    if (injection_month + timeframe) > 12:
        stop_month = injection_month + timeframe - (12*years)
        stop_year = injection_year + years
    else:
        stop_month = injection_month + timeframe
        stop_year = injection_year
    
    stop = str(stop_year) + '-' + str(stop_month) + '-' + str(injection_day) + ' ' + '23:30'
    
    times = pd.date_range(start, stop, freq='1H')

    return times




def get_solarzenith(times):

    '''
    Args:
    - times = date times from start to stop as ['1991-06-15 00:00:00']
    
    Return:
    - all_sol_pos = df with [region/lat	| time_stamp | apparent_zenith | zenith	| apparent_elevation | elevation | azimuth | equation_of_time]
    -> for every region all timesteps are iterted, than next region
    '''
    
    lat_dict = {}
    all_sol_pos = pd.DataFrame()
    
    for key, val in REGIONS_LAT.items():  
        mean = sum(val) / len(val)                                                                      # mean lat each 16 belts
        lat_dict[key] = mean                                                                            # new dict: {region & lat_mean}
        
    for key_region, val_mean_lat in lat_dict.items(): 
        
        sol_pos = pv.solarposition.get_solarposition(times, val_mean_lat, 0)                            # sol position for every timestep and mean_lat (each 16 belts)
        sol_pos = sol_pos.reset_index()                                                                 # set timesteps as column, new index
        sol_pos.rename(columns={'index': 'time_stamp'}, inplace=True)                                   # rename timestep col
        #sol_pos.insert(loc=0, column='region/lat', value=['' for i in range(sol_pos.shape[0])])         # add new col (as first col): region(of the mean_lat) -> empty col
        sol_pos.insert(0, 'region/lat', key_region)


        all_sol_pos = pd.concat([all_sol_pos, sol_pos], ignore_index=True)
    
    # sort all_sol_pos 
    all_sol_pos['region/lat'] = pd.Categorical(all_sol_pos['region/lat'], categories=LIST_REGIONS, ordered=True)

    all_sol_pos_sorted = all_sol_pos.sort_values(by=['time_stamp', 'region/lat'])
    all_sol_pos_sorted = all_sol_pos_sorted.reset_index(drop=True)

    µ = get_µ(all_sol_pos_sorted)
    all_sol_pos_sorted.insert(3, 'µ=cos(app_zenith)', µ)

    print('solar position localised')
    
    return all_sol_pos_sorted



def get_µ(solar_position):

    zenith_angle = solar_position['apparent_zenith']

    µ = np.cos(np.radians(zenith_angle))

    return µ



