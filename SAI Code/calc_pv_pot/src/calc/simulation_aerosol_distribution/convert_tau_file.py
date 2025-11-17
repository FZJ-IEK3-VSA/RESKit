import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import os

cwd = os.getcwd()

MONTH_DICT = {
        'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
        'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
    }

LIST_REGIONS = ['NH_pol', 'NH_extrop_1', 'NH_extrop_2', 'NH_extrop_3', 'NH_extrop_4', 'NH_trop_1', 'NH_trop_2', 'NH_trop_3', 'SH_trop_1', 'SH_trop_2', 'SH_trop_3', 'SH_extrop_1', 'SH_extrop_2', 'SH_extrop_3', 'SH_extrop_4', 'SH_pol']



def get_time_period(timeframe, injection_month, injection_year, injection_day):

    '''
    Gets time period in hourly steps from injection date
    Args:
    - timeframe = period of month 
    - injection_month = month of injection as int
    - injection_year = month of injection as int
    - injection_day = month of injection as int
    Return:
    - times = date times from start to stop as ['1991-06-15 00:00:00']
    '''

    # Startzeitpunkt
    start = pd.Timestamp(year=injection_year, month=injection_month, day=injection_day, hour=0, minute=30)

    # Stopzeitpunkt
    stop = start + pd.DateOffset(months=timeframe) - pd.Timedelta(hours=1)    
    times = pd.date_range(start, stop, freq='h') #, tz='UTC')

    return times




def convert_tau_file(df_tau, injection_year, injection_month, injection_day, timeframe):
    '''
    Conversion of output tau-distribution in form needed in flux calculation, hourly resolution
    Args: 
    - df = output of aerosol distribution simulation (tau)
    Return: 
    - df_transposed = mirrored df
    '''
    print('start converting')
    # (1) create transposed distribution of tau
    df_transposed = df_tau.transpose()
    df_transposed = df_transposed.reset_index()
    df_transposed.rename(columns = {'index':'Date'}, inplace = True)

    # save to excel       
    df_transposed.to_excel(cwd + '\SAI Code\calc_pv_pot/data/processed/tau_distr/transposed/df_tau_transposed.xlsx')

    # time period in hourly steps from injection date until end of timeframe: Injection at 00:00
    times = get_time_period(timeframe, injection_month, injection_year, injection_day)

    # create start date as timestamp
    start_date = pd.Timestamp(year=injection_year, month=injection_month, day=injection_day, hour=0, minute=30)

    ## NEU ##
    datetime_list = []

    for mon in range(timeframe):
        start_date_modified = start_date + pd.Timedelta(hours=23)

        if mon == 0:
            #next_month = start_date ### TEST
            #datetime_list.append(start_date)
            datetime_list.append(start_date_modified + pd.Timedelta(days=30))
        else:
            next_month = start_date_modified + relativedelta(months=mon+1) - pd.Timedelta(days=1)
            datetime_list.append(next_month)
        # else:
        #     next_month += relativedelta(months=1)
        #     datetime_list.append(next_month)


    # (3) Create new df for hourly distribution of tau 

    df_tau_hourly = pd.DataFrame(columns = ['Date'] + LIST_REGIONS)                                 # new empty df_hourly

    # List of timestamps
    timestamps = times.to_list()

    # Loop to fill df_hourly
    for stamp in timestamps:

        match_found = False

        if stamp == pd.Timestamp('2009-03-31 23:30:00'):
            print('check') 

        for idx in range(len(df_transposed)):

            #date_check = df_transposed['Date'].iloc[idx]                                           # date to check with in df_transposed
            date_to_check = datetime_list[idx]

            if stamp == date_to_check:                                                              # timestamp matches date in df_transposed
                vals_to_add = df_transposed.iloc[idx][1:]
                list_to_add = vals_to_add.to_list()
                row_to_add = pd.Series([stamp] + list_to_add, index= ['Date'] + LIST_REGIONS)

                df_tau_hourly.loc[len(df_tau_hourly)] = row_to_add

                match_found = True
                break                                                                               # exit loop once a match is found
        
        if not match_found:
                
            new_row = pd.Series([stamp] + [np.nan]*16, index= ['Date'] + LIST_REGIONS)              # create a new row with NaN values
            
            df_tau_hourly.loc[len(df_tau_hourly)] = new_row                                         # append the new row to the DataFrame

    df_tau_hourly.to_excel(cwd + '\SAI Code\calc_pv_pot/data/processed/Scenarios/Volcanic/df_tau_hourly_TEST.xlsx')
    
    print('done converting')

    print('start interpolating')


    # (4) Interpolation

    df_tau_hourly = df_tau_hourly.replace({pd.NaT: np.nan})                                         # all NaT to NaN

    for col in df_tau_hourly.columns[1:]:
        df_tau_hourly[col] = df_tau_hourly[col].interpolate(method='linear', limit_direction='forward')
    
    

    print('done interpolating')

    return df_tau_hourly, times





def convert_tau_file_lastyear(df_tau, injection_year, injection_month, injection_day, years):
    '''
    Conversion of output tau-distribution in form needed in flux calculation, hourly resolution
    Args: 
    - df = output of aerosol distribution simulation (tau)
    Return: 
    - df_transposed = mirrored df
    '''
    print('start converting')

    timeframe = 12 # only for one year (last year)

    # (1) create transposed distribution of tau
    df_transposed = df_tau.transpose()
    df_transposed = df_transposed.reset_index()
    df_transposed.rename(columns = {'index':'Date'}, inplace = True)

    # save to excel       
    df_transposed.to_excel(cwd + '\SAI Code\calc_pv_pot/data/processed/tau_distr/transposed/df_tau_transposed.xlsx')

    # time period in hourly steps from injection date until end of timeframe: Injection at 00:00
    injection_year = injection_year + years
    times = get_time_period(timeframe, injection_month, injection_year, injection_day)

    # create start date as timestamp
    start_date = pd.Timestamp(year=injection_year, month=injection_month, day=injection_day, hour=0, minute=30)

    ## NEU ##
    datetime_list = []

    for mon in range(timeframe):
        if mon == 0:
            next_month = start_date
            datetime_list.append(start_date)
        else:
            next_month += relativedelta(months=1)
            datetime_list.append(next_month)


    # (3) Create new df for hourly distribution of tau 

    df_tau_hourly = pd.DataFrame(columns = ['Date'] + LIST_REGIONS)                                 # new empty df_hourly

    # List of timestamps
    timestamps = times.to_list()

    # Loop to fill df_hourly
    for stamp in timestamps:

        match_found = False
        
        for idx in range(len(df_transposed)):

            #date_check = df_transposed['Date'].iloc[idx]                                           # date to check with in df_transposed
            date_to_check = datetime_list[idx]

            if stamp == date_to_check:                                                              # timestamp matches date in df_transposed
                vals_to_add = df_transposed.iloc[idx][1:]
                list_to_add = vals_to_add.to_list()
                row_to_add = pd.Series([stamp] + list_to_add, index= ['Date'] + LIST_REGIONS)

                df_tau_hourly.loc[len(df_tau_hourly)] = row_to_add

                match_found = True
                break                                                                               # exit loop once a match is found
        
        if not match_found:
                
            new_row = pd.Series([stamp] + [np.nan]*16, index= ['Date'] + LIST_REGIONS)              # create a new row with NaN values
            
            df_tau_hourly.loc[len(df_tau_hourly)] = new_row                                         # append the new row to the DataFrame

    df_tau_hourly.to_excel(cwd + '\SAI Code\calc_pv_pot/data/processed/Scenarios/Volcanic/df_tau_hourly_TEST.xlsx')
    
    print('done converting')

    print('start interpolating')


    # (4) Interpolation

    df_tau_hourly = df_tau_hourly.replace({pd.NaT: np.nan})                                         # all NaT to NaN

    for col in df_tau_hourly.columns[1:]:
        df_tau_hourly[col] = df_tau_hourly[col].interpolate(method='linear', limit_direction='forward')
    
    print('done interpolating')

    return df_tau_hourly, times




def add_missing_time(df_distribution_tau_flux_year, flux_year, times_flux_year):

    times_flux_year_total = get_time_period(12, 1, flux_year, 1)
    missing_times = times_flux_year_total[~times_flux_year_total.isin(times_flux_year)]

    missing_data = pd.DataFrame(0, index=range(len(missing_times)), columns=df_distribution_tau_flux_year.columns)  # Fülle mit 0
    missing_data['Date'] = missing_times

    df_distribution_tau_flux_year_total = pd.concat([missing_data, df_distribution_tau_flux_year])
    df_distribution_tau_flux_year_total.reset_index(drop=True, inplace=True)


    return df_distribution_tau_flux_year_total, times_flux_year_total



def tau_flux_year(df_tau_hourly, flux_year):

    df_distribution_tau_flux_year = df_tau_hourly[df_tau_hourly['Date'].dt.year == flux_year]

    return df_distribution_tau_flux_year