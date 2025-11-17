from collections import OrderedDict
import numpy as np
from netCDF4 import Dataset
import pvlib as pv

from calc.radiation_flux_SAI.delta_eddington import delta_eddington_param
from calc.radiation_flux_SAI.flux_delta_eddinton import flux_delta_eddington
from calc.radiation_flux_SAI.flux_delta_eddinton import calc_delta
from calc.simulation_aerosol_distribution.convert_tau_file import get_time_period

REGIONS_LON = {
    'sec_01': (0.0, 22.5),      # 0° to 22.5°
    'sec_02': (22.5, 45.0),     # 22.5° to 45.0°
    'sec_03': (45.0, 67.5),     # 45.0° to 67.5°
    'sec_04': (67.5, 90.0),     # 67.5° to 90.0°
    'sec_05': (90.0, 112.5),    # 90.0° to 112.5°
    'sec_06': (112.5, 135.0),   # 112.5° to 135.0°
    'sec_07': (135.0, 157.5),   # 135.0° to 157.5°
    'sec_08': (157.5, 180.0),   # 157.5° to 180.0°
    'sec_09': (-180.0, -157.5), # -180° to -157.5°
    'sec_10': (-157.5, -135.0), # -157.5° to -135.0°
    'sec_11': (-135.0, -112.5), # -135.0° to -112.5°
    'sec_12': (-112.5, -90.0),  # -112.5° to -90.0°
    'sec_13': (-90.0, -67.5),   # -90.0° to -67.5°
    'sec_14': (-67.5, -45.0),   # -67.5° to -45.0°
    'sec_15': (-45.0, -22.5),   # -45.0° to -22.5°
    'sec_16': (-22.5, 0.0)       # -22.5° to 0.0°
}
    
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



def create_sim_data_16x16(times, df_distribution_tau_converted): 
    '''
    Struktur: [] len = 256 = (16x16)

        sim_data_MA['regions_16x16'] = []       
        sim_data_MA['def_regions'] = []   
        sim_data_MA['def_regions_mean'] = []  

    Struktur: [] len = 256 = (16x16)

    sim_data_MA['def_regions_lat'] = []                         
    sim_data_MA['def_regions_lon'] = []                       

    Struktur: [] len = len(times) = 8760 

        sim_data_MA['time_stamps'] = []         

    Struktur: [[], [], [], ... ] len = 8760, len_inner_list = 256

        sim_data_MA['AOD'] = []                 
        sim_data_MA['cos_zen'] = []                 
        sim_data_MA['R_dir'] = []
        sim_data_MA['T_dir'] = []
        sim_data_MA['R_diff'] = []
        sim_data_MA['T_diff'] = []
        sim_data_MA['tau_star'] = []
        sim_data_MA['GHI'] = []
        sim_data_MA['DNI'] = []
        sim_data_MA['DIR'] = []
        sim_data_MA['DHI'] = []
        sim_data_MA['delta_GHI'] = []
        sim_data_MA['delta_DNI'] = []
        sim_data_MA['delta_DIR'] = []
        sim_data_MA['delta_DHI']  = []
    '''
    # Anzahl der Rasterpunkte
    num_points = 16 * 16  # 256 Rasterpunkte
    num_time_stamps = len(times)  # 12 Monate, stündlich (8760)

    # Initialisiere das OrderedDict für 256 Rasterpunkte (256xtimes) 
    sim_data_MA = OrderedDict()

    #### add REGIONS 16x16 (len 256)
    sim_data_MA['regions_16x16'] = [] 

    LIST_REGIONS = ['NH_pol', 'NH_extrop_1', 'NH_extrop_2', 'NH_extrop_3', 'NH_extrop_4', 'NH_trop_1', 'NH_trop_2', 'NH_trop_3', 'SH_trop_1', 'SH_trop_2', 'SH_trop_3', 'SH_extrop_1', 'SH_extrop_2', 'SH_extrop_3', 'SH_extrop_4', 'SH_pol']
    list_regions_16x16 = [f'{region}_sect_{str(j+1).zfill(2)}' for i, region in enumerate(LIST_REGIONS) for j in range(16)]
    sim_data_MA['regions_16x16'] = list_regions_16x16

    print('REGION NAMES added')




    #### add REGIONS RANGE (KEY, LAT_RANGE, LON_RANGE)
    sim_data_MA['def_regions'] = []                             # Für die zusammengesetzten Daten (Region + Lat/Lon-Bereiche)
    sim_data_MA['def_regions_lat'] = []                         # Für die separaten Latitude-Bereiche
    sim_data_MA['def_regions_lon'] = []                         # Für die separaten Longitude-Bereiche

    for region in sim_data_MA['regions_16x16']:                 # regions_16x16 enthält die Region-Bezeichnungen

        region_key, sect_num = region.rsplit('_sect_', 1)

        lat_range = REGIONS_LAT.get(region_key)
        lon_key = f'sec_{sect_num.zfill(2)}'  
        lon_range = REGIONS_LON.get(lon_key)

        if lat_range and lon_range:
            sim_data_MA['def_regions'].append([region, lat_range, lon_range])
            sim_data_MA['def_regions_lat'].append(lat_range)
            sim_data_MA['def_regions_lon'].append(lon_range)

    print('REGIONS RANGE added')




    #### add REGIONS MEAN (KEY, LAT_MEAN, LON_MEAN)
    sim_data_MA['def_regions_mean'] = []

    for region_data in sim_data_MA['def_regions']:

        region_name, lat_range, lon_range = region_data
        
        lat_mean = (lat_range[0] + lat_range[1]) / 2
        lon_mean = (lon_range[0] + lon_range[1]) / 2
        
        sim_data_MA['def_regions_mean'].append([region_name, lat_mean, lon_mean])

    print('REGIONS MEAN added')




    #### add TIMESTAMPS
    sim_data_MA['time_stamps'] = []

    for timestamp in times:

        sim_data_MA['time_stamps'].append(timestamp)

    print('TIMESTAMPS MEAN added')





    #### add AOD
    sim_data_MA['AOD'] = []

    for idx_row, date_row in df_distribution_tau_converted.iterrows():
        list_date = []

        for region_index, region in enumerate(LIST_REGIONS):                        # Region-Index und Name
            aod_value = date_row[region]                                            # AOD-Wert für die Region (funktioniert)
            
            for i in range(16):
                list_date.append(aod_value)

        sim_data_MA['AOD'].append(list_date)

    print('AOD added')






    #### add COS(ZEN)
    sim_data_MA['cos_zen'] = [[] for _ in range(len(times))]

    for pos_idx in range(num_points):
        pos_info = sim_data_MA['def_regions_mean'][pos_idx]

        lat = pos_info[1]
        lon = pos_info[2]

        sol_pos = pv.solarposition.get_solarposition(times, lat, lon)               # len 8760
        cos_zen = np.cos(np.radians(sol_pos['apparent_zenith'].values))             # für ein place alle µ übers Jahr
        
        for hour_idx in range(len(cos_zen)):                                        # Es gibt 8760 Werte in cos_zen
            sim_data_MA['cos_zen'][hour_idx].append(cos_zen[hour_idx])

    print('COS(ZEN) added')






    #### add DELAT-EDDINGTON-PARAMETER
    w_new = 0.98
    g_new = 0.5
    f_new = (0.776199996471405)**2
    solar_const = 1361

    sim_data_MA['R_dir'] = []
    sim_data_MA['T_dir'] = []
    sim_data_MA['R_diff'] = []
    sim_data_MA['T_diff'] = []
    sim_data_MA['tau_star'] = []
        
    for time_idx in range(len(times)):                                    # 8760 Iterationen für die Zeitstempel
        AOD_values = sim_data_MA['AOD'][time_idx]                         # AOD-Werte für den aktuellen Zeitstempel (gibt innere liste wieder)
        cos_zen_values = sim_data_MA['cos_zen'][time_idx]                 # cos_zen-Werte für den aktuellen Zeitstempel (innere liste)

        R_dir_values = []
        T_dir_values = []
        R_diff_values = []
        T_diff_values = []
        tau_star_values = []
        
        for point_idx in range(num_points):                               # 256 Iterationen für die Punkte (iteration über die innere liste)
            AOD = AOD_values[point_idx]                                   # AOD für die aktuelle Region
            if AOD == 0:
                R_dir = 0
                T_dir = 0
                R_diff = 0
                T_diff = 0
                tau_star = 0
            else:
                cos_zen = cos_zen_values[point_idx]                           # cos_zen für die aktuelle Region
            
                R_dir, T_dir, R_diff, T_diff, tau_star = delta_eddington_param(AOD, w_new, g_new, cos_zen, f_new)

            R_dir_values.append(R_dir)
            T_dir_values.append(T_dir)
            R_diff_values.append(R_diff)
            T_diff_values.append(T_diff)
            tau_star_values.append(tau_star)
            
        sim_data_MA['R_dir'].append(R_dir_values)
        sim_data_MA['T_dir'].append(T_dir_values)
        sim_data_MA['R_diff'].append(R_diff_values)
        sim_data_MA['T_diff'].append(T_diff_values)
        sim_data_MA['tau_star'].append(tau_star_values) 

    # WARNING C:\Kamp\Code\CE_PV_Potential\calc_pv_pot\src\calc\radiation_flux_SAI\delta_eddington.py:60: RuntimeWarning: invalid value encountered in scalar subtract T_dir = (alpha + gamma) * T_diff + (alpha - gamma) * R_diff * np.exp(-tau_star/µ_0) - (alpha + gamma - 1) * np.exp(-tau_star/µ_0)

    print('DELAT-EDDINGTON-PARAMETER added')





    #### add DELTA (GHI, DNI, DIR, DHI)
    sim_data_MA['GHI'] = []
    sim_data_MA['DNI'] = []
    sim_data_MA['DIR'] = []
    sim_data_MA['DHI'] = []
    sim_data_MA['delta_GHI'] = []
    sim_data_MA['delta_DNI'] = []
    sim_data_MA['delta_DIR'] = []
    sim_data_MA['delta_DHI']  = []

    for time_idx in range(len(times)):                                          # 8760 Iterationen für die Zeitstempel
        
        GHI_values = []
        DNI_values = []
        DHI_values = []
        DIR_values = []
        delta_dn_values = []
        delta_dir_values = []
        delta_diff_values = []
        delta_dir_µ_values = []
        
        for point_idx in range(num_points):                                     # 256 Iterationen für die Raster-Punkte
            cos_zen = sim_data_MA['cos_zen'][time_idx][point_idx]               # Werte für den aktuellen Punkt und Zeitstempel abrufen
            tau_star = sim_data_MA['tau_star'][time_idx][point_idx]

            if tau_star == 0:
                F_dir = 0
                F_diff = 0
                F_dn = 0
                F_DIR_µ = 0
                delta_dn = 0
                delta_dir = 0
                delta_diff = 0
                delta_dir_µ = 0
            else:
                F_dir, F_diff, F_dn, F_DIR_µ = flux_delta_eddington(cos_zen, tau_star, solar_const, sim_data_MA['T_dir'][time_idx][point_idx])
                delta_dn, delta_dir, delta_diff, delta_dir_µ = calc_delta(solar_const, F_dir, F_diff, F_dn, cos_zen)

            GHI_values.append(F_dn)
            DNI_values.append(F_dir)
            DHI_values.append(F_diff)
            DIR_values.append(F_DIR_µ)
            delta_dn_values.append(delta_dn)
            delta_dir_values.append(delta_dir)
            delta_diff_values.append(delta_diff)
            delta_dir_µ_values.append(delta_dir_µ)

        sim_data_MA['GHI'].append(GHI_values)
        sim_data_MA['DNI'].append(DNI_values)
        sim_data_MA['DHI'].append(DHI_values)
        sim_data_MA['DIR'].append(DIR_values)
        sim_data_MA['delta_GHI'].append(delta_dn_values)
        sim_data_MA['delta_DNI'].append(delta_dir_values)
        sim_data_MA['delta_DHI'].append(delta_diff_values)
        sim_data_MA['delta_DIR'].append(delta_dir_µ_values)

    print('FLUX & DELTA added')


    return sim_data_MA

