
def get_region(lat_injection):

    regions_lat = { 'NH_pol': (90, 61),
                    
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
        

    # find region for the latitude of injection

    for lat_key, lat_range in regions_lat.items():
        if lat_range[1] <= lat_injection <= lat_range[0]: # check if val is in interval
             key_region = lat_key
 
    # find index of region of injection
    LIST_REGIONS = ['NH_pol', 'NH_extrop_1', 'NH_extrop_2', 'NH_extrop_3', 'NH_extrop_4', 'NH_trop_1', 'NH_trop_2', 'NH_trop_3', 'SH_trop_1', 'SH_trop_2', 'SH_trop_3', 'SH_extrop_1', 'SH_extrop_2', 'SH_extrop_3', 'SH_extrop_4', 'SH_pol']
    index_region = LIST_REGIONS.index(key_region)

    return key_region, index_region
