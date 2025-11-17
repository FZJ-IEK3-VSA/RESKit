def flux_calc_v0(S_dir, S_diff, T_dir, T_diff, R_dir, R_diff):
    '''
    berechnet aus einfallender Strahlung (dir und diff) mit transmissivity und relfectivity die ausgehende Strahlung (dir und diff)
    '''
    # Durchgelassene Strahlung
    flux_dir_transmitted = S_dir * T_dir
    flux_diff_transmitted = S_diff * T_diff

    # Reflektierte Strahlung
    flux_dir_reflected = S_dir * R_dir
    flux_diff_reflected = S_dir * R_diff

    # Absorbierte Strahlung
    flux_dir_absorbed = S_dir - flux_dir_transmitted - flux_dir_reflected
    flux_diff_absorbed = S_diff - flux_diff_transmitted - flux_diff_reflected

    #Update einfallende Strahlung für nächsten Layer
    S_dir_new = flux_dir_transmitted
    S_diff_new = flux_diff_transmitted + flux_dir_reflected + flux_diff_reflected #IST DAS RICHTIG DIE ANNAHME?? WAS GEHT INS ALL??
    
    return S_dir_new, S_diff_new