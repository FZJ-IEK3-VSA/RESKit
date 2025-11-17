# Global horizontal irradiance is the total solar radiation per unit area measured at a horizontal surface on the earth. 
# It is typically presented in W/m2 and can be broken down into two components: direct normal irradiance (DNI) and diffuse horizontal irradiance (DHI). 

def calc_GHI(F_DHI, F_DNI, µ_0):
    '''
    Berechnung der Global horizontal irradiance (GHI)

    Args:
    - F_DHI: diffuse horizontal irradiance (F_diff)
    - F_DNI: direct normal irradiance (F_dir)
    - µ_0: cosinus solar zenith angle

    Return:
    - F_GHI: gloabl horizontal irradiance 
    '''
    F_GHI = F_DHI + F_DNI * µ_0

    return F_GHI
