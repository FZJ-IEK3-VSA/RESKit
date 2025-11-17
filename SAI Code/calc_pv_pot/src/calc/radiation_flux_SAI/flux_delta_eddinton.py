import numpy as np

def flux_delta_eddington(µ_0, tau, F_0, T_dir):

    '''
    Funktion berechnet direct&transmitted und scattered&transmitted Strahlung nach der Delta-Eddington-Approximation
    Coakley, J. A., R. D. Cess, and F. B. Yurevich, The effect of tropospheric aerosols on the Earth’s radiation budget: A parameterization for climate models, J. Atmos. Sci., 40, 116–138, 1983.
    
    Args:
    - µ: Cosinus des Sonnenzenitwinkels
    - tau: optical depth (tau_star)
    - F_0: incident solar flux on first layer (= 1361 W/m²) 
    - T_dir: total transmissivity of Layer 1 to direct radiation (output delta_eddington_approximation)
    
    Return:
    - F_dir: flux transmitted & direct (no scattering)
    - F_diff: flux transmitted & diffus (directed downwards)
    
    '''
    # Direct normal irradiance (DNI) is the portion of solar radiation that reaches the earth on a direct path from the sun.
    # Diffuse horizontal irradiance (DIF or DHI) is the portion of solar radiation that reaches the earth indirectly.
    
    # Flux transmitted (direct and diffus)

    if µ_0 == 0.196238775667957:
        print("check")

    #if µ_0 < 0:                 # cos(zenith) < 0 means sun is under the horizon (night) (zenith > 90)
    if µ_0 <= np.cos(np.radians(84)):          # set threshold to zenith_angle = 84 -> no radiation
        F_dir = 0 
        F_diff = 0
        F_dn = 0
        F_DIR_µ = 0
    
    else:
        µ_0 =  np.maximum(µ_0, 0.2)                 #see Reskit 

        F_dir = F_0 * np.exp(-tau / µ_0)            #DNI
        F_diff = F_0 * (T_dir - np.exp(-tau / µ_0)) #DHI
        F_dn = F_dir * µ_0 + F_diff                 #GHI
        F_DIR_µ = F_dir * µ_0
    
    
    return F_dir, F_diff, F_dn, F_DIR_µ


def calc_delta_alt(F_0, F_dir, F_diff, F_dn):
    # fraction of diff and dir of total dn

# night = no sun (F_dn = 0)
    if F_dir == 0:
        delta_dn = 0
        delta_dir = 0
        delta_diff = 0

# no aerosol F_dir = 1361 (incident flux)  
    elif F_dir == F_0:
        delta_dn = 0
        delta_dir = 0
        delta_diff = 0

    else:
        delta_dn = F_0 - F_dn

        x_dir = F_dir / F_dn
        delta_dir = delta_dn * x_dir

        x_diff = F_diff / F_dn
        delta_diff = delta_dn * x_diff

    return delta_dn, delta_dir, delta_diff

def calc_delta(F_0, F_dir, F_diff, F_dn, µ_0):
    '''
    Calculate difference between fluxes with and without aerosols.

    Args:
    - F_0 = solar constant
    - F_dir = DNI_aer (direction of zenith angle)
    - F_diff = DHI_aer (diffus)
    - F_dn = GHI_aer (GHI=DNI*µ+DHI)
    - µ_0 = zenith angle

    Return:
    - delta_dn = 
    - delta_dir = 
    - delta_diff = 
    '''
    # with aerosols:    GHI_aer = DNI_aer * µ + DHI_aer
    #                   F_dn    = F_dir   * µ + F_diff

    # without aerosols: GHI_no,a = DNI_no,a * µ + DHI_no,a 
    #                   GHI_no,a = F_0      * µ + 0

    # night = no sun (F_dn = 0)
    if F_dn == 0:
        delta_dn = 0
        delta_dir = 0
        delta_diff = 0
        delta_dir_µ = 0

    # no aerosol F_dir = 1361 (incident flux)  
    elif F_dir == F_0:
        delta_dn = 0
        delta_dir = 0
        delta_diff = 0
        delta_dir_µ = 0

    else:
        µ_0 =  np.maximum(µ_0, 0.2)                 #see Reskit 

        # delta_GHI = delta_DIR + delta_DHI = (µ*F_0 - µ*F_dir) + (0 - F_diff)
        delta_dn = µ_0 * F_0 - µ_0 * F_dir - F_diff

        # delta_DHI = DHI_no,a - DHI_aer 
        delta_diff = - F_diff # negativ, because more diffus radiation

        # delta_DNI = DNI_no,a - DNI_aer
        delta_dir = F_0 - F_dir

        delta_dir_µ = delta_dir * µ_0


    return delta_dn, delta_dir, delta_diff, delta_dir_µ
