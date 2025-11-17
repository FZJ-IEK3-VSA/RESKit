import math

'''
Calculation of AOD with two approaches:

(1) Easy parameter approach: 
    - from amound Injection -> global mean AOD at wv 0,55 µm
    - RF (Changes in shortwave downward)

'''


def def_AOD(F_0, F_transm):
    '''
    calculation of AOD from flux-definition
    Args:
    - F_0 = Initial flux (wv-dependent)
    - F_transm = transmitted flux (wv-dependent)'''

    tau = math.log(F_0 / F_transm)

    return tau





def easy_AOD(m_so2):
    '''
    Calculation of AOD for given sulfat_injection (gloabl mean)
    "Climatic fluctuations modeled for carbon and sulfur emissions from end-Triassic volcanism" (APDX)
    "Radiative forcing and climate impact resulting from SO2 injections based on a 200,000-year record of Plinian eruptions along the Central American Volcanic Arc"

    Args:
    - m_so2 = SO2-injection
    Return:
    - tau = Aerosol optical depth (AOD)

    '''
    conv_to_acid = 1.53                                         # conversion from SO2 to acid (SO4^2-) -> Molmassen-Verhältniss
    chem_scal = 1.25                                            # Further, we assume a volcanic sulfate aerosol composition of 75 % H2SO4 and 25 % H2O by weight (Hamill et al. 1977), such that multiplying the MSL by 1.25 yields the maximum (total) sulfate aerosol loading (MD).
    conv_to_aerosol = 0.44                                      # only a fraction of acid actually forms aerosol                      
   
    m_d = conv_to_acid * chem_scal * conv_to_aerosol * m_so2    # total sulfate aerosol loading
    #m_d = conv_to_acid * chem_scal  * m_so2                    # total sulfate aerosol loading


    if m_d < 9.4:                                               # (Mt)
        tau = 0.0067 * m_d

    else:
        tau = 0.02 * m_d ** (2/3) - 0.028

    return tau





def easy_RF(easy_tau):
    '''
    Calculation of RF for tau (from easy_AOD)
    "Climatic fluctuations modeled for carbon and sulfur emissions from end-Triassic volcanism" (APDX)
    "Radiative forcing and climate impact resulting from SO2 injections based on a 200,000-year record of Plinian eruptions along the Central American Volcanic Arc"

    Args:
    - tau = Aerosol optical depth (AOD)
    Return:
    - RF = radiative forcing [W/m²]

    '''              
    rf_scal = -24                                               # instead of (-24) [W/m²]

    RF = rf_scal * easy_tau

    return RF

