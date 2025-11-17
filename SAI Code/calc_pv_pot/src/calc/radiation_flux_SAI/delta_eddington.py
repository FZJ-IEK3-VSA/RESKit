import numpy as np

def delta_eddington_param(tau, w, g, µ_0, f):

    '''
    Funktion berechnet transmisivity und reflectivity analog zur "Delta-Eddington-Approximation"
    
    Args:
    - tau: optical depth 
    - w: single scattering albedo
    - g:  asymetrie parameter
    - µ: Cosinus des Sonnenzenitwinkels
    - f: forward scattering fraction

    Return: 
    - R_dir: reflectivity to direct 
    - T_dir: transmissivity to direct
    - R_diff: reflectivity to diffus
    - T_diff: transmissivity to diffus
    '''

    if w == 1:
        w_eps = 1 - (1e-13)

        tau_star = tau * (1 - w_eps*f)
        w_star = w_eps * ( (1-f) / (1 - w_eps*f))  
        g_star = (g-f) / (1-f)

        lam = np.sqrt(3 * (1-w_star) * (1 - w_star*g_star))   
        u = 3/2 * (1 - w_star * g_star * lam)
        N = (u+1)**2 * np.exp(lam * tau_star) - (u-1)**2 * np.exp(-lam * tau_star)
        gamma = 1/2 * w_star * ((1 + 3 * g_star * (1-w_star) * µ_0 ) / (1-lam**2 * µ_0**2))
        alpha =  3/4 * w_star * µ_0 * ((1 + g_star * (1-w_star)) / (1 - lam**2 * µ_0**2))

        # T und R bezogen auf diffuse (einfallende) Strahlung
        T_diff = (4/3) / ((4/3) + (1-g) * tau)
        R_diff = ((1-g) * tau) / ((4/3) + (1-g) * tau)

        # T und R bezogen auf direkte (einfallende) Strahlung
        T_dir = (alpha + gamma) * T_diff + (alpha - gamma) * R_diff * np.exp(-tau_star/µ_0) - (alpha + gamma - 1) * np.exp(-tau_star/µ_0)
        R_dir = (alpha - gamma) * T_diff * np.exp(-tau_star/µ_0) + (alpha - gamma) * R_diff - (alpha - gamma)
    
    
    else:
        tau_star = tau * (1 - w*f)
        w_star = w * ( (1-f) / (1 - w*f))  
        g_star = (g-f) / (1-f)
    
        lam = np.sqrt(3 * (1-w_star) * (1 - w_star*g_star))   
        u = 3/2 * (1 - w_star * g_star * lam)
        N = (u+1)**2 * np.exp(lam * tau_star) - (u-1)**2 * np.exp(-lam * tau_star)
        gamma = 1/2 * w_star * ((1 + 3 * g_star * (1-w_star) * µ_0 ) / (1-lam**2 * µ_0**2))
        alpha =  3/4 * w_star * µ_0 * ((1 + g_star * (1-w_star)) / (1 - lam**2 * µ_0**2))
        
        # T und R bezogen auf diffuse (einfallende) Strahlung
        T_diff = 4 * u / N
        R_diff = (u+1) * (u-1) * (np.exp(lam*tau_star) - np.exp(-lam*tau_star)) / N

        # T und R bezogen auf direkte (einfallende) Strahlung
        T_dir = (alpha + gamma) * T_diff + (alpha - gamma) * R_diff * np.exp(-tau_star/µ_0) - (alpha + gamma - 1) * np.exp(-tau_star/µ_0)
        R_dir = (alpha - gamma) * T_diff * np.exp(-tau_star/µ_0) + (alpha + gamma) * R_diff - (alpha - gamma)


    if R_diff < 0:
        R_diff = 0


    return R_dir, T_dir, R_diff, T_diff, tau_star