import numpy as np


def flux_two_stream(F_0, tau, w, g):
    '''
    Ansatz two-stream calculation: Näherung, aber ohne µ
    https://doi.org/10.1016/B978-0-12-382225-3.00053-0

    args: 
    - F_0: Initiale Einsrahlung
    - tau: Optische Tiefe
    - w: single scattering albedo
    - g: asymetrie factor

    Return:
    - F_up: Strahlung aufwärts (atmos)
    - F_dn: Strahlung abwärts (erde)
    - F_dn_dir: direker Anteil F_dn
    - F_dn_diff: diffuser Anteil F_dn
    '''
    b = (1 - g) / 2 # backscattering fraction
    F = F_0 * np.exp(-tau) #Strahlungstransportgleichung nach Beers law
    
    F_up = w * b * tau * F # Berechnung nach oben streute Strahlung (diff)

    F_dn_dir = (1 - tau) * F 
    F_dn_diff = (1 - b) * tau * F
    
    F_dn = F_dn_dir + F_dn_diff

    return F_up, F_dn, F_dn_dir, F_dn_diff