def interpolate_delta(delta_values, low_res_coords, high_res_coords, method='linear'):
    '''
    Interpoliert die Delta-Werte von niedriger auf hohe räumliche Auflösung (der Global Solar Atlas (GSA) data).
    
    Args:
    - delta_values: Array der Delta-Werte mit niedriger Auflösung
    - low_res_coords: Koordinaten der niedrigen Auflösung
    - high_res_coords: Koordinaten der hohen Auflösung
    - method: Interpolationsmethode ('linear', 'cubic', etc.)
    
    Return:
    - interpolated_delta: Interpolierte Delta-Werte mit hoher Auflösung
    '''
    interpolated_delta = griddata(low_res_coords, delta_values, high_res_coords, method=method)
    return interpolated_delta