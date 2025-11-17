

def header_gen(wavelengths, func_out):
    '''
    Funktion erstellt Kopfzeile der Excel
    
    Args:
    - wavelenght: 
    - func_out: Output_Variabeln für jede wv

    Return: 
    - headers: Kopfzeile der Excel

    '''

    headers = []
    for wv in wavelengths:
        for out in func_out:
            col = str(wv) + '_' + out
            headers.append(col)
    
    return headers


