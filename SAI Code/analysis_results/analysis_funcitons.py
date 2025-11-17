
########### GET REGION INFO ##############


def get_region_info(region_input, ds):
    """
    Funktion, um entweder den Index oder die Bezeichnung der Region zu erhalten.
    - Wenn region_input ein Index (int) ist, gibt sie die Regionsbezeichnung zurück.
    - Wenn region_input eine Bezeichnung (str) ist, gibt sie den Index der Region zurück.
    """
    regions = ds['regions_16x16'].values  # Extrahiere alle Regionennamen
    
    # Wenn der Input ein Index ist (int)
    if isinstance(region_input, int):
        if region_input >= len(regions) or region_input < 0:
            return f"Index {region_input} liegt außerhalb des gültigen Bereichs."
        return regions[region_input]  # Gebe die Region-Bezeichnung für diesen Index zurück

    # Wenn der Input eine Bezeichnung ist (str)
    elif isinstance(region_input, str):
        if region_input not in regions:
            return f"Region '{region_input}' wurde nicht gefunden."
        return list(regions).index(region_input)  # Gebe den Index der Region zurück

    else:
        return "Ungültiger Input: Bitte entweder einen Index (int) oder eine Regionsbezeichnung (str) eingeben."


########### FIND REGION ##############


def find_region(ds, latitude, longitude):
    """
    Überprüft, in welcher Region sich die gegebene Latitude und Longitude befinden.

    Args:
        ds (xarray.Dataset): Das Dataset, das die Regionen enthält.
        latitude (float): Die Latitude, die überprüft werden soll.
        longitude (float): Die Longitude, die überprüft werden soll.

    Returns:
        tuple: (region_name, index) der gefundenen Region oder (None, None), wenn keine Region gefunden wurde.
    """
    for i in range(len(ds['regions_16x16'])):
        region_name = ds['regions_16x16'].values[i]
        lat_range = ds['def_regions_lat'][i, :].values
        lon_range = ds['def_regions_lon'][i, :].values

        # Prüfen, ob die gegebene lat, lon in den Range fällt
        lat_in_range = (lat_range[0] <= latitude <= lat_range[1]) if lat_range[0] < lat_range[1] else (lat_range[1] <= latitude <= lat_range[0])
        lon_in_range = (lon_range[0] <= longitude <= lon_range[1]) if lon_range[0] < lon_range[1] else (lon_range[1] <= longitude <= lon_range[0])

        if lat_in_range and lon_in_range:
            return region_name, i  # Rückgabe der Region und des Index

    print(f"Die Latitude {latitude} und Longitude {longitude} befinden sich in keiner definierten Region.")  # Ausgabe, wenn keine Region gefunden wurde
    return None, None  # Rückgabe von None, wenn keine Region gefunden wurde
