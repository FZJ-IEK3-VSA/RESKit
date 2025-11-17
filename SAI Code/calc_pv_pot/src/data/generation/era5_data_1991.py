import xarray as xr

# Pfad zur NetCDF-Datei
file_path = 'C:/Kamp/Code/CE_PV_Potential/calc_pv_pot/data/raw/era5_test1991.nc'

# Öffnen der NetCDF-Datei
ds = xr.open_dataset(file_path)

# in pandas df umwandeln
df = ds.to_dataframe().reset_index()

# Anzeigen der Daten
#print(df)

'''
(time, latitude, longitude) float32

ssrdc = Surface solar radiation downward clear-sky [J/m²]
tisr = TOA incident solar radiation [J/m²]
tsrc = Top net solar radiation, clear sky [J/m²]
'''
# finde maximalwerte
max_tisr = df['tisr'].max()
max_ssrdc = df['ssrdc'].max()

print(df)


# umrechnung J/m² in W/m²
def in_watt_per_sqrm(val):
    val_wm = val / (24 * 3600)
    return val_wm

tisr_wm = in_watt_per_sqrm(max_tisr)
ssrdc_wm = in_watt_per_sqrm(max_ssrdc)

# lat 90 to 30 mittelwert (analog (30 to 0) (0 to -30) (-30 to -90))
# lat in 0,25 Schritten
#for elem in df['tisr'][0:]

indices = df[df['latitude'] == 90.00].index.tolist()

#print(df['latitude'][240])

# Regionen definieren
def define_region(latitude):
    if -90 <= latitude < -30:
        return '(-90, -30)'
    elif -30 <= latitude < 0:
        return '(-30, 0)'
    elif 0 <= latitude < 30:
        return '(0, 30)'
    elif 30 <= latitude <= 90:
        return '(30, 90)'
    else:
        return 'Undefined'

# Region zuordnen
df['region'] = df['latitude'].apply(define_region)

# Jahr extrahieren
df['year'] = df['time'].dt.year

# Jahresmittelwerte berechnen
mean_values = df.groupby(['region', 'year']).agg({'ssrdc': 'mean', 'tisr': 'mean', 'tsrc': 'mean'}).reset_index()

ssrdc_reg1 = in_watt_per_sqrm(mean_values['ssrdc'])
print(ssrdc_reg1)

# in W/m²-Funktion auf jede Spalte anwenden
    
mean_values['ssrdc'] = mean_values['ssrdc'].apply(in_watt_per_sqrm)
mean_values['tisr'] = mean_values['tisr'].apply(in_watt_per_sqrm)
mean_values['tsrc'] = mean_values['tsrc'].apply(in_watt_per_sqrm)

print(mean_values)