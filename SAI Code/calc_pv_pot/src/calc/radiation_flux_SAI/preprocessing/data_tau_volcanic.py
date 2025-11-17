import pandas as pd

data_file = r'C:\Kamp\Data\aod.xlsx'

data_tau_volcanic = pd.read_excel(data_file)

data_tau_volcanic.replace({'\s+': '', ',': '.'}, regex=True, inplace=True)

#print(data_tau_volcanic)

