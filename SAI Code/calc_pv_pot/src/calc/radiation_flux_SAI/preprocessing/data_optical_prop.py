import pandas as pd

data_file = r'C:\Kamp\Data\opt_data_S00.xlsx'

data_optical_prop = pd.read_excel(data_file)

new_column_names = ['wavelength', 'ext_coef', 'sca_coef', 'abs_coef', 'si_sc_alb', 'asym_par', 'ext_nor', 'ref_real', 'ref_imag']
data_optical_prop.columns = new_column_names

#print(data_optical_prop.columns)
