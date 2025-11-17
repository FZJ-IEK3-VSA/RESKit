def gen_excel(path, version):

    version = version + '.xlsx'

    filepath = path + version
    
    return filepath

# df_distribution_RF.to_excel('C:/Kamp/Code/CE_PV_Potential/calc_pv_pot/data/processed/RF_distr/RF_hourly_distribution_v19.xlsx')