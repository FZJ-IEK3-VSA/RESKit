import numpy as np
import os

# (1) CALCULATION OF AEROSOL DISTRIBUTION

from calc.simulation_aerosol_distribution.exchange_matrix_new import create_transilient_matrix_new
from calc.simulation_aerosol_distribution.exchange_matrix_new import create_transilient_matrix_new_new
from calc.simulation_aerosol_distribution.initial_distribution import initial_distribution
from calc.simulation_aerosol_distribution.simulate_distribution import simulate_distribution
from calc.simulation_aerosol_distribution.simulate_distribution_mult_inj import simulate_distribution_multi
from calc.simulation_aerosol_distribution.get_region import get_region
from calc.simulation_aerosol_distribution.sim_linear_buildup import linear_buildup
from calc.simulation_aerosol_distribution.convert_tau_file import convert_tau_file
from calc.radiation_flux_SAI.generation.excel_gen import gen_excel
from calc.simulation_aerosol_distribution.convert_tau_file import tau_flux_year
from calc.simulation_aerosol_distribution.convert_tau_file import add_missing_time

from calc.AOD_RF.AOD import easy_AOD
from calc.AOD_RF.AOD import easy_RF
cwd = os.getcwd()

########################### FOR SINGLE INJECTION (VOLCANO) ###########################

def create_aerosol_distr_16x16(version, 
                               injection_NH, 
                               injection_SH, 
                               injection_lat_NH,
                               injection_lat_SH,
                               injection_year, 
                               injection_month, 
                               injection_day, 
                               timeframe, 
                               flux_year):

    # create exchange matrix for each season
    matrix_w, matrix_sp, matrix_s, matrix_f = create_transilient_matrix_new()
    transport_matrices = { 'winter': matrix_w,
                            'spring': matrix_sp,
                            'summer': matrix_s,
                            'fall': matrix_f    }



    # get region from latitude input [°]
    injection_reg_NH, injection_index_NH = get_region(injection_lat_NH)
    injection_reg_SH, injection_index_SH = get_region(injection_lat_SH)

    # initial distribution after injection (16 belts)
    zero_distr = np.zeros(16)
    init_distr_NH = initial_distribution(injection_reg_NH, injection_NH)
    init_distr_SH = initial_distribution(injection_reg_SH, injection_SH)
    init_distr_total = np.add(init_distr_NH, init_distr_SH) # überflüssig, weil innit distr in distr-4month enthalten ist

    # simulate linear buildup
    distr_4month_NH = linear_buildup(injection_NH, zero_distr, injection_index_NH)
    distr_4month_SH = linear_buildup(injection_SH, zero_distr, injection_index_SH)

    distr_4month = np.array([NH + SH for NH, SH in zip(distr_4month_NH, distr_4month_SH)]) # je die ersten 4 monats-verteilung von NH und SH addieren -> 4 gesamte monatsverteilungen

    # simulation transport mechanisms and sedimentation: final distribution of aerosols in every belt
    df_distribution_aerosol_belts = simulate_distribution(distr_4month, timeframe, injection_month, transport_matrices)
    df_distribution_aerosol_belts.to_excel(gen_excel(cwd + '\SAI Code\calc_pv_pot/data/processed/aerosol_distribution/aerosol_distribution_', version))

    # calculation tau from aerosol distribution & store to excel
    df_distribution_tau = df_distribution_aerosol_belts.apply(lambda aerosol: aerosol.map(easy_AOD))
    df_distribution_tau.to_excel(gen_excel(cwd + '\SAI Code\calc_pv_pot/data/processed/tau_distr/tau_monthly_distribution_', version))

    # convert df (tau): transposed and with hourly distribution
    df_distribution_tau_converted, times = convert_tau_file(df_distribution_tau, injection_year, injection_month, injection_day, timeframe)                   # INPUT FOR (2)
    df_distribution_tau_converted.to_excel(gen_excel(cwd + '\SAI Code\calc_pv_pot/data/processed/tau_distr/hourly/df_hourly_', version))

    df_distribution_tau_flux_year = tau_flux_year(df_distribution_tau_converted, flux_year)
    times_flux_year = times[times.year == flux_year]
    df_distribution_tau_flux_year_total, times_flux_year_total = add_missing_time(df_distribution_tau_flux_year, flux_year, times_flux_year)

    return df_distribution_tau_flux_year_total, times_flux_year_total, times_flux_year



########################### FOR MULTIPLE INJECTION (SAI) ###########################

def create_aerosol_distr_16x16_SAI(version, 
                               injection_NH, 
                               injection_SH, 
                               injection_lat_NH,
                               injection_lat_SH,
                               injection_year, 
                               injection_month, 
                               injection_day, 
                               timeframe, 
                               flux_year):

    # create exchange matrix for each season
    matrix_w, matrix_sp, matrix_s, matrix_f = create_transilient_matrix_new_new()
    transport_matrices = { 'winter': matrix_w,
                            'spring': matrix_sp,
                            'summer': matrix_s,
                            'fall': matrix_f    }



    # get region from latitude input [°]
    injection_reg_NH, injection_index_NH = get_region(injection_lat_NH)
    injection_reg_SH, injection_index_SH = get_region(injection_lat_SH)

    # initial distribution after injection (16 belts)
    zero_distr = np.zeros(16)
    init_distr_NH = initial_distribution(injection_reg_NH, injection_NH)
    init_distr_SH = initial_distribution(injection_reg_SH, injection_SH)
    init_distr_total = np.add(init_distr_NH, init_distr_SH) # überflüssig, weil innit distr in distr-4month enthalten ist

    # simulation transport mechanisms and sedimentation: final distribution of aerosols in every belt
    df_distribution_aerosol_belts = simulate_distribution_multi(zero_distr, init_distr_total, timeframe, injection_month, transport_matrices)
    df_distribution_aerosol_belts.to_excel(gen_excel(cwd + '\SAI Code\calc_pv_pot/data/processed/aerosol_distribution/aerosol_distribution_', version))

   # calculation tau from aerosol distribution & store to excel
    df_distribution_tau = df_distribution_aerosol_belts.apply(lambda aerosol: aerosol.map(easy_AOD))
    df_distribution_tau.to_excel(gen_excel(cwd + '\SAI Code\calc_pv_pot/data/processed/tau_distr/tau_monthly_distribution_', version))

    # convert df (tau): transposed and with hourly distribution
    df_distribution_tau_converted, times = convert_tau_file(df_distribution_tau, injection_year, injection_month, injection_day, timeframe)                   # INPUT FOR (2)
    df_distribution_tau_converted.to_excel(gen_excel(cwd + '\SAI Code\calc_pv_pot/data/processed/tau_distr/hourly/df_hourly_', version))

    df_distribution_tau_flux_year = tau_flux_year(df_distribution_tau_converted, flux_year)

    # calculation RF from AOD-distribution & store to excel
    df_distribution_RF = df_distribution_tau_converted.iloc[:, 1:].apply(lambda tau: tau.map(easy_RF))
    df_distribution_RF.to_excel(gen_excel(cwd + '\SAI Code\calc_pv_pot/data/processed/RF_distr/RF_hourly_distribution_', version))

    return df_distribution_tau_flux_year, times