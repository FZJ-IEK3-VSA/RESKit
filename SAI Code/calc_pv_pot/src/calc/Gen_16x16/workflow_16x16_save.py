from netCDF4 import Dataset
import numpy as np
import xarray as xr
from typing import List, Dict

def to_xarray_16x16(sim_data_MA):

    ds_16x16 = xr.Dataset()

    # define dimensions
    region =  sim_data_MA['regions_16x16']
    time = sim_data_MA['time_stamps']

    # convert to array
    def_regions_lat = np.array(sim_data_MA['def_regions_lat'])
    def_regions_lon = np.array(sim_data_MA['def_regions_lon'])

    # add to xarray
    ds_16x16['regions_16x16'] = (('region',), region)
    ds_16x16['time_stamps'] = (('time',), time)

    ds_16x16['def_regions_lat'] = (('region', 'lat_range'), def_regions_lat)
    ds_16x16['def_regions_lon'] = (('region', 'lat_range'), def_regions_lon)

    ds_16x16['AOD'] = (('time', 'region'), sim_data_MA['AOD'])
    ds_16x16['cos_zen'] = (('time', 'region'), sim_data_MA['cos_zen'])

    ds_16x16['GHI'] = (('time', 'region'), sim_data_MA['GHI'])
    ds_16x16['DNI'] = (('time', 'region'), sim_data_MA['DNI'])
    ds_16x16['DHI'] = (('time', 'region'), sim_data_MA['DHI'])
    ds_16x16['DIR'] = (('time', 'region'), sim_data_MA['DIR'])

    ds_16x16['delta_GHI'] = (('time', 'region'), sim_data_MA['delta_GHI'])
    ds_16x16['delta_DNI'] = (('time', 'region'), sim_data_MA['delta_DNI'])
    ds_16x16['delta_DHI'] = (('time', 'region'), sim_data_MA['delta_DHI'])
    ds_16x16['delta_DIR'] = (('time', 'region'), sim_data_MA['delta_DIR'])

    return ds_16x16



def save_nc4_16x16(ds_16x16, nc_filepath):

    ds_16x16.to_netcdf(nc_filepath, format='NETCDF4')

    print('saved successfully')