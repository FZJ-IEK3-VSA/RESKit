"""How closely ERA5 data from the Zarr test store can match the netCDF4 fixtures.

Both hold the same ERA5 hours, each rounded by its own encoding:

- era5-zarr/era5.zarr keeps the bit-rounding of its online source: float32 with
  ZARR_MANTISSA_BITS mantissa bits, a step of 2**-ZARR_MANTISSA_BITS of the value's
  magnitude in its stored unit (K, Pa, m/s, J/m² per hour).
- the 'era5' netCDF4 fixtures are packed to 16 bit integers, a step of the file's
  scale_factor; they were themselves made from packed GRIB data.

So an element may differ by one Zarr step plus two packing steps, doubled as a margin.
The largest difference measured uses about half of that. The bound is relative to the
stored value, not to the variable's range: a temperature of 270 K is rounded to 0.25 K
however little it varies.
"""

from glob import glob
from os.path import join

import numpy as np
import xarray as xr

from reskit import data

ZARR_MANTISSA_BITS = 10
MARGIN = 2

# RESKit variable -> netCDF4 variable it is read from, and RESKit's unit minus the stored unit
STORED_AS = {
    "elevated_wind_speed": ("ws100", 0.0),
    "surface_wind_speed": ("ws10", 0.0),
    "surface_pressure": ("sp", 0.0),
    "surface_air_temperature": ("t2m", -273.15),
    "surface_dew_temperature": ("d2m", -273.15),
    "boundary_layer_height": ("blh", 0.0),
    "global_horizontal_irradiance": ("ssrd_t_adj", 0.0),
    "direct_horizontal_irradiance": ("fdir_t_adj", 0.0),
}


def netcdf_scale_factor(nc_variable, folder=data.paths("test_suite")["era5"]):
    """The packing step of a variable of the netCDF4 fixtures."""
    for path in sorted(glob(join(folder, "*.nc"))):
        with xr.open_dataset(path, mask_and_scale=False) as ds:
            if nc_variable in ds.data_vars:
                return float(ds[nc_variable].attrs["scale_factor"])
    raise KeyError(f"No netCDF4 fixture in {folder} holds '{nc_variable}'")


def encoding_tolerance(expected, variable):
    """The largest difference each element of 'expected' (netCDF4, RESKit's unit) may have from the Zarr data."""
    nc_variable, offset = STORED_AS[variable]
    stored = np.abs(np.asarray(expected) - offset)
    zarr_step = 2.0**-ZARR_MANTISSA_BITS * stored
    return MARGIN * (zarr_step + 2 * netcdf_scale_factor(nc_variable))


def assert_matches_encoding(actual, expected, variable):
    """Assert that Zarr data 'actual' differs from netCDF4 data 'expected' only by their encodings."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, variable
    used = np.abs(actual - expected) / encoding_tolerance(expected, variable)
    assert used.max() <= 1, f"{variable} differs by up to {used.max():.3g} times its encoding tolerance"
