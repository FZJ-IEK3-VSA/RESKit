"""Test helper: check that ERA5 data from the Zarr test store and the netCDF4 fixtures agree within the difference their encodings allow.

Both hold the same ERA5 hours, each rounded differently: the Zarr store by bit-rounding
(a step relative to the value), the netCDF4 fixtures by 16 bit packing (a fixed
scale_factor step). An element may differ by one Zarr step plus two packing steps,
doubled as a margin.
"""

from __future__ import annotations

from glob import glob
from os import PathLike
from os.path import join

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike, NDArray

from reskit import data

ZARR_MANTISSA_BITS = 10  # mantissa bits kept by the Zarr store's bit-rounding
MARGIN = 2  # safety factor on the summed rounding steps

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


TEST_SUITE_PATHS = data.paths("test_suite")
NETCDF_FIXTURE_FOLDER = TEST_SUITE_PATHS["era5"]


def netcdf_scale_factor(
    nc_variable: str,
    folder: str | PathLike[str] = NETCDF_FIXTURE_FOLDER,
) -> float:
    """The packing step of a variable of the netCDF4 fixtures.

    Parameters
    ----------
    nc_variable : str
        Name of the variable in the netCDF4 files, e.g. ``"t2m"``.
    folder : str or path-like, optional
        Folder with the ``*.nc`` fixtures, by default the ``era5`` handle of the
        ``test_suite`` collection.

    Returns
    -------
    float
        The ``scale_factor`` of the variable, in its stored unit, from the first
        fixture file (in sorted order) that holds it.

    Raises
    ------
    KeyError
        If no fixture file in ``folder`` holds ``nc_variable``.
    """
    fixture_pattern = join(folder, "*.nc")
    fixture_paths = glob(fixture_pattern)
    fixture_paths_sorted = sorted(fixture_paths)  # deterministic file order

    for fixture_path in fixture_paths_sorted:
        # mask_and_scale=False keeps the packing attributes in 'attrs'
        with xr.open_dataset(fixture_path, mask_and_scale=False) as fixture_dataset:
            variables_in_fixture = fixture_dataset.data_vars
            # not every fixture file holds every variable: use the first that does
            if nc_variable in variables_in_fixture:
                variable_attributes = fixture_dataset[nc_variable].attrs
                scale_factor = variable_attributes["scale_factor"]
                return float(scale_factor)

    raise KeyError(f"No netCDF4 fixture in {folder} holds '{nc_variable}'")


def encoding_tolerance(expected: ArrayLike, variable: str) -> NDArray[np.float64]:
    """The largest difference each element of 'expected' (netCDF4, RESKit's unit) may have from the Zarr data.

    Parameters
    ----------
    expected : array-like
        The netCDF4 values, in RESKit's unit.
    variable : str
        RESKit variable name, a key of ``STORED_AS``.

    Returns
    -------
    numpy.ndarray
        The tolerance per element, in RESKit's unit, with the shape of ``expected``.
        It is the Zarr step (relative to the value in its stored unit) plus two
        netCDF4 packing steps, times ``MARGIN``.

    Raises
    ------
    KeyError
        If ``variable`` is not in ``STORED_AS``, or no fixture holds its netCDF4 variable.
    """
    nc_variable, unit_offset = STORED_AS[variable]

    expected_values = np.asarray(expected)
    # the Zarr step scales with the value in the stored unit (e.g. K, not °C)
    expected_in_stored_unit = expected_values - unit_offset
    stored_magnitude = np.abs(expected_in_stored_unit)

    relative_zarr_step = 2.0**-ZARR_MANTISSA_BITS
    zarr_step = relative_zarr_step * stored_magnitude

    # the netCDF4 fixtures were packed twice (GRIB, then 16 bit), hence two steps
    netcdf_packing_step = netcdf_scale_factor(nc_variable)
    two_netcdf_packing_steps = 2 * netcdf_packing_step

    total_step = zarr_step + two_netcdf_packing_steps
    return MARGIN * total_step


def assert_matches_encoding(actual: ArrayLike, expected: ArrayLike, variable: str) -> None:
    """Assert that Zarr data 'actual' differs from netCDF4 data 'expected' only by their encodings.

    Parameters
    ----------
    actual : array-like
        The values read from the Zarr test store, in RESKit's unit.
    expected : array-like
        The values read from the netCDF4 fixtures, in RESKit's unit, with the
        shape of ``actual``.
    variable : str
        RESKit variable name, a key of ``STORED_AS``.

    Raises
    ------
    AssertionError
        If the shapes differ, or any element differs by more than
        ``encoding_tolerance``; the message gives the worst element as a multiple
        of its tolerance.
    """
    actual_values = np.asarray(actual)
    expected_values = np.asarray(expected)
    assert actual_values.shape == expected_values.shape, variable

    difference = actual_values - expected_values
    absolute_difference = np.abs(difference)

    tolerance = encoding_tolerance(expected_values, variable)
    tolerance_used = absolute_difference / tolerance  # 1 means exactly at the bound

    # the worst element decides, and is reported in the failure message
    largest_tolerance_used = tolerance_used.max()
    assert largest_tolerance_used <= 1, (
        f"{variable} differs by up to {largest_tolerance_used:.3g} times its encoding tolerance"
    )
