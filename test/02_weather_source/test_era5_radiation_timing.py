"""ERA5 preprocessed by era5_prepare must put the sun where it is.

ERA5 stores solar radiation as the energy accumulated over the hour before each
timestamp. RESKit turns it into the mean flux of that hour and puts it at the middle
of the hour (Era5Source's time index is the timestamp minus 30 minutes). Shifting it
by an hour -- which era5_prepare did -- moves the radiation-weighted middle of the day
an hour away from solar noon. This test builds clear-sky accumulations from the sun
position and checks that middle.
"""

import numpy as np
import pandas as pd
import pvlib
import xarray as xr

from reskit.weather import Era5Source
from reskit.weather.era5_source.era5_prepare import preprocess_era5_data

LATITUDES = np.array([50.75, 50.5])
LONGITUDES = np.array([6.0, 6.25])
# ERA5 timestamps of three days, the accumulations of the hours ending at them
TIMESTAMPS = pd.date_range("2015-06-01 00:00", "2015-06-03 23:00", freq="h")
TOLERANCE_MINUTES = 15


def _clear_sky_flux(times: pd.DatetimeIndex) -> np.ndarray:
    """A clear-sky flux in W/m² at the centre of the grid: 1000 W/m² times cos(zenith)."""
    position = pvlib.solarposition.get_solarposition(times, LATITUDES.mean(), LONGITUDES.mean())
    return 1000.0 * np.clip(np.cos(np.radians(position["apparent_zenith"].values)), 0, None)


def _raw_accumulations() -> np.ndarray:
    """The energy in J/m² of the hour ending at each timestamp, from the flux per minute."""
    minutes = pd.date_range(TIMESTAMPS[0] - pd.Timedelta(minutes=59), TIMESTAMPS[-1], freq="min")
    energy = _clear_sky_flux(minutes) * 60.0
    hourly = energy.reshape(len(TIMESTAMPS), 60).sum(axis=1)
    return np.broadcast_to(hourly[:, None, None], (len(TIMESTAMPS), LATITUDES.size, LONGITUDES.size)).copy()


def _midpoint_offset_minutes(time_index: pd.DatetimeIndex, flux: np.ndarray) -> float:
    """How far the radiation-weighted middle of the days lies from solar noon, in minutes."""
    days = pd.Timedelta(days=1)
    minutes = pd.date_range(
        time_index.min().floor("D"), time_index.max().floor("D") + days, freq="min", inclusive="left"
    )
    clear_sky = pd.Series(_clear_sky_flux(minutes), index=minutes)
    solar_noon = clear_sky.groupby(clear_sky.index.date).idxmax()

    weights = np.nan_to_num(flux.mean(axis=(1, 2)))
    noon = pd.DatetimeIndex(solar_noon.loc[time_index.date].values)
    offsets = (time_index - noon).total_seconds().values / 60.0
    return float(np.sum(weights * offsets) / np.sum(weights))


def _raw_dataset() -> xr.Dataset:
    accumulation = _raw_accumulations()
    dims = ("time", "latitude", "longitude")
    attrs = {"units": "J m**-2"}
    return xr.Dataset(
        data_vars={
            "ssrd": (dims, accumulation, attrs),
            "fdir": (dims, 0.8 * accumulation, attrs),
        },
        coords={"time": TIMESTAMPS, "latitude": LATITUDES, "longitude": LONGITUDES},
    )


def test_era5_prepare_output_puts_the_sun_at_solar_noon(tmp_path):
    """Preprocessed with era5_prepare and read as every solar workflow reads it."""
    raw = tmp_path / "raw.nc"
    dataset = _raw_dataset()
    dataset["time"].encoding = {"units": "hours since 1900-01-01", "dtype": "int32"}
    dataset.to_netcdf(raw)
    preprocess_era5_data(str(raw), str(tmp_path / "processed"))

    source = Era5Source(str(tmp_path / "processed"), time_index_from="direct_horizontal_irradiance", verbose=False)
    source.sload_direct_horizontal_irradiance()
    source.sload_global_horizontal_irradiance()

    for variable in ("direct_horizontal_irradiance", "global_horizontal_irradiance"):
        offset = _midpoint_offset_minutes(source.time_index, source.data[variable])
        assert abs(offset) < TOLERANCE_MINUTES, (variable, offset)
