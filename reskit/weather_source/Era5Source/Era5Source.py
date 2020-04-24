from .. import NCSource
import numpy as np
from os.path import dirname, join


class Era5Source(NCSource):

    ELEVATED_WIND_SPEED_HEIGHT = 100
    SURFACE_WIND_SPEED_HEIGHT = 10

    LONG_RUN_AVERAGE_WINDSPEED = join(
        dirname(__file__),
        "data",
        "ERA5_wind_speed_100m_mean.tiff")
    LONG_RUN_AVERAGE_WINDDIR = join(
        dirname(__file__),
        "data",
        "ERA5_wind_direction_100m_mean.tiff")
    LONG_RUN_AVERAGE_GHI = join(
        dirname(__file__),
        "data",
        "ERA5_surface_solar_radiation_downwards_mean.tiff")
    LONG_RUN_AVERAGE_DNI = join(
        dirname(__file__),
        "data",
        "ERA5_total_sky_direct_solar_radiation_at_surface_mean.tiff")

    MAX_LON_DIFFERENCE = 0.26
    MAX_LAT_DIFFERENCE = 0.26

    def __init__(self, source, bounds=None, indexPad=5, **kwargs):
        """Initialize a ERA5 style netCDF4 file source


        Parameters
        ----------
        source : str
            The path to the main data file

        bounds : Anything acceptable to geokit.Extent.load(), optional
            The boundaries of the data which is needed
              * Usage of this will help with memory mangement
              * If None, the full dataset is loaded in memory

        padExtent : numeric, optional
            The padding to apply to the boundaries 
              * Useful in case of interpolation

        timeName : str, optional
            The name of the time parameter in the netCDF4 dataset

        latName : str, optional
            The name of the latitude parameter in the netCDF4 dataset

        lonName : str, optional
            The name of the longitude parameter in the netCDF4 dataset

        timeBounds : tuple of length 2, optional
            Used to employ a slice of the time dimension
              * Expect two pandas Timestamp objects> The first indicates the point
                to start collecting data, and the second indicates the end

        """

        super().__init__(
            self,
            source=source,
            bounds=bounds,
            time_name="time",
            lat_name="latitude",
            lon_name="longitude",
            index_pad=indexPad,
            _max_lon_diff=self.MAX_LON_DIFFERENCE,
            _max_lat_diff=self.MAX_LAT_DIFFERENCE,
            tz=None,
            flip_lat=True,
            **kwargs)

    loc2Index = NCSource._lot_to_index_rect(0.25, 0.25)

    # STANDARD LOADERS
    def sload_elevated_wind_speed(self):
        return self.load(
            "ws{}".format(self.ELEVATED_WIND_SPEED_HEIGHT),
            "elevated_wind_speed")

    def sload_surface_wind_speed(self):
        return self.load(
            "ws{}".format(self.SURFACE_WIND_SPEED_HEIGHT),
            "surface_wind_speed")

    def sload_wind_speed_at_100m(self):
        return self.load("ws100", "wind_speed_at_100m")

    def sload_wind_speed_at_10m(self):
        return self.load("ws10", "wind_speed_at_10m")

    def sload_elevated_wind_direction(self):
        return self.load("wd100", "elevated_wind_direction")

    def sload_surface_pressure(self):
        return self.load("sp", name='surface_pressure')

    def sload_surface_air_temperature(self):
        return self.load("t2m", name="surface_air_temperature", processor=lambda x: x - 273.15)

    def sload_surface_dew_temperature(self):
        return self.load("d2m", name="surface_dew_temperature", processor=lambda x: x - 273.15)

    def sload_direct_horizontal_irradiance(self):
        return self.load("fdir", name="direct_horizontal_irradiance")

    def sload_global_horizontal_irradiance(self):
        return self.load("ssrd", name="global_horizontal_irradiance")
