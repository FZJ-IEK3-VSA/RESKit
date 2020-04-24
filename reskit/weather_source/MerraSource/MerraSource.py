from .. import NCSource
import numpy as np
from os.path import dirname, join
import geokit as gk


class MerraSource(NCSource):

    ELEVATED_WIND_SPEED_HEIGHT = 50
    SURFACE_WIND_SPEED_HEIGHT = 2

    LONG_RUN_AVERAGE_WINDSPEED = join(
        dirname(__file__),
        "data",
        "merra_average_windspeed_50m-shifted.tif")

    LONG_RUN_AVERAGE_GHI = join(
        dirname(__file__),
        "data",
        "merra_average_SWGDN_1994-2015_globe.tif")

    MAX_LON_DIFFERENCE = 0.5
    MAX_LAT_DIFFERENCE = 0.5

    def __init__(self, source, bounds=None, indexPad=5, **kwargs):
        """Initialize a Merra2 style netCDF4 file source


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
            lat_name="lat",
            lon_name="lon",
            index_pad=indexPad,
            _max_lon_diff=self.MAX_LON_DIFFERENCE,
            _max_lat_diff=self.MAX_LAT_DIFFERENCE,
            tz="GMT",
            **kwargs)

    loc2Index = NCSource._lot_to_index_rect(lat_step=0.5, lon_step=0.625)

    def context_area_at_index(self, latI, lonI):
        """Compute the context area surrounding the a specified index"""
        # Make and return a box
        lowLat = self.lats[latI] - 0.25
        highLat = self.lats[latI] + 0.25
        lowLon = self.lons[lonI] - 0.3125
        highLon = self.lons[lonI] + 0.3125

        return gk.geom.box(lowLon, lowLat, highLon, highLat, srs=gk.srs.EPSG4326)

    # STANDARD LOADERS
    def _load_uv(self, height):
        U = "U{}M".format(height)
        V = "V{}M".format(height)

        self.load(U)
        self.load(V)

        return self.data[U], self.data[V]

    def _load_wind_speed(self, height):
        uData, vData = self._load_uv(height=height)
        return np.sqrt(uData * uData + vData * vData)  # total speed

    def sload_elevated_wind_speed(self):
        self.data["elevated_wind_speed"] = self._load_wind_speed(
            height=self.ELEVATED_WIND_SPEED_HEIGHT)

    def sload_surface_wind_speed(self):
        self.data["surface_wind_speed"] = self._load_wind_speed(
            height=self.SURFACE_WIND_SPEED_HEIGHT)

    def sload_wind_speed_at_2m(self):
        self.data["wind_speed_at_2m"] = self._load_wind_speed(2)

    def sload_wind_speed_at_10m(self):
        self.data["wind_speed_at_10m"] = self._load_wind_speed(10)

    def sload_wind_speed_at_50m(self):
        self.data["wind_speed_at_50m"] = self._load_wind_speed(50)

    def _load_wind_dir(self, height):
        uData, vData = self._load_uv(height=height)
        return np.arctan2(vData, uData) * (180 / np.pi)  # total direction

    def sload_elevated_wind_direction(self):
        self.data["elevated_wind_speed"] = self._load_wind_speed(
            height=self.ELEVATED_WIND_SPEED_HEIGHT)

    def sload_surface_wind_direction(self):
        self.data["surface_wind_speed"] = self._load_wind_speed(
            height=self.SURFACE_WIND_SPEED_HEIGHT)

    def sload_wind_direction_at_2m(self):
        self.data["wind_speed_at_2m"] = self._load_wind_speed(2)

    def sload_wind_direction_at_10m(self):
        self.data["wind_speed_at_10m"] = self._load_wind_speed(10)

    def sload_wind_direction_at_50m(self):
        self.data["wind_speed_at_50m"] = self._load_wind_speed(50)

    def sload_surface_pressure(self):
        return self.load("PS", name='surface_pressure')

    def sload_surface_air_temperature(self):
        return self.load("T2M", name="surface_air_temperature", processor=lambda x: x - 273.15)

    def sload_surface_dew_temperature(self):
        return self.load("T2MDEW", name="surface_dew_temperature", processor=lambda x: x - 273.15)

    def sload_global_horizontal_irradiance(self):
        return self.load("SWGDN", name="global_horizontal_irradiance")
