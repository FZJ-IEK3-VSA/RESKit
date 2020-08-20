from .. import NCSource
import numpy as np
from os.path import dirname, join
import geokit as gk


class MerraSource(NCSource):
    """The MerraSource object manages weather data (as netCDF4 files) coming from the 
    `MERRA2 climate data products<https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/>`

    If furthermore allows access a number of common functionalities and constants which are
    often encountered when simulating renewable energy technologies

    Note:
    -----
    Various constants can have been set for this weather source which can impact later simulation workflows.

    These constants include:
        MAX_LON_DIFFERENCE = 0.625
            The maximum longitude difference to accept between a grid cell's center and the coordinates 
                to extract data for

        MAX_LAT_DIFFERENCE = 0.5
            The maximum latitude difference to accept between a grid cell's center and the coordinates 
                to extract data for

        WIND_SPEED_HEIGHT_FOR_WIND_ENERGY = 50
            The suggested altitude of wind speed data to use for wind-energy simulations

        WIND_SPEED_HEIGHT_FOR_SOLAR_ENERGY = 2
            The suggested altitude of wind speed data to use for wind-energy simulations

        LONG_RUN_AVERAGE_WINDSPEED : 
            <RESKit path>/weather/MerraSource/data/merra_average_windspeed_50m-shifted.tif

            A path to a raster file with the long-time average wind speed in each grid cell
            * Can be used in wind energy simulations
            * Calculated at the height specified in `WIND_SPEED_HEIGHT_FOR_WIND_ENERGY`
            * Time range includes 1980 until the end of 2017 (the time of first calculation)
            * Only a few regions have been precomputed, therefore applications which require this 
                data will not work outside of these regions. They include:
                - Europe
                - Australia
                - Iceland
                - Parts of south america
                - Parts of north america


        LONG_RUN_AVERAGE_GHI : 
            <RESKit path>/weather/MerraSource/data/merra_average_SWGDN_1994-2015_globe.tif

            A path to a raster file with the long-time average global horizontal irradiance in 
                each grid cell
            * Can be used in solar energy simulations
            * Calculated at the surface
            * Time range includes 1994 until the end of 2015 (to match the global solar atlas)
            * Only a few regions have been precomputed, therefore applications which require this 
                data will not work outside of these regions. They include:
                - Europe
                - Australia
                - Iceland
                - Parts of south america
                - Parts of north america


    See Also:
    ---------
    reskit.weather.MerraSource
    reskit.weather.SarahSource
    reskit.weather.Era5Source
    """

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

    MAX_LON_DIFFERENCE = 0.625
    MAX_LAT_DIFFERENCE = 0.5

    def __init__(self, source, bounds=None, index_pad=5, **kwargs):
        """Initialize a MERRA2 style netCDF4 file source

        Compared to the generic NCSource object, the following parameters are automatically set:
            * tz = "GMT"
            * time_name = "time"
            * lat_name = "lat"
            * lon_name = "lon"
            * flip_lat = False
            * flip_lon = False
            * time_offset_minutes = 0


        Parameters:
        -----------
        path : str or list of str
            The path to the main data file(s) to load

            If multiple files are given, or if a directory of netCDF4 files is given, then it is assumed
            that all files ending with the extension '.nc' or '.nc4' should be managed by this object.
            * Be sure that all the netCDF4 files given share the same time and spatial dimensions!

        bounds : Anything acceptable to geokit.Extent.load(), optional
            The boundaries of the data which is needed
              * Usage of this will help with memory mangement
              * If None, the full dataset is loaded in memory
              * The actual extent of the loaded data depends on the source's
                available data

        index_pad : int, optional
            The padding to apply to the boundaries
              * Useful in case of interpolation
              * Units are in longitudinal degrees

        verbose : bool, optional
            If True, then status outputs are printed when searching for and reading weather data

        forward_fill : bool, optional
            If True, then missing data in the weather file is forward-filled
            * Generally, there should be no missing data at all. This option is only intended to
                catch the rare scenarios where one or two timesteps are missing

        See Also:
        ---------
        MerraSource
        SarahSource
        Era5Source
        """

        super().__init__(
            source=source,
            bounds=bounds,
            time_name="time",
            lat_name="lat",
            lon_name="lon",
            index_pad=index_pad,
            _max_lon_diff=self.MAX_LON_DIFFERENCE,
            _max_lat_diff=self.MAX_LAT_DIFFERENCE,
            tz="GMT",
            **kwargs)

    loc_to_index = NCSource._loc_to_index_rect(lat_step=0.5, lon_step=0.625)

    def context_area_at_index(self, latI, lonI):
        """Compute the context area surrounding the specified index of the original source"""
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
        """Standard loader function for the variable 'elevated_wind_speed'

        Automatically reads the variables "U<X>M" and "V<X>M" from the given MERRA2 source, 
        computes the total windspeed, and saves it as the variable 'elevated_wind_speed' in 
        the data library

        Where '<X>' is the height specified by `MerraSource.ELEVATED_WIND_SPEED_HEIGHT`
        """
        self.data["elevated_wind_speed"] = self._load_wind_speed(
            height=self.ELEVATED_WIND_SPEED_HEIGHT)

    def sload_surface_wind_speed(self):
        """Standard loader function for the variable 'surface_wind_speed'

        Automatically reads the variables "U<X>M" and "V<X>M" from the given MERRA2 source, 
        computes the total windspeed, and saves it as the variable 'surface_wind_speed' in 
        the data library

        Where '<X>' is the height specified by `MerraSource.SURFACE_WIND_SPEED_HEIGHT`
        """
        self.data["surface_wind_speed"] = self._load_wind_speed(
            height=self.SURFACE_WIND_SPEED_HEIGHT)

    def sload_wind_speed_at_2m(self):
        """Standard loader function for the variable 'wind_speed_at_2m'

        Automatically reads the variables "U2M" and "V2M" from the given MERRA2 source, 
        computes the total windspeed, and saves it as the variable 'wind_speed_at_2m' in 
        the data library
        """
        self.data["wind_speed_at_2m"] = self._load_wind_speed(2)

    def sload_wind_speed_at_10m(self):
        """Standard loader function for the variable 'wind_speed_at_10m'

        Automatically reads the variables "U10M" and "V10M" from the given MERRA2 source, 
        computes the total windspeed, and saves it as the variable 'wind_speed_at_10m' in 
        the data library
        """
        self.data["wind_speed_at_10m"] = self._load_wind_speed(10)

    def sload_wind_speed_at_50m(self):
        """Standard loader function for the variable 'wind_speed_at_50m'

        Automatically reads the variables "U50M" and "V50M" from the given MERRA2 source, 
        computes the total windspeed, and saves it as the variable 'wind_speed_at_50m' in 
        the data library
        """
        self.data["wind_speed_at_50m"] = self._load_wind_speed(50)

    def _load_wind_dir(self, height):
        uData, vData = self._load_uv(height=height)
        return np.arctan2(vData, uData) * (180 / np.pi)  # total direction

    def sload_elevated_wind_direction(self):
        """Standard loader function for the variable 'elevated_wind_direction'

        Automatically reads the variables "U<X>M" and "V<X>M" from the given MERRA2 source, 
        computes the total windspeed, and saves it as the variable 'elevated_wind_direction' in 
        the data library

        Where '<X>' is the height specified by `MerraSource.ELEVATED_WIND_SPEED_HEIGHT`
        """
        self.data["elevated_wind_direction"] = self._load_wind_speed(
            height=self.ELEVATED_WIND_SPEED_HEIGHT)

    def sload_surface_wind_direction(self):
        """Standard loader function for the variable 'surface_wind_direction'

        Automatically reads the variables "U<X>M" and "V<X>M" from the given MERRA2 source, 
        computes the total windspeed, and saves it as the variable 'surface_wind_direction' in 
        the data library

        Where '<X>' is the height specified by `MerraSource.SURFACE_WIND_SPEED_HEIGHT`
        """
        self.data["surface_wind_direction"] = self._load_wind_speed(
            height=self.SURFACE_WIND_SPEED_HEIGHT)

    def sload_wind_direction_at_2m(self):
        """Standard loader function for the variable 'wind_direction_at_2m'

        Automatically reads the variables "U2M" and "V2M" from the given MERRA2 source, 
        computes the total windspeed, and saves it as the variable 'wind_direction_at_2m' in 
        the data library
        """
        self.data["wind_direction_at_2m"] = self._load_wind_speed(2)

    def sload_wind_direction_at_10m(self):
        """Standard loader function for the variable 'wind_direction_at_10m'

        Automatically reads the variables "U10M" and "V10M" from the given MERRA2 source, 
        computes the total windspeed, and saves it as the variable 'wind_direction_at_10m' in 
        the data library
        """
        self.data["wind_direction_at_10m"] = self._load_wind_speed(10)

    def sload_wind_direction_at_50m(self):
        """Standard loader function for the variable 'wind_direction_at_50m'

        Automatically reads the variables "U50M" and "V50M" from the given MERRA2 source, 
        computes the total windspeed, and saves it as the variable 'wind_direction_at_50m' in 
        the data library
        """
        self.data["wind_direction_at_50m"] = self._load_wind_speed(50)

    def sload_surface_pressure(self):
        """Standard loader function for the variable 'surface_pressure'

        Automatically reads the variable "PS" from the given MERRA2 source and saves it as the 
        variable 'surface_pressure' in the data library
        """
        return self.load("PS", name='surface_pressure')

    def sload_surface_air_temperature(self):
        """Standard loader function for the variable 'surface_air_temperature'

        Automatically reads the variable "T2M" from the given MERRA2 source and saves it as the 
        variable 'surface_air_temperature' in the data library

        Temperature values are also converted from kelvin to degrees celsius
        """
        return self.load("T2M", name="surface_air_temperature", processor=lambda x: x - 273.15)

    def sload_surface_dew_temperature(self):
        """Standard loader function for the variable 'surface_dew_temperature'

        Automatically reads the variable "T2MDEW" from the given MERRA2 source and saves it as the 
        variable 'surface_dew_temperature' in the data library

        Temperature values are also converted from kelvin to degrees celsius
        """
        return self.load("T2MDEW", name="surface_dew_temperature", processor=lambda x: x - 273.15)

    def sload_global_horizontal_irradiance(self):
        """Standard loader function for the variable 'global_horizontal_irradiance'

        Automatically reads the variable "SWGDN" from the given MERRA2 source and saves it as the 
        variable 'global_horizontal_irradiance' in the data library
        """
        return self.load("SWGDN", name="global_horizontal_irradiance")
