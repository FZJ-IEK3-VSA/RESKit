from .. import NCSource
import numpy as np
from os.path import dirname, join


class Era5Source(NCSource):
    """The Era5Source object manages weather data (as netCDF4 files) coming from the 
    `ERA5 climate data products<https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5>`

    If furthermore allows access a number of common functionalities and constants which are
    often encountered when simulating renewable energy technologies

    Note:
    -----
    Various constants can have been set for this weather source which can impact later simulation workflows.

    These constants include:
        MAX_LON_DIFFERENCE = 0.26
            The maximum longitude difference to accept between a grid cell's center and the coordinates 
                to extract data for

        MAX_LAT_DIFFERENCE = 0.26
            The maximum latitude difference to accept between a grid cell's center and the coordinates 
                to extract data for

        WIND_SPEED_HEIGHT_FOR_WIND_ENERGY = 100
            The suggested altitude of wind speed data to use for wind-energy simulations

        WIND_SPEED_HEIGHT_FOR_SOLAR_ENERGY = 10
            The suggested altitude of wind speed data to use for wind-energy simulations

        LONG_RUN_AVERAGE_WINDSPEED : 
            <RESKit path>/weather/Era5Source/data/ERA5_wind_speed_100m_mean.tiff

            A path to a raster file with the long-time average wind speed in each grid cell
            * Can be used in wind energy simulations
            * Calculated at the height specified in `WIND_SPEED_HEIGHT_FOR_WIND_ENERGY`
            * Time range includes 1980 until the end of 2019 (the time of first calculation)
            * The averaging is performed globally

        LONG_RUN_AVERAGE_WINDDIR : 
            <RESKit path>/weather/Era5Source/data/ERA5_wind_direction_100m_mean.tiff

            A path to a raster file with the long-time average wind direction in each grid cell
            * Can be used in wind energy simulations
            * Calculated at the height specified in `WIND_SPEED_HEIGHT_FOR_WIND_ENERGY`
            * Time range includes 1980 until the end of 2019 (the time of first calculation)
            * The averaging is performed globally

        LONG_RUN_AVERAGE_GHI : 
            <RESKit path>/weather/Era5Source/data/ERA5_surface_solar_radiation_downwards_mean.tiff

            A path to a raster file with the long-time average global horizontal irradiance in 
                each grid cell
            * Can be used in solar energy simulations
            * Calculated at the surface
            * Time range includes 1980 until the end of 2019 (the time of first calculation)
            * The averaging is performed globally

        LONG_RUN_AVERAGE_DNI : 
            <RESKit path>/weather/Era5Source/data/ERA5_total_sky_direct_solar_radiation_at_surface_mean.tiff

            A path to a raster file with the long-time average direct horizontal irradiance in 
                each grid cell
            * Can be used in solar energy simulations
            * Calculated at the surface and on a horizontal plane (not DNI!)
            * Time range includes 1980 until the end of 2019 (the time of first calculation)
            * The averaging is performed globally


    See Also:
    ---------
    reskit.weather.MerraSource
    reskit.weather.SarahSource
    reskit.weather.Era5Source
    """

    ELEVATED_WIND_SPEED_HEIGHT = 100
    SURFACE_WIND_SPEED_HEIGHT = 10

    LONG_RUN_AVERAGE_WINDSPEED = join(
        dirname(__file__), "data", "ERA5_wind_speed_100m_mean.tiff"
    )
    LONG_RUN_AVERAGE_WINDDIR = join(
        dirname(__file__), "data", "ERA5_wind_direction_100m_mean.tiff"
    )
    LONG_RUN_AVERAGE_GHI = join(
        dirname(__file__), "data", "ERA5_surface_solar_radiation_downwards_mean.tiff"
    )
    LONG_RUN_AVERAGE_DNI_archive = join(
        dirname(__file__),
        "data",
        "ERA5_total_sky_direct_solar_radiation_at_surface_mean.tiff",
    )
    LONG_RUN_AVERAGE_DNI = join(dirname(__file__), "data", "ERA5_DNI_mean.tif")
    DNI_90_PERC_QUANT = join(
        dirname(__file__), "data", "ERA5_DNI_percentile_90_2000_to_2020.tif"
    )

    MAX_LON_DIFFERENCE = 0.26
    MAX_LAT_DIFFERENCE = 0.26

    def __init__(
        self, source, bounds=None, index_pad=5, time_index_from=None, **kwargs
    ):
        """Initialize a ERA5 style netCDF4 file source

         Compared to the generic NCSource object, the following parameters are automatically set:
             * tz = None
             * time_name = "time"
             * lat_name = "latitude"
             * lon_name = "longitude"
             * flip_lat = True
             * flip_lon = False
             * time_offset_minutes = -30


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

        # translate the mos common lear names for time_index_from
        ERA5_names = {
            "global_horizontal_irradiance_archive": "ssrd",
            "global_horizontal_irradiance": "ssrd_t_adj",
            "direct_horizontal_irradiance_archive": "fdir",
            "direct_horizontal_irradiance": "fdir_t_adj",
            "surface_wind_speed": "w10",
            "elevated_wind_speed": "w100",
        }
        if time_index_from in ERA5_names.keys():
            # if time_index_from is a known clear name use the dict
            time_index_from = ERA5_names[time_index_from]
        else:
            # hope it is a well known ERA5 string. checkes in super.__init__
            pass

        super().__init__(
            source=source,
            bounds=bounds,
            time_name="time",
            lat_name="latitude",
            lon_name="longitude",
            index_pad=index_pad,
            _max_lon_diff=self.MAX_LON_DIFFERENCE,
            _max_lat_diff=self.MAX_LAT_DIFFERENCE,
            tz=None,
            flip_lat=True,
            time_offset_minutes=-30,  # time convention -30
            time_index_from=time_index_from,
            **kwargs
        )

    loc_to_index = NCSource._loc_to_index_rect(0.25, 0.25)

    # STANDARD LOADERS
    def sload_boundary_layer_height(self):
        """Standard loader function for the variable 'boundary_layer_height' in meters 
        from the surface

        """
        return self.load("blh", "boundary_layer_height")

    def sload_elevated_wind_speed(self):
        """Standard loader function for the variable 'elevated_wind_speed'

        Automatically reads the variables "ws<X>" from the given ERA5 source and saves 
        it as the variable 'elevated_wind_speed' in the data library

        Where '<X>' is the height specified by `Era5Source.ELEVATED_WIND_SPEED_HEIGHT`

        The "ws<X>" variable also needs to be precomputed from the raw variables "u<X>" 
            and "v<X>"

        TODO: Update function to also be able to handle raw ERA5 inputs for u & v
        """
        return self.load(
            "ws{}".format(self.ELEVATED_WIND_SPEED_HEIGHT), "elevated_wind_speed"
        )

    def sload_surface_wind_speed(self):
        """Standard loader function for the variable 'surface_wind_speed'

        Automatically reads the variables "ws<X>" from the given ERA5 source and saves 
        it as the variable 'surface_wind_speed' in the data library

        Where '<X>' is the height specified by `Era5Source.SURFACE_WIND_SPEED_HEIGHT`

        The "ws<X>" variable also needs to be precomputed from the raw variables "u<X>" 
            and "v<X>"

        TODO: Update function to also be able to handle raw ERA5 inputs for u & v
        """
        return self.load(
            "ws{}".format(self.SURFACE_WIND_SPEED_HEIGHT), "surface_wind_speed"
        )

    def sload_wind_speed_at_100m(self):
        """Standard loader function for the variable 'wind_speed_at_100m'

        Automatically reads the variables "ws100" from the given ERA5 source and saves 
        it as the variable 'wind_speed_at_100m' in the data library

        The "ws100" variable also needs to be precomputed from the raw variables "u100" 
            and "v100"

        TODO: Update function to also be able to handle raw ERA5 inputs for u & v
        """
        return self.load("ws100", "wind_speed_at_100m")

    def sload_wind_speed_at_10m(self):
        """Standard loader function for the variable 'wind_speed_at_10m'

        Automatically reads the variables "ws10" from the given ERA5 source and saves 
        it as the variable 'wind_speed_at_10m' in the data library

        The "ws10" variable also needs to be precomputed from the raw variables "u10" 
            and "v10"

        TODO: Update function to also be able to handle raw ERA5 inputs for u & v
        """
        return self.load("ws10", "wind_speed_at_10m")

    def sload_elevated_wind_direction(self):
        """Standard loader function for the variable 'elevated_wind_direction'

        Automatically reads the variables "wd<X>" from the given ERA5 source and saves 
        it as the variable 'elevated_wind_direction' in the data library

        Where '<X>' is the height specified by `Era5Source.ELEVATED_WIND_SPEED_HEIGHT`

        The "wd<X>" variable also needs to be precomputed from the raw variables "u<X>" 
            and "v<X>" and made available in the raw dataset

        TODO: Update function to also be able to handle raw ERA5 inputs for u & v
        """
        return self.load("wd100", "elevated_wind_direction")

    def sload_surface_pressure(self):
        """Standard loader function for the variable 'surface_pressure'

        Automatically reads the variable "sp" from the given ERA5 source and saves it as the 
        variable 'surface_pressure' in the data library
        """
        return self.load("sp", name="surface_pressure")

    def sload_surface_air_temperature(self):
        """Standard loader function for the variable 'surface_air_temperature'

        Automatically reads the variable "t2m" from the given ERA5 source and saves it as the 
        variable 'surface_air_temperature' in the data library

        Temperature values are also converted from kelvin to degrees celsius
        """
        return self.load(
            "t2m", name="surface_air_temperature", processor=lambda x: x - 273.15
        )

    def sload_surface_dew_temperature(self):
        """Standard loader function for the variable 'surface_dew_temperature'

        Automatically reads the variable "d2m" from the given ERA5 source and saves it as the 
        variable 'surface_dew_temperature' in the data library

        Temperature values are also converted from kelvin to degrees celsius
        """
        return self.load(
            "d2m", name="surface_dew_temperature", processor=lambda x: x - 273.15
        )

    def sload_direct_horizontal_irradiance_archive(self):
        """Standard loader function for the variable 'direct_horizontal_irradiance'

        Automatically reads the variable "fdir" from the given ERA5 source and saves it as the 
        variable 'direct_horizontal_irradiance' in the data library
        """
        print(
            "WARNING: Non time corrected ERA5-direct_horizontal_irradiance loaded. Only do this, if you understand the implications of this!"
        )
        return self.load("fdir", name="direct_horizontal_irradiance_archive")

    def sload_direct_horizontal_irradiance(self):
        """Standard loader function for the variable 'direct_horizontal_irradiance'

        Automatically reads the variable "fdir" from the given ERA5 source and saves it as the 
        variable 'direct_horizontal_irradiance' in the data library
        """
        return self.load("fdir_t_adj", name="direct_horizontal_irradiance")

    def sload_global_horizontal_irradiance_archive(self):
        """Archive loader function for the variable 'global_horizontal_irradiance. Uses non corrected solar inputs.
        Use only for reproduceability purposes'

        Automatically reads the variable "ssrd" from the given ERA5 source and saves it as the 
        variable 'global_horizontal_irradiance' in the data library
        """
        print(
            "WARNING: Non time corrected ERA5-GHI loaded. Only do this, if you understand the implications of this!"
        )
        return self.load("ssrd", name="global_horizontal_irradiance_archive")

    def sload_global_horizontal_irradiance(self):
        """Standard loader function for the variable 'global_horizontal_irradiance'

        Automatically reads the variable "ssrd" from the given ERA5 source and saves it as the 
        variable 'global_horizontal_irradiance' in the data library
        """
        return self.load("ssrd_t_adj", name="global_horizontal_irradiance")
