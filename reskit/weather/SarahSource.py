from . import NCSource
import numpy as np


class SarahSource(NCSource):
    """The SarahSource object manages weather data (as netCDF4 files) coming from the
    `SARAH satellite-reconstructed data products<https://climatedataguide.ucar.edu/climate-data/surface-solar-radiation-data-set-heliosat-sarah-edition-1>`

    If furthermore allows access a number of common functionalities and constants which are
    often encountered when simulating renewable energy technologies


    Note:
    -----
    Various constants can have been set for this weather source which can impact later simulation workflows.

    For this source, these constants include:
        MAX_LON_DIFFERENCE = 0.06
            The maximum longitude difference to accept between a grid cell's center and the coordinates
                to extract data for

        MAX_LAT_DIFFERENCE = 0.06
            The maximum latitude difference to accept between a grid cell's center and the coordinates
                to extract data for


    See Also:
    ---------
    reskit.weather.MerraSource
    reskit.weather.SarahSource
    reskit.weather.Era5Source
    """

    MAX_LON_DIFFERENCE = 0.06
    MAX_LAT_DIFFERENCE = 0.06

    def __init__(self, source, bounds=None, index_pad=5, **kwargs):
        """Initialize a SARAH style netCDF4 file source

        Compared to the generic NCSource object, the following parameters are automatically set:
            * tz = None
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
            tz=None,
            **kwargs
        )

    loc_to_index = NCSource._loc_to_index_rect(lat_step=0.05, lon_step=0.05)

    # STANDARD LOADERS
    def sload_direct_normal_irradiance(self):
        """Standard loader function for the variable 'direct_normal_irradiance'

        Automatically reads the variable "DNI" from the given SARAH source and saves it as the
        variable 'direct_normal_irradiance' in the data library
        """
        self.load("DNI", name="direct_normal_irradiance")
        sel = np.logical_or(
            self.data["direct_normal_irradiance"] < 0,
            np.isnan(self.data["direct_normal_irradiance"]),
        )
        self.data["direct_normal_irradiance"][sel] = 0

    def sload_global_horizontal_irradiance(self):
        """Standard loader function for the variable 'global_horizontal_irradiance'

        Automatically reads the variable "SIS" from the given SARAH source and saves it as the
        variable 'global_horizontal_irradiance' in the data library
        """
        self.load("SIS", name="global_horizontal_irradiance")
        sel = np.logical_or(
            self.data["global_horizontal_irradiance"] < 0,
            np.isnan(self.data["global_horizontal_irradiance"]),
        )
        self.data["global_horizontal_irradiance"][sel] = 0
