from . import NCSource
import numpy as np


class SarahSource(NCSource):

    MAX_LON_DIFFERENCE = 0.06
    MAX_LAT_DIFFERENCE = 0.06

    def __init__(self, source, bounds=None, index_pad=5, **kwargs):
        """Initialize a SARAH style netCDF4 file source


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
            source=source,
            bounds=bounds,
            time_name="time",
            lat_name="lat",
            lon_name="lon",
            index_pad=index_pad,
            _max_lon_diff=self.MAX_LON_DIFFERENCE,
            _max_lat_diff=self.MAX_LAT_DIFFERENCE,
            tz=None,
            **kwargs)

    loc_to_index = NCSource._loc_to_index_rect(lat_step=0.05, lon_step=0.05)

    # STANDARD LOADERS
    def sload_direct_normal_irradiance(self):
        self.load("DNI", name="direct_normal_irradiance")
        sel = np.logical_or(self.data['direct_normal_irradiance'] < 0, np.isnan(self.data['direct_normal_irradiance']))
        self.data['direct_normal_irradiance'][sel] = 0

    def sload_global_horizontal_irradiance(self):
        self.load("SIS", name="global_horizontal_irradiance")
        sel = np.logical_or(self.data['global_horizontal_irradiance'] < 0, np.isnan(self.data['global_horizontal_irradiance']))
        self.data['global_horizontal_irradiance'][sel] = 0
