from os.path import dirname, join

from reskit.weather.nc_source import NCSource


class GSAmeanSource:
    GHI_with_ERA5_pixel = join(dirname(__file__), "GSA_GHI_mean_ERA5_pixel.tif")

    DNI_with_ERA5_pixel = join(dirname(__file__), "GSA_DNI_mean_ERA5_pixel.tif")

    def __init__():
        pass
