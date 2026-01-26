"""TODO: NEEDS UPDATING!!!"""

from reskit.weather.nc_source import *

# Define constants


class CordexSource(NCSource):
    """
    Open a netCDF4 source which is at the EURO - CORDEX EUR - 11 domain

    Standard variables are:
        clt - cloud cover[]
        dpas - 2m dew point temperature[K]
        hurs - 2m relative humidity[]
        huss - 2m specific humidity[kg kg - 1]
        pr - total(convective + large scale) precipitation[kg m - 2 s - 1]
        prsn - snowfall flux[kg m - 2 s - 1]
        ps - surface pressure[Pa]
        rlen - roughness length[m]
        rsds - surface downwelling shortwave radiation[W m - 2]
        rsdt - top of atmosphere incident shortwave radiation[W m - 2]
        tas - 2m temperature[K]
        uas - 10m u - velocity[m s - 1]
        vas - 10m v - velocity[m s - 1]
        glat - geographical latitude[deg N]
        glon - geographical longitude[deg E]
        orog - surface orography[m]
        sftlf - lang area fraction[]
    """

    GWA50_CONTEXT_MEAN_SOURCE = None
    GWA100_CONTEXT_MEAN_SOURCE = None

    def __init__(self, path, bounds=None, domain="EUR11"):
        print("WARNING: CordexSource has not been updated in awhile and is almost guaranteed to fail...")

        if not bounds is None:
            if isinstance(bounds, gk.Extent):
                bounds.pad((self.MAX_LON_DIFFERENCE, self.MAX_LAT_DIFFERENCE))
            else:
                if isinstance(bounds, Bounds):
                    lon_min = bounds.lonMin
                    lat_min = bounds.latMin
                    lon_max = bounds.lonMax
                    lat_max = bounds.latMax
                else:
                    print("Consider using a Bounds object or a gk.Extent object. They are safer!")
                    lon_min, lat_min, lon_max, lat_max = bounds

                bounds = Bounds(
                    lonMin=lon_min - self.MAX_LON_DIFFERENCE,
                    latMin=lat_min - self.MAX_LAT_DIFFERENCE,
                    lonMax=lon_max + self.MAX_LON_DIFFERENCE,
                    latMax=lat_max + self.MAX_LAT_DIFFERENCE,
                )

        NCSource.__init__(
            self,
            path=path,
            bounds=bounds,
            timeName="time",
            latName="lat",
            lonName="lon",
            dependent_coordinates=True,
        )

        # set maximal differences
        if domain == "EUR11":
            self._maximal_lon_difference = 0.0625
            self._maximal_lat_difference = 0.0625
        else:
            raise ResError("Domain not understood")

    def __add__(self, o):
        out = CordexSource(None)
        return NCSource.__add__(self, o, _shell=out)

    def load_wind_speed(self, v_name="vas", u_name="uas"):
        # read raw data
        self.load(v_name, heightIdx=0)
        self.load(u_name, heightIdx=0)

        # read the data
        u_data = self.data[u_name]
        v_data = self.data[v_name]

        # combine into a single time series matrix
        speed = np.sqrt(u_data * u_data + v_data * v_data)  # total speed
        direction = np.arctan2(v_data, u_data) * (180 / np.pi)  # total direction

        # done!
        self.data["windspeed"] = speed
        self.data["winddir"] = direction

    def load_radiation(self, ghi_name="rsds"):
        # read raw data
        self.load(ghi_name, name="ghi")

    def load_temperature(self, which="air", processor=lambda x: x - 273.15):
        """Temperature variable loader"""
        if which.lower() == "air":
            var_name = "tas"
        elif which.lower() == "dew":
            var_name = "dpas"
        else:
            raise ResMerraError("sub group '%s' not understood" % which)

        # load
        self.load(var_name, name=which + "_temp", processor=processor)

    def load_pressure(self):
        self.load("ps", name="pressure")
