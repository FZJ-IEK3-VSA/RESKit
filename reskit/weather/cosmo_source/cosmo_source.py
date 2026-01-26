"""TODO: NEEDS UPDATING!!!"""

import pytz
from geokit.core.location import LocationSet

from reskit.weather.nc_source import *

# Define constants


class CosmoSource(NCSource):
    """
    Handles the sources Sev created from the COSMO-REA6 dataset (cannot handle the original sources because they're whack)
    """

    gwa50_context_mean_source = None
    gwa100_context_mean_source = None

    # a LARGE ooverestimate of how much space should be in between a given point and the nearest index
    max_lon_difference = 0.6
    # a LARGE ooverestimate of how much space should be in between a given point and the nearest index
    max_lat_difference = 0.6

    def __init__(self, source, bounds=None, index_pad=0, **kwargs):
        """Initialize a COSMO style netCDF4 file source

        * Assumes REA6 conventions

        Parameters
        ----------
        source : str
            The path to the main data file

        bounds : Anything acceptable to geokit.Extent.load(), optional
            The boundaries of the data which is needed
              * Usage of this will help with memory management
              * If None, the full dataset is loaded in memory

        padExtent : numeric, optional
            The padding to apply to the boundaries
              * Useful in case of interpolation

        timeBounds : tuple of length 2, optional
            Used to employ a slice of the time dimension
              * Expect two pandas Timestamp objects> The first indicates the point
                to start collecting data, and the second indicates the end

        """
        NCSource.__init__(
            self,
            source=source,
            bounds=bounds,
            time_name="time",
            lat_name="lat",
            lon_name="lon",
            index_pad=index_pad,
            _max_lon_diff=self.max_lon_difference,
            _max_lat_diff=self.max_lat_difference,
            tz=pytz.FixedOffset(60),
            **kwargs,
        )

    def loc2_index(self, loc, outside_okay=False, as_int=True):
        """Returns the closest X and Y indexes corresponding to a given location
        or set of locations

        Parameters
        ----------
        loc : Anything acceptable by geokit.LocationSet
            The location(s) to search for
            * A single tuple with (lon, lat) is acceptable, or a list of such tuples
            * A single point geometry (as long as it has an SRS), or a list
              of geometries is okay
            * geokit,Location, or geokit.LocationSet are best!

        outsideOkay : bool, optional
            Determines if points which are outside the source's lat/lon grid
            are allowed
            * If True, points outside this space will return as None
            * If False, an error is raised

        Returns
        -------
        If a single location is given: tuple
            * Format: (yIndex, xIndex)
            * y index can be accessed with '.yi'
            * x index can be accessed with '.xi'

        If multiple locations are given: list
            * Format: [ (yIndex1, xIndex1), (yIndex2, xIndex2), ...]
            * Order matches the given order of locations

        """
        # Set REA6 Conventions
        lon_south_pole = 18
        lat_south_pole = -39.25
        rlon_res = 0.0550000113746
        rlat_res = 0.0550001976179
        rlon_start = -28.40246773
        rlat_start = -23.40240860

        if self is None:
            _lon_start = 0
            _lat_start = 0
            _lat_n = 824
            _lon_n = 848
        else:
            _lon_start = self._lon_start
            _lat_start = self._lat_start
            _lat_n = self._lat_n
            _lon_n = self._lon_n

        # Ensure loc is a list
        locations = LocationSet(loc)

        # Convert to rotated coordinates
        rlon_coords, rlat_coords = rotateFromLatLon(
            locations.lons,
            locations.lats,
            lonSouthPole=lon_south_pole,
            latSouthPole=lat_south_pole,
        )

        # Find integer locations
        lon_i = (rlon_coords - rlon_start) / rlon_res - _lon_start
        lat_i = (rlat_coords - rlat_start) / rlat_res - _lat_start

        # Check for out of bounds
        self = (lat_i < 0) | (lat_i >= _lat_n) | (lon_i < 0) | (lon_i >= _lon_n)
        if self.any():
            if not outside_okay:
                print("The following locations are out of bounds")
                print(locations[self])
                raise ResError("Locations are outside the boundaries")

        # Make int, maybe
        if as_int:
            lon_i = np.round(lon_i).astype(int)
            lat_i = np.round(lat_i).astype(int)

        # Make output
        if locations.count == 1:
            if self[0] is True:
                return None
            else:
                return Index(yi=lat_i[0], xi=lon_i[0])
        else:
            return [None if ss else Index(yi=y, xi=x) for ss, y, x in zip(self, lat_i, lon_i)]

    def load_radiation(self):
        """frankCorrection: 'Bias correction of a novel European reanalysis data set for solar energy applications'"""
        self.load("SWDIFDS_RAD", "dhi")
        self.load("SWDIRS_RAD", "dni_flat")
        self.data["ghi"] = self.data["dhi"] + self.data["dni_flat"]

        del self.data["dni_flat"], self.data["dhi"]

    def load_wind_speed_levels(self):
        self.load("windspeed_10", name="windspeed_10")
        self.load("windspeed_50", name="windspeed_50")
        self.load("windspeed_100", name="windspeed_100")
        self.load("windspeed_140", name="windspeed_140")

    def load_wind_speed_at_height(self, height=100):
        """NEEDS UPDATING!"""
        # Check if height is on of the heights we already have
        # The 3 known heights should always be 50, 100, and 140
        if height == 10:
            self.load("windspeed_10", name="windspeed")
        elif height == 50:
            self.load("windspeed_50", name="windspeed")
        elif height == 100:
            self.load("windspeed_100", name="windspeed")
        elif height == 140:
            self.load("windspeed_140", name="windspeed")
        else:
            # projection is required
            if height <= 50:
                self.load("windspeed_10")
                self.load("windspeed_50")
                self.load("windspeed_100")

                # DO CUBIC INTERP
                raise RuntimeError("This hasn't been implemented yet :(")

                # Remove unneeded data
                del self.data["windspeed_10"]
                del self.data["windspeed_50"]
                del self.data["windspeed_100"]

            elif height < 100:
                self.load("windspeed_50")
                self.load("windspeed_100")

                fac = (height - 50) / (100 - 50)

                new_wspd = self.data["windspeed_100"] * fac + self.data["windspeed_50"] * (1 - fac)
                self.data["windspeed"] = new_wspd

                del self.data["windspeed_50"]
                del self.data["windspeed_100"]

            else:
                self.load("windspeed_100")
                self.load("windspeed_140")

                fac = (height - 100) / (140 - 100)

                new_wspd = self.data["windspeed_140"] * fac + self.data["windspeed_100"] * (1 - fac)
                self.data["windspeed"] = new_wspd

                del self.data["windspeed_100"]
                del self.data["windspeed_140"]

    def load_temperature(self, processor=lambda x: x - 273.15):
        """Load the typical pressure variable"""
        self.load("2t", name="air_temp", processor=processor)

    def load_pressure(self):
        """Load the typical pressure variable"""
        self.load("sp", name="pressure")

    def load_set_pv(self, verbose=False, _clockstart=None, _header=""):
        if verbose:
            from datetime import datetime as dt

            if _clockstart is None:
                _clockstart = dt.now()
            print(
                _header,
                "Loading radiation at: +%.2fs" % (dt.now() - _clockstart).total_seconds(),
            )
        self.load_radiation()

        if verbose:
            print(
                _header,
                "Loading wind speed at: +%.2fs" % (dt.now() - _clockstart).total_seconds(),
            )
        self.load_wind_speed_at_height(10)

        if verbose:
            print(
                _header,
                "Loading pressure at: +%.2fs" % (dt.now() - _clockstart).total_seconds(),
            )
        self.load_pressure()

        if verbose:
            print(
                _header,
                "Loading temperature at: +%.2fs" % (dt.now() - _clockstart).total_seconds(),
            )
        self.load_temperature()

    def get_wind_speed_at_heights(
        self,
        locations,
        heights,
        spatial_interpolation="near",
        force_data_frame=False,
        outside_okay=False,
        _indicies=None,
    ):
        """
        Retrieve complete time series for a variable from the source's loaded data
        table at the given location(s)

        Parameters
        ----------
            locations : Anything acceptable by geokit.LocationSet
                The location(s) to search for
                  * A single tuple with (lon, lat) is acceptable, or a list of such
                    tuples
                  * A single point geometry (as long as it has an SRS), or a list
                    of geometries is okay
                  * geokit,Location, or geokit.LocationSet are best, though

            spatialInterpolation : str, optional
                The interpolation method to use
                  * 'near' => For each location, extract the time series at the
                    closest lat/lon index
                  * 'bilinear' => For each location, use the time series of the
                    surrounding +/- 1 index locations to create an estimated time
                    series at the given location using a biliear scheme
                  * 'cubic' => For each location, use the time series of the
                    surrounding +/- 2 index locations to create an estimated time
                    series at the given location using a cubic scheme

            forceDataFrame : bool, optional
                Instructs the returned value to take the form of a DataFrame
                regardless of how many locations are specified


            outsideOkay : bool, optional
                Determines if points which are outside the source's lat/lon grid
                are allowed
                * If True, points outside this space will return as None
                * If False, an error is raised

        Returns
        -------
        If a single location is given: pandas.Series
          * Indexes match to times

        If multiple locations are given: pandas.DataFrame
          * Indexes match to times
          * Columns match to the given order of locations

        """
        k = dict(
            interpolation=spatial_interpolation,
            forceDataFrame=force_data_frame,
            outsideOkay=outside_okay,
            _indicies=_indicies,
        )

        locations = gk.LocationSet(locations)
        heights = np.array(heights)
        if heights.size == 1:
            heights = np.array([heights] * locations.count)
        elif not heights.size == locations.count:
            raise RuntimeError("Heights and locations sizes don't match")
        _0_50 = heights < 50
        _50_100 = np.logical_and(heights >= 50, heights < 100)
        _100_ = heights >= 100

        new_windspeed = np.empty((len(self.timeindex), locations.count))

        if _0_50.any():
            raise RuntimeError("This hasn't been implemented yet below 50m :(")
        if _50_100.any():
            ws50 = NCSource.get(self, "windspeed_50", locations=locations[_50_100], **k)
            ws100 = NCSource.get(self, "windspeed_100", locations=locations[_50_100], **k)

            fac = (heights[_50_100] - 50) / (100 - 50)
            tmp = ws100 * fac + ws50 * (1 - fac)

            new_windspeed[:, _50_100] = tmp

        if _100_.any():
            ws100 = NCSource.get(self, "windspeed_100", locations=locations[_100_], **k)
            ws140 = NCSource.get(self, "windspeed_140", locations=locations[_100_], **k)

            fac = (heights[_100_] - 100) / (140 - 100)
            tmp = ws140 * fac + ws100 * (1 - fac)
            if tmp.shape[0] == 1:
                tmp = tmp[0, :]
            new_windspeed[:, _100_] = tmp

        return pd.DataFrame(new_windspeed, columns=locations, index=self.timeindex)
