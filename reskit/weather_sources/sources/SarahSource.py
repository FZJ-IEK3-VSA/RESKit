from .NCSource import *


class SarahSource(NCSource):

    MAX_LON_DIFFERENCE = 0.06
    MAX_LAT_DIFFERENCE = 0.06

    def __init__(s, source, bounds=None, indexPad=5, **kwargs):
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

        NCSource.__init__(s, source=source, bounds=bounds, timeName="time", latName="lat", lonName="lon",
                          indexPad=indexPad, _maxLonDiff=s.MAX_LON_DIFFERENCE, _maxLatDiff=s.MAX_LAT_DIFFERENCE,
                          tz=None, **kwargs)

    def loc2Index(s, loc, outsideOkay=False, asInt=True):
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
        # Ensure loc is a list
        locations = LocationSet(loc)

        # get closest indices
        latI = (locations.lats - s.lats[0]) / 0.05
        lonI = (locations.lons - s.lons[0]) / 0.05

        # Check for out of bounds
        s = (latI < 0) | (latI >= s._latN) | (lonI < 0) | (lonI >= s._lonN)
        if s.any():
            if not outsideOkay:
                print("The following locations are out of bounds")
                print(locations[s])
                raise ResError("Locations are outside the boundaries")

        # As int?
        if asInt:
            latI = np.round(latI).astype(int)
            lonI = np.round(lonI).astype(int)

        # Make output
        if locations.count == 1:
            if s[0] is True:
                return None
            else:
                return Index(yi=latI[0], xi=lonI[0])
        else:
            return [None if ss else Index(yi=y, xi=x) for ss, y, x in zip(s, latI, lonI)]

    def loadRadiation(s, fill=0):
        """Load the SWGDN variable into the data table with the name 'ghi'
        """
        for ds_name, name in [("SIS", "ghi"), ("DNI", "dni")]:
            s.load(ds_name, name=name)

            sel = np.logical_or(s.data[name] < 0, np.isnan(s.data[name]))
            s.data[name][sel] = fill
            # minus_1 = src.data[name][np.roll(sel, -1, axis=0)]
            # plus_1 = src.data[name][np.roll(sel,  1, axis=0)]

            # src.data[name][sel] = (minus_1+plus_1)/2

    # STANDARD LOADERS
    def sload_direct_normal_irradiance(self):
        self.load("DNI", name="direct_normal_irradiance")
        sel = np.logical_or(self.data['direct_normal_irradiance'] < 0, np.isnan(self.data['direct_normal_irradiance']))
        self.data['direct_normal_irradiance'][sel] = 0

    def sload_global_horizontal_irradiance(self):
        self.load("SIS", name="global_horizontal_irradiance")
        sel = np.logical_or(self.data['global_horizontal_irradiance'] < 0, np.isnan(self.data['global_horizontal_irradiance']))
        self.data['global_horizontal_irradiance'][sel] = 0
