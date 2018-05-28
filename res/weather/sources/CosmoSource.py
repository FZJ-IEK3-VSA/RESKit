from ..NCSource import *

## Define constants
class CosmoSource(NCSource):
    """
    Handles the sources Sev created from the COSMO-REA6 dataset (cannot handle the orginal sources because they're whack)
    """
    
    GWA50_CONTEXT_MEAN_SOURCE = None
    GWA100_CONTEXT_MEAN_SOURCE = None

    MAX_LON_DIFFERENCE = 0.6 # a LARGE ooverestimate of how much space should be inbetween a given point and the nearest index
    MAX_LAT_DIFFERENCE = 0.6 # a LARGE ooverestimate of how much space should be inbetween a given point and the nearest index

    def __init__(s, source, bounds=None, indexPad=5, convention="REA6", **kwargs):
        """Initialize a COSMO style netCDF4 file source

        * Assumes REA6 conventions

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
              
        timeBounds : tuple of length 2, optional
            Used to employ a slice of the time dimension
              * Expect two pandas Timestamp objects> The first indicates the point
                to start collecting data, and the second indicates the end

        """

        NCSource.__init__(s, source=source, bounds=bounds, timeName="time", latName="lat", lonName="lon", 
                          indexPad=indexPad, _maxLonDiff=s.MAX_LON_DIFFERENCE, _maxLatDiff=s.MAX_LAT_DIFFERENCE,
                          **kwargs)
        
        if convention == "REA6":
            s.lonSouthPole = 18
            s.latSouthPole = -39.25
            s.rlonRes = 0.0550000113746
            s.rlatRes = 0.0550001976179
            s.rlonStart = -28.40246773
            s.rlatStart = -23.40240860

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

        # Convert to rotated coordinates
        rlonCoords, rlatCoords = rotateFromLatLon(locations.lons, locations.lats, lonSouthPole=s.lonSouthPole, latSouthPole=s.latSouthPole)
        
        # Find integer locations
        lonI = (rlonCoords - s.rlonStart)/s.rlonRes - s._lonStart
        latI = (rlatCoords - s.rlatStart)/s.rlatRes - s._latStart

        # Check for out of bounds
        s = (latI<0)|(latI>=s._latN)|(lonI<0)|(lonI>=s._lonN)
        if s.any():
            print("Im in!!")
            if not outsideOkay:
                print("The following locations are out of bounds")
                print(locations[s])
                raise ResError("Locations are outside the boundaries")

        # Make int, maybe
        if asInt:
            lonI = np.round(lonI).astype(int)
            latI = np.round(latI).astype(int)

        # Make output
        if locations.count==1:
            if s[0] is True: return None
            else: return Index(yi=latI[0],xi=lonI[0])
        else:
            return [ None if ss else Index(yi=y,xi=x) for ss,y,x in zip(s,latI,lonI) ]

    def loadWindSpeed(s, height=100):
        """NEEDS UPDATING!"""
        # Check if height is on of the heights we already have
        # The 3 known heights should always be 10, 100, and 120
        if height == 10:
              s.load("wspd", name="windspeed", heightIdx=0)
        elif height == 100:
              s.load("wspd", name="windspeed", heightIdx=1)
        elif height == 120:
              s.load("wspd", name="windspeed", heightIdx=2)
        else:
            # projection is required
            if height <= 60:
                lowIndex = 0
                highIndex = 1

                lowHeight = 10
                highHeight = 100
            else:
                lowIndex = 1
                highIndex = 2

                lowHeight = 100
                highHeight = 120

            
            s.load("wspd", name="lowWspd", heightIdx=lowIndex)
            s.load("wspd", name="highWspd", heightIdx=highIndex)

            lowData = s.data["lowWspd"]
            highData = s.data["highWspd"]

            alpha = np.log(highData/lowData)/np.log(highHeight/lowHeight)

            s.data["windspeed"] = lowData*np.power(height/lowHeight, alpha)

            del s.data["lowWspd"]
            del s.data["highWspd"]

    #def loadRadiation(s, ghiName="rsds"):
    #    # read raw data
    #    s.load(ghiName, name="ghi")

    #def loadTemperature(s, which='air', processor=lambda x: x-273.15):
    #    """Temperature variable loader"""
    #    if which.lower() == 'air': varName = "tas"
    #    elif which.lower() == 'dew': varName = "dpas"
    #    else: raise ResMerraError("sub group '%s' not understood"%which)
    #
    #    # load
    #    s.load(varName, name=which+"_temp", processor=processor)

    #def loadPressure(s): s.load("ps", name='pressure')