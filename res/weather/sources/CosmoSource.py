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

    def __init__(s, source, bounds=None, indexPad=0, **kwargs):
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
                          tz="GMT", **kwargs)
           

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
        # Set REA6 Conventions
        lonSouthPole = 18
        latSouthPole = -39.25
        rlonRes = 0.0550000113746
        rlatRes = 0.0550001976179
        rlonStart = -28.40246773
        rlatStart = -23.40240860

        if s is None: 
            _lonStart=0
            _latStart=0
            _latN = 824
            _lonN = 848
        else:
            _lonStart = s._lonStart
            _latStart = s._latStart
            _latN = s._latN
            _lonN = s._lonN

        # Ensure loc is a list
        locations = LocationSet(loc)

        # Convert to rotated coordinates
        rlonCoords, rlatCoords = rotateFromLatLon(locations.lons, locations.lats, lonSouthPole=lonSouthPole, latSouthPole=latSouthPole)
        
        # Find integer locations
        lonI = (rlonCoords - rlonStart)/rlonRes - _lonStart
        latI = (rlatCoords - rlatStart)/rlatRes - _latStart

        # Check for out of bounds
        s = (latI<0)|(latI>=_latN)|(lonI<0)|(lonI>=_lonN)
        if s.any():
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

    def loadRadiation(s):
        """frankCorrection: "Bias correction of a novel European reanalysis data set for solar energy applications" """
        s.load("SWDIFDS_RAD", "dhi")
        s.load("SWDIRS_RAD", "dni_flat")
        s.data["ghi"] = s.data["dhi"]+s.data["dni_flat"]

        del s.data["dni_flat"]

    def loadWindSpeedLevels(s):
        s.load("windspeed_10", name="windspeed_10")
        s.load("windspeed_50", name="windspeed_50")
        s.load("windspeed_100", name="windspeed_100")
        s.load("windspeed_140", name="windspeed_140")

    def loadWindSpeedAtHeight(s, height=100):
        """NEEDS UPDATING!"""
        # Check if height is on of the heights we already have
        # The 3 known heights should always be 50, 100, and 140
        if height == 10:
              s.load("windspeed_10", name="windspeed")
        elif height == 50:
              s.load("windspeed_50", name="windspeed")
        elif height == 100:
              s.load("windspeed_100", name="windspeed")
        elif height == 140:
              s.load("windspeed_140", name="windspeed")
        else:
            # projection is required
            if height <= 50:
                s.load("windspeed_10")
                s.load("windspeed_50")
                s.load("windspeed_100")

                # DO CUBIC INTERP
                raise RuntimeError("This hasn't been implemented yet :(")

                # Remove unneeded data
                del s.data["windspeed_10"]
                del s.data["windspeed_50"]
                del s.data["windspeed_100"]
            
            elif height < 100:
                s.load("windspeed_50")
                s.load("windspeed_100")

                fac = (height-50)/(100-50)

                newWspd = s.data["windspeed_100"]*fac+s.data["windspeed_50"]*(1-fac)
                s.data["windspeed"] = newWspd
                
                del s.data["windspeed_50"]
                del s.data["windspeed_100"]    

            else:
                s.load("windspeed_100")
                s.load("windspeed_140")

                fac = (height-100)/(140-100)

                newWspd = s.data["windspeed_140"]*fac+s.data["windspeed_100"]*(1-fac)
                s.data["windspeed"] = newWspd
                
                del s.data["windspeed_100"]
                del s.data["windspeed_140"]                

    def loadTemperature(s, processor=lambda x: x-273.15):
        """load the typical pressure variable"""
        s.load("2t", name="air_temp", processor=processor)

    def loadPressure(s):
        """load the typical pressure variable"""
        s.load("sp", name="pressure")

    def loadSet_PV(s):
        s.loadRadiation()
        s.loadWindSpeedAtHeight(10)
        s.loadPressure()
        s.loadTemperature()

    def getWindSpeedAtHeights(s, locations, heights, spatialInterpolation='near', forceDataFrame=False, outsideOkay=False, _indicies=None):
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
        k = dict(interpolation=spatialInterpolation, forceDataFrame=forceDataFrame, 
                 outsideOkay=outsideOkay, _indicies=_indicies)

        locations = gk.LocationSet(locations)
        heights = np.array(heights)
        _0_50 = heights<50
        _50_100 = np.logical_and(heights>=50, heights<100)
        _100_ = heights>=100

        newWindspeed = np.empty((len(s.timeindex), locations.count))

        if _0_50.any(): raise RuntimeError("This hasn't been implemented yet below 50m :(")
        if _50_100.any(): 
            ws50 = NCSource.get(s, "windspeed_50", locations=locations[_50_100], **k)
            ws100 = NCSource.get(s, "windspeed_100", locations=locations[_50_100], **k)

            fac = (heights-50)/(100-50)
            newWindspeed[:,_50_100] = ws100*fac + ws50*(1-fac)

        if _100_.any():
            ws100 = NCSource.get(s, "windspeed_100", locations=locations[_100_], **k)
            ws140 = NCSource.get(s, "windspeed_140", locations=locations[_100_], **k)

            fac = (heights-100)/(140-100)
            newWindspeed[:,_100_] = ws140*fac + ws100*(1-fac)

        return pd.DataFrame(newWindspeed, columns=ws100.columns, index=ws100.index)
