from .util import *

## Define constants
class ResMerraError(ResWeatherError): pass # this just creates an error that we can use

class MerraSource( GenericSource ):

    _LONS_MASTER = np.arange(-180, 180, 0.625)
    _LATS_MASTER = np.arange(-90, 90, 0.5)
    _WINDSPEED_NORMALIZER_SOURCE = join(dirname(__file__),"..","..","data","merra_average_windspeed_50m.tif")
    _WINDSPEED_NORMALIZER_VAR = "wspd_50"

    def __init__(s, timeframe, topDir, lat=None, lon=None, bounds=None):
        # set some useful attributes
        s.topDir = topDir
        s.source = "MERRA"
        
        # save boundaries
        if not bounds is None: # set boundaries from the given boudnaries
            try: # probably bounds is a tuple
                lonMin, latMin, lonMax, latMax = bounds
            except: # But maybe it is a geokit Extent object
                lonMin, latMin, lonMax, latMax = bounds.castTo(gk.srs.EPSG4326).xyXY

        elif not( lat is None or lon is None): # set the boundaries immidiately around the given coordinate
            latMin = lat - 0.5
            lonMin = lon - 0.625
            latMax = lat + 0.5
            lonMax = lon + 0.625

        else: # don't set any boundaries
            raise ResMerraError("Either a boundary or a specific lat/lon must be provided")

        s.lats = s._LATS_MASTER[ (s._LATS_MASTER>=latMin) & (s._LATS_MASTER<=latMax)]
        s.lons = s._LONS_MASTER[ (s._LONS_MASTER>=lonMin) & (s._LONS_MASTER<=lonMax)]

        s._latMin = s.lats.min()
        s._latMax = s.lats.max()
        s._lonMin = s.lons.min()
        s._lonMax = s.lons.max()

        # Initialize the generic datasource
        GenericSource.__init__(s, timeframe=timeframe, xCoords=s.lons, yCoords=s.lats, coordSRS='latlon')

        # Extract the days
        s.days = pd.date_range(s.timeStart, s.timeEnd, freq='D') # s.timeStart and s.timeEnd are defined in the GenericSource
        s._days = ["%4d%02d%02d"%(day.year,day.month,day.day) for day in s.days]

    def _generic_loader(s, varname, subDir, subset):
        # search for suitable files
        searchDir = s.topDir if subDir is None else join(s.topDir, subDir)
        
        files = glob(join(searchDir,"*%s_Nx.*.nc*"%subset))

        if len(files)==0: raise ResMerraError("No files found")

        # read each day
        rawdata = []
        rawindex = []
        meta = None
        for dayString in s._days:
            try:
                # get the first path which matches our day (there should only be one, anyway)
                path = next(filter(lambda x: dayString in basename(x), files)) 
            except:
                raise ResMerraError("Could not find path for day:", daystring)

            # open dataset
            ds = nc.Dataset(path)
            if meta is None:
                meta = dict(unit=ds.variables[varname].units, longName=ds.variables[varname].long_name)

            # read the time index
            index = nc.num2date(ds.variables["time"][:], ds.variables["time"].units)

            # get the slices
            lats = ds.variables["lat"][:]
            lons = ds.variables["lon"][:]

            latSelect = np.ones(lats.shape, dtype=bool)
            np.logical_and(latSelect, lats>=s._latMin, latSelect)
            np.logical_and(latSelect, lats<=s._latMax, latSelect)

            lonSelect = np.ones(lons.shape, dtype=bool)
            np.logical_and(lonSelect, lons>=s._lonMin, lonSelect)
            np.logical_and(lonSelect, lons<=s._lonMax, lonSelect)

            # Be sure lats and lons lineup with the other datasets
            if not (np.abs(s.lats - lats[latSelect]) < 1e-8).all(): raise ResMerraError("Lat mismatch in %s"%path)
            if not (np.abs(s.lons - lons[lonSelect]) < 1e-8).all(): raise ResMerraError("Lon mismatch in %s"%path)

            # fetch and save data
            rawdata.append( ds.variables[varname][:,latSelect,lonSelect])
            rawindex.append( index )

        # set index
        index = np.concatenate(rawindex)

        # combine data
        data = np.vstack( rawdata )

        # done!
        return data, index, meta

    def loadWindSpeed(s, height=50, subDir=None, subset="slv"):
        # read data for each day
        uData, uIndex, uMeta = s._generic_loader("U%dM"%height, subDir, subset)
        vData, vIndex, vMeta= s._generic_loader("V%dM"%height, subDir, subset)

        # Check indicies
        if not uIndex.shape == vIndex.shape and not (uIndex == vIndex).all():
            raise ResMerraError("Data indexes do not match")

        index = uIndex

        # combine into a single time series matrix
        speed = np.sqrt(uData*uData+vData*vData) # total speed
        direction = np.arctan2(vData,uData)*(180/np.pi)# total direction

        # set meta data
        metaSpeed = OrderedDict()
        metaSpeed["units"] = "m s-1"
        metaSpeed["height"] = height
        metaSpeed["context"] = "tavg" if subset == 'slv' else "inst"
        metaSpeed["longname"] = "wind speed at %d meters"%height

        metaDir = OrderedDict()
        metaDir["units"] = "degrees"
        metaDir["height"] = height
        metaDir["context"] = "tavg" if subset == 'slv' else "inst"
        metaDir["longname"] = "wind direction at %d meters"%height
        
        # done!
        s.setElement(name="wspd_%d"%height, data=speed, times=index, meta=metaSpeed)
        s.setElement(name="wdir_%d"%height, data=direction, times=index, meta=metaDir)

    def loadRadiation(s, subDir=None):
        """GHI/DNI variable loader"""
        ghiData, ghiIndex, ghiMeta = s._generic_loader("SWGNT", subDir, "rad")
        dniData, dniIndex, dniMeta = s._generic_loader("SWGDN", subDir, "rad")

        # set meta data
        metaGHI = OrderedDict()
        metaGHI["units"] = ghiMeta["unit"]
        metaGHI["height"] = 0
        metaGHI["context"] = "tavg"
        metaGHI["longname"] = ghiMeta["longName"]

        metaDNI = OrderedDict()
        metaDNI["units"] = dniMeta["unit"]
        metaDNI["height"] = 0
        metaDNI["context"] = "tavg"
        metaDNI["longname"] = dniMeta["longName"]

        # done!
        s.setElement(name="ghi", data=ghiData, times=ghiIndex, meta=metaGHI)
        s.setElement(name="dni", data=dniData, times=dniIndex, meta=metaDNI)

    def loadTemperature(s, which='air', height=2, subDir=None, subset="slv"):
        """Temperature variable loader"""
        if which.lower() == 'air': varName = "T%dM"%height
        elif which.lower() == 'dew': varName = "T%dMDEW"%height
        elif which.lower() == 'wet': varName = "T%dMWET"%height
        else: raise ResMerraError("sub group '%s' not understood"%which)

        data, index, _meta = s._generic_loader(varName, subDir, subset)

        # set meta data
        meta = OrderedDict()
        meta["units"] = _meta["unit"]
        meta["height"] = 0
        meta["context"] = "tavg" if subset == 'slv' else "inst"
        meta["longname"] = _meta["longName"]

        # done!
        if which.lower() == 'air': outName = "temp_%d"%height
        elif which.lower() == 'dew': outName = "tdew_%d"%height
        elif which.lower() == 'wet': outName = "twet_%d"%height

        s.setElement(name=outName, data=data, times=index, meta=meta)

    def loadPressure(s, subDir=None, subset="slv"):
        """Pressure variable loader"""
        data, index, _meta = s._generic_loader("PS", subDir, subset)

        # set meta data
        meta = OrderedDict()
        meta["units"] = _meta["unit"]
        meta["height"] = 0
        meta["context"] = "tavg" if subset == 'slv' else "inst"
        meta["longname"] = _meta["longName"]

        # done!
        s.setElement(name="pres", data=data, times=index, meta=meta)


