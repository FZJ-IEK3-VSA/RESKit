import numpy as np
import netCDF4 as nc
import geokit as gk
import ogr
from os.path import join, isfile, dirname, basename
from glob import glob
import pandas as pd
from collections import namedtuple, OrderedDict

## Define constants
class MerraError(Exception): pass # this just creates an error that we can use

MERRA_AVERAGE_50_PATH = join(dirname(__file__),"data","merra_average_windspeed_50m.tif")
MERRA_AVERAGE_10_PATH = join(dirname(__file__),"data","merra_average_windspeed_10m.tif")
_DEFAULT_MERRA_DIR = None
_DEFAULT_GWA_DIR = None

## Make some default setters
def setDefaultMerraDir(path): globals()["_DEFAULT_MERRA_DIR"] = path
def setDefaultGwaDir(path): globals()["_DEFAULT_GWA_DIR"] = path

## Create a MERRA data container
class Datasource(object):
	DataElement = namedtuple("DataElement","data index")

	def __init__(s, start, end=None, topDir=None, lat=None, lon=None, bounds=None):

		# check for default paths
		s.topDir = topDir if topDir else _DEFAULT_MERRA_DIR
		if s.topDir is None: 
			raise MerraError("topDir is None, try setting the default MERRA path using 'setDefaultMerraDir'")

		# Arange start and end date
		if isinstance(start, int): start = pd.Timestamp(year=start, month=1, day=1, hour=0)
		elif isinstance(start, dict): start = pd.Timestamp(**start)
		else: start = pd.Timestamp(start)

		if end is None: end = pd.Timestamp(year=start.year, month=12, day=31, hour=0)
		elif isinstance(end, pd.Timedelta): end = start+end
		elif isinstance(end, int): end = start+pd.Timedelta(end, 'D')
		elif isinstance(end, dict): end = pd.Timestamp(**end)
		else: end = pd.Timestamp(end)

		s.days = pd.date_range(start, end, freq='D')
		s._days = ["%4d%02d%02d"%(day.year,day.month,day.day) for day in s.days]

		# save boundaries
		if not bounds is None: # set boudnaries from the given boudnaries
			try: # probably bounds is a tuple
				s.lonMin, s.latMin, s.lonMax, s.latMax = bounds
			except: # But maybe it is a geokit Extent object
				s.lonMin, s.latMin, s.lonMax, s.latMax = bounds.castTo(gk.srs.SRSCOMMON.latlon).xyXY

		elif not( lat is None or lon is None): # set the boundaries immidiately around the given coordinate
			s.latMin = lat - 0.5
			s.lonMin = lon - 0.625
			s.latMax = lat + 0.5
			s.lonMax = lon + 0.625

		else: # don't set any boundaries
			s.latMin = None
			s.lonMin = None
			s.latMax = None
			s.lonMax = None

		s.lats=None
		s.lons=None

		# make empty data container
		s.data = OrderedDict()

	def _read3dData(s, path, parameter):
		# open dataset
		ds = nc.Dataset(path)

		# read the time index
		index = nc.num2date(ds.variables["time"][:], ds.variables["time"].units)

		# get the slices
		lats = ds.variables["lat"][:]
		lons = ds.variables["lon"][:]

		latSelect = np.ones(lats.shape, dtype=bool)
		if not s.latMin is None: np.logical_and(latSelect, lats>=s.latMin, latSelect)
		if not s.latMax is None: np.logical_and(latSelect, lats<=s.latMax, latSelect)

		lonSelect = np.ones(lons.shape, dtype=bool)
		if not s.lonMin is None: np.logical_and(lonSelect, lons>=s.lonMin, lonSelect)
		if not s.lonMax is None: np.logical_and(lonSelect, lons<=s.lonMax, lonSelect)

		# Be sure lats and lons lineup with the other datasets
		if s.lats is None: s.lats = lats[latSelect]
		if s.lons is None: s.lons = lons[lonSelect]

		if not (s.lats == lats[latSelect]).all(): raise MerraError("Coordinate mismatch")
		if not (s.lons == lons[lonSelect]).all(): raise MerraError("Coordinate mismatch")

		# fetch data
		return index, ds.variables[parameter][:,latSelect,lonSelect]

	def loadVar(s, variable, subSource, name=None, subDir=None, level=None, **kwargs):
		"""generic variable loader"""
		# search for suitable files
		searchDir = s.topDir if subDir is None else join(s.topDir, subDir)
		files = glob(join(searchDir,"*%s_Nx.*.nc4"%subSource))
		if len(files)==0: raise MerraError("No files found")

		# read data for each day
		tmp = []
		for dayString in s._days:
			try:
				path = next(filter(lambda x: dayString in x, files)) # get the first path which matches our day (there should only be one, anyway)
			except:
				raise MerraError("Could not find path for day:", daystring)

			tmp.append(s._read3dData(path, "V%dM"%level ))

		# Check indicies
		index = np.concatenate([ x[0] for x in tmp])
		
		# combine into a single time series matrix
		data = np.vstack( [ x[1] for x in tmp] )
		
		# make a time series
		values = Datasource.DataElement(data,index)

		# done!
		if name is None:
			if level is None: name = variable
			else: name = "%s_%d"%(variable,height)

		s.data[name] = values

	def loadWindSpeed(s, height=50, subDir=None):
		# search for suitable files
		searchDir = s.topDir if subDir is None else join(s.topDir, subDir)
		files = glob(join(searchDir,"*slv_Nx.*.nc4"))
		if len(files)==0: raise MerraError("No files found")

		# read data for each day
		uTmp = []
		vTmp = []
		for dayString in s._days:
			try:
				path = next(filter(lambda x: dayString in x, files)) # get the first path which matches our day (there should only be one, anyway)
			except:
				raise MerraError("Could not find path for day:", daystring)

			uTmp.append(s._read3dData(path, "U%dM"%height ))
			vTmp.append(s._read3dData(path, "V%dM"%height ))

		# Check indicies
		uIndex = np.concatenate([ x[0] for x in uTmp])
		vIndex = np.concatenate([ x[0] for x in vTmp])

		if not uIndex.shape == vIndex.shape and not (uIndex == vIndex).all():
			raise MerraError("Data indexes do not match")

		index = uIndex
		vIndex = None

		# combine into a single time series matrix
		uData = np.vstack( [ x[1] for x in uTmp] )
		vData = np.vstack( [ x[1] for x in vTmp] )

		speed = Datasource.DataElement(np.sqrt(uData*uData+vData*vData),index) # total speed
		direction = Datasource.DataElement(np.arctan2(vData,uData)*(180/np.pi), index) # total direction

		# done!
		s.data["windspeed_%d"%height] = speed
		s.data["winddir_%d"%height] = direction

	def nearestIndex(s, lat, lon):
		latIndex = np.argmin( abs(lat-s.lats) )
		lonIndex = np.argmin( abs(lon-s.lons) )

		return latIndex, lonIndex

	def nearestLoc(s, lat, lon):
		latIndex, lonIndex = s.nearestIndex(lat,lon)
		return s.lats[latIndex], s.lons[lonIndex]

	def nearestVar(s, var, lat, lon):
		latIndex, lonIndex = s.nearestIndex(lat,lon)
		v = s.data[var]
		return pd.Series(v.data[:,latIndex,lonIndex], index=v.index, name=var)

	def nearestSet(s, lat, lon):
		latIndex, lonIndex = s.nearestIndex(lat,lon)

		# load a temporary data dictionary time pandas Series objects
		tmpData = {}
		for k,v in s.data.items():
			tmpData[k] = pd.Series(v.data[:,latIndex,lonIndex], index=v.index)

		# make into a DataFrame and return
		return pd.DataFrame(tmpData)

## Function for creating wind data
def hubWindSpeed(source, loc=None, height=100, GWA_DIR=None, MERRA_DIR=None, subDir=None, **kwargs):
	# check for default paths
	GWA_DIR = GWA_DIR if GWA_DIR else _DEFAULT_GWA_DIR
	if GWA_DIR is None: raise MerraError("GWA_DIR has not been set")
	
	###################
	## Ensure location is okay
	if loc is None:
		try:
			lat = kwargs["lat"]
			lon = kwargs["lon"]
		except KeyError:
			raise MerraError("'lat' and 'lon' must be explicitly defined if not giving a 'loc' input")
	elif isinstance(loc, tuple): # Handle tuple: (lat, lon)
		lat,lon = loc
	elif isinstance(loc, ogr.Geometry): # Handle point geometry
		loc.TransformTo(gl.EPSG4326) # make sure the point is in lat-lon
		lat = loc.GetY()
		lon = loc.GetX()
	else:
		raise MerraError("'loc' input could not be understood")

	# Ensure we have a MERRA source
	if isinstance(source, int): # user just gave a year, so try to create the source
		# try to create a new source
		source = Datasource(start=source, topDir=MERRA_DIR, lat=lat, lon=lon)

	# Ensure we have the right data in our MERRA source
	if not "windspeed_50" in source.data.keys():
		# Load the wind speeds at 50 meters
		source.loadWindSpeed(height=50, subDir=subDir)
	
	# Extract the data we need
	windSpeedMerra = source.nearestVar("windspeed_50", lat, lon)
	windSpeedMerra.sort_index(inplace=True) # this shouldnt make any difference but.....just in case...

	# Get the total MERRA average at 50m
	mlat, mlon = source.nearestLoc(lat,lon)
	merraAverage50 = gk.raster.extractValues(MERRA_AVERAGE_50_PATH, (mlon, mlat), noDataOkay=False).data

	# Do normalization
	windSpeedNormalized = windSpeedMerra / merraAverage50

	###################
	## Scale the normalized time series by the GWA average (at location)

	# Get the GWA averages
	GWA_files = [join(GWA_DIR, "WS_050m_global_wgs84_mean_trimmed.tif"),
		         join(GWA_DIR, "WS_100m_global_wgs84_mean_trimmed.tif"),
		         join(GWA_DIR, "WS_200m_global_wgs84_mean_trimmed.tif")]

	for f in GWA_files: 
		if not isfile(f): 
			raise MerraError("Could not find file: "+f)

	try:
		gwaAverage50 =  gk.raster.extractValues(GWA_files[0], (lon,lat), noDataOkay=False).data
		gwaAverage100 = gk.raster.extractValues(GWA_files[1], (lon,lat), noDataOkay=False).data
		gwaAverage200 = gk.raster.extractValues(GWA_files[2], (lon,lat), noDataOkay=False).data
	except gk.util.GeoKitRasterError as e:
		if str(e) == "No data values found in extractValues with 'noDataOkay' set to False":
			raise MerraError("The given point does not appear to have valid data in the Global Wind Atlas dataset")
		else:
			raise e

	# Interpolate gwa average to desired height
	if height == 50:
		gwaAverage = gwaAverage50
	elif height < 100:
		alpha = np.log(gwaAverage50/gwaAverage100)/np.log(50.0/100)
		gwaAverage = gwaAverage50 * np.power(height/50, alpha)
	elif height == 100:
		gwaAverage = gwaAverage100
	elif height > 100 and height <200:
		alpha = np.log(gwaAverage100/gwaAverage200)/np.log(100/200)
		gwaAverage = gwaAverage100 * np.power(height/100, alpha)
	elif height == 200:
		gwaAverage = gwaAverage200
	else:
		raise MerraError("Wind speed cannot be extrapolated above 200m")
	
	# Rescale normalized wind speeds
	windSpeedScaled = gwaAverage * windSpeedNormalized

	# ALL DONE!
	return windSpeedScaled
