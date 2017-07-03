import numpy as np
import netCDF4 as nc
import geokit as gk
import ogr
from os.path import join, isfile, dirname, basename
from glob import glob
import pandas as pd
from collections import namedtuple, OrderedDict
from scipy.interpolate import RectBivariateSpline

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

	def __init__(s, timeframe, topDir=None, lat=None, lon=None, bounds=None):

		# check for default paths
		s.topDir = topDir if topDir else _DEFAULT_MERRA_DIR
		if s.topDir is None: 
			raise MerraError("topDir is None, try setting the default MERRA path using 'setDefaultMerraDir'")

		# Arange start and end date
		if isinstance(timeframe, int): # assume a year was given
			start = pd.Timestamp(year=timeframe, month=1, day=1, hour=0)
			end = pd.Timestamp(year=timeframe, month=12, day=31, hour=0)
		elif isinstance(timeframe, tuple): # timeframe must be a tuple of some sort
			startTmp, endTmp = timeframe

			if isinstance(startTmp, int): start = pd.Timestamp(year=startTmp, month=1, day=1, hour=0)
			else: start = pd.Timestamp(startTmp)  

			if isinstance(endTmp, int): end = pd.Timestamp(year=endTmp, month=1, day=1, hour=0)
			else: end = pd.Timestamp(endTmp)
		else:
			raise MerraError("Could not interperate the given timeframe")

		# Get the list of days which will be considered	
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

	@property
	def bounds(s):
		return (s.lons.min()-0.625/2, s.lats.min()-0.5/2, s.lats.max()+0.625/2, s.lons.max()+0.5/2)

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

	def loadVariable(s, variable, subSource, name=None, subDir=None, level=None):
		"""generic variable loader"""
		# search for suitable files
		searchDir = s.topDir if subDir is None else join(s.topDir, subDir)
		files = glob(join(searchDir,"*%s_Nx.*.nc*"%subSource))
		if len(files)==0: raise MerraError("No files found")

		# read data for each day
		tmp = []
		for dayString in s._days:
			try:
				# get the first path which matches our day (there should only be one, anyway)
				path = next(filter(lambda x: dayString in basename(x), files)) 
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

	def loadRadiation(s, subDir=None):
		"""GHI/DNI variable loader"""

		# search for suitable files
		searchDir = s.topDir if subDir is None else join(s.topDir, subDir)
		files = glob(join(searchDir,"*rad_Nx.*.nc*"))
		if len(files)==0: raise MerraError("No files found")

		# read data for each day
		ghiTmp = []
		dniTmp = []
		for dayString in s._days:
			try:
				# get the first path which matches our day (there should only be one, anyway)
				path = next(filter(lambda x: dayString in basename(x), files)) 
			except:
				raise MerraError("Could not find path for day:", daystring)

			ghiTmp.append(s._read3dData(path, "SWGDN" ))
			dniTmp.append(s._read3dData(path, "SWGNT" ))

		# Check indicies
		ghiIndex = np.concatenate([ x[0] for x in ghiTmp])
		dniIndex = np.concatenate([ x[0] for x in dniTmp])

		# combine into a single time series matrix
		ghiData = np.vstack( [ x[1] for x in ghiTmp] )
		dniData = np.vstack( [ x[1] for x in dniTmp] )
		
		# make a time series
		ghiValues = Datasource.DataElement(ghiData,ghiIndex)
		dniValues = Datasource.DataElement(dniData,dniIndex)

		# done!
		s.data["ghi"] = ghiValues
		s.data["dni"] = dniValues

	def loadWindSpeed(s, height=50, subDir=None, context="tavg"):
		# search for suitable files
		searchDir = s.topDir if subDir is None else join(s.topDir, subDir)
		
		if context=="tavg":
			files = glob(join(searchDir,"*slv_Nx.*.nc*"))
		elif context=="inst":
			files = glob(join(searchDir,"*asm_Nx.*.nc*"))
		else:
			raise MerraError("context not understood")

		if len(files)==0: raise MerraError("No files found")

		# read data for each day
		uTmp = []
		vTmp = []
		for dayString in s._days:
			try:
				# get the first path which matches our day (there should only be one, anyway)
				path = next(filter(lambda x: dayString in basename(x), files)) 
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

	def interpolateVar( s, var, lat, lon, mode="cubic"):
		# get nearest index
		latIndex, lonIndex = s.nearestIndex(lat, lon)

		# Determine mode
		if mode=='near':
			return s.nearestVar(var, lat, lon)
		elif mode=="cubic": 
			width=3
			kwargs = dict()
		elif mode=="linear": 
			width=1
			kwargs = dict(kx=1, ky=1)
		else: 
			raise MerraError("'mode' input not recognized")

		# Make sure the contained data is sufficient
		if latIndex<width or latIndex>s.lats.shape[0]-width or lonIndex<width or lonIndex>s.lons.shape[0]-width:
			raise MerraError("Insufficient spatial extent")

		# Setup interpolator
		lats = s.lats[latIndex-width:latIndex+width+1]
		lons = s.lons[lonIndex-width:lonIndex+width+1]

		def interpolate(timeIndex):
			data = s.data[var].data[timeIndex, latIndex-width:latIndex+width+1, lonIndex-width:lonIndex+width+1]
			rbs = RectBivariateSpline(lats,lons,data,**kwargs)
			return rbs(lat,lon)[0][0]

		# do interpolations
		values = [interpolate(i) for i in range(s.data[var].index.shape[0])]

		return pd.Series(values, index=s.data[var].index, name=var)

	def nearestSet(s, lat, lon):
		latIndex, lonIndex = s.nearestIndex(lat,lon)

		# load a temporary data dictionary time pandas Series objects
		tmpData = {}
		for k,v in s.data.items():
			tmpData[k] = pd.Series(v.data[:,latIndex,lonIndex], index=v.index)

		# make into a DataFrame and return
		return pd.DataFrame(tmpData)

	defaultInterpolatorSchemes = dict( winddir_2=None, winddir_10=None, winddir_50=None,
									   windspeed_2='cubic', windspeed_10='cubic', windspeed_50='cubic',
									   ghi='cubic', dni='cubic' )

	def interpolateSet(s, lat, lon, **modes):
		# make sure all variables have a mode
		for var in s.data.keys():
			if not var in modes: modes[var]=s.defaultInterpolatorSchemes.get(var, 'cubic')

		# load a temporary data dictionary time pandas Series objects
		tmpData = {}
		for var in s.data.keys():
			if modes[var] is None: continue
			tmpData[var] = s.interpolateVar(var, lat, lon, modes[var])

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
		source = Datasource(source, topDir=MERRA_DIR, lat=lat, lon=lon)

	# Ensure we have the right data in our MERRA source
	if not "windspeed_50" in source.data.keys():
		# Load the wind speeds at 50 meters
		source.loadWindSpeed(height=50, subDir=subDir)
	
	# Extract the data we need
	windSpeedMerra = source.nearestVar("windspeed_50", lat, lon)
	windSpeedMerra.sort_index(inplace=True) # this shouldnt make any difference but.....just in case...

	# Get the total MERRA average at 50m
	mlat, mlon = source.nearestLoc(lat,lon)
	merraAverage50 = gk.raster.extractValues(MERRA_AVERAGE_50_PATH, (mlon, mlat), noDataOkay=True).data
	if np.isnan(merraAverage50):
		print("WARNING: could not find average merra value at the given location, defaulting to the average of the current time series")
		merraAverage50 = windSpeedMerra.mean()

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
			# Try to get the surrounding points and average
			gwaAverage50 =  gk.raster.extractValues(GWA_files[0], (lon,lat), noDataOkay=True, winRange=3).data.mean()
			if np.isnan(gwaAverage50):
				# the point is likely an offshore location, therefor using the Merra Average is probably okay
				print("Could not extract from GlobalWindAtlas, assuming its far offshore, using the MERRA average with a roughness of 0.0005")
				gwaAverage50 = merraAverage50
				gwaAverage100 = merraAverage50 * np.log(100/0.0005)/np.log(50/0.0005)
				gwaAverage200 = merraAverage50 * np.log(200/0.0005)/np.log(50/0.0005)
			else:
				gwaAverage100 = gk.raster.extractValues(GWA_files[1], (lon,lat), noDataOkay=True, winRange=3).data.mean()
				gwaAverage200 = gk.raster.extractValues(GWA_files[2], (lon,lat), noDataOkay=True, winRange=3).data.mean()
			#raise MerraError("The given point does not appear to have valid data in the Global Wind Atlas dataset")
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
