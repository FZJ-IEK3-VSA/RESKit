import numpy as np
import netCDF4 as nc
import glaes as gl
import ogr
from os.path import join, isfile, dirname
from glob import glob
import pandas as pd

## Define constants
class RSError(Exception): pass # this just creates an error that we can use
MERRA_AVERAGE_50_PATH = join(dirname(__file__),"data","merra_average_windspeed_50m.tif")
MERRA_AVERAGE_10_PATH = join(dirname(__file__),"data","merra_average_windspeed_10m.tif")

## Function for creating wind data
def weatherGenWind(loc, year, height=100, MERRA_PATH="D:\Data\weather\merra\slv", GWA_PATH="D:\Data\weather\global_wind_atlas"):
	
	###################
	## Generate file lists and check if they exist
	merra_files = glob(join(MERRA_PATH,"*slv_Nx.%d*.nc4"%year))

	if not (len(merra_files) == 365 or len(merra_files) == 366):
		raise RSError("Could not find a complete year of MERRA data at: "+MERRA_PATH)

	GWA_files = [join(GWA_PATH, "WS_050m_global_wgs84_mean_trimmed.tif"),
	             join(GWA_PATH, "WS_100m_global_wgs84_mean_trimmed.tif"),
	             join(GWA_PATH, "WS_200m_global_wgs84_mean_trimmed.tif")]

	for f in GWA_files:
		if not isfile(f):
			raise RSError("Could not find file: "+f)

	###################
	## Ensure location is okay
	if isinstance(loc, tuple): # Handle tuple: (lat, lon)
		lat,lon = loc
	elif isinstance(loc, ogr.Geometry): # Handle point geometry
		loc.TransformTo(gl.EPSG4326) # make sure the point is in lat-lon
		lat = loc.GetY()
		lon = loc.getX()

	###################
	## Find the MERRA index associated with location

	# open one merra file
	tmpDS = nc.Dataset(merra_files[0])

	# read the latitudes and longitudes
	lats = tmpDS.variables["lat"][:]
	lons = tmpDS.variables["lon"][:]

	# check boundaries
	if ( lat < lats.min() or lat > lats.max() or lon < lons.min() or lon > lons.max()):
		raise RSError("Given location is outside of the MERRA data boundary")

	# Determine the best index to use
	latIndex = np.argmin(abs(lats-lat))
	lonIndex = np.argmin(abs(lons-lon))

	# cleanup
	tmpDS = None
	lats = None
	lons = None

	###################
	## Normalize the MERRA year by the total MERRA average (at index)

	# Create the full MERRA time series
	speeds = []
	dates = []
	for f in merra_files:
		# open file
		tmpDS = nc.Dataset(f)

		# read Northward and Eastward wind speeds
		u = tmpDS.variables["U50M"][:,latIndex,lonIndex]
		v = tmpDS.variables["V50M"][:,latIndex,lonIndex]
		
		# Calculate the total wind speeds
		total = np.sqrt(u*u+v*v)

		# Get the start date
		timeSteps = nc.num2date(tmpDS.variables["time"][:], tmpDS.variables["time"].units)

		# append to series
		speeds.append(total)
		dates.append(timeSteps)
	
	speeds = np.concatenate(speeds)
	dates  = np.concatenate(dates )
	
	windSpeedMerra = pd.Series(speeds, index=dates)
	windSpeedMerra.sort_index(inplace=True)

	# Get the total MERRA average at 50m
	merraAverage50 = gl.valueAtPoints(MERRA_AVERAGE_50_PATH, (lon,lat))[0][0][0][0]

	# Do normalization
	windSpeedNormalized = windSpeedMerra / merraAverage50


	###################
	## Scale the normalized time series by the GWA average (at location)

	# Get the GWA averages
	gwaAverage50 =  gl.valueAtPoints(GWA_files[0], (lon,lat) )[0][0][0][0]
	gwaAverage100 = gl.valueAtPoints(GWA_files[1], (lon,lat) )[0][0][0][0]
	gwaAverage200 = gl.valueAtPoints(GWA_files[2], (lon,lat) )[0][0][0][0]

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
		raise RSError("Wind speed cannot be extrapolated above 200m")
	
	# Rescale normalized wind speeds
	windSpeedScaled = gwaAverage * windSpeedNormalized

	# ALL DONE!
	return windSpeedScaled
