import netCDF4 as nc
import numpy as np
from os.path import join
import geokit as gk

from res.weather import MerraSource, computeContextMean
from res.util import ResError, LatLonLocation

## Make testing globals
raw = nc.Dataset(join("data","merra-like.nc4"))
rawLats = raw["lat"][:]
rawLons = raw["lon"][:]
rawTimes = nc.num2date(raw["time"][:], raw["time"].units)

aachenExt = gk.Extent.fromVector(join("data","aachenShapeFile.shp")).pad(0.5).fit(0.01)
aachenLats = np.array([50.0, 50.5, 51.0])
aachenLons = np.array([5.625, 6.250, 6.875])
aacehnLatSel = np.s_[2:5]
aacehnLonSel = np.s_[1:4]

loc = LatLonLocation(lat=50.1, lon=6.0)
locInAachen = LatLonLocation(lat=50.763, lon=6.202)
locOutsideAachen = LatLonLocation(lat=51.6, lon=5.6)
locs = [loc, locOutsideAachen,LatLonLocation(lat=50.6, lon=6.6)]

## Make testing scripts
def unbounded_initialization():
	print("")
	print("Testing Unbounded Initialization...")
	#### Initialize a MerraSource with no boundaries
	ms = MerraSource(join("data","merra-like.nc4"))

	## ensure lats, lons and times are okay
	if (ms.lats==rawLats).all(): print("  lat match: Success")
	else: raise RuntimeError("lat match: Fail")

	if (ms.lons==rawLons).all(): print("  lon match: Success")
	else: raise RuntimeError("lon match: Fail")

	if (ms.timeindex==rawTimes).all(): print("  times match: Success")
	else: raise RuntimeError("times match: Fail")

	#### Initialize a MerraSource with no boundaries
	print("Testing bounded initialization...")
	ms = MerraSource(join("data","merra-like.nc4"), bounds=aachenExt)

	## ensure lats, lons and times are okay
	if (ms.lats==aachenLats).all(): print("  lat match: Success")
	else: raise RuntimeError("lat match: Fail")

	if (ms.lons==aachenLons).all(): print("  lon match: Success")
	else: raise RuntimeError("lon match: Fail")

	if (ms.timeindex==rawTimes).all(): print("  times match: Success")
	else: raise RuntimeError("times match: Fail")

def load_variable():
	print("")
	print("Testing variable loading in unbounded state")

	#### Trying unbounded first
	ms = MerraSource(join("data","merra-like.nc4"))

	# Test for failure
	try:
		ms.load("stupid_var", name="U50M_")
		passed = False
	except ResError as e:
		if str(e) == "stupid_var not in source":
			passed = True

	if passed: print("  Caught bad variable name: Success")
	else: raise RuntimeError("  Caught bad variable name: Fail")

	# Test if raw data matches source
	ms.load("U50M", name="U50M_")

	if (ms.data["U50M_"] == raw["U50M"][:]).all(): print("  Raw data match: Success")
	else: raise RuntimeError("  Raw data match: Fail")

	# Test if processed data matches source
	ms.load("V50M", name="V50M_", processor=lambda x: x*x)

	if (ms.data["V50M_"] == raw["V50M"][:]*raw["V50M"][:]).all(): print("  Processed data match: Success")
	else: raise RuntimeError("  Processed data match: Fail")

	#### Trying bounded
	print("Testing variable loading in bounded state...")
	ms = MerraSource(join("data","merra-like.nc4"), bounds=aachenExt)
	
	# Test if raw data matches source
	ms.load("U50M", name="U50M_")
	if (ms.data["U50M_"] == raw["U50M"][:,aacehnLatSel, aacehnLonSel]).all(): print("  Raw data match: Success")
	else: raise RuntimeError("  Raw data match: Fail")

	# Test if processed data matches source
	ms.load("V50M", name="V50M_", processor=lambda x: x*x)

	if (ms.data["V50M_"] == raw["V50M"][:,aacehnLatSel, aacehnLonSel]*raw["V50M"][:,aacehnLatSel, aacehnLonSel]).all(): print("  Processed data match: Success")
	else: raise RuntimeError("  Processed data match: Fail")

def load_windspeed():
	print("")
	print("Testing windspeed loading...")

	#### only testing unbounded
	ms = MerraSource(join("data","merra-like.nc4"))
	ms.loadWindSpeed()

	wsRaw = np.sqrt(raw["U50M"][:]*raw["U50M"][:]+raw["V50M"][:]*raw["V50M"][:])
	if (np.abs(ms.data["windspeed"] - wsRaw) < 1e-5).all():
		print("  Computed windspeed magnitude: Success")
	else:
		raise RuntimeError("  Computed windspeed magnitude: Fail")

def get_index_from_location():
	print("")
	print("Testing index retrieving...")

	ms = MerraSource(join("data","merra-like.nc4"))
	if ms.loc2Index(loc) == (2,2): print("  Unbounded single access: Success")
	else: raise RuntimeError("  Unbounded single access: Fail")

	idx = ms.loc2Index(locs)
	if idx[0] == (2,2) and idx[1] == (5,1) and idx[2] == (3,3): 
	    print("  Unbounded multiple access: Success")
	else: 
		raise RuntimeError("  Unbounded multiple access: Fail")

	# testing bounded
	ms = MerraSource(join("data","merra-like.nc4"), bounds=aachenExt)

	try:
		idx = ms.loc2Index(locOutsideAachen)
		caught = False
	except ResError as e:
		caught = True
	if caught: print("  Out of bounds caught: Success")
	else: raise RuntimeError("  Out of bounds caught: Fail")

	if ms.loc2Index(loc) == (0,1): print("  Bounded single access: Success")
	else: raise RuntimeError("  Bounded single access: Fail")

	idx = ms.loc2Index(locs, outsideOkay=True)
	if idx[0] != (0,1) or idx[1] is None or idx[2] != (1,2): print("  Unbounded multiple access: Success")
	else: raise RuntimeError("  Unbounded multiple access: Fail")


def get_variable_at_location():
	print("")
	print("Testing variable getting...")

	ms = MerraSource(join("data","merra-like.nc4"), bounds=aachenExt)
	ms.load("U50M")

	# Get a single location
	ws = ms.get("U50M",loc)
	if ((ws - raw["U50M"][:,2,2])<1e-6).all():print("  Single get: Success")
	else: raise RuntimeError("  Single get: Fail")

	# Get a group of locations with a fail case
	ws = ms.get("U50M",locs, outsideOkay=True)
	if (np.abs(ws[locs[0]] - raw["U50M"][:,2,2])<1e-6).all() and (np.abs(ws[locs[2]] - raw["U50M"][:,3,3])<1e-6).all():
		print("  Multiple get: Success")
	else: raise RuntimeError("  Multiple get: Fail")

def computeContextMeans():
	print("")
	print("Testing context area...")

	ms = MerraSource(join("data","merra-like.nc4"))#, bounds=aachenExt)

	# Compute a single mean
	context = ms.contextAreaAt(locInAachen)
	contextMean = computeContextMean(join("data","gwa100-like.tif"), context)

	if (contextMean-6.05447)<1e-4: # Manually checked via QGIS
		print("  Context mean computation: Success")
	else: raise RuntimeError("  Context mean computation: Fail")

	# Compare against the precomputed value...
	print(gk.raster.extractValues(ms.GWA100_CONTEXT_MEAN_SOURCE, locInAachen, pointSRS="latlon"))

if __name__ == "__main__":
	unbounded_initialization()
	load_variable()
	load_windspeed()
	get_index_from_location()
	get_variable_at_location()
	computeContextMeans()
