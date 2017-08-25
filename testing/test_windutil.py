import netCDF4 as nc
import numpy as np
from os.path import join
import geokit as gk

from res.weather.windutil import *
from res.util import ResError, LatLonLocation

## setup some common inputs
loc = LatLonLocation(lat=50.105, lon=6.005)
locInAachen = LatLonLocation(lat=50.763, lon=6.202)
locOutsideAachen = LatLonLocation(lat=51.255, lon=5.605)
locs = [loc, locOutsideAachen,LatLonLocation(lat=50.605, lon=6.605)]

windspeed = np.arange(30)
windspeeds = np.column_stack( [windspeed, windspeed+0.3333, windspeed+0.6667] )

## Make testing functions!!
def test_adjustLraToGwa():
	print("MAKE TEST FOR: adjustLraToGwa()")

def test_adjustContextMeanToGwa():
	print("MAKE TEST FOR: adjustContextMeanToGwa()")

def test_projectByLogLaw():
	print("Testing projectByLogLaw...")

	# testing single wind speed array
	wsOut = projectByLogLaw(windspeed, measuredHeight=20, targetHeight=100, roughness=0.01)
	if abs(wsOut.sum() - 527.10820631304512) < 1e-6: # checked by hand computation
		print("  Single array: Success")
	else: raise RuntimeError("Single array: Fail")

	# testing multiple wind speed array
	wsOut = projectByLogLaw(windspeeds, measuredHeight=20, targetHeight=100, roughness=0.01)
	if abs(wsOut[:,0].sum() - 527.10820631304512) < 1e-6 and \
	   abs(wsOut[:,1].sum() - 539.22442460022705) < 1e-6 and \
	   abs(wsOut[:,2].sum() - 551.34427811641797) < 1e-6 : # checked by hand computation
		print("  Multiple arrays: Success")
	else: raise RuntimeError("Multiple arrays: Fail")

	# testing multiple wind speed arrays with multiple roughnesses 
	wsOut = projectByLogLaw(windspeeds, measuredHeight=20, targetHeight=100, roughness=np.array([0.01, 0.121, 0.005]))
	if abs(wsOut[:,0].sum() - 527.10820631304512) < 1e-6 and \
	   abs(wsOut[:,1].sum() - 585.21841016121857) < 1e-6 and \
	   abs(wsOut[:,2].sum() - 543.29271410486263) < 1e-6 : # checked by hand computation
		print("  Multiple arrays and roughnesses: Success")
	else: raise RuntimeError("Multiple arrays and roughnesses: Fail")

def test_projectByPowerLaw():
	print("Testing projectByPowerLaw...")

	# testing single wind speed array
	wsOut = projectByPowerLaw(windspeed, measuredHeight=20, targetHeight=100, alpha=0.2)
	if abs(wsOut.sum() - 600.18240273562856) < 1e-6: # checked by hand computation
		print("  Single array: Success")
	else: raise RuntimeError("Single array: Fail")

	# testing multiple wind speed array
	wsOut = projectByPowerLaw(windspeeds, measuredHeight=20, targetHeight=100, alpha=0.2)
	if abs(wsOut[:,0].sum() - 600.18240273562856) < 1e-6 and \
	   abs(wsOut[:,1].sum() - 613.97831962057921) < 1e-6 and \
	   abs(wsOut[:,2].sum() - 627.77837569451424) < 1e-6 : # checked by hand computation
		print("  Multiple arrays: Success")
	else: raise RuntimeError("Multiple arrays: Fail")

	# testing multiple wind speed arrays with multiple alphas 
	wsOut = projectByPowerLaw(windspeeds, measuredHeight=20, targetHeight=100, alpha=np.array([0.2, 0.23, 0.25]))
	if abs(wsOut[:,0].sum() - 600.18240273562856) < 1e-6 and \
	   abs(wsOut[:,1].sum() - 644.35044981969804) < 1e-6 and \
	   abs(wsOut[:,2].sum() - 680.38519080443655) < 1e-6 : # checked by hand computation
		print("  Multiple arrays and alphas: Success")
	else: raise RuntimeError("Multiple arrays and alphas: Fail")

def test_alphaFromLevels():
	print("Testing alphaFromLevels...")

	a = alphaFromLevels(lowWindSpeed=5, lowHeight=10, highWindSpeed=7.9244659623055682, highHeight=100)
	if a-0.2 < 1e-6: print("  Single Conversion: Success")
	else: raise RuntimeError("Single Conversion: Fail")

	a = alphaFromLevels(lowWindSpeed=np.array([5,5,5]), lowHeight=10, highHeight=100, 
		                highWindSpeed=np.array([7.9244659623055682, 7.0626877231137719, 6.6676071608166199]))
	if (a[0]-0.2) < 1e-6 and (a[1]-0.15) < 1e-6 and (a[2]-0.125) < 1e-6 : 
		print("  Multiple windspeed conversion: Success")
	else: raise RuntimeError("Multiple windspeed conversion: Fail")

	a = alphaFromLevels(lowWindSpeed=np.array([5,5,5]), 
		                    lowHeight=np.array([10,15,20]), 
		                    highWindSpeed=np.array([7.9244659623055682, 6.6459384598839435, 6.1142227249692596]), 
		                    highHeight=100)

	if (a[0]-0.2) < 1e-6 and (a[1]-0.15) < 1e-6 and (a[2]-0.125) < 1e-6 :  # checked by hand computation
		print("  Multiple height conversion: Success")
	else: raise RuntimeError("Multiple height conversion: Fail")

def test_alphaFromGWA():
	print("Testing alphaFromGWA...")

	a = alphaFromGWA(gwaDir="data", loc=loc, _structure="gwa%d-like.tif", pairID=0)
	if a-0.2180788 < 1e-6: print("  Single location: Success")
	else: raise RuntimeError("Single location: Fail")

	a = alphaFromGWA(gwaDir="data", loc=locs, _structure="gwa%d-like.tif", pairID=0)
	if abs(a[0]-0.2180788) < 1e-6 and abs(a[1]-0.223474) < 1e-6 and abs(a[2]-0.220947) < 1e-6: 
		print("  Multiple locations: Success")
	else: raise RuntimeError("Multiple locations: Fail")

def test_roughnessFromLevels():
	print("Testing roughnessFromLevels...")

	r = roughnessFromLevels(lowWindSpeed=5, lowHeight=10, highWindSpeed=7.5, highHeight=100)
	if r-0.1 < 1e-6: print("  Single Conversion: Success")
	else: raise RuntimeError("Single Conversion: Fail")

	r = roughnessFromLevels(lowWindSpeed=np.array([5,5,5]), lowHeight=10, highWindSpeed=np.array([7.5, 6.25, 6.0]), highHeight=100)
	if (r[0]-0.1) < 1e-6 and (r[1]-0.001) < 1e-6 and (r[2]-0.0001) < 1e-6 : 
		print("  Multiple windspeed conversion: Success")
	else: raise RuntimeError("Multiple windspeed conversion: Fail")

	r = roughnessFromLevels(lowWindSpeed=np.array([5,5,5]), 
		                    lowHeight=np.array([10,15,20]), 
		                    highWindSpeed=np.array([7.5, 6.25, 6.0]), 
		                    highHeight=100)

	if (r[0]-0.1) < 1e-6 and (r[1]-0.00759375) < 1e-6 and (r[2]-0.0064) < 1e-6: # checked by hand computation
		print("  Multiple height conversion: Success")
	else: raise RuntimeError("Multiple height conversion: Fail")

def test_roughnessFromGWA():
	print("Testing roughnessFromGWA...")

	r = roughnessFromGWA(gwaDir="data", loc=loc, _structure="gwa%d-like.tif", pairID=0)
	if abs(r-0.71488771) < 1e-6: print("  Single location: Success")
	else: raise RuntimeError("Single location: Fail")

	r = roughnessFromGWA(gwaDir="data", loc=locs, _structure="gwa%d-like.tif", pairID=0)
	if abs(r[0]-0.71488771) < 1e-6 and abs(r[1]-0.798400880115) < 1e-6 and abs(r[2]-0.75864270) < 1e-6: 
		print("  Multiple locations: Success")
	else: raise RuntimeError("Multiple locations: Fail")

def test_roughnessFromCLC():
	print("Testing roughnessFromCLC...")
	loc1 = LatLonLocation(lat=50.370680, lon=5.752684) # grid value: 24 -> code: 312 -> rough: 0.75
	loc2 = LatLonLocation(lat=50.52603, lon=6.10476) # grid value: 36 -> code: 412 -> rough: 0.0005
	loc3 = LatLonLocation(lat=50.59082, lon=5.86483) # grid value: 1 -> code: 111 -> rough: 1.2

	r = roughnessFromCLC(clcPath=join("data","clc-aachen_clipped.tif"), loc=loc1)
	if r-0.75 < 1e-6: print("  Single location: Success")
	else: raise RuntimeError("Single location: Fail")

	r = roughnessFromCLC(clcPath=join("data","clc-aachen_clipped.tif"), loc=[loc1,loc2,loc3])
	if abs(r[0]-0.75) < 1e-6 and abs(r[1]-0.0005) < 1e-6 and abs(r[2]-1.2) < 1e-6: 
		print("  Multiple locations: Success")
	else: raise RuntimeError("Multiple locations: Fail")

def test_roughnessFromLandCover():
	print("MAKE TEST FOR: roughnessFromLandCover()")

if __name__ == '__main__':
	test_adjustLraToGwa(); print("")
	test_adjustContextMeanToGwa(); print("")
	test_projectByLogLaw(); print("")
	test_projectByPowerLaw(); print("")
	test_alphaFromLevels(); print("")
	test_alphaFromGWA(); print("")
	test_roughnessFromLevels(); print("")
	test_roughnessFromGWA(); print("")
	test_roughnessFromCLC(); print("")
	test_roughnessFromLandCover(); print("")