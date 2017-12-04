import numpy as np
import pandas as pd
import pvlib
#from scipy.interpolate import splrep, splev
#from scipy.stats import norm
from collections import namedtuple, OrderedDict
#from glob import glob
#from os.path import join, dirname

from res.util import *
from res.weather import NCSource

sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')

def simulatePVModule(source, locs, elev, module="Canadian_Solar_CS5P_220M___2009_", inverter="ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_", extract="capacity-factor"):
	# normalize location
	locs = Location.ensureLocation(locs, forceAsArray=True)
	if isinstance(elev, str):
		elev = gk.raster.extractValues(elev, locs).data.values
	else:
		elev = np.array(elev)

	pvLibLocs = [ pvlib.location.Location(l.lat, l.lon, tz='GMT', altitude=e) for l,e in zip(locs,elev) ]

	# Ensure the source contains the correct data
	if not "ghi" in source.data: raise ResError("Source does not contain a 'ghi' field")
	if not "dni" in source.data: raise ResError("Source does not contain a 'dni' field")
	if not "windspeed" in source.data: raise ResError("Source does not contain a 'windspeed' field")
	if not "pressure" in source.data: raise ResError("Source does not contain a 'pressure' field")
	if not "air_temp" in source.data: raise ResError("Source does not contain an 'air_temp' field")

	# Construct system
	module = sandia_modules[module]
	if not inverter is None:
		inverter = sapm_inverters[inverter]

	system = {'module': module, 'inverter': inverter, 'surface_azimuth': 180}

	### Begin simulations with basic pvlib workflow
	if extract=="capacity-factor": getCF = True
	elif extract=="production": getCF = False
	else: raise ResError("extract method not understood")

	if getCF: outputs = []
	else: outputs = OrderedDict()

	for loc in locs:
		system['surface_tilt'] = loc.lat
	
		# Compute solar angles
		solpos = pvlib.solarposition.get_solarposition(source.timeindex, loc.lat, loc.lon)
		airmass = pvlib.atmosphere.relativeairmass( solpos['apparent_zenith'] )
		pressure = source.get("pressure", loc, interpolation="bilinear")

		am_abs = pvlib.atmosphere.absoluteairmass(airmass, pressure)

		aoi = pvlib.irradiance.aoi(system['surface_tilt'], system['surface_azimuth'], solpos['apparent_zenith'], solpos['azimuth'])

		# Compute irradiances
		dni_extra = pd.Series(pvlib.irradiance.extraradiation(source.timeindex), index=source.timeindex)

		dni = source.get("dni", loc, interpolation="bilinear")
		ghi = source.get("ghi", loc, interpolation="bilinear")
		dhi = ghi - dni*np.sin(solpos.apparent_elevation*np.pi/180)
		dhi[dhi<0] = 0

		total_irrad = pvlib.irradiance.total_irrad(system['surface_tilt'],system['surface_azimuth'], solpos['apparent_zenith'],
		                                           solpos['azimuth'], dni, ghi, dhi, dni_extra=dni_extra, 
		                                           model='haydavies')

		effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(total_irrad['poa_direct'], total_irrad['poa_diffuse'], 
		                                                                am_abs, aoi, module)

		# Compute cell temperature
		wspd = source.get("windspeed", loc, interpolation="bilinear")
		airT = source.get("air_temp", loc, interpolation="bilinear")-273.15
		temps = pvlib.pvsystem.sapm_celltemp(total_irrad['poa_global'], wspd, airT)


		# Compute production
		dc = pvlib.pvsystem.sapm(effective_irradiance, temps['temp_cell'], module)
		
		if not inverter is None: 
			output = pvlib.pvsystem.snlinverter(dc['v_mp'], dc['p_mp'], inverter)
		else:
			output = dc.p_mp

		if getCF:
			cf = output.mean()/(module.Impo*module.Vmpo)
			outputs.append(cf)
		else:
			outputs[loc] = output

	# Done!
	if getCF:
		return pd.Series(outputs, index=locs)
	else:
		return pd.DataFrame(outputs)/1000 # output in kWh
