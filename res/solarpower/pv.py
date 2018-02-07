import numpy as np
import pandas as pd
import pvlib
#from scipy.interpolate import splrep, splev
#from scipy.stats import norm
from collections import namedtuple, OrderedDict
from res.util import *
from res.weather import NCSource

sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')

def simulatePVModule(locs, elev, source=None, capacity=None, module="Canadian_Solar_CS5P_220M___2009_", azimuth=180, tilt="latitude", inverter="ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_", extract="capacity-factor", interpolation="bilinear", loss=0.00, ghi=None, dni=None, windspeed=None, pressure=None, air_temp=None, timeindex=None):
    """
    Performs a simple PV simulation

    For module options, see: res.solarpower.ModuleLibrary
    For inverter options, see: res.solarpower.InverterLibrary

    interpolation options are:
        - near
        - bilinear
        - cubic
    """
    # normalize location
    locs = Location.ensureLocation(locs, forceAsArray=True)
    if isinstance(elev, str): elev = gk.raster.extractValues(elev, locs).data.values
    else: elev = np.array(elev)


    pvLibLocs = [ pvlib.location.Location(l.lat, l.lon, tz='GMT', altitude=e) for l,e in zip(locs,elev) ]

    # Ensure the source contains the correct data
    if timeindex is None:
        timeindex = source._timeindex()

    if ghi is None: ghi = source.get("ghi", locs, interpolation=interpolation)
    if dni is None: dni = source.get("dni", locs, interpolation=interpolation)
    if windspeed is None: windspeed = source.get("windspeed", locs, interpolation=interpolation)
    if pressure is None: pressure = source.get("pressure", locs, interpolation=interpolation)
    if air_temp is None: air_temp = source.get("air_temp", locs, interpolation=interpolation)-273.15

    # Construct system
    module = sandia_modules[module]
    moduleCap = module.Impo*module.Vmpo
    if capacity is None: capacity = pd.Series([moduleCap/1000,]*len(locs), index=locs)

    if not inverter is None:
        inverter = sapm_inverters[inverter]

    system = {'module': module, 'inverter': inverter}
    
    # Check the (potentially) uniquely defined input
    if not isinstance(capacity, pd.Series):
        if isinstance(capacity, float) or isinstance(capacity, int):
            capacity = pd.Series([capacity,]*len(locs), index=locs)
        else:
            capacity = pd.Series(capacity, index=locs)

    if not isinstance(azimuth, pd.Series):
        if isinstance(azimuth, float) or isinstance(azimuth, int):
            azimuth = pd.Series([azimuth,]*len(locs), index=locs)
        else:
            azimuth = pd.Series(azimuth, index=locs)

    if isinstance(tilt,str):
        if tilt=="latitude": tilt=pd.Series([l.lat for l in locs], index=locs)
        elif tilt=="half-latitude": tilt=pd.Series([l.lat/2 for l in locs], index=locs)
        else:
            return ResError("tilt directive '%s' not recognized"%tilt)
    else:
        if not isinstance(tilt, pd.Series):
            if isinstance(tilt, float) or isinstance(tilt, int):
                tilt = pd.Series([tilt,]*len(locs), index=locs)
            else:
                tilt = pd.Series(tilt, index=locs)

    ### Begin simulations with basic pvlib workflow
    if extract=="capacity-factor": getCF = True
    elif extract=="production": getCF = False
    elif extract=="capacity-production": getCF = False
    else: raise ResError("extract method not understood")

    if getCF: outputs = []
    else: outputs = OrderedDict()

    for loc in locs:
        system['surface_azimuth'] = azimuth[loc]
        system['surface_tilt'] = tilt[loc]
    
        # Compute solar angles
        solpos = pvlib.solarposition.get_solarposition(timeindex, loc.lat, loc.lon)
        airmass = pvlib.atmosphere.relativeairmass( solpos['apparent_zenith'] )

        am_abs = pvlib.atmosphere.absoluteairmass(airmass, pressure[loc])
        #return system['surface_tilt'], system['surface_azimuth'], solpos['apparent_zenith'], solpos['azimuth']
        aoi = pvlib.irradiance.aoi(system['surface_tilt'], system['surface_azimuth'], solpos['apparent_zenith'], solpos['azimuth'])

        # Compute irradiances
        dni_extra = pd.Series(pvlib.irradiance.extraradiation(timeindex), index=timeindex)

        dhi = ghi[loc] - dni[loc]*np.sin(solpos.apparent_elevation*np.pi/180)
        dhi[dhi<0] = 0

        total_irrad = pvlib.irradiance.total_irrad(system['surface_tilt'],system['surface_azimuth'], solpos['apparent_zenith'],
                                                   solpos['azimuth'], dni[loc], ghi[loc], dhi, dni_extra=dni_extra, 
                                                   model='haydavies')

        effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(total_irrad['poa_direct'], total_irrad['poa_diffuse'], 
                                                                        am_abs, aoi, module)

        # Compute cell temperature
        temps = pvlib.pvsystem.sapm_celltemp(total_irrad['poa_global'], windspeed[loc], air_temp[loc])

        # Compute production
        dc = pvlib.pvsystem.sapm(effective_irradiance, temps['temp_cell'], module)
        
        if not inverter is None: 
            ########################################
            ## INVERTER DOES NTO TAKE INTO ACCOUNT 
            ## MULTIPLE MODULES PER INVERTER
            ##  - It should be added!
            ########################################
            output = pvlib.pvsystem.snlinverter(dc['v_mp'], dc['p_mp'], inverter)*(1-loss)
        else:
            output = dc.p_mp*(1-loss)

        if getCF:
            outputs.append(output.mean())
        else:
            outputs[loc] = output

    # Done!
    if getCF:
        return pd.Series(outputs, index=locs)/moduleCap
    else:
        if extract == "capacity-production":
            return pd.DataFrame(outputs)/moduleCap # output in capacity factor
        else:
            return pd.DataFrame(outputs)/(capacity/moduleCap) # output in kWh
