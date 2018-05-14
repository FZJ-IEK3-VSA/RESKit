import numpy as np
import pandas as pd
from collections import namedtuple, OrderedDict
from res.util import *
from res.weather import NCSource
import pvlib
from types import FunctionType

class _SolarLibrary:
    def __init__(s):
        s._cecmods = None
        s._sandiamods = None
        s._cecinverters = None
        s._sandiainverters = None

    def modules(s, group='cec'): 
        name = "_"+group.lower()+"mods"
        if getattr(s, name) is None:
            setattr(s, name, pvlib.pvsystem.retrieve_sam(group+"mod"))
        return getattr(s, name)

    def inverters(s, group='cec'): 
        name = "_"+group.lower()+"inverters"
        if getattr(s, name) is None:
            setattr(s, name, pvlib.pvsystem.retrieve_sam(group+"mod"))
        return getattr(s, name)
    
SolarLibrary = _SolarLibrary()

def simulatePVModule(locs, elev, source=None, module="SunPower_SPR_X21_255", azimuth=180, tilt="latitude", totalSystemCapacity=None, tracking="fixed", modulesPerString=1, inverter=None, stringsPerInverter=1, rackingModel='open_rack_cell_glassback', airMassModel='kastenyoung1989', transpositionModel='haydavies', cellTempModel="sandia", generationModel="single-diode", inverterModel="sandia", interpolation="bilinear", loss=0.00, trackingGCR=2/7, trackingMaxAngle=60, ghi=None, dni=None, windspeed=None, pressure=None, air_temp=None, albedo=0.2, timeindex=None):
    """
    Performs a simple PV simulation

    For module options, see: res.solarpower.ModuleLibrary
    For inverter options, see: res.solarpower.InverterLibrary

    interpolation options are:
        - near
        - bilinear
        - cubic
    """
    ### Check a few inputs so it doesn't need to be done repeatedly
    if cellTempModel.lower() == "sandia": sandiaCellTemp = True
    elif cellTempModel.lower() == "noct": sandiaCellTemp = False
    else: raise ValueError("cellTempModel parameter not understood")

    if generationModel.lower() == "sapm": sandiaGenerationModel = True
    elif generationModel.lower() == "single-diode": sandiaGenerationModel = False
    else: raise ValueError("generationModel parameter not understood")

    if inverterModel.lower() == "sandia": sandiaInverterModel = True
    elif inverterModel.lower() == "driesse": sandiaInverterModel = False
    else: raise ValueError("generationModel parameter not understood")

    if tracking.lower() == "single-axis": singleAxis = True
    elif tracking.lower() == "fixed": singleAxis = False
    else: raise ValueError("tracking parameter not understood")

    ### Normalize location
    locs = LocationSet(locs)

    ### Collect weather data
    if timeindex is None:
        times = source.timeindex

    if ghi is None: ghi = source.get("ghi", locs, interpolation=interpolation)
    if dni is None: dni = source.get("dni", locs, interpolation=interpolation)
    if windspeed is None: windspeed = source.get("windspeed", locs, interpolation=interpolation)
    if pressure is None: pressure = source.get("pressure", locs, interpolation=interpolation)
    if air_temp is None: air_temp = source.get("air_temp", locs, interpolation=interpolation)-273.15

    ### Identify module and inverter
    if isinstance(module, str):
        if sandiaGenerationModel: 
            module = SolarLibrary.modules("sandia")[module] # Extract module parameters
            moduleCap = module.Impo*module.Vmpo # Max capacity of a single module
        elif not sandiaGenerationModel: 
            module = SolarLibrary.modules("cec")[module] # Extract module parameters
            moduleCap = module.I_mp_ref*module.V_mp_ref # Max capacity of a single module

            ## Check if we need to add the Desoto parameters
            # defaults for EgRef and dEgdT taken from the note in the docstring for 
            #  'pvlib.pvsystem.calcparams_desoto'
            if not "EgRef" in module: module['EgRef'] =  1.121
            if not "dEgdT" in module: module['dEgdT'] = -0.0002677

    if not inverter is None and isinstance(inverter, str):
        try:
            inverter = SolarLibrary.inverters("cec")[inverter]
        except KeyError:
            inverter = SolarLibrary.inverters("sandia")[inverter]
        else:
            raise KeyError("Could not load an inverter with name: "+inverter)

    ### Construct Generic system (tilt and azimuth given at sim time)
    if singleAxis:
        genericSystem = pvlib.tracking.SingleAxisTracker(axis_tilt=None, axis_azimuth=None, 
                            max_angle=trackingMaxAngle, module_parameters=module, albedo=albedo, 
                            modules_per_string=modulesPerString, strings_per_inverter=stringsPerInverter, 
                            inverter_parameters=inverter, racking_model=rackingModel, gcr=trackingGCR)
    else:
        genericSystem = pvlib.pvsystem.PVSystem(surface_tilt=None, surface_azimuth=None, 
                            module_parameters=module, albedo=albedo, modules_per_string=modulesPerString, 
                            strings_per_inverter=stringsPerInverter, inverter_parameters=inverter, 
                            racking_model=rackingModel)
    
    ### Check the (potentially) uniquely defined inputs
    if not totalSystemCapacity is None:
        if isinstance(totalSystemCapacity, pd.Series): pass
        elif isinstance(totalSystemCapacity, float) or isinstance(totalSystemCapacity, int):
            totalSystemCapacity = pd.Series([totalSystemCapacity,]*locs.count, index=locs)
        else:
            totalSystemCapacity = pd.Series(totalSystemCapacity, index=locs)

    if isinstance(azimuth, pd.Series): pass
    elif isinstance(azimuth, float) or isinstance(azimuth, int):
        azimuth = pd.Series([azimuth,]*locs.count, index=locs)
    else:
        azimuth = pd.Series(azimuth, index=locs)

    if isinstance(tilt, pd.Series): pass
    if isinstance(tilt,str):
        if tilt=="latitude": tilt=pd.Series([l.lat for l in locs], index=locs)
        elif tilt=="half-latitude": tilt=pd.Series([l.lat/2 for l in locs], index=locs)
        else: return ValueError("tilt directive '%s' not recognized"%tilt)
    elif isinstance(tilt, FunctionType):
        tilt = tilt=pd.Series([tilt(l,e) for l,e in zip(locs, elev)], index=locs)
    elif isinstance(tilt, float) or isinstance(tilt, int):
        tilt = pd.Series([tilt,]*locs.count, index=locs)
    else:
        tilt = pd.Series(tilt, index=locs)

    if isinstance(elev, pd.Series): pass
    elif isinstance(elev, str): 
        elev = gk.raster.extractValues(elev, locs).data
    elif isinstance(elev, float) or isinstance(elev, int):
        elev = pd.Series([elev,]*locs.count, index=locs)
    else:
        elev = pd.Series(elev, index=locs)

    ### Begin simulations
    outputs = {}
    for loc in locs:
        ## Make a localized system
        system = genericSystem.localize(latitude=loc.lat, longitude=loc.lon, altitude=elev[loc])
        system.surface_tilt = tilt[loc]
        system.surface_azimuth = azimuth[loc]
    
        ## Compute cascading dependancies
        solpos = system.get_solarposition(times, pressure[loc], air_temp[loc]) # Solar position
        
        dni_extra = pvlib.irradiance.extraradiation(times) # Extraterrestrial DNI

        amRel = system.get_airmass(times, solpos, model=airMassModel)["airmass_relative"] # Relative airmass
        amAbs = pvlib.atmosphere.absoluteairmass(amRel, pressure[loc]) # Absolute airmass 
        
        aoi = system.get_aoi(solpos['apparent_zenith'], solpos['azimuth']) # Angle of incidence

        # Compute DHI (THIS IS MY OWN THING. IT MIGHT NOT BE RIGHT AND NEEDS TO BE CHECKED!)
        dhi = ghi[loc] - dni[loc]*np.sin(solpos.apparent_elevation*np.pi/180)
        dhi[dhi<0] = 0

        # Plane of array irradiance
        # Note that not all parameters are used for all models
        poa = pvlib.irradiance.total_irrad(surface_tilt=system.surface_tilt, surface_azimuth=system.surface_azimuth,
                                           apparent_zenith=solpos['apparent_zenith'], azimuth=solpos['azimuth'],
                                           dni=dni[loc], ghi=ghi[loc], dhi=dhi,
                                           dni_extra=dni_extra,
                                           albedo=albedo, airmass=amAbs,
                                           model=transpositionModel) 

        # Cell temp
        if sandiaCellTemp:
            moduleTemp = system.sapm_celltemp(poa.poa_global, windspeed[loc], air_temp[loc])
        else:
            # TODO: Implement NOCT model
            raise ResError("NOCT celltemp module not yet implemented :(")

        ## Add irradiance losses
        # TODO: Look into this. Is it necessary???
        # See pvsystem.physicaliam, pvsystem.ashraeiam, and pvsystem.sapm_aoi_loss

        ## Do DC Generation calculation
        if sandiaGenerationModel:
            effectiveIrradiance = system.sapm_effective_irradiance( poa_direct=total_irrad['poa_direct'], 
                                        poa_diffuse=total_irrad['poa_diffuse'], airmass_absolute=amAbs, aoi=aoi)
            rawDCGeneration = system.sapm(effective_irradiance=effectiveIrradiance, 
                                       temp_cell=moduleTemp['temp_cell'])
        else:
            sotoParams = system.calcparams_desoto(poa.poa_global, moduleTemp.temp_cell)
            photoCur, satCur, resSeries, resShunt, nNsVth = sotoParams
            rawDCGeneration = system.singlediode(photocurrent=photoCur, saturation_current=satCur, 
                                    resistance_series=resSeries, resistance_shunt=resShunt, nNsVth=nNsVth)

        ## Simulate inverter interation
        if not inverter is None: 
            if sandiaInverterModel: 
                generation   = system.snlinverter(v_dc=rawDCGeneration['v_mp']*system.modules_per_string, 
                                    p_dc=rawDCGeneration['p_mp']*system.modules_per_string*system.strings_per_inverter)

            else: generation = system.adrinverter(v_dc=rawDCGeneration['v_mp']*system.modules_per_string, 
                                    p_dc=rawDCGeneration['p_mp']*system.modules_per_string*system.strings_per_inverter)
            
            # normalize to a single module
            generation = generation/system.modules_per_string/system.strings_per_inverter
        else:
            generation = rawDCGeneration.p_mp

        generation *= (1-loss) # apply a flat loss

        if totalSystemCapacity is None:
            outputs[loc] = generation/moduleCap # outputs in capacity-factor
        else:
            # outputs in whatever unit totalSystemCapacity is in
            outputs[loc] = generation/moduleCap*totalSystemCapacity[loc] 

    # Done!
    return pd.DataFrame(outputs)
    
def simulatePVRooftopDistribution(locs, tilts, azimuths, probabilityDensity, rackingModel="roof_mount_cell_glassback", **kwargs):
    """
    Simulate a distribution of pv rooftops and combine results
    """
    output = None
    for ti, tilt in enumerate(tilts):
        output.append([])
        for ai, azimuth in enumerate(tilts):
            simResult = simulatePVModule(locs=locs, tilt=tilt, azimuth=azimuth, 
                                         rackingModel=rackingModel, extract=extract, **kwargs)

            if output is None: output=simResult*probabilityDensity[ti,ai]
            else: finalResult += simResult*probabilityDensity[ti,ai]
    return finalResult