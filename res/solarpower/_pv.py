import numpy as np
import pandas as pd
from collections import namedtuple, OrderedDict
from res.util.util_ import *
from res.weather import NCSource
import pvlib
from types import FunctionType
from datetime import datetime as dt
from os.path import isfile
from scipy.interpolate import RectBivariateSpline

class _SolarLibrary:
    def __init__(s):
        s._cecmods = None
        s._sandiamods = None
        s._cecinverters = None
        s._sandiainverters = None

    def modules(s, group='cec'): 
        name = "_"+group.lower()+"mods"
        if getattr(s, name) is None:
            #setattr(s, name, pvlib.pvsystem.retrieve_sam(group+"mod"))
            setattr(s, name, pd.read_csv(join(DATADIR,"modules","sam-library-cec-modules-2017-6-5.csv"), skiprows=[1,2], index_col=0))
        return getattr(s, name)

    def inverters(s, group='sandia'): 
        name = "_"+group.lower()+"inverters"
        if getattr(s, name) is None:
            setattr(s, name, pvlib.pvsystem.retrieve_sam(group+"inverter").T)
        return getattr(s, name)

SolarLibrary = _SolarLibrary()

def _sapm_celltemp(poa_global, wind_speed, temp_air, model='open_rack_cell_glassback'):
    """ Cell temp function updated from PVLIB version! """
    temp_models = {'open_rack_cell_glassback': [-3.47, -.0594, 3],
                   'roof_mount_cell_glassback': [-2.98, -.0471, 1],
                   'open_rack_cell_polymerback': [-3.56, -.0750, 3],
                   'insulated_back_polymerback': [-2.81, -.0455, 0],
                   'open_rack_polymer_thinfilm_steel': [-3.58, -.113, 3],
                   '22x_concentrator_tracker': [-3.23, -.130, 13]
                   }

    if isinstance(model, str):
        model = temp_models[model.lower()]
    elif isinstance(model, list):
        model = model
    elif isinstance(model, (dict, pd.Series)):
        model = [model['a'], model['b'], model['deltaT']]

    a = model[0]
    b = model[1]
    deltaT = model[2]

    E0 = 1000.  # Reference irradiance

    temp_module = poa_global*np.exp(a + b*wind_speed) + temp_air

    temp_cell = temp_module + (poa_global / E0)*(deltaT)

    return temp_cell

def ensureSeries(var, locs):
    if isinstance(var, pd.Series): pass
    elif isinstance(var, float) or isinstance(var, int):
        var = pd.Series([var,]*locs.count, index=locs)
    elif isinstance(var, str): 
        var = gk.raster.extractValues(var, locs).data
    else:
        var = pd.Series(var, index=locs)

    return var

def frankCorrectionFactors(ghi, dni_extra, times, solarElevation):
    transmissivity = ghi/dni_extra
    sigmoid = 1/(1+np.exp( -(transmissivity-0.5)/0.03 ))

    # Adjust cloudy regime
    months = times.month
    cloudyFactors = np.empty(months.shape)

    cloudyFactors[months==1] = 0.7776553729824053
    cloudyFactors[months==2] = 0.7897164461247639
    cloudyFactors[months==3] = 0.8176553729824052
    cloudyFactors[months==4] = 0.8406805293005672
    cloudyFactors[months==5] = 0.8761808928311765
    cloudyFactors[months==6] = 0.9094139886578452
    cloudyFactors[months==7] = 0.9350856478115459
    cloudyFactors[months==8] = 0.9191682419659737
    cloudyFactors[months==9] = 0.912703795259561
    cloudyFactors[months==10]= 0.8775035625999711
    cloudyFactors[months==11]= 0.8283158353933402
    cloudyFactors[months==12]= 0.7651417769376183
    cloudyFactors = np.broadcast_to(cloudyFactors.reshape( (cloudyFactors.size,1) ), ghi.shape)

    cloudyFactors = cloudyFactors*(1-sigmoid)

    # Adjust clearsky regime
    e = solarElevation
    clearSkyFactors = np.ones(e.shape)


    clearSkyFactors[np.where((e>=10)&(e<20))] = 1.17612920884004
    clearSkyFactors[np.where((e>=20)&(e<30))] = 1.1384180020822825
    clearSkyFactors[np.where((e>=30)&(e<40))] = 1.1022951259566156
    clearSkyFactors[np.where((e>=40)&(e<50))] = 1.0856852748290704
    clearSkyFactors[np.where((e>=50)&(e<60))] = 1.0779254457050245
    clearSkyFactors[np.where(e>=60)] = 1.0715262914980628

    clearSkyFactors *= sigmoid

    # Apply to ghi
    totalCorrectionFactor = clearSkyFactors+cloudyFactors

    del clearSkyFactors, cloudyFactors, totalCorrectionFactor, e, months, sigmoid, transmissivity

    return totalCorrectionFactor

def locToTilt(locs, convention="latitude*0.76", **k):
    locs = gk.LocationSet(locs)

    if convention=="ninja": 
        lats = locs.lats
        tilt = np.zeros( lats.size ) + 40

        s = lats <= 25
        tilt[ s ] = lats[s]*0.87

        s = np.logical_and(lats > 25, lats <= 50)
        tilt[ s ] = (lats[s]*0.76)+3.1

    elif isfile(convention):
        tilt = gk.raster.interpolateValues(convention, locs, **k)

    else:
        try:
            tilt = eval(convention, {}, {"latitude":locs.lats})
        except:
            raise ResError("Failed to apply tilt convention")

    tilt=pd.Series(tilt, index=locs[:])

    return tilt

def _presim(locs, source, elev=300, module="WINAICO WSx-240P6", azimuth=180, tilt="ninja", totalSystemCapacity=None, tracking="fixed", modulesPerString=1, inverter=None, stringsPerInverter=1, rackingModel='open_rack_cell_glassback', airmassModel='kastenyoung1989', transpositionModel='perez', cellTempModel="sandia", generationModel="single-diode", inverterModel="sandia", interpolation="bilinear", loss=0.16, trackingGCR=2/7, trackingMaxAngle=60, frankCorrection=False,):

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

    #addTime("param check")

    ### Normalize location
    locs = LocationSet(locs)
    #addTime("arrange locations")

    ### Collect weather data
    if isinstance(source, NCSource):
        times = source.timeindex

        idx = source.loc2Index(locs, asInt=False)
        k = dict( locations=locs, interpolation=interpolation, forceDataFrame=True, _indicies=idx )

        ghi = source.get("ghi", **k)
        dhi = source.get("dhi", **k) if "dhi" in source.data else None
        dni = source.get("dni", **k) if "dni" in source.data else None

        windspeed = source.get("windspeed", **k)
        pressure = source.get("pressure", **k)
        air_temp = source.get("air_temp", **k)
        dew_temp = source.get("dew_temp", **k)

        if "albedo" in source.data: 
            albedo = source.get("albedo", **k)
        else: 
            albedo = 0.2

    else: # source should be a dictionary
        times = source["times"]
        
        ghi = source["ghi"]
        dhi = source["dhi"] if "dhi" in source else None
        dni = source["dni"] if "dni" in source else None

        windspeed = source["windspeed"]
        pressure = source["pressure"]
        air_temp = source["air_temp"]
        dew_temp = source["dew_temp"]
        if "albedo" in source: 
            albedo = source["albedo"]
        else: 
            albedo = 0.2

    #addTime("Extract weather data")

    ### Identify module and inverter
    if isinstance(module, str):
        if sandiaGenerationModel: 
            module = SolarLibrary.modules("sandia").loc[module] # Extract module parameters
            moduleCap = module.Impo*module.Vmpo # Max capacity of a single module
        elif not sandiaGenerationModel:
            if isinstance(module, str):
                if module == "WINAICO WSx-240P6":
                    module = pd.Series(dict(
                        BIPV      =           "N",
                        Date      =    "6/2/2014",
                        T_NOCT    =            43,
                        A_c       =         1.663,
                        N_s       =            60,
                        I_sc_ref  =          8.41,
                        V_oc_ref  =         37.12,
                        I_mp_ref  =          7.96,
                        V_mp_ref  =          30.2,
                        alpha_sc  =      0.001164,
                        beta_oc   =      -0.12357,
                        a_ref     =        1.6704,
                        I_L_ref   =         8.961,
                        I_o_ref   =      1.66e-11,
                        R_s       =         0.405,
                        R_sh_ref  =        326.74,
                        Adjust    =         4.747,
                        gamma_r   =        -0.383,
                        Version   =      "NRELv1",
                        PTC       =         220.2,
                        Technology=  "Multi-c-Si",
                    ))
                    module.name="WINAICO WSx-240P6"
                elif module == "LG Electronics LG370Q1C-A5":
                    module = pd.Series(dict(
                        BIPV      =            "N",
                        Date      =   "12/14/2016",
                        T_NOCT    =          45.7,
                        A_c       =         1.673,
                        N_s       =            60,
                        I_sc_ref  =         10.82,
                        V_oc_ref  =          42.8,
                        I_mp_ref  =         10.01,
                        V_mp_ref  =            37,
                        alpha_sc  =      0.003246,
                        beta_oc   =      -0.10272,
                        a_ref     =        1.5532,
                        I_L_ref   =        10.829,
                        I_o_ref   =      1.12e-11,
                        R_s       =         0.079,
                        R_sh_ref  =         92.96,
                        Adjust    =            14,
                        gamma_r   =         -0.32,
                        Version   =       "NRELv1",
                        PTC       =         347.2,
                        Technology=    "Mono-c-Si",
                    ))
                    module.name="LG Electronics LG370Q1C-A5"
                else:

                    module = SolarLibrary.modules("cec").loc[module].copy() # Extract module parameters

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

    #addTime("Create generic module")

    ### Check the (potentially) uniquely defined inputs
    if not totalSystemCapacity is None: totalSystemCapacity = ensureSeries(totalSystemCapacity, locs)
    
    azimuth = ensureSeries(azimuth, locs)
    elev = ensureSeries(elev, locs)

    if isinstance(tilt, str): tilt = locToTilt(locs, tilt)
    else: tilt = ensureSeries(tilt, locs)
    
    #addTime("Check unique inputs")

    ### Begin simulations
    # get solar position
    _solpos = {}
    checkedSolPosValues = {}

    for i,loc in enumerate(locs):
        solposLat = np.round(loc.lat, 1) # Only perform solor position calc for every .1 degrees
        solposLon = np.round(loc.lon, 1) # Only perform solor position calc for every .1 degrees
        solposElev = np.round(elev[loc], -2) # Only perform solor position calc for every 100 meters
        if isinstance(solposElev, pd.Series): raise ResError("%s is not unique"%str(loc))

        solposKey = (solposLon, solposLat, solposElev)
        if not solposKey in checkedSolPosValues:
            # Pressure and temperature should not change much between evaluated locations
            checkedSolPosValues[solposKey] = pvlib.solarposition.spa_python(times, 
                                                                            latitude=solposLat,
                                                                            longitude=solposLon,
                                                                            altitude=solposElev,
                                                                            pressure=pressure[loc],  
                                                                            temperature=air_temp[loc])

        _solpos[loc] = checkedSolPosValues[solposKey]
    solpos = {}
    for c in ['apparent_zenith', 'azimuth', 'apparent_elevation']:
        solpos[c] = pd.DataFrame(np.column_stack([_solpos[loc][c].copy() for loc in locs]), columns=locs, index=times)
    del _solpos, checkedSolPosValues
    #addTime("Solar position")

    # DNI Extraterrestrial
    dni_extra = pvlib.irradiance.extraradiation(times).values
    if len(ghi.shape) > 1:
        dni_extra = np.broadcast_to(dni_extra.reshape((ghi.shape[0],1)), ghi.shape)
    #addTime("DNI Extra")
    
    # Apply Frank corrections when dealing with COSMO data?
    if frankCorrection:
        ghi *= frankCorrectionFactors(ghi, dni_extra, times, solpos["apparent_elevation"])

    #addTime("Frank Correction")

    # Compute DHI or DNI
    if dni is None:
        dni = pd.DataFrame(0, index=times, columns=ghi.columns)
        for c in ghi.columns:
            dni[c] = pvlib.irradiance.dirint(ghi[c], solpos["apparent_zenith"][c], times, pressure=pressure[c], 
                                             temp_dew=None if dew_temp is None else dew_temp[c] )
            # TODO: This needs to be updated to adapt to COSMO data (Which has relative humidity instead of dew_point temp)
        dni.fillna(0, inplace=True)

    if dhi is None:
        dhi = ghi - dni*np.sin( np.radians(solpos["apparent_elevation"]))
        dhi[dhi<0] = 0
        dhi.fillna(0, inplace=True)

    #addTime("DHI calc")

    # Airmass
    amRel = np.full_like(solpos["apparent_zenith"].values, 100)
    s = solpos["apparent_zenith"].values < 90
    amRel[s] = pvlib.atmosphere.relativeairmass(solpos["apparent_zenith"].values[s], model=airmassModel)
    #addTime("Airmass")

    return dict(singleAxis=singleAxis,
                tilt=tilt.values,
                module=module,
                azimuth=azimuth.values,
                inverter=inverter,
                moduleCap=moduleCap,
                modulesPerString=modulesPerString,
                stringsPerInverter=stringsPerInverter,
                loss=loss,
                
                locs=locs,
                times=times,

                dni=dni.values,
                ghi=ghi.values,
                dhi=dhi.values,
                amRel=amRel,
                solpos=solpos,
                pressure=pressure,
                air_temp=air_temp,
                windspeed=windspeed,
                dni_extra=dni_extra,

                sandiaCellTemp=sandiaCellTemp,
                transpositionModel=transpositionModel,
                totalSystemCapacity=totalSystemCapacity,
                sandiaGenerationModel=sandiaGenerationModel,)

def _simulation(singleAxis, tilt, module, azimuth, inverter, moduleCap, modulesPerString, stringsPerInverter, locs, times, dni, ghi, dhi, amRel, solpos, pressure, air_temp, windspeed, dni_extra, sandiaCellTemp, transpositionModel, totalSystemCapacity, sandiaGenerationModel, loss, approximateSingleDiode):

    # Get tilt and azimuths
    if singleAxis:
        axis_tilt = tilt
        axis_azimuth = azimuth

        tilt = pd.DataFrame(index=times, columns=locs)
        azimuth = pd.DataFrame(index=times, columns=locs)

        for loc, at, aa in zip(locs, axis_tilt, axis_azimuth):
            tmp = pvlib.tracking.singleaxis(apparent_zenith=solpos["apparent_zenith"][loc], 
                                            apparent_azimuth=solpos["azimuth"][loc], axis_tilt=at, axis_azimuth=aa, 
                                            max_angle=genericSystem.max_angle, backtrack=genericSystem.backtrack,
                                            gcr=genericSystem.gcr)
            tilt[loc] = tmp["surface_tilt"].copy()
            azimuth[loc] = tmp["surface_azimuth"].copy()

        del axis_azimuth, axis_tilt, tmp

    # Angle of Incidence
    aoi = pvlib.irradiance.aoi(tilt, azimuth, solpos['apparent_zenith'], solpos['azimuth'])
    
    # Compute Total irradiation
    poa = pvlib.irradiance.total_irrad(surface_tilt=tilt,
                                       surface_azimuth=azimuth,
                                       apparent_zenith=solpos['apparent_zenith'].values,
                                       azimuth=solpos['azimuth'].values,
                                       dni=dni, ghi=ghi, dhi=dhi,
                                       dni_extra=dni_extra,
                                       model=transpositionModel,
                                       airmass=amRel)

    del dni, ghi, dhi, azimuth, dni_extra, solpos
    
    # Cell temp
    if sandiaCellTemp:
        cellTemp = _sapm_celltemp(poa['poa_global'], windspeed, air_temp)
    else:
        # TODO: Implement NOCT model
        raise ResError("NOCT celltemp module not yet implemented :(")
    del air_temp, windspeed
    #addTime("Cell temp")

    ## Do DC Generation calculation
    if sandiaGenerationModel:
        # Not guarenteed to work with 2D inputs
        amAbs = pvlib.atmosphere.absoluteairmass(amRel, pressure)
        effectiveIrradiance = pvlib.pvsystem.sapm_effective_irradiance( poa_direct=poa['poa_direct'], 
                                    poa_diffuse=poa['poa_diffuse'], airmass_absolute=amAbs, aoi=aoi, 
                                    module=module)
        rawDCGeneration = pvlib.pvsystem.sapm(effective_irradiance=effectiveIrradiance, 
                                   temp_cell=cellTemp, module=module)
        del amAbs, aoi
    else:
        ## Add irradiance losses due to angle of incidence
        poa_total = 0
        poa_total += poa["poa_direct"] * pvlib.pvsystem.physicaliam(aoi)

        # Effective angle of incidence values from "Solar-Engineering-of-Thermal-Processes-4th-Edition"
        poa_total += poa["poa_ground_diffuse"] * pvlib.pvsystem.physicaliam( 90 - 0.5788*tilt + 0.002693*np.power(tilt, 2) ) 
        poa_total += poa["poa_sky_diffuse"] * pvlib.pvsystem.physicaliam( 59.7 - 0.1388*tilt + 0.001497*np.power(tilt, 2) ) 
        sel = poa_total>0

        if approximateSingleDiode:
            # Use RectBivariateSpline to speed up simulation, but at the cost of accuracy (should still be >99.996%)
            maxpoa = np.nanmax(poa_total)
            _poa = np.concatenate([np.logspace(-1, np.log10(maxpoa/10), 20, endpoint=False), 
                                   np.linspace(maxpoa/10, maxpoa, 80)])
            _temp = np.linspace(cellTemp.values[sel].min(), cellTemp.values[sel].max(), 100)
            poaM, tempM = np.meshgrid(_poa, _temp)

            sotoParams = pvlib.pvsystem.calcparams_desoto(poa_global=poaM.flatten(), 
                                                          temp_cell=tempM.flatten(), 
                                                          alpha_isc=module.alpha_sc, 
                                                          module_parameters=module, 
                                                          EgRef=module.EgRef, 
                                                          dEgdT=module.dEgdT)

            photoCur, satCur, resSeries, resShunt, nNsVth = sotoParams
            gen = pvlib.pvsystem.singlediode(photocurrent=photoCur, saturation_current=satCur, 
                                             resistance_series=resSeries, resistance_shunt=resShunt, 
                                             nNsVth=nNsVth)

            rawDCGeneration=OrderedDict()

            interpolator = RectBivariateSpline( _temp, _poa, gen['p_mp'].reshape(poaM.shape), kx=3, ky=3)
            rawDCGeneration['p_mp'] = pd.DataFrame(index=times, columns=locs)
            rawDCGeneration['p_mp'].values[sel] = interpolator( cellTemp.values[sel], poa_total[sel], grid=False )

            if not inverter is None: 
                interpolator = RectBivariateSpline( _temp, _poa, gen['v_mp'].reshape(poaM.shape), kx=3, ky=3)
                rawDCGeneration['v_mp'] = pd.DataFrame(index=times, columns=locs)
                rawDCGeneration['v_mp'].values[sel] = interpolator( cellTemp.values[sel], poa_total[sel], grid=False )

            del photoCur, satCur, resSeries, resShunt, nNsVth, gen, poa_total, aoi, poaM, tempM, interpolator

        else:
            sotoParams = pvlib.pvsystem.calcparams_desoto(poa_global=poa_total[sel], 
                                                          temp_cell=cellTemp.values[sel], 
                                                          alpha_isc=module.alpha_sc, 
                                                          module_parameters=module, 
                                                          EgRef=module.EgRef, 
                                                          dEgdT=module.dEgdT)

            photoCur, satCur, resSeries, resShunt, nNsVth = sotoParams
            
            gen = pvlib.pvsystem.singlediode(photocurrent=photoCur, saturation_current=satCur, 
                                             resistance_series=resSeries, resistance_shunt=resShunt, 
                                             nNsVth=nNsVth)
            
            rawDCGeneration = OrderedDict()
            for k in ["p_mp", "v_mp", ]:
                rawDCGeneration[k] = pd.DataFrame(index=times, columns=locs)
                rawDCGeneration[k].values[sel] = gen[k]

            del photoCur, satCur, resSeries, resShunt, nNsVth, gen, poa_total, aoi
        
    del poa, cellTemp, tilt, amRel, sel
    #addTime("DC Sim")

    ## Simulate inverter interation
    if not inverter is None: 
        if sandiaInverterModel: 
            generation = genericSystem.snlinverter(v_dc=rawDCGeneration['v_mp']*modulesPerString, 
                                                   p_dc=rawDCGeneration['p_mp']*modulesPerString*stringsPerInverter)

        else: 
            generation = genericSystem.adrinverter(v_dc=rawDCGeneration['v_mp']*modulesPerString, 
                                                   p_dc=rawDCGeneration['p_mp']*modulesPerString*stringsPerInverter)
        
        # normalize to a single module
        generation = generation/modulesPerString/stringsPerInverter
    else:
        generation = rawDCGeneration["p_mp"]
    del rawDCGeneration
    #addTime("Inverter Sim")

    generation *= (1-loss) # apply a flat loss

    if totalSystemCapacity is None:
        output = generation/moduleCap # outputs in capacity-factor
    else:
        # output in whatever unit totalSystemCapacity is in
        output = generation/moduleCap*totalSystemCapacity

    #addTime("Finalize")

    # Done!
    #addTime("total",True)
    return output.fillna(0)

def simulatePVModule(locs, source, elev=300, module="WINAICO WSx-240P6", azimuth=180, tilt="ninja", totalSystemCapacity=None, tracking="fixed", interpolation="bilinear", loss=0.16, rackingModel="open_rack_cell_glassback", approximateSingleDiode=True, **kwargs):
    """
    Performs a simple PV simulation

    For module options, see: res.solarpower.ModuleLibrary
    For inverter options, see: res.solarpower.InverterLibrary


    interpolation options are:
        - near
        - bilinear
        - cubic
    """

    # Perform pre-sim procedures and unpack
    k = _presim(locs=locs, source=source, elev=elev, module=module, azimuth=azimuth, 
               tilt=tilt, totalSystemCapacity=totalSystemCapacity, tracking=tracking, 
               interpolation=interpolation, loss=loss, **kwargs)
    # return k

    # Do regular simulation procedure
    result = _simulation(approximateSingleDiode=approximateSingleDiode, **k)

    # Done! 
    return result


def simulatePVModuleDistribution(locs, tilts, source, elev=300, azimuths=180, occurrence=None, rackingModel="roof_mount_cell_glassback", module="LG Electronics LG370Q1C-A5", approximateSingleDiode=True, **kwargs):
    """
    Simulate a distribution of pv rooftops and combine results
    """
    # Arrange tilts and azimuths
    tilts = np.array(tilts)
    azimuths = np.array(azimuths)

    if azimuths.size ==1 and tilts.size > 1:
        azimuths = np.full_like(tilts, azimuths)
    if tilts.size ==1 and azimuths.size > 1:
        tilts = np.full_like(azimuths, tilts)

    if not azimuths.size == tilts.size:
        raise ResError("Tilts and azmiuths sizes do not match")

    # Arrange occurences
    if occurrence is None:
        occurrence = np.ones( tilts.size )
    else:
        occurrence = np.array(occurrence)

    if not occurrence.size == tilts.size:
        raise ResError("occurrence input does not have the correct shape")

    occurrence = occurrence / occurrence.sum()

    # Perform pre-sim procedures
    k = _presim(locs=locs, source=source, elev=elev, rackingModel=rackingModel, 
                azimuth=None, tilt=None, totalSystemCapacity=1, tracking='fixed', **kwargs)
    trash = k.pop("tilt")
    trash = k.pop("azimuth")

    # Do Simulation procedure multiple times
    result = 0
    for tilt, azimuth, occ in zip(tilts, azimuths, occurrence):
        result += _simulation(tilt=tilt, azimuth=azimuth, 
                              approximateSingleDiode=approximateSingleDiode, 
                              **k) * occ

    # Done!
    return result
