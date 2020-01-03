import numpy as np
import pandas as pd
from collections import namedtuple, OrderedDict
from reskit.util.util_ import *
from reskit.weather import NCSource
import pvlib
from types import FunctionType
from datetime import datetime as dt
from os.path import isfile
from scipy.interpolate import RectBivariateSpline
from scipy.special import lambertw

class _SolarLibrary:
    """Provides access to module and inverter parameters from the CEC and Sandia databases.

    **This simply exposes the associated functions from PVLib, and is only here for convenience**
    
    TODO: Add citation!
    """
    def __init__(self):
        self._cecmods = None
        self._sandiamods = None
        self._cecinverters = None
        self._sandiainverters = None

    def modules(self, group='cec'): 
        name = "_"+group.lower()+"mods"
        if getattr(self, name) is None:
            # setattr(self, name, pvlib.pvsystem.retrieve_sam(group+"mod"))
            setattr(self, name, pd.read_csv(join(DATADIR,"modules","sam-library-cec-modules-2017-6-5.csv"), skiprows=[1,2], index_col=0))
        return getattr(self, name)

    def inverters(self, group='sandia'): 
        name = "_"+group.lower()+"inverters"
        if getattr(self, name) is None:
            setattr(self, name, pvlib.pvsystem.retrieve_sam(group+"inverter").T)
        return getattr(self, name)

SolarLibrary = _SolarLibrary()

def my_sapm_celltemp(poa_global, wind_speed, temp_air, model='open_rack_cell_glassback'):
    """ Cell temp function slightly adapted from the PVLib version 

    TODO: Add citation!
    """
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

def spencerSolPos(times, lat, lon):
    """Compute solar position from the method proposed by Spencer

    TODO: Add citation!
    """
    lat_r = lat * np.pi/180
    
    times += pd.Timedelta(hours=lon/180*12 )
    
    xx = 2*np.pi * (times.dayofyear-1)/365
    d = 0.006918 -0.399912*np.cos(xx) +0.07257*np.sin(xx) -0.006758*np.cos(2*xx) +0.000907*np.sin(2*xx) -0.006758*np.cos(3*xx) +0.00907*np.sin(3*xx)
    #d = d*np.pi/180
    
    x = 360 / 365 * (times.dayofyear-81) *np.pi/180
    #solarTime = (times.hour*60+times.minute+times.second/60) + (meridian-lon)*4*s + 9.87*np.sin(2*x) - 7.53*np.cos(x) -1.5*np.sin(x)
    solarTime = (times.hour*60+times.minute+times.second/60) + 9.87*np.sin(2*x) - 7.53*np.cos(x) -1.5*np.sin(x)
    
    hourAngle = (solarTime/60 - 12)*15
    hourAngle_r = hourAngle * np.pi/180
    
    zenith = np.degrees(np.arccos( np.cos(lat_r)*np.cos(d)*np.cos(hourAngle_r)+np.sin(lat_r)*np.sin(d) ))
    
    # Azimuth!
    num = np.sin(hourAngle_r)
    denom = (np.cos(hourAngle_r) * np.sin(lat_r) - np.tan(d) * np.cos(lat_r))
    gamma = np.degrees(np.arctan2(num, denom))
    azimuth = (gamma % 360 + 180) % 360
    
    return dict(apparent_zenith=zenith, apparent_elevation=90-zenith, azimuth=azimuth)

def myDisc(ghi, zenith, I0, am, pressure):
    """Diffuse irradiance estimation using the DISC model.

    Slightly Adapted from PVLIB to better fit my use-case

    TODO: Add citation to PVLib and DISC!
    """
    # this is the I0 calculation from the reference
    I0h = I0 * np.cos(np.radians(zenith))

    am = am*pressure/101325  # convert to absolute air mass

    kt = ghi / I0h
    kt = np.maximum(kt, 0)
    # powers of kt will be used repeatedly, so compute only once
    kt2 = kt * kt  # about the same as kt ** 2
    kt3 = kt2 * kt  # 5-10x faster than kt ** 3

    bools = (kt <= 0.6)
    a = np.where(bools,
                 0.512 - 1.56*kt + 2.286*kt2 - 2.222*kt3,
                 -5.743 + 21.77*kt - 27.49*kt2 + 11.56*kt3)
    b = np.where(bools,
                 0.37 + 0.962*kt,
                 41.4 - 118.5*kt + 66.05*kt2 + 31.9*kt3)
    c = np.where(bools,
                 -0.28 + 0.932*kt - 2.048*kt2,
                 -47.01 + 184.2*kt - 222.0*kt2 + 73.81*kt3)

    delta_kn = a + b * np.exp(c*am)

    Knc = 0.866 - 0.122*am + 0.0121*am**2 - 0.000653*am**3 + 1.4e-05*am**4
    Kn = Knc - delta_kn

    dni = Kn * I0

    dni = np.where((zenith > 87) | (ghi < 0) | (dni < 0), 0, dni)

    output = OrderedDict()
    output['dni'] = dni
    output['kt'] = kt
    output['airmass'] = am

    return output

dirintCoeffs = pvlib.irradiance._get_dirint_coeffs()
def myDirint(ghi, zenith, pressure, use_delta_kt_prime, temp_dew, amRel, I0):
    """Diffuse irradiance estimation using the Dirint model.

    Slightly Adapted from PVLIB to better fit my use-case

    TODO: Add citation to PVLib and Dirint!
    """
    disc_out = myDisc(ghi=ghi, zenith=zenith, I0=I0, am=amRel, pressure=pressure)

    dni = disc_out['dni']
    kt = disc_out['kt']
    am = disc_out['airmass']

    kt_prime = kt / (1.031 * np.exp(-1.4 / (0.9 + 9.4 / am)) + 0.1)
    kt_prime = np.minimum(kt_prime, 0.82)  # From SRRL code
    # wholmgren:
    # the use_delta_kt_prime statement is a port of the MATLAB code.
    # I am confused by the abs() in the delta_kt_prime calculation.
    # It is not the absolute value of the central difference.
    # current implementation requires that kt_prime is a Series
    if use_delta_kt_prime:
        vforward  = np.abs((kt_prime - np.roll(kt_prime, 1, axis=0)))
        vbackward = np.abs((kt_prime - np.roll(kt_prime,-1, axis=0)))
        delta_kt_prime = 0.5*(vforward+vbackward)
    else:
        delta_kt_prime = np.full_like(ghi, -1)

    if not temp_dew is None:
        w = np.exp(0.07 * temp_dew - 0.075)
    else:
        w = np.full_like(ghi,-1)

    # @wholmgren: the following bin assignments use MATLAB's 1-indexing.
    # Later, we'll subtract 1 to conform to Python's 0-indexing.

    # Create kt_prime bins
    kt_prime_bin = np.zeros_like(ghi, dtype=np.uint8)
    kt_prime_bin[(kt_prime >= 0) & (kt_prime < 0.24)] = 1
    kt_prime_bin[(kt_prime >= 0.24) & (kt_prime < 0.4)] = 2
    kt_prime_bin[(kt_prime >= 0.4) & (kt_prime < 0.56)] = 3
    kt_prime_bin[(kt_prime >= 0.56) & (kt_prime < 0.7)] = 4
    kt_prime_bin[(kt_prime >= 0.7) & (kt_prime < 0.8)] = 5
    kt_prime_bin[(kt_prime >= 0.8) & (kt_prime <= 1)] = 6

    # Create zenith angle bins
    zenith_bin = np.zeros_like(ghi, dtype=np.uint8)
    zenith_bin[(zenith >= 0)  & (zenith < 25)] = 1
    zenith_bin[(zenith >= 25) & (zenith < 40)] = 2
    zenith_bin[(zenith >= 40) & (zenith < 55)] = 3
    zenith_bin[(zenith >= 55) & (zenith < 70)] = 4
    zenith_bin[(zenith >= 70) & (zenith < 80)] = 5
    zenith_bin[(zenith >= 80)] = 6

    # Create the bins for w based on dew point temperature
    w_bin = np.zeros_like(ghi, dtype=np.uint8)
    w_bin[(w >= 0) & (w < 1)] = 1
    w_bin[(w >= 1) & (w < 2)] = 2
    w_bin[(w >= 2) & (w < 3)] = 3
    w_bin[(w >= 3)] = 4
    w_bin[(w == -1)] = 5

    # Create delta_kt_prime binning.
    delta_kt_prime_bin = np.zeros_like(ghi, dtype=np.uint8)
    delta_kt_prime_bin[(delta_kt_prime >= 0) & (delta_kt_prime < 0.015)] = 1
    delta_kt_prime_bin[(delta_kt_prime >= 0.015) &
                       (delta_kt_prime < 0.035)] = 2
    delta_kt_prime_bin[(delta_kt_prime >= 0.035) & (delta_kt_prime < 0.07)] = 3
    delta_kt_prime_bin[(delta_kt_prime >= 0.07) & (delta_kt_prime < 0.15)] = 4
    delta_kt_prime_bin[(delta_kt_prime >= 0.15) & (delta_kt_prime < 0.3)] = 5
    delta_kt_prime_bin[(delta_kt_prime >= 0.3) & (delta_kt_prime <= 1)] = 6
    delta_kt_prime_bin[delta_kt_prime == -1] = 7

    # subtract 1 to account for difference between MATLAB-style bin
    # assignment and Python-style array lookup. 
    kt_prime_bin = kt_prime_bin.flatten()
    zenith_bin = zenith_bin.flatten()
    w_bin = w_bin.flatten()
    delta_kt_prime_bin = delta_kt_prime_bin.flatten()
    
    dirint_coeffs = dirintCoeffs[kt_prime_bin-1, zenith_bin-1,
                                 delta_kt_prime_bin-1, w_bin-1]

    # convert unassigned bins to nan
    dirint_coeffs = np.where((kt_prime_bin == 0) | (zenith_bin == 0) |
                             (w_bin == 0) | (delta_kt_prime_bin == 0),
                             np.nan, dirint_coeffs)
    
    dni *= dirint_coeffs.reshape(dni.shape)
    return dni

def ensureSeries(var, locs):
    """*Internal function to ensure a given object is a pandas series"""
    if isinstance(var, pd.Series): pass
    elif isinstance(var, float) or isinstance(var, int):
        var = pd.Series([var,]*locs.count, index=locs)
    elif isinstance(var, str): 
        var = gk.raster.extractValues(var, locs).data
    else:
        var = pd.Series(var, index=locs)

    return var

def frankCorrectionFactors(ghi, dni_extra, times, solarElevation):
    """Applies the proposed transmissivity-based irradiance corrections to COSMO
    data based on Frank et al.

    TODO: Add citation to Frank!
    """
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

    del clearSkyFactors, cloudyFactors, e, months, sigmoid, transmissivity

    return totalCorrectionFactor

def locToTilt(locs, convention="latitude*0.76", **k):
    """Simple system tilt estimators based off latitude and longitude coordinates

    TODO: Add citation to Pfenninger for 'ninja' method
    
    **The role of this function probably needs to be readdressed since it tries to
        do a lot of things currently**
    """
    locs = gk.LocationSet(locs)

    if convention=="ninja": 
        lats = locs.lats
        tilt = np.zeros( lats.size ) + 40

        s = lats <= 25
        tilt[ s ] = lats[s]*0.87

        s = np.logical_and(lats > 25, lats <= 50)
        tilt[ s ] = (lats[s]*0.76)+3.1
    
    elif convention=='bestTilt':
        try:
            tilt = gk.raster.interpolateValues(join(DATADIR, "bestTilt_europe_int.tif"), locs, **k)
        except:
            raise ResError("Could not load best tilt for all locations. They may not have been included in the preprocessed data")

    elif isfile(convention):
        tilt = gk.raster.interpolateValues(convention, locs, **k)

    else:
        try:
            tilt = eval(convention, {}, {"latitude":locs.lats})
        except:
            raise ResError("Failed to apply tilt convention")

    tilt=pd.Series(tilt, index=locs[:])

    return tilt

def _presim(locs, source, elev=300, module="WINAICO WSx-240P6", azimuth=180, tilt="ninja", totalSystemCapacity=None, tracking="fixed", modulesPerString=1, inverter=None, stringsPerInverter=1, rackingModel='open_rack_cell_glassback', airmassModel='kastenyoung1989', transpositionModel='perez', cellTempModel="sandia", generationModel="single-diode", inverterModel="sandia", interpolation="bilinear", loss=0.18, trackingGCR=2/7, trackingMaxAngle=60, frankCorrection=False, ghiScaling=None):

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
    if isinstance(source, NCSource):
        times = source.timeindex

        idx = source.loc2Index(locs, asInt=False)
        k = dict( locations=locs, interpolation=interpolation, forceDataFrame=True, _indicies=idx )

        ghi = source.get("ghi", **k)
        if ghiScaling:
            merraAvg = gk.raster.interpolateValues( source.LONG_RUN_AVERAGE_GHI_SOURCE, locs, mode="linear-spline" )*24/1000 # make into kW/m2/day
            worldBankAvg = gk.raster.interpolateValues( ghiScaling, locs )
            scaling = worldBankAvg / merraAvg
            scaling[np.isnan(scaling)] = 0.9
            scaling[locs.lats>=59.9] = 0.9 # Default scaling value defined from near-edge average in Norway, Sweden, and Finland
            scaling[locs.lats<=-55.1] = 0.9 # Assumed to be the same as above

            #print("SCALING GHI", scaling.mean())

            ghi *= scaling
        dhi = source.get("dhi", **k) if "dhi" in source.data else None
        dni = source.get("dni", **k) if "dni" in source.data else None

        windspeed = source.get("windspeed", **k)
        pressure = source.get("pressure", **k)
        air_temp = source.get("air_temp", **k)
        if "dew_temp" in source.data: dew_temp = source.get("dew_temp", **k)
        else: dew_temp = None

        if "albedo" in source.data: albedo = source.get("albedo", **k)
        else: albedo = 0.2

    else: # source should be a dictionary
        times = source["times"]
        
        ghi = source["ghi"]
        dhi = source["dhi"] if "dhi" in source else None
        dni = source["dni"] if "dni" in source else None

        windspeed = source["windspeed"]
        pressure = source["pressure"]
        air_temp = source["air_temp"]

        if "dew_temp" in source: dew_temp = source["dew_temp"]
        else: dew_temp = None
        if "albedo" in source:  albedo = source["albedo"]
        else:  albedo = 0.2

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
    else:
        genericSystem = None
    ### Check the (potentially) uniquely defined inputs
    if not totalSystemCapacity is None:
        totalSystemCapacity = ensureSeries(totalSystemCapacity, locs)
    
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

    ## MAKE TIME SELECTION
    goodTimes = (solpos["apparent_zenith"] < 92).any(axis=1).values
    solpos['azimuth']= solpos['azimuth'][goodTimes].values
    solpos['apparent_zenith'] = solpos['apparent_zenith'][goodTimes].values
    solpos['apparent_elevation']= solpos['apparent_elevation'][goodTimes].values
    
    # filter GHI, pressure, windspeed, and temperature
    ghi = ghi[goodTimes].values
    if not dni is None: dni = dni[goodTimes].values
    if not dhi is None: dhi = dhi[goodTimes].values
    windspeed= windspeed[goodTimes].values
    pressure = pressure[goodTimes].values
    air_temp = air_temp[goodTimes].values
    if not dew_temp is None: dew_temp = dew_temp[goodTimes].values

    # DNI Extraterrestrial
    dni_extra = pvlib.irradiance.extraradiation(times[goodTimes], 1370, method='spencer').values
    if len(ghi.shape) > 1:
        dni_extra = np.broadcast_to(dni_extra.reshape((ghi.shape[0],1)), ghi.shape)

    # Apply Frank corrections when dealing with COSMO data?
    if frankCorrection:
        ghi *= frankCorrectionFactors(ghi, dni_extra, times[goodTimes], solpos["apparent_elevation"])

    # Airmass
    amRel = np.zeros_like(ghi)
    s = solpos["apparent_zenith"]<89
    amRel[s] = pvlib.atmosphere.relativeairmass(solpos["apparent_zenith"][s], model=airmassModel)

    # Compute DHI or DNI
    if dni is None:
        dni = myDirint(ghi=ghi, zenith=solpos["apparent_zenith"], pressure=pressure, 
                       use_delta_kt_prime=True, amRel=amRel, I0=dni_extra,
                       temp_dew=dew_temp)
    # elif dni is None and not dhi is None:
    #     dni = (ghi - dhi)/np.sin( np.radians(solpos["apparent_elevation"]))

    if dhi is None:
        dhi = ghi - dni*np.sin( np.radians(solpos["apparent_elevation"]))
        dhi[dhi<0] = 0

    return dict(singleAxis=singleAxis,
                genericSystem=genericSystem,
                tilt=tilt if isinstance(tilt, np.ndarray) else tilt[locs[:]].values,
                module=module,
                azimuth=azimuth if isinstance(azimuth, np.ndarray) else azimuth[locs[:]].values,
                inverter=inverter,
                moduleCap=moduleCap,
                modulesPerString=modulesPerString,
                stringsPerInverter=stringsPerInverter,
                loss=loss,
                
                locs=locs,
                times=times,
                goodTimes=goodTimes,

                dni=dni,
                ghi=ghi,
                dhi=dhi,
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

def my_i_from_v(resistance_shunt, resistance_series, nNsVth, voltage,
                saturation_current, photocurrent):
    '''
    Internal function to estimate module output current at a given voltage. 

    * Slightly adapted from PVLIB to better fit my use case

    TODO: Add citation to PVLib

    **Should be hidden**
    '''

    # This transforms Gsh=1/Rsh, including ideal Rsh=np.inf into Gsh=0., which
    #  is generally more numerically stable
    conductance_shunt = 1./resistance_shunt

    # Ensure that we are working with read-only views of numpy arrays
    # Turns Series into arrays so that we don't have to worry about
    #  multidimensional broadcasting failing
    Gsh, Rs, a, V, I0, IL = (conductance_shunt, resistance_series, nNsVth,
                            voltage, saturation_current, photocurrent)

    # LambertW argument, cannot be float128, may overflow to np.inf
    argW = Rs*I0/(a*(Rs*Gsh + 1.)) * np.exp((Rs*(IL + I0) + V) / (a*(Rs*Gsh + 1.)))

    # lambertw typically returns complex value with zero imaginary part
    # may overflow to np.inf
    lambertwterm = lambertw(argW).real

    # Eqn. 2 in Jain and Kapoor, 2004
    #  I = -V/(Rs + Rsh) - (a/Rs)*lambertwterm + Rsh*(IL + I0)/(Rs + Rsh)
    # Recast in terms of Gsh=1/Rsh for better numerical stability.
    I = (IL + I0 - V*Gsh) / (Rs*Gsh + 1.) - (a/Rs)*lambertwterm

    return I


def my_golden_sect_DataFrame(params, VL, VH, func):
    '''
    Slightly adapted from pvlib to better fit my needs

    TODO: Add citation to PVLib

    **Should be hidden**
    '''
    df = params

    df['VH'] = VH
    df['VL'] = VL

    err = df['VH'] - df['VL']
    errflag = True
    iterations = 0

    finalVals = np.zeros_like(df["r_sh"])
    index = np.arange(finalVals.size, dtype=int)

    ALLKEYS = ["r_sh", "nNsVth", "i_0", "i_l", "VL", "VH"]
    while errflag:
        phi = ((np.sqrt(5)-1)/2)*(df['VH']-df['VL'])
        df['V1'] = df['VL'] + phi
        df['V2'] = df['VH'] - phi

        df['f1'] = func(df, 'V1')
        df['f2'] = func(df, 'V2')

        df['SW_Flag'] = df['f1'] > df['f2']

        df['VL'] = df['V2']* df['SW_Flag'] + df['VL']*(~df['SW_Flag'])
        df['VH'] = df['V1']*~df['SW_Flag'] + df['VH']*( df['SW_Flag'])

        err = df['V1'] - df['V2']

        ### ADDED BY ME to maybe save a little bit of time...
        isgood = np.abs(err) <= 0.01
        isbad = ~isgood
        if isgood.any():
           finalVals[index[isgood]] = df["V1"][isgood]
           
           for k in ALLKEYS: 
               df[k] = df[k][isbad]
           index = index[isbad]
        
        errflag = isbad.any()
        ####
        iterations += 1

        if iterations > 50:
            raise Exception("EXCEPTION:iterations exeeded maximum (50)")

    return finalVals #df['V1']

def my_pwr_optfcn(df, loc):
    '''
    Slightly adapted from pvlib to better fit my needs

    TODO: Add citation to PVLib

    **Should be hidden**
    '''
    I = np.zeros_like(df['r_sh'])

    I = my_i_from_v(df['r_sh'], df['r_s'], df['nNsVth'], 
                          df[loc], df['i_0'], df['i_l'])

    return I * df[loc]

def mysinglediode(photocurrent, saturation_current, resistance_series,
                resistance_shunt, nNsVth, ivcurve_pnts=None):
    '''
    Slightly adapted from pvlib to better fit my needs

    TODO: Add citation to PVLib

    **Should be hidden**
    '''

    # Compute open circuit voltage
    v_oc = pvlib.pvsystem.v_from_i(resistance_shunt, resistance_series, nNsVth, 0.,
                                   saturation_current, photocurrent)

    params = {'r_sh': resistance_shunt,
              'r_s': resistance_series,
              'nNsVth': nNsVth,
              'i_0': saturation_current,
              'i_l': photocurrent}


    v_mp = my_golden_sect_DataFrame(params, 0, v_oc, my_pwr_optfcn)

    # Invert the Power-Current curve. Find the current where the inverted power
    # is minimized. This is i_mp. Start the optimization at v_oc/2
    i_mp = my_i_from_v(resistance_shunt, resistance_series, nNsVth, v_mp,
                    saturation_current, photocurrent)

    out = OrderedDict()
    out['i_mp'] = i_mp
    out['v_mp'] = v_mp
    out['p_mp'] = v_mp*i_mp

    if isinstance(photocurrent, pd.Series) and not ivcurve_pnts:
        out = pd.DataFrame(out, index=photocurrent.index)
    return out

def _simulation(singleAxis, genericSystem, tilt, module, azimuth, inverter, moduleCap, modulesPerString, stringsPerInverter, locs, times, dni, ghi, dhi, amRel, solpos, pressure, air_temp, windspeed, dni_extra, sandiaCellTemp, transpositionModel, totalSystemCapacity, sandiaGenerationModel, loss, approximateSingleDiode, goodTimes):

    # Get tilt and azimuths
    if singleAxis:
        axis_tilt = tilt
        axis_azimuth = azimuth

        tilt = np.zeros_like(ghi) #pd.DataFrame(index=times, columns=locs)
        azimuth = np.zeros_like(ghi) #pd.DataFrame(index=times, columns=locs)

        for loc, at, aa, i in zip(locs, axis_tilt, axis_azimuth, range(locs.count)):
            
            # These fail if it isn't a pandas type :/
            zen = pd.Series(solpos["apparent_zenith"][:,i],)
            azi = pd.Series(solpos["azimuth"][:,i],)

            tmp = pvlib.tracking.singleaxis(apparent_zenith= zen, apparent_azimuth=azi, axis_tilt=at, axis_azimuth=aa, 
                                            max_angle=genericSystem.max_angle, backtrack=genericSystem.backtrack,
                                            gcr=genericSystem.gcr)
            tilt[:,i] = tmp["surface_tilt"].copy()
            azimuth[:,i] = tmp["surface_azimuth"].copy()

        del axis_azimuth, axis_tilt, tmp

    # Angle of Incidence
    aoi = pvlib.irradiance.aoi(tilt, azimuth, solpos['apparent_zenith'], solpos['azimuth'])
    
    # Compute Total irradiation
    poa = pvlib.irradiance.total_irrad(surface_tilt=tilt,
                                       surface_azimuth=azimuth,
                                       apparent_zenith=solpos['apparent_zenith'],
                                       azimuth=solpos['azimuth'],
                                       dni=dni, ghi=ghi, dhi=dhi,
                                       dni_extra=dni_extra,
                                       model=transpositionModel,
                                       airmass=amRel)

    del dni, ghi, dhi, azimuth, dni_extra, solpos
    
    # Cell temp
    if sandiaCellTemp:
        cellTemp = my_sapm_celltemp(poa['poa_global'], windspeed, air_temp)
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
            if maxpoa > 2000: raise RuntimeError("Why is POA so huge???")
            _poa = np.concatenate([np.logspace(-1, np.log10(maxpoa/10), 20, endpoint=False), 
                                   np.linspace(maxpoa/10, maxpoa, 80)])
            _temp = np.linspace(cellTemp[sel].min(), cellTemp[sel].max(), 100)
            poaM, tempM = np.meshgrid(_poa, _temp)

            sotoParams = pvlib.pvsystem.calcparams_desoto(poa_global=poaM.flatten(), 
                                                          temp_cell=tempM.flatten(), 
                                                          alpha_isc=module.alpha_sc, 
                                                          module_parameters=module, 
                                                          EgRef=module.EgRef, 
                                                          dEgdT=module.dEgdT)

            photoCur, satCur, resSeries, resShunt, nNsVth = sotoParams
            gen = mysinglediode(photocurrent=photoCur, saturation_current=satCur, 
                                             resistance_series=resSeries, resistance_shunt=resShunt, 
                                             nNsVth=nNsVth)

            rawDCGeneration=OrderedDict()

            interpolator = RectBivariateSpline( _temp, _poa, gen['p_mp'].reshape(poaM.shape), kx=3, ky=3)
            rawDCGeneration['p_mp'] = pd.DataFrame(index=times[goodTimes], columns=locs)
            rawDCGeneration['p_mp'].values[sel] = interpolator( cellTemp[sel], poa_total[sel], grid=False )

            if not inverter is None: 
                interpolator = RectBivariateSpline( _temp, _poa, gen['v_mp'].reshape(poaM.shape), kx=3, ky=3)
                rawDCGeneration['v_mp'] = pd.DataFrame(index=times[goodTimes], columns=locs)
                rawDCGeneration['v_mp'].values[sel] = interpolator( cellTemp[sel], poa_total[sel], grid=False )

            del photoCur, satCur, resSeries, resShunt, nNsVth, gen, poa_total, aoi, poaM, tempM, interpolator

        else:
            sotoParams = pvlib.pvsystem.calcparams_desoto(poa_global=poa_total[sel], 
                                                          temp_cell=cellTemp[sel], 
                                                          alpha_isc=module.alpha_sc, 
                                                          module_parameters=module, 
                                                          EgRef=module.EgRef, 
                                                          dEgdT=module.dEgdT)

            photoCur, satCur, resSeries, resShunt, nNsVth = sotoParams
            
            gen = mysinglediode(photocurrent=photoCur, saturation_current=satCur, 
                                             resistance_series=resSeries, resistance_shunt=resShunt, 
                                             nNsVth=nNsVth)
            
            rawDCGeneration = OrderedDict()
            for k in ["p_mp", "v_mp", ]:
                rawDCGeneration[k] = pd.DataFrame(index=times[goodTimes], columns=locs)
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

    # Arrange back to the full time series
    output = output.reindex(times).fillna(0)

    # Done!
    #addTime("total",True)
    return output

def simulatePVModule(locs, source, elev=300, module="WINAICO WSx-240P6", azimuth=180, tilt="ninja", totalSystemCapacity=None, tracking="fixed", interpolation="bilinear", loss=0.18, rackingModel="open_rack_cell_glassback", approximateSingleDiode=True, **kwargs):
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
