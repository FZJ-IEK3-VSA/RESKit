from ._util import *
from ._simulator import *
from ._powerCurveConvoluter import *
from res.weather.sources import MerraSource

from multiprocessing import Pool, cpu_count, Manager
from multiprocessing.managers import BaseManager

## Make helpers
class WSManager(BaseManager): pass
WSManager.register('MerraSource', MerraSource, exposed=["get", "loadWindSpeed"] )

Result = namedtuple("Result", "c o")
ncattr = {
    "time":{"longname":"time","units":"minutes since 1900-01-01 00:00:00","timezone":"GMT"},
    "locationID":{},
    "lat":{"longname":"latitude","units":"degrees-N"},
    "lon":{"longname":"longitude","units":"degrees-W"},
    "capfac":{"longname":"capacity-factor","units":""},
    "capacity":{"longname":"turbine-capacity","units":"kW"},
    "rotordiam":{"longname":"turbine-rotor-diameter","units":"m"},
    "hubHeight":{"longname":"turbine-hub-height","units":"m"},
    "pcKey":{"longname":"power-curve-key","units":""},
    "cutout":{"longname":"cutout-wind-speed","units":"m s-1"},
}

class PoollikeResult(object):
    def __init__(s, obj): s.obj = obj
    def get(s): return s.obj

##############################
## Extractor functions

def passfunc(*a,**k): return

def raw_combiner(r1, r2):
    if r1 is None and not r2 is None: return r2
    elif not r1 is None and r2 is None: return r1
    else: 
        output = pd.concat([r1.o, r2.o], axis=1)
        count = r1.c+r2.c
        return Result(count, output)

def capacityFactor_combiner(r1, r2):
    if r1 is None and not r2 is None: return r2
    elif not r1 is None and r2 is None: return r1
    else: 
        output = pd.concat([r1.o, r2.o], axis=0)
        count = r1.c+r2.c
        return Result(count, output)

def averageProduction_combiner(r1, r2):
    if r1 is None and not r2 is None: return r2
    elif not r1 is None and r2 is None: return r1
    else: 
        output = (r1.c*r1.o + r2.c*r2.o)/(r1.c+r2.c)
        count = r1.c+r2.c
        return Result(count, output)

def raw_finalizer(production, capacityFactor, **kwargs):
    output = production
    output.columns = [str(v) for v in output.columns]
    count = capacityFactor.size        
    return Result(count, output)

def capacityFactor_finalizer(production, capacityFactor, **kwargs):
    output = capacityFactor
    output.index = [str(v) for v in output.index]
    count = capacityFactor.size        
    return Result(count, output)
    
def averageProduction_finalizer(production, capacityFactor, **kwargs):
    output = production.mean(axis=1)
    count = capacityFactor.size
    return Result(count, output)

def writeCSV(output, const, mainData, **kwargs):
    fo = open(output,"w")
    if len(const)>0:
        fo.write("#### CONSTANTS ####\n")
        for k,v in const.items(): fo.write("%s,%s\n"%(k,str(v)))

    fo.write("#### RESULT-OUTPUT ####\n")
    mainData.to_csv(fo, **kwargs)

    fo.close()

def output_function(func):
    def wrapped(output, result, k):
        name,ext = splitext(basename(output))
        
        const = OrderedDict()
        const["TurbineCount"]=result.c
        dim1 = OrderedDict()
        dim2 = OrderedDict()

        if len(result.o.shape)==1:
            N1 = result.o.shape[0]
            N2 = -1
        elif len(result.o.shape)==2:
            N1,N2 = result.o.shape

        for key,v in k.items(): 
            if v is None: continue
            if isinstance(v, str) or isinstance(v, int) or isinstance(v, float) or (isinstance(v, np.ndarray) and v.shape==()):
                const[key] = str(v)
            else:
                if np.array([_v is None for _v in v]).all(): continue
                
                if (isinstance(v, np.ndarray)) and v.shape == (N1,):
                    dim1[key] = v
                elif (isinstance(v, np.ndarray)) and v.shape == (N2,):
                    dim2[key] = v

        func(output, result.o, ext=ext, const=const, dim1=dim1, dim2=dim2)
    return wrapped

@output_function
def raw_output(output, result, ext, const, dim1, dim2):
    const["extract"]="production"
    const["units"]="kWh"

    if ext == ".csv":
        writeCSV(output, const, result)

    elif ext == ".nc" or ext=="nc4":
        ds = nc.Dataset(output, mode="w")
        
        var = ds.createVariable("constants", "i4")
        var.setncatts(const)
        var[:] = len(const)

        # Set time dimension variables
        ds.createDimension("time", len(result.index))
        timeV = ds.createVariable("time", "u4", dimensions=("time",), contiguous=True)
        timeV.setncatts(ncattr["time"])
        times = result.index.tz_localize(None)
        timeV[:] = nc.date2num(times.to_pydatetime(), timeV.units)

        for k,v in dim1.items():
            var = ds.createVariable(k, v.dtype, dimensions=("time",))
            var.setncatts(ncattr[k])
            var[:] = v

        # Set location dimension variables
        locs = gk.Location.ensureLocation(result.columns, forceAsArray=True)
        ds.createDimension("locationID", len(locs))

        lon = ds.createVariable("lon", "f", dimensions=("locationID",))
        lon.setncatts(ncattr["lon"])
        lon[:] = [l.lon for l in locs]

        lat = ds.createVariable("lat", "f", dimensions=("locationID",))
        lat.setncatts(ncattr["lat"])
        lat[:] = [l.lat for l in locs]

        for k,v in dim2.items():
            var = ds.createVariable(k, v.dtype, dimensions=("locationID",))
            var.setncatts(ncattr[k])
            var[:] = v

        # Write main data
        production = ds.createVariable("production", "f4", dimensions=("time","locationID",))
        production.setncatts({"longname":"energy production","units":"kWh"})
        production[:] = result.values

        ds.close()
    else:
        raise RuntimeError("Extensions unknown or unavailable for this extraction")

@output_function
def capacityFactor_output(output, result, ext, const, dim1, dim2):
    const["extract"]="capacityFactor"
    const["units"]="% of max capacity"

    locs = gk.Location.ensureLocation(result.index, forceAsArray=True)
    
    finalResult = OrderedDict()
    finalResult["lat"] = np.array([l.lat for l in locs])
    finalResult["lon"] = np.array([l.lon for l in locs])
    finalResult["capfac"] = result.values
    for k,v in dim1.items():
        finalResult[k]=v

    # Do writing
    if ext == ".shp":
        gk.vector.createVector([l.geom for l in locs], fieldVals=finalResult, output=output)

    elif ext == ".csv":
        writeCSV(output, const, pd.DataFrame(finalResult), index=False)

    elif ext == ".nc" or ext=="nc4":
        ds = nc.Dataset(output, mode="w")
        
        var = ds.createVariable("constants", "i4")
        var.setncatts(const)
        var[:] = len(const)

        locs = gk.Location.ensureLocation(result.index, forceAsArray=True)
        ds.createDimension("locationID", len(locs))

        for k,v in finalResult.items():
            var = ds.createVariable(k, v.dtype, dimensions=("locationID",))
            var.setncatts(ncattr[k])
            var[:] = v

        ds.close()
    else:
        raise RuntimeError("Extensions unknown or unavailable for this extraction")

@output_function
def averageProduction_output(output, result, ext, const, dim1, dim2):
    const["extract"]="averageProduction"
    const["units"]="kWh"

    if ext == ".csv":
        result.name="production"
        writeCSV(output, const, result, header=True, index_label="time")

    elif ext==".nc" or ext==".nc4":
        ds = nc.Dataset(output, mode="w")

        var = ds.createVariable("constants", "i4")
        var.setncatts(const)
        var[:] = len(const)

        ds.createDimension("time", len(result.index))
        timeV = ds.createVariable("time", "u4", dimensions=("time",), contiguous=True)
        timeV.setncatts(ncattr["time"])
        times = result.index.tz_localize(None)

        timeV[:] = nc.date2num(times.to_pydatetime(), timeV.units)

        production = ds.createVariable("avgProduction", "f", dimensions=("time",))
        production.setncatts({"longname":"average energy production per turbine","units":"kWh"})
        production[:] = result.values

        ds.close()

class Extractor(object):
    def __init__(s, method, outputPath=None):
        s.method = method
        s.skipFinalOutput=False
        if method=="p" or method == "production" or method == "raw":
            s._combine = "raw_combiner"
            s._finalize = "raw_finalizer"
            s._output = "raw_output"
        elif method=="cf" or method == "capacityFactor":
            s._combine = "capacityFactor_combiner"
            s._finalize = "capacityFactor_finalizer"
            s._output = "capacityFactor_output"
        elif method=="ap" or method == "averageProduction":
            s._combine = "averageProduction_combiner"
            s._finalize = "averageProduction_finalizer"
            s._output = "averageProduction_output"
        elif method=="batch":
            try:
                r = outputPath%10
            except:
                raise RuntimeError("output path string should handle integer formating")

            s.outputPath = outputPath
            s.skipFinalOutput=True
            s._combine = "passfunc"
            s._finalize = "passfunc"
            s._output = "passfunc"
        else:
            raise ResError('''Don't know extraction type. Try using... 
                'production' (or just 'p')
                'capacityFactor' (or just 'cf')
                'averageProduction' ('wa')"
                ''')

    def combine(s, *a,**k): return globals()[s._combine](*a,**k)
    def finalize(s, *a,**k): return globals()[s._finalize](*a,**k)
    def output(s, *a,**k): return globals()[s._output](*a,**k)

##################################################################
## Make a typical simulator function
def simulateLocations(**k):
    # Unpack inputs
    pickleable = k.pop('pickleable')
    hubHeight = k.pop('hubHeight')
    lctype = k.pop('lctype')
    pcKey = k.pop('pcKey')
    capacity = k.pop('capacity')
    landcover = k.pop('landcover')
    locations = k.pop('locations')
    rotordiam = k.pop('rotordiam')
    cutout = k.pop('cutout')
    gwa = k.pop('gwa')
    minCF = k.pop("minCF")
    wsSource = k.pop('wsSource')
    powerCurve = k.pop('powerCurve')
    verbose = k.pop('verbose')
    extractor = k.pop('extractor')
    gid = k.pop("gid")
                            
    if verbose: 
        startTime = dt.now()
        globalStart = k.get("globalStart", startTime)
        print(" %s: Starting at +%.2fs"%(str(gid), (startTime-globalStart).total_seconds()))
    
    # read wind speeds
    locations = Location.ensureLocation(locations)

    if len(locations) == 0 : 
        if verbose: print( " %s: No locations found"%(str(gid)))
        return None

    if pickleable: Location.makePickleable(locations)
    ws = wsSource.get("windspeed", locations, forceDataFrame=True)

    # spatially adjust windspeeds
    if not k.get("skipLRA",False):
        ws = windutil.adjustLraToGwa( ws, locations, longRunAverage=MerraSource.LONG_RUN_AVERAGE_50M_SOURCE, gwa=gwa)

    # apply wind speed corrections to account (somewhat) for local effects not captured on the MERRA context
    factors = (1-0.3)*(1-np.exp(-0.2*ws))+0.3 # dampens lower wind speeds
    ws = factors*ws
    factors = None

    # Get roughnesses from Land Cover
    if lctype == "clc":
        winRange = int(k.get("lcRange",0))
        roughnesses = windutil.roughnessFromCLC(landcover, locations, winRange=winRange)
    else:
        lcVals = gk.raster.extractValues(landcover, locations).data
        roughnesses = windutil.roughnessFromLandCover(lcVals, lctype)

    # Project WS to hub height
    ws = windutil.projectByLogLaw(ws, measuredHeight=50, targetHeight=hubHeight, roughness=roughnesses)

    # do simulations
    loss = 0.04
    if pcKey is None:
        capacityGeneration = simulateTurbine(ws, powerCurve=powerCurve, loss=loss)
    else:
        capacityGeneration = pd.DataFrame(-1*np.ones(ws.shape),index=ws.index, columns=ws.columns)
        for key in np.unique(pcKey):
            tmp = simulateTurbine(ws.iloc[:,pcKey==key], powerCurve[key], loss=loss)
            capacityGeneration.update( tmp )
        if (capacityGeneration.values<0).any(): raise RuntimeError("Some placements were not evaluated")

    capacityFactor = capacityGeneration.mean(axis=0)
    production = capacityGeneration*capacity

    if verbose:
        endTime = dt.now()
        simSecs = (endTime - startTime).total_seconds()
        globalSecs = (endTime - globalStart).total_seconds()
        print(" %s: Finished %d turbines +%.2fs (%.2f turbines/sec)"%(str(gid), len(locations), globalSecs, len(locations)/simSecs))
    
    # Apply Capacity Factor Filter
    if minCF > 0:
        sel = capacityFactor >= minCF
        capacityFactor = capacityFactor[sel]
        production = production.ix[:,sel]

    # Done!
    if extractor.method == "batch":
        outputVars = OrderedDict()
        outputVars["lctype"] = lctype
        outputVars["minCF"] = minCF
        outputVars["pcKey"] = pcKey
        outputVars["hubHeight"] = hubHeight
        outputVars["capacity"] = capacity
        outputVars["rotordiam"] = rotordiam
        outputVars["cutout"] = cutout
        
        result = raw_finalizer(production, capacityFactor)
        raw_output(extractor.outputPath%gid, result, outputVars)
        output = None
    else:
        output = extractor.finalize(production, capacityFactor)
    return output

##################################################################
## Distributed Wind production from a Merra wind source
def WindWorkflow(placements, merra, landcover, gwa, hubHeight=None, powerCurve=None, turbine=None, capacity=None, rotordiam=None, cutout=None, lctype="clc", extract="averageProduction", output=None, minCF=0, jobs=1, batchSize=None, verbose=True, **kwargs):
    """
    WARNING: DOC STRING NEEDS UPDATING!!

    Apply the wind simulation method developed by Severin Ryberg, Dilara Caglayan, and Sabrina Schmitt. This method 
    works as follows for a given simulation point:
        1. The nearest time-series in the provided MERRA climate data is extracted
            * reads windspeeds at 50 meters
        2. The time series is adjusted so that the long-run-average (all MERRA data at this point) matches the value
           given by the Global Wind Atlas at the simulation point
        3. A roughness factor is assumed from the land cover and is used to project the wind speeds to the indicated
           hub height
        4. Low windspeeds are depressed slightly, ending around 10 m/s
        5. The wind speeds are fed through the power-curve of the indicated turbine
            * The power curve has been convoluted to incorporate a stochastic spread of windspeeds using
        6. A 4% loss is applied

    inputs:
        placements: a list of (lon,lat) coordinates to simulate

        merraSource - str : A path to the MERRA data which will be used for the simulation
            * MUST have the fields 'U50M' and 'V50M'

        turbine: The turbine to simulate
            str : An indicator from the TurbineLibrary (res.windpower.TurbineLibrary)
            [(float, float), ...] : An explicit performance curve given as (windspeed, power output) pairs

        landcover - str : The path to the land cover source

        gwa - str : The path to the global wind atlas mean windspeeds (at 50 meters)

        hubHeight - float : The hub height to simulate at

        lctype - str: The land cover type to use
            * Options are "clc", "globCover", and "modis"

        extract - str: Determines the extraction method and the form of the returned information
            * Options are:
                "production" - returns the timeseries production for each location
                "capacityFactor" - returns only the resulting capacity factor for each location
                "averageProduction" - returns the average time series of all locations

        cfMin - float : The minimum capacity factor to accept
            * Must be between 0..1

        jobs - int : The number of parallel jobs

        batchSize - int : The number of placements to simulate in each job
            * Use this to tune performance to your specific machine

        verbose - bool : False means silent, True means noisy
    """
    if verbose: 
        startTime = dt.now()
        print("Starting at: %s"%str(startTime))

    if jobs==1: # use only a single process
        cpus = 1
        pool = None
        useMulti = False
    elif jobs > 1: # uses multiple processes (equal to jobs)
        cpus = jobs
        useMulti = True
    else: # uses multiple processes (equal to the number of available processors - jobs)
        cpus = cpu_count()-jobs
        if cpus <=0: raise ResError("Bad jobs count")
        useMulti = True
    
    extractor = Extractor(extract, outputPath=output)
    if not turbine is None: powerCurve=turbine

    ### Determine the total extent which will be simulated (also make sure the placements input is okay)
    if verbose: print("Arranging placements at +%.2fs"%((dt.now()-startTime).total_seconds()))
    if isinstance(placements, str): # placements is a path to a point-type shapefile
        placements = gk.vector.extractAsDataFrame(placements, outputSRS='latlon')
        placements["lat"] = placements.geom.apply(lambda x: x.GetY())
        placements["lon"] = placements.geom.apply(lambda x: x.GetX())
    
    if isinstance(placements, pd.DataFrame):
        if "powerCurve" in placements.columns and powerCurve is None: powerCurve = placements.powerCurve.values
        if "turbine" in placements.columns and powerCurve is None: powerCurve = placements.turbine.values
        if "hubHeight" in placements.columns and hubHeight is None: hubHeight = placements.hubHeight.values
        if "capacity" in placements.columns and capacity is None: capacity = placements.capacity.values
        if "rotordiam" in placements.columns and rotordiam is None: rotordiam = placements.rotordiam.values
        if "cutout" in placements.columns and cutout is None: cutout = placements.cutout.values

        placements = placements[["lon","lat"]].values

    placements = Location.ensureLocation(placements, forceAsArray=True)

    hubHeight = None if hubHeight is None else np.array(hubHeight)
    capacity = None if capacity is None else np.array(capacity)
    rotordiam = None if rotordiam is None else np.array(rotordiam)
    cutout = None if cutout is None else np.array(cutout)

    allLats = np.array([p.lat for p in placements])
    allLons = np.array([p.lon for p in placements])

    latMin = allLats.min()
    latMax = allLats.max()
    lonMin = allLons.min()
    lonMax = allLons.max()

    if verbose: print("Pre-loading windspeeds at +%.2fs"%((dt.now()-startTime).total_seconds()))
    totalExtent = gk.Extent((lonMin,latMin,lonMax,latMax,), srs=LATLONSRS)
    
    # Setup manager if needed
    if useMulti:
        manager = WSManager()
        manager.start()
        wsSource = manager.MerraSource(path=merra, bounds=Bounds(*totalExtent.pad(1).xyXY))
    else:
        wsSource = MerraSource(path=merra, bounds=totalExtent.pad(1))

    wsSource.loadWindSpeed(height=50)

    ### Convolute turbine
    if verbose: print("Convolving power curves at +%.2fs"%( (dt.now()-startTime).total_seconds()) )
    
    pcKey = None
    if powerCurve is None: # no turbine given, so a synthetic turbine will need to be constructed
        if capacity is None and rotordiam is None:
            raise RuntimeError("powerCurve, capacity, and rotordiam cannot all be None")

        # Compute specific capacity
        #  - Round to the nearest 10 to save memory and time for convolution
        specificCapacity = np.array(capacity*1000/(np.pi*rotordiam**2/4))
        specificCapacity = np.round(specificCapacity, -1).astype(int)

        if specificCapacity.size == 1:
            powerCurve = SyntheticPowerCurve( specificCapacity=specificCapacity, cutout=cutout)
            pcKey = "%d:%d"%(specificCapacity,25 if cutout is None else cutout)
        else:
            powerCurve = dict()
            pcKey = []
            if isinstance(cutout, int) or isinstance(cutout, float) or cutout is None: 
                cutout = [cutout]*specificCapacity.size
            for sp,co in zip(specificCapacity,cutout):
                key = "%d:%d"%(sp,25 if co is None else co)
                pcKey.append( key )
                if not key in powerCurve.keys():
                    powerCurve[key] = SyntheticPowerCurve( sp, co)

    elif isinstance(powerCurve, str):
        pcKey = powerCurve
        capacity = TurbineLibrary.ix[powerCurve].Capacity
        powerCurve = TurbineLibrary.ix[powerCurve].PowerCurve

    else: # powerCurve is either a (ws,power) list or is a list of turbine names
        if isinstance(powerCurve[0],str): # assume entire list is a list of names
            pcKey = powerCurve
            powerCurve = dict()
            capacity = []

            for name in pcKey:
                capacity.append(TurbineLibrary.ix[name].Capacity)
                if not name in powerCurve.keys():
                    powerCurve[name] = TurbineLibrary.ix[name].PowerCurve
        
        else: # powerCurve is a single power curve definition
            tmp = np.array(powerCurve)
            powerCurve = PowerCurve(tmp[:,0], tmp[:,1])
            pcKey = None

        turbine = None # remove turbine so it doesn't show up in output
    pcKey = pcKey if pcKey is None or isinstance(pcKey, str) else np.array(pcKey)

    if isinstance(powerCurve, dict):
        if verbose: 
            print("   Convolving %d power curves..."%(len(powerCurve)))

        if useMulti:
            pool = Pool(cpus)
            res = []
        
            for k,v in powerCurve.items():
                kwargs = dict(stdScaling=0.1, stdBase=0.6, powerCurve=v)
                res.append((k,pool.apply_async(convolutePowerCurveByGuassian, (), kwargs)))

            for k,r in res:
                powerCurve[k] = r.get()
            
            pool.close()
            pool.join()
            pool = None
        else:
            for k,v in powerCurve.items():
                powerCurve[k] = convolutePowerCurveByGuassian(stdScaling=0.1, stdBase=0.6, powerCurve=v)
    else:
        powerCurve = convolutePowerCurveByGuassian(stdScaling=0.1, stdBase=0.6, powerCurve=powerCurve )

    ### Do simulations
    # Check inputs
    if hubHeight is None:
        raise RuntimeError("hubHeight has not been provided")

    ### initialize simulations
    if verbose: print("Initializing simulations at +%.2fs"%((dt.now()-startTime).total_seconds()))
    if useMulti: pool = Pool(cpus)
    simGroups = []
    I = np.arange(placements.shape[0])
    if batchSize is None and cpus==1: # do everything in one big batch
        simGroups.append( I )
    elif cpus>1 and (batchSize is None or placements.shape[0] < batchSize): # Split evenly to all cpus
        for simPlacements in np.array_split( I, cpus):
            simGroups.append( simPlacements )
    else: # split the area in to equal size groups, and simulate one group at a time
        for simPlacements in np.array_split(I, len(placements)//(batchSize/cpus)):
            simGroups.append( simPlacements )

    if verbose: 
        print("Simulating %d groups at +%.2fs"%(len(simGroups), (dt.now()-startTime).total_seconds() ))

    results = []
    if useMulti: Location.makePickleable(placements)
    for i,sel in enumerate(simGroups):
        # Construct arguments for each submission
        inputs = {}
        inputs["wsSource"]=wsSource
        inputs["landcover"]=landcover
        inputs["lctype"]=lctype
        inputs["gwa"]=gwa
        inputs["minCF"]=minCF
        inputs["verbose"]=verbose
        inputs["extractor"]=extractor
        inputs["powerCurve"] = powerCurve
        inputs["pcKey"] = None if (pcKey is None or isinstance(pcKey, str)) else pcKey[sel]
        inputs["gid"]=i

        if verbose:
            inputs["globalStart"]=startTime

        def add(val,name):
            if isinstance(val, list): val = np.array(val)
            if isinstance(val , np.ndarray) and val.size>1: inputs[name] = val[sel]
            else: inputs[name] = val

        add(placements, "locations")
        add(capacity, "capacity")
        add(hubHeight, "hubHeight")
        add(rotordiam, "rotordiam")
        add(cutout, "cutout")
        
        if useMulti:
            inputs["pickleable"]=True
            results.append( pool.apply_async( simulateLocations, (), inputs ))
        else:    
            inputs["pickleable"]=False
            results.append( PoollikeResult(simulateLocations(**inputs)) )

    # Read each result as it becomes available
    finalResult = None
    for r in results:
        tmp = r.get()
        if tmp is None: continue
        finalResult = extractor.combine(finalResult, r.get())
       
    if useMulti:
        # Do some cleanup
        pool.close()
        pool.join()

    if verbose:
        endTime = dt.now()
        totalSecs = (endTime - startTime).total_seconds()
        print("Finished simulating %d turbines (%d surviving) at +%.2fs (%.2f turbines/sec)"%(len(placements), finalResult.c, totalSecs, len(placements)/totalSecs))

    ### Give the results
    if not output is None and not extractor.skipFinalOutput:
        if verbose:
            endTime = dt.now()
            totalSecs = (endTime - startTime).total_seconds()
            print("Writing output at +%.2fs"%totalSecs)
        outputVars = OrderedDict()
        outputVars["lctype"] = lctype
        outputVars["minCF"] = minCF
        outputVars["pcKey"] = pcKey
        outputVars["hubHeight"] = hubHeight
        outputVars["capacity"] = capacity
        outputVars["rotordiam"] = rotordiam
        outputVars["cutout"] = cutout
        
        extractor.output(output, finalResult, outputVars)

    #### Formulate result
    if verbose:
        endTime = dt.now()
        totalSecs = (endTime - startTime).total_seconds()
        print("Done at +%.2fs!"%totalSecs)

    if extract == "batch":  return

    outputResult = finalResult.o
    outputResult.TurbineCount = finalResult.c
    return outputResult
    