from res.util import *
from res.weather import windutil
from res.weather.sources import MerraSource
from res.production import wind

from collections import namedtuple
from multiprocessing import Pool, cpu_count, Manager
from multiprocessing.managers import BaseManager
from datetime import datetime as dt
from types import FunctionType

class WSManager(BaseManager): pass
WSManager.register('MerraSource', MerraSource, exposed=["get", "loadWindSpeed"] )

##################################################################
## Make some typical simulator functions
Result = namedtuple("Result", "count output")
def simulateLocations(locations, wsSource, lcSource, lcType, gwaSource, performance, capacity, hubHeight, extract, cfMin=0, pickleable=False, verbose=True, **kwargs):
    if verbose: 
        startTime = dt.now()
        gid = kwargs["gid"]
        globalStart = kwargs.get("globalStart", startTime)
        print(" %s: Starting at +%.2fs"%(str(gid), (startTime-globalStart).total_seconds()))
    
    # read wind speeds
    locations = Location.ensureLocation(locations)

    if len(locations) == 0 : 
        if verbose: print( " %s: No locations found"%(str(gid)))
        return None

    if pickleable:
        Location.makePickleable(locations)
    ws = wsSource.get("windspeed", locations)

    # spatially adjust windspeeds
    ws = windutil.adjustLraToGwa( ws, locations, longRunAverage=MerraSource.LONG_RUN_AVERAGE_50M_SOURCE, gwa=gwaSource)
    #ws = windutil.adjustContextMeanToGwa( ws, locations, contextMean=MerraSource.GWA50_CONTEXT_MEAN_SOURCE , gwa=gwaSource)

    # apply wind speed corrections to account (somewhat) for local effects not captured on the MERRA context
    factors = (1-0.3)*(1-np.exp(-0.2*ws))+0.3 # dampens lower wind speeds
    ws = factors*ws
    factors = None

    # Get roughnesses from Land Cover
    if lcType == "clc":
        winRange = int(kwargs.get("lcRange",0))
        roughnesses = windutil.roughnessFromCLC(lcSource, locations, winRange=winRange)
    else:
        lcVals = gk.raster.extractValues(lcSource, locations).data
        roughnesses = windutil.roughnessFromLandCover(lcVals, lcType)

    # do simulations
    res = wind.simulateTurbine(ws, performance=performance, capacity=capacity, measuredHeight=50, hubHeight=hubHeight, roughness=roughnesses, loss=0.04)
    capacityFactor = res.capacityFactor
    production = res.production

    if verbose:
        endTime = dt.now()
        simSecs = (endTime - startTime).total_seconds()
        globalSecs = (endTime - globalStart).total_seconds()
        print(" %s: Finished %d turbines +%.2fs (%.2f turbines/sec)"%(str(gid), len(locations), globalSecs, len(locations)/simSecs))
    
    # Apply Capacity Factor Filter
    sel = capacityFactor >= cfMin
    if not sel.all():
        capacityFactor = capacityFactor[sel]
        production = production.ix[:,sel]

    # Done!
    if extract=="p" or extract == "production":
        output = production
        output.columns = [str(v) for v in output.columns]

    elif extract=="cf" or extract == "capacityFactor":
        output = capacityFactor
        output.index = [str(v) for v in output.index]

    elif extract=="ap" or extract == "averageProduction":
        output = production.mean(axis=1)
        #output.columns = [str(v) for v in output.columns]

    else:
        raise ResError("Don't know extraction type. Try using 'production' (or just 'p'), 'capacityFactor' (or just 'cf'), or 'averageProduction' ('ap')")
    print(sel.sum())
    return Result(count=sel.sum(), output=output)

##################################################################
## Distributed Wind production from a Merra wind source
def windProductionFromMerraSource(placements, merraSource, turbine, lcSource, gwaSource, hubHeight, lcType="clc", extract="averageProduction", cfMin=0, jobs=1, batchSize=None, verbose=True, **kwargs):
    """
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
            str : An indicator from the TurbineLibrary (res.production.wind.TurbineLibrary)
            [(float, float), ...] : An explicit performance curve given as (windspeed, power output) pairs

        lcSource - str : The path to the land cover source

        gwaSource - str : The path to the global wind atlas mean windspeeds (at 50 meters)

        hubHeight - float : The hub height to simulate at

        lcType - str: The land cover type to use
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
        pool = None
        useManager = False
    elif jobs > 1: # uses multiple processes (equal to jobs)
        pool = Pool(jobs)
        useManager = True
    else: # uses multiple processes (equal to the number of available processors - jobs)
        cpus = cpu_count()-jobs
        if cpus <=0: raise ResError("Bad jobs count")
        pool = Pool( cpus )
        useManager = True
    
    ### Determine the total extent which will be simulated (also make sure the placements input is okay)
    if verbose: print("Arranging placements at +%.2fs"%((dt.now()-startTime).total_seconds()))
    if isinstance(placements, str): # placements is a path to a point-type shapefile
        placements = np.array([ (placement.GetX(), placement.GetY()) for placement in gk.vector.extractFeatures(placements, onlyGeom=True, outputSRS='latlon')])

    placements = Location.ensureLocation(placements, forceAsArray=True)

    allLats = np.array([p.lat for p in placements])
    allLons = np.array([p.lon for p in placements])

    #allLons = placements[:,0] 
    #allLats = placements[:,1]

    latMin = allLats.min()
    latMax = allLats.max()
    lonMin = allLons.min()
    lonMax = allLons.max()

    if verbose: print("Pre-loading windspeeds at +%.2fs"%((dt.now()-startTime).total_seconds()))
    totalExtent = gk.Extent((lonMin,latMin,lonMax,latMax,), srs=LATLONSRS)
    

    # Setup manager if needed
    if useManager:
        manager = WSManager()
        manager.start()
        wsSource = manager.MerraSource(path=merraSource, bounds=Bounds(*totalExtent.pad(1).xyXY))
    else:
        wsSource = MerraSource(path=merraSource, bounds=totalExtent.pad(1))

    wsSource.loadWindSpeed(height=50)

    #del wsSource.data["winddir"]
    #del wsSource.data["U50M"]
    #del wsSource.data["V50M"]

    ### initialize simulations
    if verbose: print("Initializing simulations at +%.2fs"%((dt.now()-startTime).total_seconds()))

    simGroups = []
    if batchSize is None: # do everything in one big batch
        simGroups.append( placements )

    else: # split the area in to equal size groups, and simulate one group at a time
        for simPlacements in np.array_split(placements, len(placements)//batchSize+1):
            tmp = []
            #for i in simPlacements:
            #    tmp.append( (i[0], i[1]) )
            #simGroups.append( tmp )

            simGroups.append( simPlacements )

    ### Set up combiner 
    if extract=="p" or extract == "production":
        def combiner(result, newResult):
            if result is None: return newResult
            else: return Result(count=result.count+newResult.count,
                                output=pd.concat([result.output, newResult.output], axis=1))

    elif extract=="cf" or extract == "capacityFactor":
        def combiner(result, newResult):
            if result is None: return newResult
            else: return Result(count=result.count+newResult.count,
                                output=pd.concat([result.output, newResult.output], axis=0))
        
    elif extract=="ap" or extract == "averageProduction":
        def combiner(result, newResult):
            if result is None: return newResult
            else: 
                return Result( count=result.count+newResult.count,
                               output=(result.count*result.output + newResult.count*newResult.output)/(result.count+newResult.count))
    else:
        raise ResError('''Don't know extraction type. Try using... 
            'production' (or just 'p')
            'capacityFactor' (or just 'cf')
            'averageProduction' ('wa')"
            ''')

    ### Convolute turbine
    if verbose: print("Convolving power curve at +%.2fs"%( (dt.now()-startTime).total_seconds()) )

    if isinstance(turbine, str):
        performance = np.array(wind.TurbineLibrary.ix[turbine].Performance)
        capacity = wind.TurbineLibrary.ix[turbine].Capacity
    else:
        performance = np.array(turbine)
        capacity = performance[:,1].max()

    performance = wind.convolutePowerCurveByGuassian(stdScaling=0.1, stdBase=0.6, performance=performance )

    ### Do simulations
    totalC = 0
    if verbose: 
        print("Simulating %d groups at +%.2fs"%(len(simGroups), (dt.now()-startTime).total_seconds() ))

    # Construct arguments
    staticKwargs = OrderedDict()
    staticKwargs["wsSource"]=wsSource
    staticKwargs["lcSource"]=lcSource
    staticKwargs["lcType"]=lcType

    staticKwargs["gwaSource"]=gwaSource
    staticKwargs["performance"]=performance
    staticKwargs["capacity"]=capacity
    staticKwargs["hubHeight"]=hubHeight

    staticKwargs["extract"]=extract
    staticKwargs["cfMin"]=cfMin
    staticKwargs["verbose"]=verbose

    staticKwargs.update(kwargs)

    result = None
    if jobs==1:
        for i,locs in enumerate(simGroups):

            if verbose:
                staticKwargs["globalStart"]=startTime
                staticKwargs["gid"]=i
            
            staticKwargs["pickleable"]=False
            staticKwargs["locations"] = locs

            tmp = simulateLocations(**staticKwargs)

            if tmp is None:continue
            totalC+=tmp.count
            result = combiner(result, tmp)
    else:
        results_ = []
        # submit all jobs to the queue
        for i,locs in enumerate(simGroups):

            if verbose:
                staticKwargs["globalStart"]=startTime
                staticKwargs["gid"]=i
            
            staticKwargs["pickleable"]=True

            Location.makePickleable(locs)
            staticKwargs["locations"] = locs

            results_.append( 
                pool.apply_async( simulateLocations, (), staticKwargs.copy() )
            )

        # Read each result as it becomes available
        for r in results_:
            tmp = r.get()
            if tmp is None: continue
            totalC+=tmp.count
            result = combiner(result, tmp)

        # Do some cleanup
        pool.close()
        pool.join()


    if verbose:
        endTime = dt.now()
        totalSecs = (endTime - startTime).total_seconds()
        print("Finished simulating %d turbines (%d surviving) at +%.2fs (%.2f turbines/sec)"%(len(placements), totalC, totalSecs, len(placements)/totalSecs))


    ### Give the results
    return result
    
