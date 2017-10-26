from res.util import *
from res.weather import windutil
from res.weather.sources import MerraSource
from res.production import wind

from collections import namedtuple
from multiprocessing import Pool, cpu_count, Manager
from multiprocessing.managers import BaseManager
from datetime import datetime as dt


class WSManager(BaseManager): pass
WSManager.register('MerraSource', MerraSource, exposed=["get", "loadWindSpeed"] )

##################################################################
## Make some typical simulator functions
Result = namedtuple("Result", "count output")
def simulateLocations(locations, wsSource, clcSource, gwaSource, performance, capacity, hubHeight, extract, verbose=True, **kwargs):
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

    ws = wsSource.get("windspeed", locations)

    # spatially adjust windspeeds
    ws = windutil.adjustLraToGwa( ws, locations, longRunAverage=MerraSource.LONG_RUN_AVERAGE_50M_SOURCE, gwa=gwaSource)
    #ws = windutil.adjustContextMeanToGwa( ws, locations, contextMean=MerraSource.GWA50_CONTEXT_MEAN_SOURCE , gwa=gwaSource)


    # apply wind speed corrections to account (somewhat) for local effects not captured on the MERRA context
    factors = (1-0.3)*(1-np.exp(-0.2*ws))+0.3 # dampens lower wind speeds
    ws = factors*ws
    factors = None

    # Get roughnesses from CLC
    winRange = int(kwargs.get("clcRange",0)/100)
    roughnesses = windutil.roughnessFromCLC(clcSource, locations, winRange=winRange)

    # do simulations
    res = wind.simulateTurbine(ws, performance=performance, capacity=capacity, measuredHeight=50, hubHeight=hubHeight, roughness=roughnesses, loss=0.04)

    if verbose:
        endTime = dt.now()
        simSecs = (endTime - startTime).total_seconds()
        globalSecs = (endTime - globalStart).total_seconds()
        print(" %s: Finished %d turbines +%.2fs (%.2f turbines/sec)"%(str(gid), len(locations), globalSecs, len(locations)/simSecs))
    
    # Done!

    if extract=="p" or extract == "production":
        output = res.production
        output.columns = [str(v) for v in output.columns]

    elif extract=="cf" or extract == "capacityFactor":
        output = res.capacityFactor
        output.index = [str(v) for v in output.index]

    elif extract=="wa" or extract == "weightedAverage":
        output = res.production.mean(axis=1)
        output.columns = [str(v) for v in output.columns]

    else:
        raise ResError("Don't know extraction type. Try using 'production' (or just 'p'), 'capacityFactor' (or just 'cf'), or 'weightedAverage' ('wa')")

    
    return Result(count=len(locations), output=output)

##################################################################
## Distributed Wind production from a Merra wind source
def windProductionFromMerraSource(placements, merraSource, turbine, clcSource, gwaSource, hubHeight, jobs=1, batchSize=None, extract="weightedAverage", verbose=True, **kwargs):
    if verbose: 
        startTime = dt.now()
        print("Starting at: %s"%str(startTime))

    if jobs==1: # use only a single process
        pool = None
        useManager = True
    elif jobs > 1: # uses multiple processes (equal to jobs)
        pool = Pool(jobs)
        useManager = True
    else: # uses multiple processes (equal to the number of available processors - jobs)
        cpus = cpu_count()-jobs
        if cpus <=0: raise ResError("Bad jobs count")
        pool = Pool( cpus )
        useManager = False
    
    ### Determine the total extent which will be simulated (also make sure the placements input is okay)
    if verbose: print("Arranging placements at +%.2fs"%((dt.now()-startTime).total_seconds()))
    if isinstance(placements, str): # placements is a path to a point-type shapefile
        placements = np.array([ (placement.GetX(), placement.GetY()) for placement in gk.vector.extractFeatures(placements, onlyGeom=True, outputSRS='latlon')])

    #placements = Location.ensureLocation(placements, forceAsArray=True)

    #allLats = np.array([p.lat for p in placements])
    #allLons = np.array([p.lon for p in placements])

    allLons = placements[:,0] 
    allLats = placements[:,1]

    latMin = allLats.min()
    latMax = allLats.max()
    lonMin = allLons.min()
    lonMax = allLons.max()

    if verbose: print("Pre-loading windspeeds at +%.2fs"%((dt.now()-startTime).total_seconds()))
    totalExtent = gk.Extent((lonMin,latMin,lonMax,latMax,), srs=LATLONSRS)
    
    with WSManager() as manager:

        wsSource = manager.MerraSource(path=merraSource, bounds=Bounds(*totalExtent.pad(1).xyXY))
        #else:
        #    wsSource = MerraSource(path=merraSource, bounds=totalExtent.pad(1))
    
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
                for i in simPlacements:
                    tmp.append( (i[0], i[1]) )
                simGroups.append( tmp )
    
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
            
        elif extract=="wa" or extract == "weightedAverage":
            def combiner(result, newResult):
                if result is None: return newResult
                else: 
                    return Result( count=result.count+newResult.count,
                                   output=(result.count*result.output + newResult.count*newResult.output)/(result.count+newResult.count))
        else:
            raise ResError('''Don't know extraction type. Try using... 
                'production' (or just 'p')
                'capacityFactor' (or just 'cf')
                'weightedAverage' ('wa')"
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
        staticKwargs["clcSource"]=clcSource
    
        staticKwargs["gwaSource"]=gwaSource
        staticKwargs["performance"]=performance
        staticKwargs["capacity"]=capacity
        staticKwargs["hubHeight"]=hubHeight
    
        staticKwargs["extract"]=extract
        staticKwargs["verbose"]=verbose
    
        staticKwargs.update(kwargs)
    
        result = None
        if jobs==1:
            for i,locs in enumerate(simGroups):
    
                if verbose:
                    staticKwargs["globalStart"]=startTime
                    staticKwargs["gid"]=i
    
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
        print("Finished simulating %d turbines at +%.2fs (%.2f turbines/sec)"%(totalC, totalSecs, totalC/totalSecs))


    ### Give the results
    return result
    
