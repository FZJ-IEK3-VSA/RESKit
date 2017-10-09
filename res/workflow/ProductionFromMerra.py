from res.util import *
from res.weather import MerraSource, windutil
from res.production import wind

from collections import namedtuple
from multiprocessing import Pool, cpu_count
from datetime import datetime as dt

##################################################################
## Make some typical simulator functions
Result = namedtuple("Result", "count output")
def simulateLocations(locations, extent, wsSource, clcSource, gwaSource, turbine, hubHeight, extract, verbose=True, **kwargs):
    if verbose: 
        startTime = dt.now()
        gid = kwargs.get("gid",extent)
        globalStart = kwargs.get("globalStart", startTime)
        print(" %s: Starting at +%.2fs"%(str(gid), (startTime-globalStart).total_seconds()))

    if not extent is None:
        extent = gk.Extent(extent, srs=LATLONSRS)
    
    # make merra source and load data
    if not isinstance(wsSource, MerraSource):
        print(" building new source")
        if extent is None: raise ResError("extent cannot be None when building a new source")
        wsSource = MerraSource(path=wsSource, bounds=extent.pad(1))
        wsSource.loadWindSpeed(height=50)

    # read windspeeds
    if isinstance(locations, str): # locations points to a point-type shapefile
        locations = Location.ensureLocation([g for g,a in gk.vector.extractFeatures( locations, geom=extent.box, outputSRS=LATLONSRS )])
    else:
        locations = Location.ensureLocation(locations)

    if len(locations) == 0 : 
        if verbose: print( " %s: No locations found"%(str(gid)))
        return None
    ws = wsSource.get("windspeed", locations)

    # adjust ws 
    ws = windutil.adjustContextMeanToGwa( ws, locations, contextMean=wsSource.GWA50_CONTEXT_MEAN_SOURCE, 
            gwa=gwaSource)

    # Get roughnesses from clc
    roughnesses = windutil.roughnessFromCLC(clcSource, locations)

    # do simulations
    res = wind.simulateTurbine(ws, performance=turbine, measuredHeight=50, hubHeight=hubHeight, roughness=roughnesses)

    if verbose:
        endTime = dt.now()
        simSecs = (endTime - startTime).total_seconds()
        globalSecs = (endTime - globalStart).total_seconds()

        print(" %s: Finished %d turbines +%.2fs (%.2f turbines/sec)"%(str(gid), len(locations), globalSecs, len(locations)/simSecs))

    # Done!
    if extract=="p" or extract == "production":
        return Result(count=len(locations), output=res.production)
    elif extract=="cf" or extract == "capacityFactor":
        return Result(count=len(locations), output=res.capacityFactor)
    elif extract=="wa" or extract == "weightedAverage":
        return Result(count=len(locations), output=res.production.mean(axis=1))
    else:
        raise ResError("Don't know extraction type. Try using 'production' (or just 'p'), 'capacityFactor' (or just 'cf'), or 'weightedAverage' ('wa')")

##################################################################
## Distributed Wind production from a Merra wind source
def windProductionFromMerraSource(placements, merraSource, turbine, clcSource, gwaSource, hubHeight, jobs=1, batchSize=None, extract="weightedAverage", verbose=True, **kwargs):
    if verbose: 
        startTime = dt.now()
        print("Starting at: %s"%str(startTime))
    ### Determine the total extent which will be simulated (also make sure the placements input is okay)
    if isinstance(placements, str): # placements is a path to a point-type shapefile
        placements = gk.vector.extractFeatures(placements, onlyGeom=True)

    placements = Location.ensureLocation(placements, forceAsArray=True)

    allLats = np.array([p.lat for p in placements])
    allLons = np.array([p.lon for p in placements])

    latMin = allLats.min()
    latMax = allLats.max()
    lonMin = allLons.min()
    lonMax = allLons.max()

    if verbose: print("Loading windspeed at +%.2fs"%((dt.now()-startTime).total_seconds()))
    totalExtent = gk.Extent((lonMin,latMin,lonMax,latMax,), srs=LATLONSRS)
    wsSource = MerraSource(path=merraSource, bounds=totalExtent.pad(1))
    wsSource.loadWindSpeed(height=50)

    ### initialize simulations
    if verbose: print("Initializing simulations at +%.2fs"%((dt.now()-startTime).total_seconds()))

    if jobs==1: # use only a single process
        pass
    elif jobs > 1: # uses multiple processes (equal to jobs)
        pool = Pool(jobs)
    else: # uses multiple processes (equal to the number of available processors - jobs)
        pool = Pool( cpu_count()-jobs )
    
    simGroups = []
    if batchSize is None: # do everything in one big batch
        simGroups.append( placements )
    else: # split the area in to eual size extent boxes, and simulate one box at a time
        for simPlacements in np.array_split(placements, placements.size/batchSize):
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
            '''


    ### Do simulations
    totalC = 0
    if verbose: 
        print("Simulating %d groups at +%.2fs"%(len(simGroups), (dt.now()-startTime).total_seconds() ))

    result = None
    if jobs==1:
        for i,locs in enumerate(simGroups):

            if verbose:
                _kwargs = dict(globalStart=startTime, gid=i)
                _kwargs.update(kwargs)
            else:
                _kwargs = kwargs
            
            tmp = simulateLocations(locations=locs, extent=None, wsSource=wsSource, clcSource=clcSource, 
                                    gwaSource=gwaSource, turbine=turbine, hubHeight=hubHeight, 
                                    extract=extract, verbose=verbose, **_kwargs)

            if tmp is None:continue
            totalC+=tmp.count
            result = combiner(result, tmp)
    else:
        results_ = []
        # submit all jobs to the queue
        for i,locs in enumerate(simGroups):

            if verbose:
                _kwargs = dict(globalStart=startTime, gid=i)
                _kwargs.update(kwargs)
            else:
                _kwargs = kwargs

            results_.append( 
                pool.apply_async(
                    simulateLocations, (
                        locs, None, wsSource, clcSource, gwaSource, turbine, hubHeight, extract, verbose
                    ), _kwargs
                )
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
    