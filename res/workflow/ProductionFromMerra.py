from res.util import *
from res.weather import MerraSource, windutil
from res.production import wind

from collections import namedtuple
from multiprocessing import Pool, cpu_count
from datetime import datetime as dt

##################################################################
## Make some typical simulator functions
Result = namedtuple("Result", "count output")
def simulateLocations(locations, extent, wsSource, clcSource, turbine, hubHeight, extract, verbose=True, **kwargs):
    if verbose: 
        startTime = dt.now()
        gid = kwargs.get("gid",extent)
        globalStart = kwargs.get("globalStart", startTime)
        print(" %s: Starting at +%.2fs"%(str(gid), (startTime-globalStart).total_seconds()))

    extent = gk.Extent(extent, srs=LATLONSRS)
    # make merra source and load data
    if not isinstance(wsSource, MerraSource):
        print(" building new source")
        wsSource = MerraSource(path=wsSource, bounds=extent.pad(1))
        wsSource.loadWindSpeed(height=50)

    # read windspeeds
    if isinstance(locations, str): # locations points to a point-type shapefile
        locations = [g for g,a in gk.vector.extractFeatures( locations, geom=extent.box, outputSRS=LATLONSRS )]
    else:
        locations = ensureGeom(locations)

    if len(locations) == 0 : 
        if verbose: print( " %s: No locations found"%(str(gid)))
        return None
    ws = wsSource.get("windspeed", locations)

    # adjust ws 
    ws = windutil.adjustContextMeanToGwa( ws, locations, contextMean=wsSource.GWA50_CONTEXT_MEAN_SOURCE, 
            gwa=r"D:\Data\weather\global_wind_atlas\WS_050m_global_wgs84_mean_trimmed_europe.tif")

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
    else:
        raise ResError("Don't know extraction type. Try using 'production' (or just 'p') or 'capacityFactor' (or just 'cf')")

##################################################################
## Distributed Wind production from a Merra wind source
def windProductionFromMerraSource(placements, merraSource, turbine, clcSource, hubHeight, jobs=1, batchSize=None, extract="production", verbose=True, **kwargs):
    if verbose: 
        startTime = dt.now()
        print("Starting at: %s"%str(startTime))
    ### Determine the total extent which will be simulated (also make sure the placements input is okay)
    if isinstance(placements, str): # placements is a path to a point-type shapefile
        totalExtent = gk.Extent.fromVector(placements).castTo(LATLONSRS)
        
        latMin = totalExtent.yMin
        latMax = totalExtent.yMax
        lonMin = totalExtent.xMin
        lonMax = totalExtent.xMax
    else:
        placements = np.array(ensureLoc(ensureList(placements))) # ensure placements are a list of locations

        latMin = placements[:,1].min()
        latMax = placements[:,1].max()
        lonMin = placements[:,0].min()
        lonMax = placements[:,0].max()

    if verbose: print("Loading windspeed at +%.2fs"%((dt.now()-startTime).total_seconds()))
    totalExtent = gk.Extent((lonMin,latMin,lonMax,latMax,), srs=LATLONSRS)
    wsSource = MerraSource(path=merraSource, bounds=totalExtent.pad(1))
    wsSource.loadWindSpeed(height=50)

    ### initialize simulations
    if verbose: print("Initializing simulations at +%.2fs"%((dt.now()-startTime).total_seconds()))
    result = None
    if jobs==1: # use only a single process
        pass
    elif jobs > 1: # uses multiple processes (equal to jobs)
        pool = Pool(jobs)
    else: # uses multiple processes (equal to the number of available processors - jobs)
        pool = Pool( cpu_count()-jobs )
    
    simGroups = []
    if batchSize is None: # do everything in one big batch
        simGroups.append( (placements, (lonMin, latMin, lonMax, latMax)) )
    else: # split the area in to eual size extent boxes, and simulate one box at a time
        
        for lat in np.arange(latMin,latMax, batchSize):
            for lon in np.arange(lonMin,lonMax, batchSize):
                simExtent = (lon, lat, lon+batchSize, lat+batchSize)
                if isinstance(placements, str):
                    simPlacements = placements
                else:
                    sel = placements[:,0] >= lon
                    sel = sel & (placements[:,0] < lon+batchSize)
                    sel = sel & (placements[:,1] >= lat)
                    sel = sel & (placements[:,1] < lat+batchSize)
                    simPlacements = placements[:,sel]

                simGroups.append( (simPlacements, simExtent))

    ### Do simulations
    totalC = 0
    if verbose: 
        print("Simulating %d groups at +%.2fs"%(len(simGroups), (dt.now()-startTime).total_seconds() ))

    if jobs==1:
        for i,sg in enumerate(simGroups):
            locs,ext = sg

            if verbose:
                _kwargs = dict(globalStart=startTime, gid=i)
                _kwargs.update(kwargs)
            else:
                _kwargs = kwargs
            
            tmp = simulateLocations(locations=locs, extent=ext, wsSource=wsSource, clcSource=clcSource, 
                                    turbine=turbine, hubHeight=hubHeight, extract=extract, verbose=verbose, 
                                    **_kwargs)

            if tmp is None:continue
            totalC += tmp.count

            if result is None: 
                result = tmp.output
                axis = 0 if isinstance(result, pd.Series) else 1
            else: 
                result = pd.concat( [result, tmp.output], axis=axis )
    else:
        results_ = []
        # submit all jobs to the queue
        for i,sg in enumerate(simGroups):
            locs,ext = sg

            if verbose:
                _kwargs = dict(globalStart=startTime, gid=i)
                _kwargs.update(kwargs)
            else:
                _kwargs = kwargs

            results_.append( 
                pool.apply_async(
                    simulateLocations, (
                        locs, ext, wsSource, clcSource, turbine, hubHeight, extract, verbose
                    ), _kwargs
                )
            )

        # Read each result as it becomes available
        for r in results_:
            tmp = r.get()
        
            if tmp is None: continue
            totalC += tmp.count

            if result is None: 
                result = tmp.output
                axis = 0 if isinstance(result, pd.Series) else 1
            else: 
                result = pd.concat( [result, tmp.output], axis=axis )

        # Do some cleanup
        pool.close()
        pool.join()

    if verbose:
        endTime = dt.now()
        totalSecs = (endTime - startTime).total_seconds()
        print("Finished simulating %d turbines at +%.2fs (%.2f turbines/sec)"%(totalC, totalSecs, totalC/totalSecs))


    ### Give the results
    return result
    