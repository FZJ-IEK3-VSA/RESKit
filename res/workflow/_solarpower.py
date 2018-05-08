from ._util import *
from res.solarpower import *

##################################################################
## Make a typical simulator function
def simulateLocations(source, elev, loss, minCF, verbose, extractor, module, inverter, capacity, azimuth, tilt, interpolation, gid, 
                      locationID, globalStart, locations, pickleable):
    if verbose: 
        startTime = dt.now()
        globalStart = globalStart
        print(" %s: Starting at +%.2fs"%(str(gid), (startTime-globalStart).total_seconds()))
    
    # prepare locations
    locations = LocationSet(locations)

    if len(locations) == 0 : 
        if verbose: print( " %s: No locations found"%(str(gid)))
        return None

    if pickleable: locations.makePickleable()
    
    # do simulations
    capacityGeneration = simulatePVModule(locations, elev, source, module=module, azimuth=azimuth, tilt=tilt, inverter=inverter, extract="capacity-production", interpolation=interpolation, loss=loss)
    capacityFactor = capacityGeneration.mean(0)
    
    if capacity is None: capacity=1
    production = capacityGeneration*capacity
    
    if verbose:
        endTime = dt.now()
        simSecs = (endTime - startTime).total_seconds()
        globalSecs = (endTime - globalStart).total_seconds()
        print(" %s: Finished %d locations +%.2fs (%.2f locations/sec)"%(str(gid), len(locations), globalSecs, len(locations)/simSecs))
    
    # Apply Capacity Factor Filter
    if minCF > 0:
        sel = capacityFactor >= minCF
        capacityFactor = capacityFactor[sel]
        production = production.ix[:,sel]

    # Done!
    if extractor.method == "batch":
        outputVars = OrderedDict()
        outputVars["minCF"] = minCF
        outputVars["module"] = module
        outputVars["azimuth"] = azimuth
        outputVars["tilt"] = tilt
        outputVars["capacity"] = capacity
        outputVars["inverter"] = inverter
        outputVars["interpolation"] = interpolation
        outputVars["loss"] = loss
        outputVars["locationID"] = locationID
        
        result = raw_finalizer(production, capacityFactor)

        try:
            outputPath = extractor.outputPath%gid
        except:
            if extractor.outputPath.format(0) == extractor.outputPath.format(2):
                raise ResError("output is not integer-formatable. Be sure there is a single %d or a {}")
            outputPath = extractor.outputPath.format(gid)

        raw_output(outputPath, result, outputVars)
    
    output = extractor.finalize(production, capacityFactor)
    return output

##################################################################
## Distributed PV production from a weather source
def PVWorkflowTemplate(placements, source, elev, module, azimuth, tilt, inverter, interpolation, 
                       capacity, extract, output, loss, minCF, jobs, batchSize, verbose, padding=2):
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

    ### Determine the total extent which will be simulated (also make sure the placements input is okay)
    if verbose: print("Arranging placements at +%.2fs"%((dt.now()-startTime).total_seconds()))
    if isinstance(placements, str): # placements is a path to a point-type shapefile
        placements = gk.vector.extractAsDataFrame(placements, outputSRS='latlon')
        placements["lat"] = placements.geom.apply(lambda x: x.GetY())
        placements["lon"] = placements.geom.apply(lambda x: x.GetX())
    
    if isinstance(placements, pd.DataFrame):
        if "elev" in placements.columns and elev is None: elev = placements.elev.values
        if "capacity" in placements.columns and capacity is None: capacity = placements.capacity.values
        if "azimuth" in placements.columns and azimuth is None: azimuth = placements.azimuth.values
        if "tilt" in placements.columns and tilt is None: tilt = placements.tilt.values

        try:
            placements = placements[["lon","lat"]].values
        except:
            placements = placements["geom"].values

    placements = LocationSet(placements)

    lonMin, latMin, lonMax, latMax = placements.getBounds()
    latMin = latMin -padding
    latMax = latMax +padding
    lonMin = lonMin -padding
    lonMax = lonMax +padding

    elev = elev if isinstance(elev, str) else np.array(elev)
    capacity = None if capacity is None else np.array(capacity)

    if verbose: print("Pre-loading weather data at +%.2fs"%((dt.now()-startTime).total_seconds()))
    totalExtent = gk.Extent((lonMin,latMin,lonMax,latMax,), srs=LATLONSRS)
    
    # Setup manager if needed
    if useMulti:
        manager = WSManager()
        manager.start()
        weatherSource = manager.MerraSource(path=source, bounds=Bounds(*totalExtent.xyXY))
    else:
        weatherSource = MerraSource(path=source, bounds=totalExtent)

    weatherSource.loadSet_PV()

    ### Do simulations
    # initialize simulations
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
        for simPlacements in np.array_split(I, max(1,len(placements)//(batchSize/cpus))):
            simGroups.append( simPlacements )

    if verbose: 
        print("Simulating %d groups at +%.2fs"%(len(simGroups), (dt.now()-startTime).total_seconds() ))

    results = []
    if useMulti: placements.makePickleable()
    for i,sel in enumerate(simGroups):
        # Construct arguments for each submission
        inputs = {}
        inputs["source"] = weatherSource
        inputs["loss"] = loss
        inputs["minCF"] = minCF
        inputs["verbose"] = verbose
        inputs["extractor"] = extractor
        inputs["module"] = module
        inputs["inverter"] = inverter
        inputs["capacity"] = capacity
        inputs["interpolation"] = interpolation
        inputs["gid"] = i
        inputs["locationID"] = sel

        if verbose:
            inputs["globalStart"]=startTime

        def add(val,name):
            if isinstance(val, list): val = np.array(val)
            if isinstance(val , np.ndarray) and val.size>1: inputs[name] = val[sel]
            else: inputs[name] = val

        add(placements, "locations")
        add(capacity, "capacity")
        add(azimuth, "azimuth")
        add(tilt, "tilt")
        add(elev, "elev")

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
        print("Finished simulating %d locations (%d surviving) at +%.2fs (%.2f locations/sec)"%(len(placements), finalResult.c, totalSecs, len(placements)/totalSecs))

    ### Give the results
    if not output is None and not extractor.skipFinalOutput:
        if verbose:
            endTime = dt.now()
            totalSecs = (endTime - startTime).total_seconds()
            print("Writing output at +%.2fs"%totalSecs)

        outputVars = OrderedDict()
        outputVars["minCF"] = minCF
        outputVars["module"] = module
        outputVars["capacity"] = capacity
        outputVars["inverter"] = inverter
        outputVars["azimuth"] = azimuth
        outputVars["tilt"] = tilt
        outputVars["interpolation"] = interpolation
        outputVars["loss"] = loss
        
        extractor.output(output, finalResult, outputVars)

    #### Formulate result
    if verbose:
        endTime = dt.now()
        totalSecs = (endTime - startTime).total_seconds()
        print("Done at +%.2fs!"%totalSecs)

    if extract == "batch":  return

    outputResult = finalResult.o
    outputResult.name = extractor.title
    outputResult.TurbineCount = finalResult.c
    return outputResult
    
def PVOpenFieldWorkflow(placements, source, elev, capacity=None, module="Canadian_Solar_CS5P_220M___2009_", azimuth=180, tilt="latitude", inverter=None, interpolation="bilinear", extract="totalProduction", loss=0.00, output=None, minCF=0, jobs=1, batchSize=None, verbose=True):
    return PVWorkflowTemplate(placements=placements, source=source, elev=elev, module=module, azimuth=azimuth, 
                              tilt=tilt, inverter=inverter, interpolation=interpolation, extract=extract, 
                              output=output, loss=loss, minCF=minCF, jobs=jobs, batchSize=batchSize, 
                              verbose=verbose, capacity=capacity)
                         
                                                  
