from res.util.util_ import *
from ._pv import *
from res.weather import MerraSource
from res.weather.windutil import *


def _batch_simulator(source, elev, loss, verbose, module, inverter, capacity, azimuth, tilt, 
                     gid, locationID, globalStart, placements, batchSize, extract):
    if verbose: 
        startTime = dt.now()
        globalStart = globalStart
        print(" %s: Starting at +%.2fs"%(str(gid), (startTime-globalStart).total_seconds()))

    if len(placements) == 0 : 
        if verbose: print( " %s: No locations found"%(str(gid)))
        return None

    ### Open Source and load weather data
    if isinstance(source, str):
        ext = gk.Extent.fromLocationSet(placements).castTo(gk.srs.EPSG4326).pad(1) # Pad to make sure we only select the data we need
                                                                                   # Otherwise, the NCSource might pull EVERYTHING when
                                                                                   # a smalle area is simulated. IDKY???
        source = MerraSource(source, bounds=ext, indexPad=2)
        source.loadSet_PV()

    # do simulations
    res = []
    if batchSize is None: batchSize = 1e10
    for i,batchStart in enumerate(np.arange(0, placements.count, batchSize, dtype=int)):
        s = np.s_[batchStart: min(batchStart+batchSize,placements.count) ]

        _placements = placements[s]
        _elev = elev if isinstance(elev, str) else elev[s]
        _tilt = tilt if isinstance(tilt, str) else tilt[s]
        _azimuth = azimuth[s]

        capacityGeneration = simulatePVModule(locs=_placements, elev=_elev, source=source, module=module, 
            azimuth=_azimuth, tilt=_tilt, totalSystemCapacity=1, tracking="fixed", modulesPerString=1, 
            inverter=inverter, stringsPerInverter=1, rackingModel='open_rack_cell_glassback', 
            airMassModel='kastenyoung1989', transpositionModel='haydavies', cellTempModel="sandia", 
            generationModel="single-diode", inverterModel="sandia", interpolation="bilinear", loss=loss, 
            trackingGCR=2/7, trackingMaxAngle=60, frankCorrection=False)


        # Arrange output
        if extract == "capacityFactor": tmp = capacityGeneration.mean(0)
        elif extract == "totalProduction": tmp = (capacityGeneration*capacity[s]).sum(1)
        elif extract == "raw": tmp = capacityGeneration*capacity[s]
        elif extract == "batchfile": tmp = None
        else:
            raise ResError("extract method '%s' not understood"%extract)

        res.append(tmp)
    del source

    if verbose:
        endTime = dt.now()
        simSecs = (endTime - startTime).total_seconds()
        globalSecs = (endTime - globalStart).total_seconds()
        print(" %s: Finished %d locations +%.2fs (%.2f locations/sec)"%(str(gid), len(placements), globalSecs, len(placements)/simSecs))
    
    placements.makePickleable()
    return res

##################################################################
## Distributed PV production from a weather source
def PVWorkflowTemplate(placements, source, elev, module, azimuth, tilt, inverter, 
                       capacity, extract, output, loss, jobs, batchSize, verbose):
    if verbose: 
        startTime = dt.now()
        print("Starting at: %s"%str(startTime))

    if jobs==1: # use only a single process
        jobs = 1
        pool = None
        useMulti = False
    elif jobs > 1: # uses multiple processes (equal to jobs)
        jobs = jobs
        useMulti = True
    else: # uses multiple processes (equal to the number of available processors - jobs)
        jobs = cpu_count()-jobs
        if jobs <=0: raise ResError("Bad jobs count")
        useMulti = True 
    
    ### Determine the total extent which will be simulated (also make sure the placements input is okay)
    if verbose: print("Arranging placements at +%.2fs"%((dt.now()-startTime).total_seconds()))
    if isinstance(placements, str): # placements is a path to a point-type shapefile
        placements = gk.vector.extractFeatures(placements, outputSRS='latlon')
    
    if isinstance(placements, pd.DataFrame):
        if "elev" in placements.columns and elev is None: elev = placements.elev.values
        if "capacity" in placements.columns and capacity is None: capacity = placements.capacity.values
        if "azimuth" in placements.columns and azimuth is None: azimuth = placements.azimuth.values
        if "tilt" in placements.columns and tilt is None: tilt = placements.tilt.values

        try:
            placements = placements[["lon","lat"]].values
        except:
            placements = placements["geom"].values

    placements = gk.LocationSet(placements)

    elev = elev if isinstance(elev, str) else pd.Series(elev, index=placements)
    capacity = pd.Series(elev, index=placements)
    tilt = tilt if isinstance(tilt, str) else pd.Series(tilt, index=placements)
    azimuth = pd.Series(azimuth, index=placements)

    ### Do simulations
    # initialize simulations
    if verbose: print("Initializing simulations at +%.2fs"%((dt.now()-startTime).total_seconds()))
    locationID=pd.Series(np.arange(placements.shape[0]), index=placements)

    simKwargs = {}
    simKwargs["source"] = source
    simKwargs["loss"] = loss
    simKwargs["verbose"] = verbose
    simKwargs["module"] = module
    simKwargs["inverter"] = inverter
    simKwargs["globalStart"]=startTime
    simKwargs["extract"]=extract

    if batchSize is None: batchSize = 1e10
    if useMulti:
        from multiprocessing import Pool
        pool = Pool(jobs)
        placements.makePickleable()
        pool = Pool(jobs)
        res = []

        # Split locations into groups
        groups = []
        for grp in placements.splitKMeans(jobs):
            if grp.count > (batchSize/jobs)*3:
                subgroups = np.round(grp.count/(3*batchSize/jobs))
                for sgi in range(int(subgroups)):
                    groups.append( gk.LocationSet(grp[sgi::subgroups]) )
            else:
                groups.append( grp )

        # Submit groups
        if verbose: print("Submitting %d simulation groups at +%.2fs"%( len(groups), (dt.now()-startTime).total_seconds()) )
        for i,grp in enumerate(groups):
            kwargs = simKwargs.copy()
            kwargs["placements"] = grp
            kwargs["capacity"] = capacity[grp].values
            kwargs["tilt"] = tilt if isinstance(tilt, str) else tilt[grp].values
            kwargs["azimuth"] = azimuth[grp].values
            kwargs["elev"] = elev if isinstance(elev, str) else elev[grp].values
            kwargs["locationID"] = locationID[grp]
            kwargs["gid"] = i
            kwargs["batchSize"] = int(np.round(batchSize/jobs))
            
            res.append(pool.apply_async(_batch_simulator, (), kwargs))

        finalRes = []
        for r in res: finalRes.extend(r.get())
        res = finalRes

        pool.close()
        pool.join()
        pool = None

    else:
        simKwargs["placements"] = placements
        simKwargs["capacity"] = capacity.values
        simKwargs["tilt"] = tilt if isinstance(tilt, str) else tilt.values
        simKwargs["azimuth"] = azimuth.values
        simKwargs["elev"] = elev if isinstance(elev, str) else elev.values
        simKwargs["locationID"] = locationID
        simKwargs["gid"] = 0
        simKwargs["batchSize"] = batchSize
        res = _batch_simulator(**simKwargs)

    ## Finalize
    if extract == "capacityFactor": res = pd.concat(res)
    elif extract == "totalProduction": res = sum(res)
    elif extract == "raw": res = pd.concat(res, axis=1)
    elif extract == "batchfile": 
        pass
    else:
        raise ResError("extract method '%s' not understood"%extract)

    endTime = dt.now()
    totalSecs = (endTime - startTime).total_seconds()
    if verbose: print("Finished simulating %d locations at +%.2fs (%.2f locations/sec)"%(placements.count, totalSecs, placements.count/totalSecs))

    return res
    
def workflowOpenField(placements, source, elev, capacity=None, module="SunPower_SPR_X21_255", azimuth=180, tilt="latitude", inverter=None, extract="totalProduction", loss=0.00, output=None, jobs=1, batchSize=None, verbose=True):
    return PVWorkflowTemplate(placements=placements, source=source, elev=elev, module=module, azimuth=azimuth, 
                              tilt=tilt, inverter=inverter, extract=extract, 
                              output=output, loss=loss, jobs=jobs, batchSize=batchSize, 
                              verbose=verbose, capacity=capacity)
                         

