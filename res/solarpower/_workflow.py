from res.util.util_ import *
from ._pv import *
from res.weather import MerraSource
from res.weather.windutil import *
import warnings

def _batch_simulator(source, loss, verbose, module, globalStart, extract, 
                     frankCorrection, tracking, interpolation, cellTempModel, 
                     rackingModel, airmassModel, transpositionModel, 
                     generationModel, placements, capacity, tilt, azimuth, 
                     elev, locationID, gid, batchSize, trackingGCR, 
                     trackingMaxAngle, output):
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
                                                                                   # a smaller area is simulated. IDKY???
        source = MerraSource(source, bounds=ext.xyXY, indexPad=2)
        source.loadSet_PV(verbose=verbose, _clockstart=globalStart, _header=" %s:"%str(gid))

    # do simulations
    res = []
    if batchSize is None: batchSize = 1e10
    for i,batchStart in enumerate(np.arange(0, placements.count, batchSize, dtype=int)):
        if verbose: 
           batchStartTime = dt.now()
           print(" %s: Starting batch %d of %d at +%.2fs"%(str(gid), i+1, placements.count//batchSize+1, (batchStartTime-globalStart).total_seconds()))

        s = np.s_[batchStart: min(batchStart+batchSize,placements.count) ]

        _placements = placements[s]
        _elev = elev if isinstance(elev, str) else elev[s]
        _tilt = tilt if isinstance(tilt, str) else tilt[s]
        _azimuth = azimuth[s]

        warnings.filterwarnings("ignore")
        capacityGeneration = simulatePVModule(
                                locs=_placements, 
                                elev=_elev, 
                                source=source, 
                                module=module, 
                                azimuth=_azimuth, 
                                tilt=_tilt, 
                                totalSystemCapacity=1, 
                                tracking=tracking,
                                modulesPerString=1, 
                                inverter=None, 
                                stringsPerInverter=1, 
                                rackingModel=rackingModel,
                                airmassModel=airmassModel,
                                transpositionModel=transpositionModel,
                                cellTempModel=cellTempModel,
                                generationModel=generationModel,
                                inverterModel="sandia", # not actually used...
                                interpolation=interpolation,
                                loss=loss,
                                trackingGCR=trackingGCR,
                                trackingMaxAngle=trackingMaxAngle,
                                frankCorrection=frankCorrection,)
        warnings.simplefilter('default')

        # Arrange output        
        if extract == "capacityFactor": tmp = capacityGeneration.mean(0)
        elif extract == "totalProduction": tmp = (capacityGeneration*capacity[s]).sum(1)
        elif extract == "raw": tmp = capacityGeneration*capacity[s]
        elif extract == "batchfile": tmp = capacityGeneration
        else:
            raise ResError("extract method '%s' not understood"%extract)

        res.append(tmp)
    del source

    if extract == "batchfile": 
        res = pd.concat(res, axis=1)
        _save_to_nc( output=output+"_%d.nc"%gid,
                     capacityGeneration=res[placements[:]],
                     lats=[p.lat for p in placements],
                     lons=[p.lon for p in placements], 
                     capacity=capacity,
                     tilt=tilt,
                     azimuth=azimuth,
                     identity=locationID,
                     module=module,
                     tracking=tracking,
                     loss=loss)
        res = None

    if verbose:
        endTime = dt.now()
        simSecs = (endTime - startTime).total_seconds()
        globalSecs = (endTime - globalStart).total_seconds()
        print(" %s: Finished %d locations +%.2fs (%.2f locations/sec)"%(str(gid), len(placements), globalSecs, len(placements)/simSecs))
    
    placements.makePickleable()
    return res

##################################################################
## Distributed PV production from a weather source
def PVWorkflowTemplate( placements, source, elev, module, azimuth, tilt, extract, output, jobs, batchSize, verbose, capacity, frankCorrection, tracking, loss, interpolation, rackingModel, airmassModel, transpositionModel, cellTempModel, generationModel, trackingMaxAngle, trackingGCR):

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
    capacity = pd.Series(capacity, index=placements)
    tilt = tilt if isinstance(tilt, str) else pd.Series(tilt, index=placements)
    azimuth = pd.Series(azimuth, index=placements)

    ### Do simulations
    # initialize simulations
    if verbose: print("Initializing simulations at +%.2fs"%((dt.now()-startTime).total_seconds()))
    locationID=pd.Series(np.arange(placements.shape[0]), index=placements)

    simKwargs = dict(source=source, loss=loss, verbose=verbose, 
                     module=module, globalStart=startTime, extract=extract,
                     frankCorrection=frankCorrection, tracking=tracking,
                     interpolation=interpolation, cellTempModel=cellTempModel,
                     rackingModel=rackingModel, airmassModel=airmassModel,
                     transpositionModel=transpositionModel, trackingGCR=trackingGCR, 
                     generationModel=generationModel, trackingMaxAngle=trackingMaxAngle,
                     output=output,
                    )

    if batchSize is None: batchSize = 1e10
    if useMulti:
        from multiprocessing import Pool
        pool = Pool(jobs)
        placements.makePickleable()
        res = []

        # Split locations into groups
        groups = []
        for grp in placements.splitKMeans(jobs):
            if grp.count > (batchSize/jobs)*3: # Should the batch be broken into smaller groupings?
                subgroups = int(np.round(grp.count/(3*batchSize/jobs)))
                for sgi in range(int(subgroups)):
                    groups.append( gk.LocationSet(grp[sgi::subgroups]) )
            else:
                groups.append( grp )

        # Submit groups
        if verbose: print("Submitting %d simulation groups at +%.2fs"%( len(groups), (dt.now()-startTime).total_seconds()) )
        for i,grp in enumerate(groups):
            kwargs = simKwargs.copy()
            kwargs["placements"] = grp
            kwargs["capacity"] = capacity[grp[:]].values
            kwargs["tilt"] = tilt if isinstance(tilt, str) else tilt[grp[:]].values
            kwargs["azimuth"] = azimuth[grp[:]].values
            kwargs["elev"] = elev if isinstance(elev, str) else elev[grp[:]].values
            kwargs["locationID"] = locationID[grp[:]].values
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
        simKwargs["locationID"] = locationID.values
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
    
def workflowOpenFieldFixed(placements, source, elev=None, module="SunPower_SPR_X21_255", azimuth=180, tilt="latitude", extract="totalProduction", output=None, jobs=1, batchSize=None, verbose=True, capacity=None, frankCorrection=False,):
                            
    return PVWorkflowTemplate(# Controllable args
                              placements=placements, source=source, elev=elev, module=module, azimuth=azimuth, 
                              tilt=tilt, extract=extract, output=output, 
                              jobs=jobs, batchSize=batchSize, verbose=verbose, capacity=capacity,
                              frankCorrection=frankCorrection,

                              # Set args
                              tracking="fixed",  loss=0.16, interpolation="bilinear",
                              rackingModel='open_rack_cell_glassback', airmassModel='kastenyoung1989', 
                              transpositionModel='perez', cellTempModel="sandia", generationModel="single-diode", 
                              trackingMaxAngle=None, trackingGCR=None)
                         
def workflowOpenFieldTracking(placements, source, elev=None, module="SunPower_SPR_X21_255", azimuth=180, tilt="latitude", extract="totalProduction", output=None, jobs=1, batchSize=None, verbose=True, capacity=None, frankCorrection=False,):
    return PVWorkflowTemplate(# Controllable args
                              placements=placements, source=source, elev=elev, module=module, azimuth=azimuth, 
                              tilt=tilt, extract=extract, output=output, 
                              jobs=jobs, batchSize=batchSize, verbose=verbose, capacity=capacity,
                              frankCorrection=frankCorrection, 

                              # Set args
                              tracking="single-axis", trackingMaxAngle=60, loss=0.16,
                              rackingModel='open_rack_cell_glassback', airmassModel='kastenyoung1989', 
                              transpositionModel='perez', cellTempModel="sandia", generationModel="single-diode", 
                              interpolation="bilinear", trackingGCR=2/7, 
                              )



def _save_to_nc(output, capacityGeneration, lats, lons, capacity, tilt, azimuth, identity, module, tracking, loss):

    ds = nc.Dataset(output, mode="w")
    try:
        # Make the dimensions
        ds.createDimension("time",      size=capacityGeneration.shape[0])
        ds.createDimension("parkID",    size=capacityGeneration.shape[1])

        # Make the time variable
        timeV = ds.createVariable("time", "u4", dimensions=("time",))
        timeV.units = "minutes since 1900-01-01 00:00:00"

        times = capacityGeneration.index
        if capacityGeneration.index[0].tz is None:
            timeV.tz = "unknown"
        else:
            timeV.tz = capacityGeneration.index[0].tzname()
            times = times.tz_localize(None)

        timeV[:] = nc.date2num(times.to_pydatetime(), timeV.units)

        # Make the data variables
        var = ds.createVariable("capfac", "u2", dimensions=("time", "parkID",), zlib=True)
        
        var.scale_factor = 1/32768
        var.units = "capacity_factor"
        var.description = "Hourly generation of each park, scaled from 0 to max capacity (1)"
        var.longname = "CapacityFactor"
        
        var[:] = capacityGeneration.values

        # Make the descriptor variables
        var = ds.createVariable("parkID", "u4", dimensions=("parkID",))
        var.units = "-"
        var.description = "ID number for each park"
        var.longname = "Park ID"
        var[:] = identity

        if isinstance(tilt, str):
            var = ds.createVariable("systemTilt", str, dimensions=())
            var.description = "system tilt convention"
            var.longname = "system tilt"
            var[0] = tilt
        else:
            var = ds.createVariable("systemTilt", "u2", dimensions=("parkID",))
            var.units = "degrees"
            var.scale_factor = 1/500
            var.description = "system tilt of each park"
            var.longname = "system tilt"
            var[:] = tilt

        var = ds.createVariable("systemAzimuth", "u2", dimensions=("parkID",))
        var.units = "degrees"
        var.scale_factor = 1/150
        var.description = "system azimuth of each park"
        var.longname = "system azimuth"
        var[:] = azimuth

        var = ds.createVariable("capacity", "u2", dimensions=("parkID",))
        var.units = "kW"
        var.description = "Capacity of each park"
        var.longname = "Capacity"
        var[:] = capacity

        var = ds.createVariable("latitude", "f", dimensions=("parkID",))
        var.units = "degrees latitude"
        var.description = "Latitude location of each turbine"
        var.longname = "Latitude"
        var[:] = lats

        var = ds.createVariable("longitude", "f", dimensions=("parkID",))
        var.units = "degrees longitude"
        var.description = "Longitude location of each turbine"
        var.longname = "Longitude"
        var[:] = lons

        var = ds.createVariable("tracking", str, dimensions=())
        var.description = "Tracking type"
        var.longname = "Tracking type"
        var[0] = tracking

        var = ds.createVariable("module", str, dimensions=())
        var.description = "Module identifier"
        var.longname = "Module identifier"
        var[0] = module

        var = ds.createVariable("loss", "f", dimensions=())
        var.description = "Loss factor"
        var.longname = "Loss factor"
        var[0] = loss

        # Done!
        ds.close()

    except Exception as e: 
        ds.close()
        raise e
