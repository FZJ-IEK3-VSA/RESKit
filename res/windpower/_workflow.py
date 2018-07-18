from ._util import *
from ._powerCurveConvoluter import *
from ._simulator import *
from res.weather import MerraSource
from res.weather.windutil import *

def _batch_simulator(source, landcover, gwa, adjustMethod, roughness, loss, convScale, convBase, lowBase, lowSharp, lctype, 
                     verbose, extract, powerCurve , pcKey , gid, globalStart, densityCorrection, placements, hubHeight, 
                     capacity, rotordiam, cutout, batchSize):
    if verbose: 
        groupStartTime = dt.now()
        globalStart = globalStart
        print(" %s: Starting at +%.2fs"%(str(gid), (groupStartTime-globalStart).total_seconds()))

    ### Open Source and load weather data
    if isinstance(source, str):
        ext = gk.Extent.fromLocationSet(placements).castTo(gk.srs.EPSG4326).pad(1) # Pad to make sure we only select the data we need
                                                                                   # Otherwise, the NCSource might pull EVERYTHING when
                                                                                   # a smalle area is simulated. IDKY???
        source = MerraSource(source, bounds=ext, indexPad=2)
        source.loadWindSpeed(50)
        if densityCorrection:
            source.loadPressure()
            source.loadTemperature('air')

    ### Loop over batch size
    res = []
    if batchSize is None: batchSize = 1e10
    for i,batchStart in enumerate(np.arange(0, placements.count, batchSize)):
        if verbose: 
           batchStartTime = dt.now()
           print(" %s: Starting batch %d of %d at +%.2fs"%(str(gid), i+1, placements.count//batchSize+1, (batchStartTime-globalStart).total_seconds()))

        s = np.s_[batchStart: min(batchStart+batchSize,placements.count) ]

        ### Read windspeed data and adjust to local context
        # read and spatially spatially adjust windspeeds
        if adjustMethod == "lra":
            ws = source.get("windspeed", placements[s], forceDataFrame=True)
            ws = windutil.adjustLraToGwa( ws, placements[s], longRunAverage=MerraSource.LONG_RUN_AVERAGE_50M_SOURCE, gwa=gwa)

        elif adjustMethod == "near" or adjustMethod == "bilinear" or adjustMethod == "cubic":
            ws = source.get("windspeed", placements[s], interpolation=adjustMethod, forceDataFrame=True)

        elif adjustMethod is None:
            ws = source.get("windspeed", placements[s], forceDataFrame=True)
        
        else: raise ResError("adjustMethod not recognized")

        # Look for bad values
        badVals = np.isnan(ws)
        if badVals.any().any():
            print("%d locations have invalid wind speed values:"%badVals.any().sum())
            sel = badVals.any()
            for loc in placements[s][sel]: print("  ", loc)
            raise RuntimeError("Bad windspeed values")

        # Get roughnesses from Land Cover
        if roughness is None and not lctype is None:
            lcVals = gk.raster.extractValues(landcover, placements[s]).data
            roughnesses = windutil.roughnessFromLandCover(lcVals, lctype)

            if np.isnan(roughnesses).any():
                raise RuntimeError("%d locations are outside the given landcover file"%np.isnan(roughnesses).sum())

        elif not roughness is None:
            roughnesses = roughness
        else:
            raise ResError("roughness and lctype are both given or are both None")

        # Project WS to hub height
        ws = windutil.projectByLogLaw(ws, measuredHeight=50, targetHeight=hubHeight[s], roughness=roughnesses)
        
        # Density correction to windspeeds
        if densityCorrection:
            t =  source.get("air_temp", placements[s], interpolation='bilinear', forceDataFrame=True)
            p =  source.get("pressure", placements[s], interpolation='bilinear', forceDataFrame=True)
            ws = densityAdjustment(ws, pressure=p, temperature=t, height=hubHeight[s])

        ### Do simulations
        if not isinstance(powerCurve, dict):
            capacityGeneration = simulateTurbine(ws, powerCurve=powerCurve, loss=0)
        else:
            capacityGeneration = pd.DataFrame(-1*np.ones(ws.shape), index=ws.index, columns=ws.columns)
            tmpPCKey = pcKey[s]
            for key in np.unique(tmpPCKey):
                tmp = simulateTurbine(ws.iloc[:,tmpPCKey==key], powerCurve[key], loss=0)
                capacityGeneration.update( tmp )

            if (capacityGeneration.values<0).any(): raise RuntimeError("Some placements were not evaluated")
        
        # apply wind speed corrections to account (somewhat) for local effects not captured on the MERRA context
        if not (lowBase is None and lowSharp is None):
            factors = (1-lowBase)*(1-np.exp(-lowSharp*capacityGeneration))+lowBase # dampens lower wind speeds
            capacityGeneration = factors*capacityGeneration
            factors = None

        capacityGeneration *= (1-loss)
        
        # Arrange output
        if extract == "capacityFactor": tmp = capacityGeneration.mean(0)
        elif extract == "totalProduction": tmp = (capacityGeneration*capacity[s]).sum(1)
        elif extract == "raw": tmp = capacityGeneration*capacity[s]
        elif extract == "batchfile": pass
        else:
            raise ResError("extract method '%s' not understood"%extract)

        res.append(tmp)
    del source
    placements.makePickleable()
    # All done!
    if verbose:
        endTime = dt.now()
        simSecs = (endTime - groupStartTime).total_seconds()
        globalSecs = (endTime - globalStart).total_seconds()
        print(" %s: Finished %d turbines +%.2fs (%.2f turbines/sec)"%(str(gid), placements.count, globalSecs, placements.count/simSecs))
    return res

def workflowTemplate(placements, source, landcover, gwa, convScale, convBase, lowBase, lowSharp, adjustMethod, hubHeight, 
                     powerCurve, capacity, rotordiam, cutout, lctype, extract, output, jobs, groups, batchSize, verbose, 
                     roughness, loss, densityCorrection):
    startTime = dt.now()
    if verbose:
        print("Starting at: %s"%str(startTime))

    ### Configre multiprocessing
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

        try:
            placements = placements[["lon","lat"]].values
        except:
            placements = placements["geom"].values

    placements = gk.LocationSet(placements)
    if useMulti: placements.makePickleable()

    hubHeight = None if hubHeight is None else pd.Series(hubHeight, index=placements)
    capacity = None if capacity is None else pd.Series(capacity, index=placements)
    rotordiam = None if rotordiam is None else pd.Series(rotordiam, index=placements)
    cutout = None if cutout is None else pd.Series(cutout, index=placements)

    ### Convolute turbine
    if verbose: print("Convolving power curves at +%.2fs"%( (dt.now()-startTime).total_seconds()) )
    
    pcKey = None
    if isinstance(powerCurve, PowerCurve):
        pcKey = 'user-defined'
    elif powerCurve is None: # no turbine given, so a synthetic turbine will need to be constructed
        if capacity is None and rotordiam is None:
            raise RuntimeError("powerCurve, capacity, and rotordiam cannot all be None")

        # Compute specific capacity
        #  - Round to the nearest 1 to save time for convolution
        specificCapacity = np.array(capacity*1000/(np.pi*rotordiam**2/4))
        specificCapacity = np.round(specificCapacity).astype(int)

        if specificCapacity.size == 1:
            powerCurve = SyntheticPowerCurve( specificCapacity=specificCapacity, cutout=cutout)
            pcKey = "%d:%d"%(specificCapacity,25 if cutout is None else cutout)
        else:
            powerCurve = dict()
            pcKey = []


            for i,sp in enumerate(specificCapacity):
                co = 25 if cutout is None else cutout[i]
                key = "%d:%d"%(sp,25 if co is None else co)
                pcKey.append( key )
                if not key in powerCurve.keys():
                    powerCurve[key] = SyntheticPowerCurve( sp, co)
            pcKey = pd.Series(pcKey, index=placements)

    elif isinstance(powerCurve, str):
        pcKey = powerCurve
        capacity = pd.Series(TurbineLibrary.ix[powerCurve].Capacity, index=placements)
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
            if capacity is None:
                raise RuntimeError("capacity cannot be None when giving a user-defined power curve")
            tmp = np.array(powerCurve)
            powerCurve = PowerCurve(tmp[:,0], tmp[:,1])
            pcKey = "user-defined"

    convolutionKwargs = dict(stdScaling=convScale, stdBase=convBase, extendBeyondCutoff=False)
    if isinstance(powerCurve, dict):
        if verbose: 
            print("   Convolving %d power curves..."%(len(powerCurve)))
        

        if useMulti:
            from multiprocessing import Pool
            pool = Pool(cpus)
            res = []
        
            for k,v in powerCurve.items():
                res.append((k,pool.apply_async(convolutePowerCurveByGuassian, (v, ), convolutionKwargs)))

            for k,r in res:
                powerCurve[k] = r.get()
            
            pool.close()
            pool.join()
            pool = None
        else:
            for k,v in powerCurve.items():
                powerCurve[k] = convolutePowerCurveByGuassian(v, **convolutionKwargs)
    else:
        powerCurve = convolutePowerCurveByGuassian(powerCurve, **convolutionKwargs)

    ### Do simulations
    if verbose: print("Starting simulations at at +%.2fs"%( (dt.now()-startTime).total_seconds()) )

    simKwargs = dict(
        source=source,
        landcover=landcover,
        gwa=gwa,
        adjustMethod=adjustMethod,
        roughness=roughness,
        loss=loss,
        lowBase=lowBase,
        lowSharp=lowSharp,
        convBase=convBase,
        convScale=convScale,
        lctype=lctype,
        verbose=verbose,
        extract=extract,
        powerCurve = powerCurve,
        globalStart=startTime,
        densityCorrection=densityCorrection,
        )

    if useMulti:
        placements.makePickleable()
        pool = Pool(jobs)
        res = []
        if groups is None: groups = jobs
        for i,placementGroup in enumerate(placements.splitKMeans(groups)):
            kwargs = simKwargs.copy()
            kwargs.update(dict(
                placements=placementGroup,
                hubHeight=None if hubHeight is None else hubHeight[placementGroup[:]].values,
                capacity=None if capacity is None else capacity[placementGroup[:]].values,
                rotordiam=None if rotordiam is None else rotordiam[placementGroup[:]].values,
                cutout=None if cutout is None else cutout[placementGroup[:]].values,
                pcKey = pcKey if isinstance(pcKey, str) else pcKey[placementGroup[:]].values,
                batchSize=batchSize//jobs,
                gid=i,
                ))
            res.append(pool.apply_async(_batch_simulator, (), kwargs))

        finalRes = []
        for r in res: finalRes.extend(r.get())
        res = finalRes

        pool.close()
        pool.join()
        pool = None
    else:
        simKwargs.update(dict(
            placements=placements,
            hubHeight=None if hubHeight is None else hubHeight.values,
            capacity=None if capacity is None else capacity.values,
            rotordiam=None if rotordiam is None else rotordiam.values,
            cutout=None if cutout is None else cutout.values,
            pcKey = pcKey if isinstance(pcKey, str) else pcKey.values,
            batchSize=batchSize,
            gid=0,
            ))
        res = _batch_simulator(**simKwargs)

    ## Finalize
    if extract == "capacityFactor": res = pd.concat(res)
    elif extract == "totalProduction": res = sum(res)
    elif extract == "raw": res = pd.concat(res, axis=1)
    elif extract == "batchfile": pass
    else:
        raise ResError("extract method '%s' not understood"%extract)


    endTime = dt.now()
    totalSecs = (endTime - startTime).total_seconds()
    print("Finished simulating %d turbines at +%.2fs (%.2f turbines/sec)"%(placements.count, totalSecs, placements.count/totalSecs))

    return res

def workflowOnshore(placements, source, landcover, gwa, hubHeight=None, powerCurve=None, capacity=None, rotordiam=None, cutout=None, lctype="clc", extract="totalProduction", output=None, jobs=1, groups=None, batchSize=10000, verbose=True):

    kwgs = dict()
    kwgs["loss"]=0.00
    kwgs["convScale"]=0.06
    kwgs["convBase"]=0.1
    kwgs["lowBase"]=0.0
    kwgs["lowSharp"]=5
    kwgs["adjustMethod"]="lra"
    kwgs["roughness"]=None
    kwgs["densityCorrection"]=True

    return workflowTemplate(placements=placements, source=source, landcover=landcover, gwa=gwa, hubHeight=hubHeight, 
                            powerCurve=powerCurve, capacity=capacity, rotordiam=rotordiam, cutout=cutout, lctype=lctype, 
                            extract=extract, output=output, jobs=jobs, groups=groups, batchSize=batchSize, verbose=verbose, 
                            **kwgs)


def workflowOffshore(placements, source, hubHeight=None, powerCurve=None, capacity=None, rotordiam=None, cutout=None, extract="totalProduction", output=None, jobs=1, groups=None, batchSize=10000, verbose=True):

    kwgs = dict()
    kwgs["loss"]=0.00
    kwgs["convScale"]=0.04
    kwgs["convBase"]=0.5
    kwgs["lowBase"]=0.1
    kwgs["lowSharp"]=3.5
    kwgs["adjustMethod"]="bilinear"
    kwgs["roughness"]=0.0002
    kwgs["lctype"]="clc" # This isn't actually used since adjustment is bilinear...
    kwgs["densityCorrection"]=False

    return workflowTemplate(placements=placements, source=source, landcover=None, gwa=None, hubHeight=hubHeight, 
                            powerCurve=powerCurve, capacity=capacity, rotordiam=rotordiam, cutout=cutout, 
                            extract=extract, output=output, jobs=jobs, groups=groups, batchSize=batchSize, verbose=verbose, 
                            **kwgs)