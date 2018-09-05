from ._util import *
from ._powerCurveConvoluter import *
from ._simulator import *
from res.weather import MerraSource
from res.weather.windutil import *

def _batch_simulator(source, landcover, gwa, adjustMethod, roughness, loss, convScale, convBase, lowBase, lowSharp, lctype, 
                     verbose, extract, powerCurves, pcKey, gid, globalStart, densityCorrection, placements, hubHeight, 
                     capacity, rotordiam, batchSize, turbineID, output):
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
        capacityGeneration = pd.DataFrame(-1*np.ones(ws.shape), index=ws.index, columns=ws.columns)
        tmpPCKey = pcKey[s]
        for key in np.unique(tmpPCKey):
            tmp = simulateTurbine(ws.iloc[:,tmpPCKey==key], powerCurves[key], loss=0)
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
        elif extract == "batchfile": tmp = capacityGeneration
        else:
            raise ResError("extract method '%s' not understood"%extract)

        res.append(tmp)
    del source

    if extract == "batchfile": 
        tmp = pd.concat(tmp, axis=1)
        _save_to_nc( output=output+"_%d.nc"%gid,
                    capacityGeneration=tmp[placements[:]],
                    lats=[p.lat for p in placements],
                    lons=[p.lon for p in placements], 
                    capacity=capacity,
                    hubHeight=hubHeight,
                    rotordiam=rotordiam,
                    identity=turbineID,
                    pckey=pcKey)
        res = None

    placements.makePickleable()
    # All done!
    if verbose:
        endTime = dt.now()
        simSecs = (endTime - groupStartTime).total_seconds()
        globalSecs = (endTime - globalStart).total_seconds()
        print(" %s: Finished %d turbines +%.2fs (%.2f turbines/sec)"%(str(gid), placements.count, globalSecs, placements.count/simSecs))
    return res

def workflowTemplate(placements, source, landcover, gwa, convScale, convBase, lowBase, lowSharp, adjustMethod, hubHeight, 
                     powerCurve, capacity, rotordiam, cutout, lctype, extract, output, jobs, batchSize, verbose, 
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
        placements = gk.vector.extractFeatures(placements, srs=gk.srs.EPSG4326)
    
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
    
    ### Convolute turbine
    if verbose: print("Convolving power curves at +%.2fs"%( (dt.now()-startTime).total_seconds()) )
    
    pcKey = None
    powerCurves = {}
    if isinstance(powerCurve, PowerCurve):
        if capacity is None: raise ResError("Capacity cannot be undefined when a power curve is given")
        capacity = pd.Series(capacity, index=placements)

        pcKey = pd.Series(['user-defined',] * placements.shape[0], index=placements)
        powerCurves['user-defined'] = PowerCurve

    elif powerCurve is None: # no turbine given, so a synthetic turbine will need to be constructed
        if capacity is None and rotordiam is None:
            raise RuntimeError("powerCurve, capacity, and rotordiam cannot all be None")
        capacity = pd.Series(capacity, index=placements)
        rotordiam = pd.Series(rotordiam, index=placements)

        # Compute specific capacity
        #  - Round to the nearest 1 to save time for convolution
        specificCapacity = np.array(capacity*1000/(np.pi*rotordiam**2/4))
        specificCapacity = np.round(specificCapacity).astype(int)

        if specificCapacity.size == 1:
            key = "%d:%d"%(specificCapacity,25 if cutout is None else cutout)
            powerCurves[key] = SyntheticPowerCurve( specificCapacity=specificCapacity, cutout=cutout)
            pcKey = pd.Series([key,] * placements.shape[0], index=placements)
        else:
            pcKey = []

            for i,sp in enumerate(specificCapacity):
                co = 25 if cutout is None else cutout[i]
                key = "%d:%d"%(sp,25 if co is None else co)
                pcKey.append( key )
                if not key in powerCurves.keys():
                    powerCurves[key] = SyntheticPowerCurve( sp, co)
            pcKey = pd.Series(pcKey, index=placements)

    elif isinstance(powerCurve, str):
        pcKey = pd.Series([powerCurve,] * placements.shape[0], index=placements)
        capacity = pd.Series(TurbineLibrary.ix[powerCurve].Capacity, index=placements)

        tmp = TurbineLibrary.ix[powerCurve].Rotordiameter
        if isinstance(tmp,float): rotordiam = pd.Series(tmp, index=placements)
        else: rotordiam = 0
        
        powerCurves[powerCurve] = TurbineLibrary.ix[powerCurve].PowerCurve

    else: # powerCurve is either a (ws,power) list or is a list of turbine names
        if isinstance(powerCurve[0],str): # assume entire list is a list of names
            pcKey = pd.Series(powerCurve, index=placements)
            capacity = []
            rotordiam = []

            for name in pcKey:
                # TODO: I SHOULD CHECK FOR THE "spPow:cutout" notation here, so that library and synthetic turbines can be mixed 
                capacity.append(TurbineLibrary.ix[name].Capacity)

                tmp = TurbineLibrary.ix[powerCurve].Rotordiameter
                if isinstance(tmp,float): rotordiam = pd.Series(tmp, index=placements)
                else: rotordiam = 0

                if not name in powerCurves:
                    powerCurves[name] = TurbineLibrary.ix[name].PowerCurve

            capacity = pd.Series(capacity, index=placements)
            rotordiam = pd.Series(rotordiam, index=placements)

        else: # powerCurve is a single power curve definition
            if capacity is None:
                raise RuntimeError("capacity cannot be None when giving a user-defined power curve")
            capacity = pd.Series(capacity, index=placements)

            pcKey = pd.Series(['user-defined',] * placements.shape[0], index=placements)
            
            tmp = np.array(powerCurve)
            powerCurve = PowerCurve(tmp[:,0], tmp[:,1])
            powerCurves['user-defined'] = powerCurve
    
    if not rotordiam is None and isinstance(rotordiam, np.ndarray): 
        rotordiam = pd.Series(rotordiam, index=placements)

    if verbose: 
        print("   Convolving %d power curves..."%(len(powerCurves)))

    convolutionKwargs = dict(stdScaling=convScale, stdBase=convBase, extendBeyondCutoff=False)
    
    if useMulti:
        from multiprocessing import Pool
        pool = Pool(cpus)
        res = []
    
        for k,v in powerCurves.items():
            res.append((k,pool.apply_async(convolutePowerCurveByGuassian, (v, ), convolutionKwargs)))

        for k,r in res:
            powerCurves[k] = r.get()
        
        pool.close()
        pool.join()
        pool = None
    else:
        for k,v in powerCurves.items():
            powerCurves[k] = convolutePowerCurveByGuassian(v, **convolutionKwargs)

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
        powerCurves = powerCurves,
        globalStart=startTime,
        densityCorrection=densityCorrection,
        output=output,
        )
    
    turbineID=pd.Series(np.arange(placements.shape[0]), index=placements)

    if useMulti:
        placements.makePickleable()
        pool = Pool(jobs)
        res = []

        # Split locations into groups
        groups = []
        for grp in placements.splitKMeans(jobs):
            if grp.count > (batchSize/jobs)*3:
                subgroups = int(np.round(grp.count/(3*batchSize/jobs)))
                for sgi in range(int(subgroups)):
                    groups.append( gk.LocationSet(grp[sgi::subgroups]) )
            else:
                groups.append( grp )

        # Submit groups
        if verbose: print("Submitting %d simulation groups at +%.2fs"%( len(groups), (dt.now()-startTime).total_seconds()) )
        for i,placementGroup in enumerate(groups):
            kwargs = simKwargs.copy()
            kwargs.update(dict(
                placements=placementGroup,
                hubHeight=hubHeight[placementGroup[:]].values,
                capacity=capacity[placementGroup[:]].values,
                rotordiam=None if rotordiam is None else rotordiam[placementGroup[:]].values,
                pcKey = pcKey[placementGroup[:]].values,
                batchSize=batchSize//jobs,
                gid=i,
                turbineID=turbineID[placementGroup[:]].values,
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
            hubHeight=hubHeight.values,
            capacity=capacity.values,
            rotordiam=None if rotordiam is None else rotordiam.values,
            pcKey = pcKey.values,
            batchSize=batchSize,
            gid=0,
            turbineID=turbineID.values,
            ))
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
    if verbose: print("Finished simulating %d turbines at +%.2fs (%.2f turbines/sec)"%(placements.count, totalSecs, placements.count/totalSecs))

    return res

def workflowOnshore(placements, source, landcover, gwa, hubHeight=None, powerCurve=None, capacity=None, rotordiam=None, cutout=None, lctype="clc", extract="totalProduction", output=None, jobs=1, groups=None, batchSize=10000, verbose=True):
    """
    Apply the wind simulation method developed by Severin Ryberg, Dilara Caglayan, and Sabrina Schmitt. 
    This method works as follows for a given simulation point:
        1. The nearest time-series in the provided MERRA climate data source is extracted
            * reads windspeeds at 50 meters
        2. The time series is adjusted so that the long-run-average (across all MERRA data at this point) 
           matches the value given by the Global Wind Atlas at the simulation point
        3. A roughness factor is assumed from the land cover and is used to project the wind speeds 
           to the indicated hub height
        4. The wind speeds are 'density corrected' so that they correspond to windspeeds at standard
           air density
        5. The wind speeds are fed through the power-curve of the indicated turbine
            * The power curve has been convoluted to incorporate a stochastic spread of windspeeds
        6. Low production periods are depressed slightly, ending around a 60% capacity factor
        
    Notes:
        * hubHeight must always be given, either as an argument or contained within the placements 
          object (see below)
        * When giving a user-defined power curve, the capacity must also be given 
        * When powerCurve isn't given, capacity, rotordiam and cutout are used to generate a synthetic 
          power curve using res.windpower.SyntheticPowerCurve. In this case, rotordiam and capacity 
          must be given, but cutout can be left as None (implying the default of 25 m/s)

    Parameters:
    ===========
        placements : DataFrame, geokit.LocationSet, [ (lon,lat), ], or str
            A list of (lon,lat) coordinates to simulate
            * If str -> A path to a point-type shapefile indicating the turbines to simulate
            * If DataFrame -> A datafrom containing per-turbine characteristics
              - Must include a 'lon' and 'lat' column
              - Unless specified later 'powercurve', capacity', 'hubHeight', 'rotordiam', and 
                'cutout' columns are used for simulating each individual turbine 
            
        source : str, res.NCSource
            The weather data to use for simulation
            * If str -> A path to the MERRA data which will be used for the simulation
            * MUST have the fields 'U50M', 'V50M', 'SP', and 'T2M'

        landcover : str
            The path to the land cover source

        gwa : str
            The path to the global wind atlas source at 50m

        powerCurve : str, list; optional
            The normalized power curve to use for simulation
            * If str -> Expects a key from the TurbineLibrary (res.windpower.TurbineLibrary)
            * If list -> expects (windspeed, capacity-factor) pairs for all points in the power curve
              - Pairs must be given in order of increasing wind speed 

        hubHeight : float; optional 
            The hub height in meters to simulate at 
            * This input is only optional when hub heights are uniquely defined in the 'placements' 
              data frame
            * When given, this input will be applied to all simulation locations 

        capacity : float; optional 
            The nameplate capacity to use for simulation in kW
            * This input is only optional when capacities are uniquely defined in the 'placements' 
              data frame
            * When given, this input will be applied to all simulation locations 

        rotordiam : float ; optional
            The turbine rotor diameter in meters
            * This input is only useful when Synthetic Power curves are generated
            
        cutout : float ; optional 
            The turbine cutout windspeed in m/s
            * This input is only useful when Synthetic Power curves are generated
        
        lctype : str ; optional
            The land cover type to use
            * Options are "clc", "globCover", and "modis"

        extract : str ; optional
            Determines the extraction method and the form of the returned information
            * Options are:
              "raw" - returns the timeseries production for each location
              "capacityFactor" - returns only the resulting capacity factor for each location
              "averageProduction" - returns the average time series of all locations
              "batch" - returns nothing, but the full production data is written for each batch

        jobs : int ; optional
            The number of parallel jobs

        batchSize : int; optional
            The number of placements to simulate across all concurrent jobs
            * Use this to tune performance to your specific machine

        verbose : bool; optional
            If True, output progress reports

        outputHeader : str ; optional
            The path of the output NC4 file to create
            * Only useful when using the "batch" extract option
    """

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
                            extract=extract, output=output, jobs=jobs, batchSize=batchSize, verbose=verbose, 
                            **kwgs)


def workflowOffshore(placements, source, hubHeight=None, powerCurve=None, capacity=None, rotordiam=None, cutout=None, extract="totalProduction", output=None, jobs=1, batchSize=10000, verbose=True, groups=None):

    kwgs = dict()
    kwgs["loss"]=0.00
    kwgs["convScale"]=0.04
    kwgs["convBase"]=0.5
    kwgs["lowBase"]=0.1
    kwgs["lowSharp"]=3.5
    kwgs["adjustMethod"]="bilinear"
    kwgs["roughness"]=0.0002
    kwgs["lctype"]=None # This isn't actually used since adjustment is bilinear...
    kwgs["densityCorrection"]=False

    return workflowTemplate(placements=placements, source=source, landcover=None, gwa=None, hubHeight=hubHeight, 
                            powerCurve=powerCurve, capacity=capacity, rotordiam=rotordiam, cutout=cutout, 
                            extract=extract, output=output, jobs=jobs, batchSize=batchSize, verbose=verbose, 
                            **kwgs)


def _save_to_nc(output, capacityGeneration, lats, lons, capacity, hubHeight, rotordiam, identity, pckey):

    ds = nc.Dataset(output, mode="w")
    try:
        # Make the dimensions
        ds.createDimension("time",      size=capacityGeneration.shape[0])
        ds.createDimension("turbineID", size=capacityGeneration.shape[1])

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
        var = ds.createVariable("capfac", "u2", dimensions=("time", "turbineID",))
        
        var.scale_factor = 1/32768
        var.units = "capacity_factor"
        var.description = "Hourly generation of each turbine, scaled from 0 to max capacity (1)"
        var.longname = "CapacityFactor"
        
        var[:] = capacityGeneration.values

        # Make the descriptor variables
        var = ds.createVariable("turbineID", "u4", dimensions=("turbineID",))
        var.units = "-"
        var.description = "ID number for each turbine"
        var.longname = "Turbine ID"
        var[:] = identity

        var = ds.createVariable("hubHeight", "u2", dimensions=("turbineID",))
        var.units = "m"
        var.scale_factor = 1/10
        var.description = "Hub height of each turbine"
        var.longname = "Hub Height"
        var[:] = hubHeight

        var = ds.createVariable("capacity", "u2", dimensions=("turbineID",))
        var.units = "kW"
        var.description = "Capacity of each turbine"
        var.longname = "Capacity"
        var[:] = capacity

        if not rotordiam is None:
            var = ds.createVariable("rotordiam", "u2", dimensions=("turbineID",))
            var.units = "m"
            var.scale_factor = 1/10
            var.description = "Rotor diameter of each turbine"
            var.longname = "Rotor Diameter"
            var[:] = rotordiam

        var = ds.createVariable("powerCurveKey", str, dimensions=("turbineID",))
        var.units = "-"
        var.convention = "Specific Power : Cutout Wind Speed"
        var.description = "Key used to identify the power curve for each turbine"
        var.longname = "Power Curve Key"
        for i,k in enumerate(pckey): var[i] = k

        var = ds.createVariable("latitude", "f", dimensions=("turbineID",))
        var.units = "degrees latitude"
        var.description = "Latitude location of each turbine"
        var.longname = "Latitude"
        var[:] = lats

        var = ds.createVariable("longitude", "f", dimensions=("turbineID",))
        var.units = "degrees longitude"
        var.description = "Longitude location of each turbine"
        var.longname = "Longitude"
        var[:] = lons

        # Done!
        ds.close()

    except Exception as e: 
        ds.close()
        raise e
