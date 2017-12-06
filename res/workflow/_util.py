from res.util import *
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

def totalProduction_combiner(r1, r2):
    if r1 is None and not r2 is None: return r2
    elif not r1 is None and r2 is None: return r1
    else: 
        output = r1.o+r2.o
        count = r1.c+r2.c
        return Result(count, output)

def batch_combiner(r1, r2):
    if r1 is None and not r2 is None: return r2
    elif not r1 is None and r2 is None: return r1
    else: 
        output = 0
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

def totalProduction_finalizer(production, capacityFactor, **kwargs):
    output = production.sum(axis=1)
    count = capacityFactor.size
    return Result(count, output)

def batch_finalizer(production, capacityFactor, **kwargs):
    output = 0
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
        const["UnitCount"]=result.c
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

@output_function
def totalProduction_output(output, result, ext, const, dim1, dim2):
    const["extract"]="totalProduction"
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

        production = ds.createVariable("production", "f", dimensions=("time",))
        production.setncatts({"longname":"total energy production","units":"kWh"})
        production[:] = result.values

        ds.close()

class Extractor(object):
    def __init__(s, method, outputPath=None):
        s.skipFinalOutput=False

        if method=="p" or method == "production" or method == "raw":
            s.title = "production"
            s.method = "raw"
            s._combine = "raw_combiner"
            s._finalize = "raw_finalizer"
            s._output = "raw_output"
        elif method=="cf" or method == "capacityFactor":
            s.title = "capacityFactor"
            s.method = "capacityFactor"
            s._combine = "capacityFactor_combiner"
            s._finalize = "capacityFactor_finalizer"
            s._output = "capacityFactor_output"
        elif method=="ap" or method == "averageProduction":
            s.title = "avgProduction"
            s.method = "averageProduction"
            s._combine = "averageProduction_combiner"
            s._finalize = "averageProduction_finalizer"
            s._output = "averageProduction_output"
        elif method=="tp" or method == "totalProduction":
            s.title = "production"
            s.method = "totalProduction"
            s._combine = "totalProduction_combiner"
            s._finalize = "totalProduction_finalizer"
            s._output = "totalProduction_output"
        elif method=="batch":
            try:
                r = outputPath%10
            except:
                try:
                    r = outputPath.format(10)
                except:
                    raise RuntimeError("output path string should handle integer formating")

            s.outputPath = outputPath
            s.skipFinalOutput=True
            s._combine = "batch_combiner"
            s._finalize = "batch_finalizer"
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