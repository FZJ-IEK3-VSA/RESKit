from res.windpower import *

import netCDF4 as nc
import geokit as gk
import numpy as np
import pandas as pd
from os.path import join

MERRA = join("data","merra-like.nc4")
GWA = join("data","gwa50-like.tif")
CLC = join("data","clc-aachen_clipped.tif")

tmp = gk.Extent.fromVector(join("data","aachenShapefile.shp")).castTo(4326)
np.random.seed(0)
x = np.random.random(500)*(tmp.xMax-tmp.xMin) + tmp.xMin
y = np.random.random(500)*(tmp.yMax-tmp.yMin) + tmp.yMin

LOCS = list(zip(x,y))
DF = pd.DataFrame({"lon":x,"lat":y})
#######################################
## Start Tests
def test_parameterizations():
    print("\nTesting basic parameterizations...")
    # Basic parameters
    r = WindWorkflow(placements=LOCS, merra=MERRA, landcover=CLC, gwa=GWA, lctype="clc", verbose=False,
                      hubHeight=100, powerCurve="E-126_EP4", extract="capacityFactor")

    if not (np.isclose(r.mean(),0.250334187462) and np.isclose(r.std(), 0.0551108465389)):
        raise RuntimeError("Failed")
    else:
        print("  Loc list okay")

    # Synthetic turbine
    rs = WindWorkflow(placements=LOCS, merra=MERRA, landcover=CLC, gwa=GWA, lctype="clc", verbose=False,
                      hubHeight=100, capacity=4000, rotordiam=120, extract="capacityFactor")

    if not (np.isclose(rs.mean(),0.241973338828) and np.isclose(rs.std(), 0.0548289231397)):
        raise RuntimeError("Failed")
    else:
        print("  Synthetic okay")

    # From shapefile
    path = join("data","turbinePlacements.shp")
    rshp = WindWorkflow(placements=path, merra=MERRA, landcover=CLC, gwa=GWA, lctype="clc", verbose=False,
                      powerCurve="E-126_EP4", hubHeight=100, extract="capacityFactor")

    if not (np.isclose(rshp.mean(),0.272063331016) and np.isclose(rshp.std(), 0.0442215953326)):
        raise RuntimeError("Failed")
    else:
        print("  Shapefile okay")

    # Locs as dataFrame
    rt = WindWorkflow(placements=DF, merra=MERRA, landcover=CLC, gwa=GWA, lctype="clc", verbose=False,
                      hubHeight=100, powerCurve="E-126_EP4", extract="capacityFactor")
    if not np.isclose(r,rt).all():
        raise RuntimeError("Failed")
    else:
        print("  DataFrame okay")

    # Everything in a DF
    DF2 = DF.copy()
    DF2["hubHeight"] = [100]*DF2.shape[0]
    DF2["turbine"] = ["E-126_EP4",]*DF2.shape[0]
    rt = WindWorkflow(placements=DF2, merra=MERRA, landcover=CLC, gwa=GWA, lctype="clc", verbose=False,
                      extract="capacityFactor")
    if not np.isclose(r,rt).all():
        raise RuntimeError("Failed")
    else:
        print("  Filled DataFrame okay")

    DF2 = DF.copy()
    DF2["hubHeight"] = [100]*DF2.shape[0]
    DF2["capacity"] = [4000,]*DF2.shape[0]
    DF2["rotordiam"] = [120,]*DF2.shape[0]
    rt = WindWorkflow(placements=DF2, merra=MERRA, landcover=CLC, gwa=GWA, lctype="clc", verbose=False,
                      extract="capacityFactor")
    if not np.isclose(rs,rt).all():
        raise RuntimeError("Failed")
    else:
        print("  Filled DataFrame and Synthetic okay")

    # Everything in a DF, multiple turbines
    DF2 = DF.copy()
    DF2["hubHeight"] = [100]*DF2.shape[0]
    DF2["turbine"] = ["E-126_EP4","V136-3450"]*int(DF2.shape[0]//2)
    rt = WindWorkflow(placements=DF2, merra=MERRA, landcover=CLC, gwa=GWA, lctype="clc", verbose=False,
                      extract="capacityFactor")
    if not np.isclose(r[::2],rt[::2]).all() or np.isclose(r[1::2],rt[1::2]).all():
        raise RuntimeError("Failed")
    else:
        print("  Filled DataFrame, Multi-turbine okay")

    DF2 = DF.copy()
    DF2["hubHeight"] = [100]*DF2.shape[0]
    DF2["capacity"] = [4000,]*DF2.shape[0]
    DF2["rotordiam"] = [120,100,]*int(DF2.shape[0]//2)
    rt = WindWorkflow(placements=DF2, merra=MERRA, landcover=CLC, gwa=GWA, lctype="clc", verbose=False,
                      extract="capacityFactor")
    if not np.isclose(rs[::2],rt[::2]).all() or np.isclose(rs[1::2],rt[1::2]).all() :
        raise RuntimeError("Failed")
    else:
        print("  Filled DataFrame, Synthetic, Multi-turbine okay")

def test_multicore():
    print("\nTesting multicore simulations...")
    # Basic parameters
    r = WindWorkflow(placements=DF, merra=MERRA, landcover=CLC, gwa=GWA, lctype="clc", verbose=False,
                     hubHeight=100, powerCurve="E-126_EP4", extract="capacityFactor", jobs=2, batchSize=100)

    if not (np.isclose(r.mean(),0.250334187462) and np.isclose(r.std(), 0.0551108465389)):
        raise RuntimeError("Failed")
    else:
        print("  Loc list okay")

    # Synthetic turbine
    rs = WindWorkflow(placements=LOCS, merra=MERRA, landcover=CLC, gwa=GWA, lctype="clc", verbose=False,
                      hubHeight=100, capacity=4000, rotordiam=120, extract="capacityFactor", jobs=2, batchSize=100)

    if not (np.isclose(rs.mean(),0.241973338828) and np.isclose(rs.std(), 0.0548289231397)):
        raise RuntimeError("Failed")
    else:
        print("  Synthetic okay")

    # Everything in a DF, multiple turbines
    DF2 = DF.copy()
    DF2["hubHeight"] = [100]*DF2.shape[0]
    DF2["turbine"] = ["E-126_EP4","V136-3450"]*int(DF2.shape[0]//2)
    rt = WindWorkflow(placements=DF2, merra=MERRA, landcover=CLC, gwa=GWA, lctype="clc", verbose=False,
                      extract="capacityFactor", jobs=2, batchSize=100)
    if not np.isclose(r[::2],rt[::2]).all() or np.isclose(r[1::2],rt[1::2]).all():
        raise RuntimeError("Failed")
    else:
        print("  Filled DataFrame, Multi-turbine okay")

    DF2 = DF.copy()
    DF2["hubHeight"] = [100]*DF2.shape[0]
    DF2["capacity"] = [4000,]*DF2.shape[0]
    DF2["rotordiam"] = [120,100,]*int(DF2.shape[0]//2)
    rt = WindWorkflow(placements=DF2, merra=MERRA, landcover=CLC, gwa=GWA, lctype="clc", verbose=False,
                      extract="capacityFactor", jobs=2, batchSize=100)
    if not np.isclose(rs[::2],rt[::2]).all() or np.isclose(rs[1::2],rt[1::2]).all() :
        raise RuntimeError("Failed")
    else:
        print("  Filled DataFrame, Synthetic, Multi-turbine okay")

def test_extractions():
    print("\nTesting extraction schemes...")
    DF2 = DF.copy()
    DF2["capacity"] = [4000,3000]*int(DF2.shape[0]//2)
    DF2["rotordiam"] = [120,100,]*int(DF2.shape[0]//2)

    # capacityFactor parameters
    rcf = WindWorkflow(placements=DF2, merra=MERRA, landcover=CLC, gwa=GWA, lctype="clc", verbose=False,
                      hubHeight=100, extract="capacityFactor", jobs=2, batchSize=100)

    if not (np.isclose(rcf.mean(), 0.234663877658) and np.isclose(rcf.std(), 0.0545827949705)):
        raise RuntimeError("Failed")
    else:
        print("  capacityFactor okay")

    # raw production
    rp = WindWorkflow(placements=DF2, merra=MERRA, landcover=CLC, gwa=GWA, lctype="clc", verbose=False,
                      hubHeight=100, extract="production", jobs=2, batchSize=100)

    if not np.isclose( rp.mean(axis=0)/DF2.capacity.values, rcf ).all():
        raise RuntimeError("Failed")
    else:
        print("  production okay")

    # average production
    rap = WindWorkflow(placements=DF2, merra=MERRA, landcover=CLC, gwa=GWA, lctype="clc", verbose=False,
                       hubHeight=100, extract="averageProduction", jobs=2, batchSize=100)

    if not np.isclose( rp.mean(axis=1), rap ).all():
        raise RuntimeError("Failed")
    else:
        print("  averageProduction okay")

def test_outputs():
    print("\nTesting output schemes...")
    DF2 = DF.copy()
    DF2["capacity"] = [4000,3000]*int(DF2.shape[0]//2)
    DF2["rotordiam"] = [120,100,]*int(DF2.shape[0]//2)

    raw = WindWorkflow(placements=DF2, merra=MERRA, landcover=CLC, gwa=GWA, lctype="clc", verbose=False,
                       hubHeight=100, extract="production", jobs=2, batchSize=100)

    # capacityFactor outputs
    out = join("outputs","cf.csv")
    WindWorkflow(placements=DF2, merra=MERRA, landcover=CLC, gwa=GWA, lctype="clc", verbose=False,
                 hubHeight=100, extract="capacityFactor", jobs=2, batchSize=100, output=out)

    with open(out) as fi:
        line = fi.readline()
        while not "RESULT-OUTPUT" in line: line = fi.readline()
        r = pd.read_csv(fi)

    if not np.isclose( raw.mean(axis=0)/DF2.capacity.values, r.capfac ).all():
        raise RuntimeError("Failed")
    else:
        print("  capacityFactor to CSV okay")


    out = join("outputs","cf.shp")
    WindWorkflow(placements=DF2, merra=MERRA, landcover=CLC, gwa=GWA, lctype="clc", verbose=False,
                 hubHeight=100, extract="capacityFactor", jobs=2, batchSize=100, output=out)

    r = gk.vector.extractAsDataFrame(out)

    if not np.isclose( raw.mean(axis=0)/DF2.capacity.values, r.capfac ).all():
        raise RuntimeError("Failed")
    else:
        print("  capacityFactor to SHP okay")

    out = join("outputs","cf.nc")
    WindWorkflow(placements=DF2, merra=MERRA, landcover=CLC, gwa=GWA, lctype="clc", verbose=False,
                 hubHeight=100, extract="capacityFactor", jobs=2, batchSize=100, output=out)

    with nc.Dataset(out) as ds:
        r = pd.DataFrame({"capfac":ds["capfac"][:]})

    if not np.isclose( raw.mean(axis=0)/DF2.capacity.values, r.capfac ).all():
        raise RuntimeError("Failed")
    else:
        print("  capacityFactor to NC okay")

    # averageProduction outputs
    out = join("outputs","ap.csv")
    WindWorkflow(placements=DF2, merra=MERRA, landcover=CLC, gwa=GWA, lctype="clc", verbose=False,
                 hubHeight=100, extract="averageProduction", jobs=2, batchSize=100, output=out)

    with open(out) as fi:
        line = fi.readline()
        while not "RESULT-OUTPUT" in line: line = fi.readline()
        r = pd.read_csv(fi, index_col=0)

    if not np.isclose( raw.mean(axis=1), r["production"] ).all():
        raise RuntimeError("Failed")
    else:
        print("  averageProduction to CSV okay")

    out = join("outputs","cf.nc")
    WindWorkflow(placements=DF2, merra=MERRA, landcover=CLC, gwa=GWA, lctype="clc", verbose=False,
                 hubHeight=100, extract="averageProduction", jobs=2, batchSize=100, output=out)

    ds = nc.Dataset(out)

    if not np.isclose( raw.mean(axis=1).values, ds["avgProduction"][:] ).all():
        raise RuntimeError("Failed")
    else:
        print("  averageProduction to NC okay")

    # production outputs
    out = join("outputs","raw.csv")
    WindWorkflow(placements=DF2, merra=MERRA, landcover=CLC, gwa=GWA, lctype="clc", verbose=False,
                 hubHeight=100, extract="raw", jobs=2, batchSize=100, output=out)

    with open(out) as fi:
        line = fi.readline()
        while not "RESULT-OUTPUT" in line: line = fi.readline()
        r = pd.read_csv(fi, index_col=0)

    if not np.isclose( raw, r ).all():
        raise RuntimeError("Failed")
    else:
        print("  raw to CSV okay")

    out = join("outputs","raw.nc")
    WindWorkflow(placements=DF2, merra=MERRA, landcover=CLC, gwa=GWA, lctype="clc", verbose=False,
                 hubHeight=100, extract="raw", jobs=2, batchSize=100, output=out)

    ds = nc.Dataset(out)

    if not np.isclose( raw.values, ds["production"][:] ).all():
        raise RuntimeError("Failed")
    else:
        print("  raw to NC okay")

    # batch outputs
    out = join("outputs","batch_%02d.csv")
    WindWorkflow(placements=DF2, merra=MERRA, landcover=CLC, gwa=GWA, lctype="clc", verbose=False,
                 hubHeight=100, extract="batch", jobs=2, batchSize=100, output=out)

    print("  batch to CSV okay")

    out = join("outputs","batch_%02d.nc")
    WindWorkflow(placements=DF2, merra=MERRA, landcover=CLC, gwa=GWA, lctype="clc", verbose=False,
                 hubHeight=100, extract="batch", jobs=2, batchSize=100, output=out)

    print("  batch to NC okay")

if __name__ == "__main__":
    test_parameterizations()
    test_multicore()
    test_extractions()
    test_outputs()

