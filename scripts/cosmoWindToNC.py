#### Setup command line arguments
import argparse

parser = argparse.ArgumentParser(description='Process single-timestep 3D wind grib files and construct an output NC file')
parser.add_argument(dest="uFile", type=str, help='path to the file containing 3D U wind speeds')
parser.add_argument(dest="vFile", type=str, help='path to the file containing 3D V wind speeds')
parser.add_argument('-o', dest='oFile', type=str, default="CHANGEME", 
                    help='path to the desired output file')

parser.add_argument(dest='cFile', type=str, help='path to the file containing domain constants')

args = parser.parse_args()

#### Imports
import netCDF4 as nc
import pygrib
from collections import OrderedDict, defaultdict
import pandas as pd
from datetime import datetime as dt
import numpy as np

#### Arrange constants
constDS = nc.Dataset(args.cFile)
levels = constDS["level"][:]
if levels[1] < levels[0]: raise RuntimeError("levels need to be in increasing order")

midHeights = OrderedDict()
for l,mh in zip(levels,constDS["cell_mean"][:]):
    midHeights[l] = mh

#######

# find levels around desired heights
heights = [10, 100, 140]
heightLevels = []
levelsToSaveU = {}
levelsToSaveV = {}
for h in heights:
    for i,l in enumerate(levels):
        if midHeights[l] <= h:
            heightLevels.append((levels[i-1],levels[i]))

            # add levels to the search list
            levelsToSaveU[levels[i-1]] = False
            levelsToSaveU[levels[i]] = False
            
            levelsToSaveV[levels[i-1]] = False
            levelsToSaveV[levels[i]] = False

            break

#### Read WS data
grbs = pygrib.open(args.uFile)

# find data for each height
rawWSU = OrderedDict()
timeVal = None

grbs.seek(0)
for grb in grbs:
    # Check if we have a wind speed
    if not grb.indicatorOfParameter == 33:
        continue

    gLvl = grb.bottomLevel

    # Check if we want this level
    if not gLvl in levelsToSaveU: continue

    # Make sure we have the right time
    tmp = dt(year=grb.year, month=grb.month, day=grb.day, hour=grb.hour, minute=grb.minute)
    timeValTmp = (tmp - dt(year=1900, month=1, day=1, hour=0, minute=0)).total_seconds()//60
    if timeVal is None:
        timeVal = timeValTmp
    else: 
        if timeVal != timeValTmp: raise RuntimeError("Time mismatch")

    # Store data
    rawWSU[gLvl] = grb.values
    Ny,Nx = rawWSU[gLvl].shape
    levelsToSaveU[gLvl]=True

#### Read WS data
grbs = pygrib.open(args.vFile)

# find data for each height
rawWSV = OrderedDict()

grbs.seek(0)
for grb in grbs:
    # Check if we have a wind speed
    if not grb.indicatorOfParameter == 34:
        continue

    gLvl = grb.bottomLevel

    # Check if we want this level
    if not gLvl in levelsToSaveV: continue

    # Make sure we have the right time
    tmp = dt(year=grb.year, month=grb.month, day=grb.day, hour=grb.hour, minute=grb.minute)
    timeValTmp = (tmp - dt(year=1900, month=1, day=1, hour=0, minute=0)).total_seconds()//60
    if timeVal != timeValTmp: raise RuntimeError("Time mismatch")

    # Store data
    rawWSV[gLvl] = grb.values
    Ny,Nx = rawWSV[gLvl].shape
    levelsToSaveV[gLvl]=True

# make sure we found all required levels
for level,found in levelsToSaveU.items():
    if not found:
        raise RuntimeError("level %d not found"%level)
for level,found in levelsToSaveV.items():
    if not found:
        raise RuntimeError("level %d not found"%level)

# Do interpolations
ws = np.zeros((1,len(heights), Ny, Nx) )
for i,h in enumerate(heights):
    lvl0 = heightLevels[i][0]
    lvl1 = heightLevels[i][1]

    ws0 = np.sqrt(rawWSU[lvl0]*rawWSU[lvl0] + rawWSV[lvl0]*rawWSV[lvl0])
    ws1 = np.sqrt(rawWSU[lvl1]*rawWSU[lvl1] + rawWSV[lvl1]*rawWSV[lvl1])

    a = np.log(ws0/ws1)/np.log(midHeights[lvl0]/midHeights[lvl1])

    ws[0,i,:,:] = ws0 * np.power(h/midHeights[lvl0], a)

#### Make output NC file

if args.oFile == "CHANGEME":
    oFile = args.uFile.replace(".grb",".nc")
else:
    oFile = args.oFile

ods = nc.Dataset(oFile,"w")

# set dimensions
ods.createDimension("time", ws.shape[0] )
ods.createDimension("height", ws.shape[1] )
ods.createDimension("latI", ws.shape[2])
ods.createDimension("lonI", ws.shape[3])

# set time
var = ods.createVariable("time", int, ("time", ))
var.setncatts(dict(
    name="time",
    units="Minutes since 1900-01-01 00:00:00"
    ))
var[:] = [ int(timeVal), ]

# set heights
var = ods.createVariable("height", int, ("height", ))
var.setncatts(dict(
    name="Height above surface",
    units="m"
    ))
var[:] = heights

# set windspeed
var = ods.createVariable("wspd", "f", ("time", "height", "latI", "lonI"))
var.setncatts(dict(
    name="Wind speed",
    units="m/s",
    ))
var[:] = ws

# DONE!
ods.close()