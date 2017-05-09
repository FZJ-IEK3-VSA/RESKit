from glob import glob
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np

import glaes as gl


files = glob("D:/Data/weather/merra/slv/*.nc4")[::-1]


windSpeeds = 0
N = len(files)

for f,i in zip(files,range(len(files))):
    
    if( i%100==0 ):
        print(100*i/N, f)
        
    ds = nc.Dataset(f)

    u = ds.variables["U10M"][:]
    v = ds.variables["V10M"][:]
    
    total = np.sqrt(u*u+v*v).mean(0)
    
    windSpeeds += total/N
    
    u = None
    v = None
    total = None
    ds = None

    

gl.createRaster( bounds=( -27.5-0.625/2, 25-0.5/2, 47.5+0.625/2, 80+0.5/2), output="merra_average_10m.tif", pixelWidth=0.625, pixelHeight=0.5, dtype=float, 
				 srs="latlon", data=windSpeeds[::-1,:], overwrite=True )
#np.savetxt("merra_average_50m.csv", windSpeeds, delimiter=",")