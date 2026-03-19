# %% [markdown]
# # Suggest Offshore Wind Turbine 
# 
# 

# %%
import reskit as rk
from reskit.util.local_values import waterDepthFromLocation
from reskit.wind import calculateSpecificOffshoreCapex
from reskit.util.local_values import distanceToCoastline

import geokit as gk


# %%
waterDepthFilePath = gk.raster.rasterInfo(rk.TEST_DATA["water_depth_northsea.tif"]).source
lon = 6.5
lat = 54.8

depth = abs(waterDepthFromLocation(
    latitude=lat,
    longitude=lon,
    waterDepthFilePath=waterDepthFilePath,
))

print(f"Water depth: {depth:.0f} m")


coastDistanceFilePath = gk.raster.rasterInfo(rk.TEST_DATA["coast_distance_northsea.tif"]).source

distance = distanceToCoastline(
    latitude=lat,
    longitude=lon,
    distancetoCoastFilePath=coastDistanceFilePath,
)

print(f"Distance: {distance:.0f} m")

# %%
capex = calculateSpecificOffshoreCapex(
    baseSpecCapex = 1000,
    capacity = 10,
    rotorDiam = 50,
    hubHeight = 100,
    waterDepth = int(depth),
    coastDistance = int(distance),
    portDistance =int(distance),
    maxMonopileDepth=25,
    maxJacketDepth=55,
    baseDepth=17,
    baseDistCoast=27,
    baseWFSize=106858,
    baseCap=13000,
    baseHubHeight=150,
    baseRotorDiam=250,
    defaultOffshoreParamsFp=None,
    techYear=2050,
)


print(f"CAPEX = {capex}")


