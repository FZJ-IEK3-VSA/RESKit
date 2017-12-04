import netCDF4 as nc
import numpy as np
import pandas as pd
from os.path import join
import geokit as gk
import matplotlib.pyplot as plt

from res.windpower import *
from res.weather import MerraSource, computeContextMean
from res.util import ResError, Location

## make some constants
windspeed = np.linspace(0,35,351)
windspeeds = np.column_stack([windspeed, windspeed+2, windspeed+5])

perf = np.array([(1,0),(2,0),(3,58),(4,185),(5,400),(6,745),(7,1200),(8,1790),(9,2450),(10,3120),(11,3660),(12,4000),
        (13,4150),(14,4200),(15,4200),(16,4200),(17,4200),(18,4200),(19,4200),(20,4200),(21,4200),(22,4200),
        (23,4200),(24,4200),(25,4200)])

## Testing scripts
def test_turbineReading():
    print("Testing reading from the Turbine Library...")

    turb = TurbineLibrary.ix["E-126_EP4"]

    if not turb.Capacity == 4200: raise RuntimError("Capacity mismatch")
    if not turb.Usage == "Onshore": raise RuntimError("Usage mismatch")
    if not turb.Manufacturer == "Enercon": raise RuntimError("Manufacturer mismatch")

    for i,tp in enumerate(perf):
        if not (tp==turb.Performance[i]).all(): raise RuntimError("Performance mismatch")

    print("  Success!")

def test_simulateTurbine():
    print("Testing simple simulation of single turbines...")

    # test simple setup
    p,c = simulateTurbine( windspeed, performance=perf, loss=0)
    if abs(c-0.476653977818)<1e-6: # Manually evaluated
        print("  Single Simulation: Success")
    else: raise RuntimeError("Single Simulation: Fail")

    # Test adding losses
    p,c = simulateTurbine( windspeed, performance=perf, loss=0.08)
    if abs(c-0.438533499183)<1e-6: # Manually evaluated
        print("  Single simulation with loss: Success")
    else: raise RuntimeError("Single simulation with loss: Fail")

    # Test choosing turbine by name
    p,c = simulateTurbine( windspeed, performance="G80", loss=0.08)
    perfG80 = np.array(TurbineLibrary.ix["G80"].Performance)
    if abs(c-0.42599635)<1e-6: # Manually evaluated
        print("  Single simulation turbine identified by name: Success")
    else: raise RuntimeError("Single simulation turbine identified by name: Fail")

    # Test projecting to a hub height by the log law
    p,c = simulateTurbine( windspeed, performance="G80", loss=0.08, measuredHeight=20, hubHeight=100, roughness=0.02)
    #plt.plot(windspeed*np.log(100/0.02)/np.log(20/0.02), p,'o', markersize=12)
    #plt.plot(perfG80[:,0],perfG80[:,1]*0.92)
    #plt.show()
    #print(c)
    if abs(c-0.343758865647)<1e-6: # Manually evaluated
        print("  Single simulation turbine with log-law projection: Success")
    else: raise RuntimeError("Single simulation turbine with log-law projection: Fail")

    # Test projecting to a hub height by the log law
    p,c = simulateTurbine( windspeed, performance="G80", loss=0.08, measuredHeight=20, hubHeight=100, alpha=0.14)
    #plt.plot(windspeed*np.power(100/20,0.14), p,'o', markersize=12)
    #plt.plot(perfG80[:,0],perfG80[:,1]*0.92)
    #plt.show()
    #print(c)
    if abs(c-0.338839384712)<1e-6: # Manually evaluated
        print("  Single simulation turbine with power-law projection: Success")
    else: raise RuntimeError("Single simulation turbine with power-law projection: Fail")

    # Test multiple simultaneous simulations
    p,c = simulateTurbine( windspeeds, performance="G80", loss=0.08)
    #plt.plot(windspeeds[:,0], p[:,0],'o', markersize=12)
    #plt.plot(windspeeds[:,1], p[:,1],'o', markersize=8)
    #plt.plot(windspeeds[:,2], p[:,2],'o', markersize=3)
    #plt.plot(perfG80[:,0],perfG80[:,1]*0.92)
    #plt.show()
    #print(c)
    if abs(c[0]-0.42599636)<1e-6 and abs(c[1]-0.42599636)<1e-6 and abs(c[2]-0.42446723)<1e-6: # Manually evaluated
        print("  Multiple turbine simulation: Success")
    else: raise RuntimeError("Multiple turbine simulation: Fail")

    # Test multiple simultaneous simulations with different paramaters
    p,c = simulateTurbine( windspeeds, performance="G80", loss=0.08, measuredHeight=np.array([20,15,10]), 
                           hubHeight=100, roughness=np.array([0.02, 0.3, 0.005]))
    #plt.plot(windspeeds[:,0]*np.log(100/0.02)/np.log(20/0.02), p[:,0],'o', markersize=12)
    #plt.plot(windspeeds[:,1]*np.log(100/0.3)/np.log(15/0.3), p[:,1],'o', markersize=8)
    #plt.plot(windspeeds[:,2]*np.log(100/0.005)/np.log(10/0.005), p[:,2],'o', markersize=3)
    #plt.plot(perfG80[:,0],perfG80[:,1]*0.92)
    #plt.show()
    #print(c)
    if abs(c[0]-0.34375887)<1e-6 and abs(c[1]-0.28637327)<1e-6 and abs(c[2]-0.32007049)<1e-6: # Manually evaluated
        print("  Multiple turbine simulation plus other variable parameters: Success")
    else: raise RuntimeError("Multiple turbine simulation plus other variable parameters: Fail")


def test_singleTurbineWorkflow():
    print("Testing single turbine workflow...")

    # make a datasource
    ms = MerraSource(join("data","merra-like.nc4"))
    ms.loadWindSpeed()

    loc1 = Location(lat=50.370680, lon=5.752684) 
    loc2 = Location(lat=50.52603, lon=6.10476) 
    loc3 = Location(lat=50.59082, lon=5.86483) 

    #### single turbine workflow
    # Get windspeeds
    windspeed = ms.get("windspeed", loc1)
    
    # Get roughnesses 
    r = windutil.roughnessFromCLC(join("data","clc-aachen_clipped.tif"), loc1)

    # simulate
    p,c = simulateTurbine(windspeed, performance="E-126_EP4", measuredHeight=50, hubHeight=100, roughness=r)

    #perfEP4 = np.array(TurbineLibrary.ix["E-126_EP4"].Performance)
    #plt.plot(windspeed*np.log(100/r)/np.log(50/r), p,'o', markersize=3)
    #plt.plot(perfEP4[:,0],perfEP4[:,1]*0.92)
    #plt.show()
    #print(c)
    if isinstance(p,pd.Series) and abs(c-0.593399911942)<1e-6: # Manually evaluated
        print("  Simple workflow with single turbine: Success")
    else: raise RuntimeError("Simple workflow with single turbine: Fail")

    #### multi-turbine workflow
    # Get windspeeds
    windspeeds = ms.get("windspeed", [loc1,loc2,loc3])
    
    # Get roughnesses 
    rs = windutil.roughnessFromCLC(join("data","clc-aachen_clipped.tif"), [loc1,loc2,loc3])

    # simulate
    p,c = simulateTurbine(windspeeds, performance="E-126_EP4", measuredHeight=50, hubHeight=100, roughness=rs)

    #perfEP4 = np.array(TurbineLibrary.ix["E-126_EP4"].Performance)
    #plt.plot(windspeeds[loc1]*np.log(100/rs[0])/np.log(50/rs[0]), p[loc1],'o', markersize=10)
    #plt.plot(windspeeds[loc2]*np.log(100/rs[1])/np.log(50/rs[1]), p[loc2],'o', markersize=7)
    #plt.plot(windspeeds[loc3]*np.log(100/rs[2])/np.log(50/rs[2]), p[loc3],'o', markersize=3)
    #plt.plot(perfEP4[:,0],perfEP4[:,1]*0.92)
    #plt.show()
    #for v in c: print(v)
    
    if isinstance(p,pd.DataFrame) and abs(c[loc1]-0.593399913759)<1e-6 and \
       abs(c[loc2]-0.559694601923)<1e-6 and abs(c[loc3]-0.605636343663)<1e-6: # Manually evaluated
        print("  Basic workflow with multiple turbines: Success")
    else: raise RuntimeError("Basic workflow with multiple turbines: Fail")

    
if __name__ == '__main__':
    test_turbineReading()
    test_simulateTurbine()
    test_singleTurbineWorkflow()
