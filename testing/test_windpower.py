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

perf = [(1,0.0),(2,0.0),(3,0.0138095238095),(4,0.0440476190476),(5,0.0952380952381),(6,0.177380952381),
        (7,0.285714285714),(8,0.42619047619),(9,0.583333333333),(10,0.742857142857),(11,0.871428571429),
        (12,0.952380952381),(13,0.988095238095),(14,1.0),(15,1.0),(16,1.0),(17,1.0),(18,1.0),(19,1.0),
        (20,1.0),(21,1.0),(22,1.0),(23,1.0),(24,1.0),(25,1.0),]

## Testing scripts
def test_SyntheticPowerCurve():
    print("Testing synthetic power curve creation...")

    rotordiameter = 120
    capacity = 3000

    pc = SyntheticPowerCurve( capacity=capacity, rotordiam=rotordiameter, cutout=30)
    if not (pc.ws[0]==0 and pc.cf[0]==0).all(): raise RuntimeError("Failed")
    if not (pc.ws[-1]==30 and pc.cf[-1]==1.0).all(): raise RuntimeError("Failed")

    if not np.isclose(pc.ws.sum(),794.083249603): raise RuntimeError("Failed")
    if not np.isclose(pc.cf.sum(),51.5): raise RuntimeError("Failed")

    print("  Success!")

def test_simulateTurbine():
    print("Testing simple simulation of single turbines...")

    # test simple setup
    p = simulateTurbine( windspeed, powerCurve=perf, loss=0)
    if abs(p.mean()-0.476666846938)<1e-6: # Manually evaluated
        print("  Single Simulation: Success")
    else: raise RuntimeError("Single Simulation: Fail")

    # Test adding losses
    p = simulateTurbine( windspeed, powerCurve=perf, loss=0.08)
    if abs(p.mean()-0.438533499183)<1e-6: # Manually evaluated
        print("  Single simulation with loss: Success")
    else: raise RuntimeError("Single simulation with loss: Fail")

    # Test choosing turbine by name
    p = simulateTurbine( windspeed, powerCurve="G80", loss=0.08)
    perfG80 = np.array(TurbineLibrary.ix["G80"].PowerCurve)
    if abs(p.mean()-0.42599635)<1e-6: # Manually evaluated
        print("  Single simulation turbine identified by name: Success")
    else: raise RuntimeError("Single simulation turbine identified by name: Fail")

    # Test projecting to a hub height by the log law
    p = simulateTurbine( windspeed, powerCurve="G80", loss=0.08, measuredHeight=20, hubHeight=100, roughness=0.02)

    if abs(p.mean()-0.343758865647)<1e-6: # Manually evaluated
        print("  Single simulation turbine with log-law projection: Success")
    else: raise RuntimeError("Single simulation turbine with log-law projection: Fail")

    # Test projecting to a hub height by the log law
    p = simulateTurbine( windspeed, powerCurve="G80", loss=0.08, measuredHeight=20, hubHeight=100, alpha=0.14)

    if abs(p.mean()-0.338839384712)<1e-6: # Manually evaluated
        print("  Single simulation turbine with power-law projection: Success")
    else: raise RuntimeError("Single simulation turbine with power-law projection: Fail")

    # Test multiple simultaneous simulations
    p = simulateTurbine( windspeeds, powerCurve="G80", loss=0.08)

    if abs(p.mean()[0]-0.42599636)<1e-6 and abs(p.mean()[1]-0.42599636)<1e-6 and abs(p.mean()[2]-0.42446723)<1e-6: # Manually evaluated
        print("  Multiple turbine simulation: Success")
    else: raise RuntimeError("Multiple turbine simulation: Fail")

    # Test multiple simultaneous simulations with different paramaters
    p = simulateTurbine( windspeeds, powerCurve="G80", loss=0.08, measuredHeight=np.array([20,15,10]), 
                           hubHeight=100, roughness=np.array([0.02, 0.3, 0.005]))

    if abs(p.mean()[0]-0.34375887)<1e-6 and abs(p.mean()[1]-0.28637327)<1e-6 and abs(p.mean()[2]-0.32007049)<1e-6: # Manually evaluated
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
    p = simulateTurbine(windspeed, powerCurve="E-126_EP4", measuredHeight=50, hubHeight=100, roughness=r)

    if isinstance(p,pd.Series) and abs(p.mean()-0.593399911942)<1e-6: # Manually evaluated
        print("  Simple workflow with single turbine: Success")
    else: raise RuntimeError("Simple workflow with single turbine: Fail")

    #### multi-turbine workflow
    # Get windspeeds
    windspeeds = ms.get("windspeed", [loc1,loc2,loc3])
    
    # Get roughnesses 
    rs = windutil.roughnessFromCLC(join("data","clc-aachen_clipped.tif"), [loc1,loc2,loc3])

    # simulate
    p = simulateTurbine(windspeeds, powerCurve="E-126_EP4", measuredHeight=50, hubHeight=100, roughness=rs)

    if isinstance(p,pd.DataFrame) and \
       abs(p.mean()[loc1]-0.593399913759)<1e-6 and \
       abs(p.mean()[loc2]-0.559694601923)<1e-6 and \
       abs(p.mean()[loc3]-0.605636343663)<1e-6: # Manually evaluated
        print("  Basic workflow with multiple turbines: Success")
    else: raise RuntimeError("Basic workflow with multiple turbines: Fail")

def test_WindWorkflow():
    print("##################################")
    print("## Make test for WindWorkflow!!!")
    print("##################################")


if __name__ == '__main__':
    test_SyntheticPowerCurve()
    test_simulateTurbine()
    test_singleTurbineWorkflow()
    test_WindWorkflow()
