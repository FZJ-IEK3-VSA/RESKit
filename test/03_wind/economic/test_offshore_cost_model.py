from reskit.wind.economic.offshore_cost_model import *
import numpy as np
import pytest
from reskit.default_paths import DEFAULT_PATHS
from reskit.parameters.parameters import OffshoreParameters
import reskit._test.data as pkg_data
from pathlib import Path
from importlib.resources import files, as_file

def test_calculateSpecificOffshoreCapex():
    
    c1= calculateSpecificOffshoreCapex(
        baseSpecCapex=1500, 
        capacity=10000,
        rotorDiam=250,
        hubHeight=150,
        waterDepth=100,
        coastDistance=41,
        portDistance=71,
        maxMonopileDepth=25,
        maxJacketDepth=55,
        baseDepth=17, 
        baseDistCoast=27, 
        baseWFSize=106858, 
        baseCap=None,
        baseHubHeight=None,
        baseRotorDiam=None,
        defaultOffshoreParamsFp=None,
        techYear=2050,
    )

    assert np.isclose(c1, 3393, rtol=0.05), "Error in calculateSpecificOffshoreCapex"

    #test Missing port distance
    c2= calculateSpecificOffshoreCapex(
        baseSpecCapex=1500, 
        capacity=10000,
        rotorDiam=250,
        hubHeight=150,
        waterDepth=100,
        coastDistance=41,
        portDistance=None,
        maxMonopileDepth=25,
        maxJacketDepth=55,
        baseDepth=17, 
        baseDistCoast=27, 
        baseWFSize=106858, 
        baseCap=None,
        baseHubHeight=None,
        baseRotorDiam=None,
        defaultOffshoreParamsFp=None,
        techYear=2050,
    )

    assert np.isclose(c2, 3520, rtol=0.05), "Error in calculateSpecificOffshoreCapex"
    
    
    
    c3= calculateSpecificOffshoreCapex(
        baseSpecCapex=[1500, 1501.5],
        capacity=[10000, 12000.5],
        rotorDiam=[250, 260.5],
        hubHeight=[150, 180.7],
        waterDepth=[30, 30.6],
        coastDistance=[41, 27.3],
        portDistance=[71, 29.8],
        maxMonopileDepth=25,
        maxJacketDepth=55,
        baseDepth=17, 
        baseDistCoast=27,
        baseWFSize=106858, 
        baseCap=None,
        baseHubHeight=None,
        baseRotorDiam=None,
        defaultOffshoreParamsFp=None,
        techYear=2050,
    )
    expected=np.array([1820, 1765])
    np.testing.assert_allclose(c3, expected, rtol=0.05), "Error in calculateSpecificOffshoreCapex"
    
  


def test_getSpecificOffshoreCableCost():
    c1 = getSpecificOffshoreCableCost(
        distance=1000,
        capacity= 14000,
        voltageType='dc',
        variableCostFactor=1.35,
        fixedCost=0,
        year=2050)
    assert np.isclose(c1, 1350), "Error in getCableCostfuncion, possibly due to adapted function"

    c2 = getSpecificOffshoreCableCost(
        distance=np.array([10000, 2000, 3000]),
        capacity= np.array([10000, 5, 5000]),
        voltageType=np.array(['dc', 'ac', 'dc']),
        variableCostFactor=np.array([2, 0.4, 20]),
        fixedCost=0,
        year=2050)
    expected = np.array([20000, 800, 60000])

    np.testing.assert_allclose(c2, expected, rtol=0.05), "Error in getCableCostfuncion, possibly due to adapted function"




def test_getOffshoreTurbineFoundationCost():
    
    c1=getOffshoreTurbineFoundationCost(
        depth=10.8,
        maxMonopileDepth=47.9,
        maxJacketDepth=60,
        year=2050,
        returnType=False
    )  
    assert np.isclose(c1, 195.57, rtol=0.05), "Error in getoffshoreTurbineFoundationCostFunction, possibly due to adapted function"

    c2=getOffshoreTurbineFoundationCost(
        depth=np.array([10, 30, 60]),
        maxMonopileDepth=25,
        maxJacketDepth=55,
        year=2050,
        returnType=False
    )

    expected=np.array([192.31, 301.4, 883.48])
    np.testing.assert_allclose(c2, expected, rtol=0.05), "Error in getoffshoreTurbineFoundationCostFunction, possibly due to adapted function"




def test_getSpecificOffshorePlatformCost():
    c1 = getSpecificOffshorePlatformCost(
        applicationType= "ac",
        capacity= 10000,
        waterDepth= 55,
        portDistance=100,
        foundationType= "jacket",
        maxJacketDepth= 55,
        convention="RogeauEtAl2023"
    )
    assert np.isclose(c1, 147, rtol=0.05)

    c2 = getSpecificOffshorePlatformCost(
        capacity=10000,
        applicationType="dc",  # DC substation offshore
        waterDepth=56,  # floating water depth
        foundationType="jacket",  # jacket given but too deep -> warning, no error
        maxJacketDepth=55,
        portDistance=100,
        convention="RogeauEtAl2023",  # Rogeau et al
    )
    assert np.isclose(c2, 88, rtol=0.05)

    c3 = getSpecificOffshorePlatformCost(
        applicationType=np.array(["electrolysis","ac","dc"]),  # central offshore electrolysis
        capacity=np.array([10000, 10000, 10000]),

        waterDepth=np.array([55, 55, 55]),  # jacket water depth
        foundationType=np.array(["floating","floating","floating",]),  # jacket would have been possibel, too, but floating allowed
        maxJacketDepth=55,
        portDistance=np.array([100, 100, 100]),
        convention="RogeauEtAl2023",  # Rogeau et al
    )
    expected = np.array([264.31700901, 155.13950901, 191.53200901])

    np.testing.assert_allclose(c3, expected, rtol=0.05)

    # TEST MUST-FAIL CASES
    with pytest.raises(Exception):
        getSpecificOffshorePlatformCost(
            applicationType= "ac",
            capacity= 10000,
            waterDepth= None,
            portDistance=100,
            foundationType= "jacket",
            maxJacketDepth= 55,
            convention="RogeauEtAl2023",
        )

    with pytest.raises(Exception):
        getSpecificOffshorePlatformCost(
            capacity=10000,
            applicationType="ac",
            waterDepth=50,
            foundationType="does_not_exist",  # must fail
            maxJacketDepth=55,
            portDistance=100,
            convention="RogeauEtAl2023",
        )

    with pytest.raises(Exception):
        getSpecificOffshorePlatformCost(
            capacity=10000,
            applicationType="aC",
            waterDepth=-1,  # must fail
            foundationType=None,
            maxJacketDepth=55,
            portDistance=100,
            convention="RogeauEtAl2023",
        )


def test_getSpecificConverterStationCost():
 
    c1 = getSpecificConverterStationCost(
        capacity=10000,
        waterDepth=20,  
        voltageType="ac",
        portDistance=1000,
        maxJacketDepth= 55,
        convention="RogeauEtAl2023",
    )

    assert np.isclose(c1, 246, rtol=0.05)


    # test None as waterdepth
    c2 = getSpecificConverterStationCost(
        capacity=10000,
        waterDepth=None,  
        voltageType="ac",
        portDistance=1000,
        maxJacketDepth= 55,
        convention="RogeauEtAl2023",
    )

    assert np.isclose(c2, 23.2, rtol=0.05)
    # array test
    c3 = getSpecificConverterStationCost(
        capacity=np.array([10000,20000,30000]),
        waterDepth=np.array([20, 25, 30]),
        voltageType=np.array(["ac", "ac", "dc"]),
        portDistance=np.array([1000, 3000, 10]),
        maxJacketDepth= 55,
        convention="RogeauEtAl2023",
    )

    expected = np.array([245.92559009, 238.06263514, 193.16463363])

    np.testing.assert_allclose(c3, expected, rtol=0.05)

    # TEST MUST-FAIL CASES

    with pytest.raises(Exception):
        getSpecificConverterStationCost(
            capacity=10000,
            waterDepth=55,  # jacket depth
            voltageType="does_not_exist", 
            portDistance=1000, # must fail
            maxJacketDepth=55,
            convention="RogeauEtAl2023",
        )


def test_getSpecificOffshoreConnectionCost():
    c1= getSpecificOffshoreConnectionCost(
        capacity=np.array([10000,20000,30000]),
        waterDepth=np.array([20, 25, 30]),
        coastDistance=np.array([1000, 3000, 10]),
        year=2050,
        voltageType='ac',
        portDistance=np.array([1000, 3000, 10]),
        baseWFSize=20000,
        maxJacketDepth= 55,
    )
    expected = (np.array([7135.17077252, 21411.09138514,   261.09967568]),'ac')
    np.testing.assert_allclose(c1[0], expected[0], rtol=0.05), "Error in getSpecificOffshoreConnectionCost, possibly due to adapted function"
    assert c1[1] == expected[1], "Error in getSpecificOffshoreConnectionCost, possibly due to adapted function"
