from reskit.wind.economic.offshore_cost_model import *
import numpy as np
import pytest
from reskit.default_paths import DEFAULT_PATHS
from reskit.parameters.parameters import OffshoreParameters
import reskit._test.data as pkg_data
from pathlib import Path
from importlib.resources import files, as_file


def test_calculateSpecificOffshoreCapex():
    # test normal behaviour of the function
    c1 = calculateSpecificOffshoreCapex(
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

    assert np.isclose(c1, 2654, rtol=0.05), "Error in calculateSpecificOffshoreCapex"

    # test missing port distance
    c2 = calculateSpecificOffshoreCapex(
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

    assert np.isclose(c2, 2654.89, rtol=0.05), "Error in calculateSpecificOffshoreCapex"

    # test arrays as input
    c3 = calculateSpecificOffshoreCapex(
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
    expected = np.array([2143, 2086.02])
    np.testing.assert_allclose(c3, expected, rtol=0.05), "Error in calculateSpecificOffshoreCapex"


def test_getSpecificOffshoreCableCost():
    # test normal behaviour of the function
    c1 = getSpecificOffshoreCableCost(
        distance=1000, capacity=14000, voltageType="dc", variableCostFactor=1.35, fixedCost=0, year=2050
    )
    assert np.isclose(c1, 1620), "Error in getCableCostfuncion, possibly due to adapted function"

    # test arrays as input
    c2 = getSpecificOffshoreCableCost(
        distance=np.array([10000, 2000, 3000]),
        capacity=np.array([10000, 5, 5000]),
        voltageType=np.array(["dc", "ac", "dc"]),
        variableCostFactor=np.array([2, 0.4, 20]),
        fixedCost=0,
        year=2050,
    )
    expected = np.array([24000, 960, 72000])

    (
        np.testing.assert_allclose(c2, expected, rtol=0.05),
        "Error in getCableCostfuncion, possibly due to adapted function",
    )


def test_getOffshoreTurbineFoundationCost():
    # test normal behaviour of the function
    c1 = getOffshoreTurbineFoundationCost(
        depth=10.8, maxMonopileDepth=47.9, maxJacketDepth=60, year=2050, returnType=False
    )
    assert np.isclose(c1, 195.57, rtol=0.05), (
        "Error in getoffshoreTurbineFoundationCostFunction, possibly due to adapted function"
    )

    # test arrays as input
    c2 = getOffshoreTurbineFoundationCost(
        depth=np.array([10, 30, 60]), maxMonopileDepth=25, maxJacketDepth=55, year=2050, returnType=False
    )

    expected = np.array([192.31, 301.4, 883.48])
    (
        np.testing.assert_allclose(c2, expected, rtol=0.05),
        "Error in getoffshoreTurbineFoundationCostFunction, possibly due to adapted function",
    )


def test_getSpecificOffshorePlatformCost():
    # test normal behaviour of the function
    c1 = getSpecificOffshorePlatformCost(
        applicationType="ac",
        capacity=10000,
        waterDepth=55,
        portDistance=100,
        foundationType="jacket",
        maxJacketDepthPlatform=55,
        convention="RogeauEtAl2023",
    )
    assert np.isclose(c1, 36.4, rtol=0.05)

    # test DC substation with floating foundation, even though jacket would be possible, but water depth is above max monopile depth
    c2 = getSpecificOffshorePlatformCost(
        capacity=10000,
        applicationType="dc",  # DC substation offshore
        waterDepth=56,  # floating water depth
        foundationType="jacket",  # jacket given but too deep -> warning, no error
        maxJacketDepthPlatform=55,
        portDistance=100,
        convention="RogeauEtAl2023",  # Rogeau et al
    )
    assert np.isclose(c2, 61.62, rtol=0.05)

    # test arrays as input
    c3 = getSpecificOffshorePlatformCost(
        applicationType=np.array(["electrolysis", "ac", "dc"]),
        capacity=np.array([10000, 10000, 10000]),
        waterDepth=np.array([55, 55, 55]),
        foundationType=np.array(
            [
                "floating",
                "floating",
                "floating",
            ]
        ),  # jacket would have been possibel, too, but floating allowed
        maxJacketDepthPlatform=55,
        portDistance=np.array([100, 100, 100]),
        convention="RogeauEtAl2023",
    )
    expected = np.array([152, 42.99, 79.4])

    np.testing.assert_allclose(c3, expected, rtol=0.05)

    # TEST MUST-FAIL CASES

    # test wrong foundation type
    with pytest.raises(Exception):
        getSpecificOffshorePlatformCost(
            capacity=10000,
            applicationType="ac",
            waterDepth=50,
            foundationType="does_not_exist",  # must fail
            maxJacketDepthPlatform=55,
            portDistance=100,
            convention="RogeauEtAl2023",
        )

    # test negative water depth
    with pytest.raises(Exception):
        getSpecificOffshorePlatformCost(
            capacity=10000,
            applicationType="aC",
            waterDepth=-1,  # must fail
            foundationType=None,
            maxJacketDepthPlatform=55,
            portDistance=100,
            convention="RogeauEtAl2023",
        )


def test_getSpecificConverterStationCost():
    # test normal behaviour of the function
    c1 = getSpecificConverterStationCost(
        capacity=10000,
        waterDepth=20,
        voltageType="ac",
        portDistance=1000,
        maxJacketDepthPlatform=55,
        convention="RogeauEtAl2023",
    )

    assert np.isclose(c1, 59.95, rtol=0.05)

    # test None as waterdepth
    c2 = getSpecificConverterStationCost(
        capacity=10000,
        waterDepth=None,
        voltageType="ac",
        portDistance=1000,
        maxJacketDepthPlatform=55,
        convention="RogeauEtAl2023",
    )

    assert np.isclose(c2, 23.2, rtol=0.05)
    # test arrays as input
    c3 = getSpecificConverterStationCost(
        capacity=np.array([10000, 20000, 30000]),
        waterDepth=np.array([20, 25, 30]),
        voltageType=np.array(["ac", "ac", "dc"]),
        portDistance=np.array([1000, 3000, 10]),
        maxJacketDepthPlatform=55,
        convention="RogeauEtAl2023",
    )

    expected = np.array([59.95, 70.4, 163.16])

    np.testing.assert_allclose(c3, expected, rtol=0.05)

    # TEST MUST-FAIL CASES

    # test wrong voltage type
    with pytest.raises(Exception):
        getSpecificConverterStationCost(
            capacity=10000,
            waterDepth=55,  # jacket depth
            voltageType="does_not_exist",
            portDistance=1000,  # must fail
            maxJacketDepthPlatform=55,
            convention="RogeauEtAl2023",
        )


def test_getSpecificOffshoreConnectionCost():
    # test normal behaviour of the function and arrays as input
    c1 = getSpecificOffshoreConnectionCost(
        capacity=np.array([10000, 20000, 30000]),
        waterDepth=np.array([20, 25, 30]),
        coastDistance=np.array([1000, 3000, 10]),
        year=2050,
        voltageType="ac",
        portDistance=np.array([1000, 3000, 10]),
        baseWFSize=20000,
        maxJacketDepthPlatform=55,
    )
    expected = (np.array([8543, 25473, 163.55]), "ac")
    (
        np.testing.assert_allclose(c1[0], expected[0], rtol=0.05),
        "Error in getSpecificOffshoreConnectionCost, possibly due to adapted function",
    )
    assert c1[1] == expected[1], "Error in getSpecificOffshoreConnectionCost, possibly due to adapted function"
