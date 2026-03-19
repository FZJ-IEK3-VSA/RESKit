from reskit.wind.economic.offshore_cost_model import *
import numpy as np
import pytest
from reskit.default_paths import DEFAULT_PATHS
from reskit.parameters.parameters import OffshoreParameters
import reskit._test.data as pkg_data
from pathlib import Path
from importlib.resources import files, as_file




def test_waterDepthFromLocation():
    # GPS Coordiantes for location in Aachen to test Reading from tif file
    lon = 5.983
    lat = 51.205


def get_pkg_file(name: str) -> Path:
    root = files(pkg_data)
    # erst übliche Stellen probieren
    for rel in (name, f"data/{name}"):
        cand = root.joinpath(*rel.split("/"))
        if cand.is_file():
            with as_file(cand) as p:
                return Path(p)
    # Fallback: rekursiv suchen
    for child in root.iterdir():
        if child.is_dir():
            for sub in child.iterdir():
                if sub.name == name and sub.is_file():
                    with as_file(sub) as p:
                        return Path(p)
    raise FileNotFoundError(name)

    tif_file = get_pkg_file("DEM-like.tif")
    value_exact = 19
    depth = waterDepthFromLocation(lat, lon, waterDepthFolderPath=tif_file)

    assert np.isclose(depth, value_exact), "the waterdepthfile is not working correct"


def test_calculateSpecificOffshoreCapex():
    comparedCAPEX = 2934.26
    calculatedCAPEX = calculateSpecificOffshoreCapex(
        baseCapex=3000,
        capacity=14000,
        hubHeight=150,
        waterDepth=25,
        coastDistance=25,
        rotorDiam=230,
        techYear=None,
        shareTurb=0.449,
        shareFound=0.204,
        shareCable=0.181,
        shareOverhead=0.166,
        maxMonopileDepth=25,
        maxJacketDepth=55,
        baseDepth=17,
        baseDistCoast=27,
        baseCap=13000,
        baseHubHeight=150,
        baseRotorDiam=250,
        defaultOffshoreParamsFp=None,
    )

    assert np.isclose(calculatedCAPEX, comparedCAPEX)



def test_getSpecificOffshoreCableCost():
    value1 = getSpecificOffshoreCableCost(10, 14000, variableCostFactor=1.35, fixedCost=0)

    value2 = getSpecificOffshoreCableCost(10, 10000, variableCostFactor=1.35, fixedCost=0)

    assert np.isclose(value1, 189000), (
        "Error in getCableCostfuncion, possibly due to adapted function"
    )
    assert np.isclose(value2, 135000), (
        "Error in getCableCostfuncion, possibly due to adapted function"
    )

def test_getOffshoreTurbineFoundationCost():
    pass

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
    # test onshore AC substation
    c1 = getSpecificConverterStationCost(
        capacity=10000,
        waterDepth=20,  
        portDistance=40,
        foundationType= "jacket",
        maxJacketDepth= 55,
        convention="RogeauEtAl2023",
    )
    # i wanmt to check wheter it is close to c1 andf not exactly c1, because the function is adapted and not exactly the same as in the paper, but the cost should be in the same order of magnitude
    assert np.isclose(c1, 1130, rtol=0.1)


    # test offshore DC substation
    c2 = getSpecificConverterStationCost(
        capacity=10000,
        waterDepth=55,  # jacket depth
        voltageType="dc",
        maxJacketDepth=55,
        convention="RogeauEtAl2023",
    )

    assert np.isclose(c2, 177, rtol=0.1)

    # TEST MUST-FAIL CASES

    with pytest.raises(Exception):
        getSpecificConverterStationCost(
            capacity=10000,
            waterDepth=55,  # jacket depth
            voltageType="does_not_exist",  # must fail
            maxJacketDepth=55,
            convention="RogeauEtAl2023",
        )

def test_getSpecificOffshoreConnectionCost():
    pass