def test___init__():
    # (s, source, bounds=None, indexPad=0, **kwargs):
    print( "__init__ not tested...")

def test_loc2Index():
    # (s, loc, outsideOkay=False, asInt=True):
    print( "loc2Index not tested...")

def test_loadRadiation():
    # (s):
    print( "loadRadiation not tested...")

def test_loadWindSpeedLevels():
    # (s):
    print( "loadWindSpeedLevels not tested...")

def test_loadWindSpeedAtHeight():
    # (s, height=100):
    print( "loadWindSpeedAtHeight not tested...")

def test_loadTemperature():
    # (s, processor=lambda x: x-273.15):
    print( "loadTemperature not tested...")

def test_loadPressure():
    # (s):
    print( "loadPressure not tested...")

def test_loadSet_PV():
    # (s):
    print( "loadSet_PV not tested...")

def test_getWindSpeedAtHeights():
    # (s, locations, heights, spatialInterpolation='near', forceDataFrame=False, outsideOkay=False, _indicies=None):
    print( "getWindSpeedAtHeights not tested...")


if __name__ == "__main__":
    test___init__()
    test_loc2Index()
    test_loadRadiation()
    test_loadWindSpeedLevels()
    test_loadWindSpeedAtHeight()
    test_loadTemperature()
    test_loadPressure()
    test_loadSet_PV()
    test_getWindSpeedAtHeights()