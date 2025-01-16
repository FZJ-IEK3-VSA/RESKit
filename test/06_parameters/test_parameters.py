from reskit.parameters.parameters import OnshoreParameters, OffshoreParameters
from reskit import TEST_DATA


def test_OnshoreParameters_default():
    OnshoreParams = OnshoreParameters()
    assert OnshoreParams.min_tip_height == 20 # test optional args
    assert OnshoreParams.base_rotor_diam == 136 # test mandatory args
    
def test_OnshoreParameters_custom():
    OnshoreParams = OnshoreParameters(fp=TEST_DATA["baseline_turbine_testdummy.csv"], year=2030)
    assert OnshoreParams.min_tip_height == 0
    assert OnshoreParams.base_rotor_diam == 118


def test_OffshoreParameters():
    OffshoreParams = OffshoreParameters(fp=None)
    assert OffshoreParams.min_tip_height == 30
    assert OffshoreParams.distance_to_bus == 3
