from reskit.parameters.parameters import OnshoreParameters, OffshoreParameters
from reskit import TEST_DATA


def test_OnshoreParameters():
    OnshoreParams = OnshoreParameters(fp=TEST_DATA["baseline_turbine_test.json"])
    assert OnshoreParams.min_tip_height == 10
    assert OnshoreParams.base_rotor_diam == 136


def test_OffshoreParameters():
    OnshoreParams = OffshoreParameters(fp=TEST_DATA["baseline_turbine_test.json"])
    assert OnshoreParams.min_tip_height == 10
    assert OnshoreParams.distance_to_bus == 3
