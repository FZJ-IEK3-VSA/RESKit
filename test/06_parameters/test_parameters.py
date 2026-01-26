from reskit import TEST_DATA
from reskit.parameters.parameters import OffshoreParameters, OnshoreParameters


def test_onshore_parameters_default():
    onshore_params = OnshoreParameters()
    assert onshore_params.min_tip_height == 20  # test optional args
    assert onshore_params.base_rotor_diam == 136  # test mandatory args


def test_onshore_parameters_custom():
    onshore_params = OnshoreParameters(fp=TEST_DATA["baseline_turbine_testdummy.csv"], year=2030)
    assert onshore_params.min_tip_height == 0
    assert onshore_params.base_rotor_diam == 118


def test_offshore_parameters():
    offshore_params = OffshoreParameters(fp=None)
    assert offshore_params.min_tip_height == 30
    assert offshore_params.distance_to_bus == 3
