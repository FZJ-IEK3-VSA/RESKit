# from reskit import TEST_DATA
from reskit.wind.core.design_turbine import onshore_turbine_from_avg_wind_speed
import numpy as np
import pandas as pd


def test_onshore_turbine_from_avg_wind_speed():
    output = onshore_turbine_from_avg_wind_speed(11)
    assert isinstance(output, dict)
    assert np.isclose(output["capacity"], 5483.067776983042)
    assert np.isclose(output["hub_height"], 88.0)
    assert np.isclose(output["rotor_diam"], 136)
    assert np.isclose(output["specific_power"], 377.44694637315786)

    output = onshore_turbine_from_avg_wind_speed(2)
    assert np.isclose(output["capacity"], 2614.8103974358564)
    assert np.isclose(output["hub_height"], 335.2328677954964)
    assert np.isclose(output["rotor_diam"], 136)
    assert np.isclose(output["specific_power"], 180)

    output = onshore_turbine_from_avg_wind_speed(4, constant_rotor_diam=False)
    assert np.isclose(output["capacity"], 4200)
    assert np.isclose(output["hub_height"], 186.01221649255768)
    assert np.isclose(output["rotor_diam"], 156.22991526535841)
    assert np.isclose(output["specific_power"], 219.09426750262875)

    output = onshore_turbine_from_avg_wind_speed(
        [3, 4, 5, 6], constant_rotor_diam=False
    )
    assert isinstance(output, pd.DataFrame)
    assert np.isclose(output.capacity.mean(), 4200.000000)
    assert np.isclose(output.hub_height.mean(), 177.304384)
    assert np.isclose(output.rotor_diam.mean(), 153.062676)
    assert np.isclose(output.specific_power.mean(), 231.569929)

    # test some custom baseline turbine parameters
    output = onshore_turbine_from_avg_wind_speed(
        4.7, base_capacity=4750, base_hub_height=145
    )
    assert np.isclose(output["capacity"], 3925.5608093815567)
    assert np.isclose(output["hub_height"], 195.98029637154556)
    assert np.isclose(output["rotor_diam"], 136)
    assert np.isclose(output["specific_power"], 270.230279939834)
