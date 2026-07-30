# from reskit import TEST_DATA
from reskit.wind.core.design_turbine import onshore_turbine_from_avg_wind_speed, turbine_design_from_avg_wind_speed
from reskit.wind.core.data import DATAFOLDER

import numpy as np
import pandas as pd
import os


def test_onshore_turbine_from_avg_wind_speed():  # TODO move most tests to test_turbine_design_from_avg_wind_speed and leave only one test for wrapper
    output = onshore_turbine_from_avg_wind_speed(wind_speed=11)
    assert isinstance(output, dict)
    assert np.isclose(output["capacity"], 5483.067776983042)
    assert np.isclose(output["hub_height"], 88.0)
    assert np.isclose(output["rotor_diam"], 136)
    assert np.isclose(output["specific_power"], 377.44694637315786)

    output = onshore_turbine_from_avg_wind_speed(wind_speed=2)
    assert np.isclose(output["capacity"], 2614.8103974358564)
    assert np.isclose(output["hub_height"], 335.2328677954964)
    assert np.isclose(output["rotor_diam"], 136)
    assert np.isclose(output["specific_power"], 180)

    output = onshore_turbine_from_avg_wind_speed(wind_speed=2, max_hub_height=199)
    assert np.isclose(output["capacity"], 2614.8103974358564)
    assert np.isclose(output["hub_height"], 199)
    assert np.isclose(output["rotor_diam"], 136)
    assert np.isclose(output["specific_power"], 180)

    output = onshore_turbine_from_avg_wind_speed(wind_speed=4, constant_rotor_diam=False)
    assert np.isclose(output["capacity"], 4200)
    assert np.isclose(output["hub_height"], 186.01221649255768)
    assert np.isclose(output["rotor_diam"], 156.22991526535841)
    assert np.isclose(output["specific_power"], 219.09426750262875)

    output = onshore_turbine_from_avg_wind_speed(wind_speed=[3, 4, 5, 6], constant_rotor_diam=False)
    assert isinstance(output, pd.DataFrame)
    assert np.isclose(output.capacity.mean(), 4200.000000)
    assert np.isclose(output.hub_height.mean(), 177.3043837272176)  # checked
    assert np.isclose(output.rotor_diam.mean(), 153.062676)
    assert np.isclose(output.specific_power.mean(), 231.569929)

    # test some custom baseline turbine parameters
    output = onshore_turbine_from_avg_wind_speed(wind_speed=4.7, base_capacity=4750, base_hub_height=145)
    assert np.isclose(output["capacity"], 3925.5608093815567)
    assert np.isclose(output["hub_height"], 195.98029637154556)
    assert np.isclose(output["rotor_diam"], 136)
    assert np.isclose(output["specific_power"], 270.230279939834)


def test_turbine_design_from_avg_wind_speed():
    # replicate the old onshore_turbine_from_avg_wind_speed base test
    output = turbine_design_from_avg_wind_speed(
        wind_speed=11,
        technology="onshore",
        constant_rotor_diam=None,
        base_capacity=None,
        base_hub_height=None,
        base_rotor_diam=None,
        reference_wind_speed=None,
        min_tip_height=None,
        min_specific_power=None,
        max_hub_height=None,
        tech_year=2050,
        baseline_turbine_fp=None,
        convention="RybergEtAl2019",
    )
    assert isinstance(output, dict)
    assert np.isclose(output["capacity"], 5483.067776983042)
    assert np.isclose(output["hub_height"], 88.0)
    assert np.isclose(output["rotor_diam"], 136)
    assert np.isclose(output["specific_power"], 377.44694637315786)

    # test default reference ws for spec pow, must yield default capacity and specpow
    output = turbine_design_from_avg_wind_speed(
        wind_speed=8.29,
        technology="onshore",
        constant_rotor_diam=None,
        base_capacity=None,
        base_hub_height=None,
        base_rotor_diam=None,
        reference_wind_speed=None,
        min_tip_height=None,
        min_specific_power=None,
        max_hub_height=None,
        tech_year=2035,
        baseline_turbine_fp=os.path.join(
            DATAFOLDER, "Baseline_plant_wind_turbine_offshore_Global_Winkler2025_v20251108.csv"
        ),
        convention="WinklerEtAl2026",
    )
    assert isinstance(output, dict)
    assert np.isclose(output["capacity"], 20000)  # default
    assert np.isclose(
        output["hub_height"], 159.9192741935484
    )  # slightly lower than default since ref ws slightly higher than ref ws for hub height
    assert np.isclose(output["rotor_diam"], 270)
    assert np.isclose(output["specific_power"], 349.3112605583437)  # default

    output = turbine_design_from_avg_wind_speed(
        wind_speed=7.72,
        technology="onshore",
        constant_rotor_diam=None,
        base_capacity=None,
        base_hub_height=None,
        base_rotor_diam=None,
        reference_wind_speed=None,
        min_tip_height=None,
        min_specific_power=None,
        max_hub_height=None,
        tech_year=2035,
        baseline_turbine_fp=os.path.join(
            DATAFOLDER, "Baseline_plant_wind_turbine_offshore_Global_Winkler2025_v20251108.csv"
        ),
        convention="WinklerEtAl2026",
    )
    assert isinstance(output, dict)
    assert np.isclose(
        output["capacity"], 19092.081687353104
    )  # slightly lower than default since ref ws slightly lower than ref ws for spec pow
    assert np.isclose(output["hub_height"], 170)  #
    assert np.isclose(output["rotor_diam"], 270)
    assert np.isclose(
        output["specific_power"], 333.4539560446091
    )  # slightly lower than default since ref ws slightly lower than ref ws for spec pow

    # TODO add tests: data year out of range, custom params as args, wrong convention, other wind speed, offshore
