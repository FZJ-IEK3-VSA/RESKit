from reskit import TEST_DATA
from reskit.wind.core.power_curve import PowerCurve, compute_specific_power

import numpy as np
import pandas as pd
import pytest


def test_compute_specific_power():
    assert np.isclose(compute_specific_power(4200, 136), 289.1223014645158)

    output = compute_specific_power(np.linspace(3000, 5000, 10), 136)
    assert output.shape == (10,)
    assert np.isclose(output.mean(), 275.35457282334835)


@pytest.fixture
def pt_power_curve():
    perf = np.array([(1, 0.0), (2, 0.0), (3, 0.0138095238095), (4, 0.0440476190476), (5, 0.0952380952381), (6, 0.177380952381),
                     (7, 0.285714285714), (8, 0.42619047619), (9,
                                                               0.583333333333), (10, 0.742857142857), (11, 0.871428571429),
                     (12, 0.952380952381), (13, 0.988095238095), (14, 1.0), (15,
                                                                             1.0), (16, 1.0), (17, 1.0), (18, 1.0), (19, 1.0),
                     (20, 1.0), (21, 1.0), (22, 1.0), (23, 1.0), (24, 1.0), (25, 1.0), ])
    return PowerCurve(perf[:, 0], perf[:, 1])


def test_PowerCurve_from_specific_power():
    pc = PowerCurve.from_specific_power(200)

    assert pc.wind_speed.shape == (121,)
    assert pc.wind_speed.shape == pc.capacity_factor.shape
    assert np.isclose(pc.wind_speed.mean(), 8.555716631349762)
    assert np.isclose(pc.capacity_factor.mean(), 0.5743801652892562)


def test_PowerCurve_from_capacity_and_rotor_diam():
    pc = PowerCurve.from_capacity_and_rotor_diam(4200, 140)

    assert pc.wind_speed.shape == (121,)
    assert pc.wind_speed.shape == pc.capacity_factor.shape
    assert np.isclose(pc.wind_speed.mean(), 9.328335673742838)
    assert np.isclose(pc.capacity_factor.mean(), 0.5743801652892562)


def test_PowerCurve_simulate(pt_power_curve):
    wind_speeds = np.linspace(15, 0, 100)
    output = pt_power_curve.simulate(wind_speeds)

    assert output.shape == (100,)
    assert np.isclose(output.mean(), 0.44624976720716636)
    assert np.isclose(output.std(), 0.39795242761035865)


def test_expected_capacity_factor_from_weibull(pt_power_curve):
    output = pt_power_curve.expected_capacity_factor_from_weibull(
        mean_wind_speed=5,
        weibull_shape=2,
    )

    assert np.isclose(output, 0.17405615945580358)


def test_expected_capacity_factor_from_distribution(pt_power_curve):
    output = pt_power_curve.expected_capacity_factor_from_distribution(
        wind_speed_values=[0, 2, 4, 6, 8, 10],
        wind_speed_counts=[5, 20, 5, 2, 5, 1])

    assert np.isclose(output, 0.0907581453633421)


def test_PowerCurve_convolute_by_gaussian(pt_power_curve):
    pc = pt_power_curve.convolute_by_gaussian(scaling=0.06, base=0.1, )

    assert pc.wind_speed.shape == (100,)
    assert pc.wind_speed.shape == pc.capacity_factor.shape
    assert np.isclose(pc.wind_speed.mean(), 19.81)
    assert np.isclose(pc.capacity_factor.mean(), 0.41840421118709586)


def test_PowerCurve_apply_loss_factor(pt_power_curve):
    pc = pt_power_curve.apply_loss_factor(0.1)
    assert pc.wind_speed.shape == (25,)
    assert pc.wind_speed.shape == pc.capacity_factor.shape
    assert np.isclose(pc.wind_speed.mean(), 13.0)
    assert np.isclose(pc.capacity_factor.mean(), 0.6184971428571072)

    pc = pt_power_curve.apply_loss_factor(lambda cf: (1 - cf) * 0.5)
    assert pc.wind_speed.shape == (25,)
    assert pc.wind_speed.shape == pc.capacity_factor.shape
    assert np.isclose(pc.wind_speed.mean(), 13.0)
    assert np.isclose(pc.capacity_factor.mean(), 0.660425526077062)
