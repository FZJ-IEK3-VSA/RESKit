from reskit.util.economic.lcoe import (
    levelized_cost_of_electricity,
    levelized_cost_of_electricity_simplified,
)
import numpy as np


def test_levelized_cost_of_electricity():
    annual_expenditures = np.full(30, 100)
    annual_expenditures[0] = 500

    np.random.seed(0)
    annual_generation = np.random.random(30) * 1000 + 2000

    lcoe = levelized_cost_of_electricity(
        expenditures=annual_expenditures,
        productions=annual_generation,
        discount_rate=0.08,
    )

    assert np.isclose(lcoe, 0.05126993383326904)


def test_levelized_cost_of_electricity_simplified():
    lcoe = levelized_cost_of_electricity_simplified(
        capex=1100,
        mean_production=2300,
        opex_per_capex=0.02,
        lifetime=25,
        discount_rate=0.08,
    )

    assert np.isclose(lcoe, 0.05436811172050649)

    lcoe = levelized_cost_of_electricity_simplified(
        capex=np.array([1100, 1200, 1300, 1400]),
        mean_production=np.array([2300, 2200, 2100, 2000]),
        opex_per_capex=0.02,
        lifetime=25,
        discount_rate=0.08,
    )

    assert np.isclose(lcoe, [0.05436811, 0.06200661, 0.07037258, 0.07957515]).all()
