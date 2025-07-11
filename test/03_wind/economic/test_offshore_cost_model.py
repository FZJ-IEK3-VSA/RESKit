from reskit.wind.economic.offshore_cost_model import *
import numpy as np



def test_offshore_turbine_capex():
    capex = offshore_turbine_capex(
        capacity=4200, hub_height=120, rotor_diam=136, depth=30, distance_to_shore=15
    )

    assert np.isclose(capex / 4200, 1949.3177174432028)

