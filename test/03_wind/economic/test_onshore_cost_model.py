from reskit.wind.economic.onshore_cost_model import onshore_turbine_capex
import numpy as np


def test_onshore_turbine_capex():
    capex = onshore_turbine_capex(capacity=4200, hub_height=120, rotor_diam=136)

    assert np.isclose(capex / 4200, 1100)

    capex = onshore_turbine_capex(
        capacity=4200,
        hub_height=120,
        rotor_diam=136,
        base_capacity=5000,
        base_hub_height=130,
        base_rotor_diam=140,
    )

    assert np.isclose(capex / 4200, 905.2822613512992)

    capex = onshore_turbine_capex(
        capacity=[4200, 4100, 4000, 3900],
        hub_height=[120, 120, 120, 120],
        rotor_diam=[136, 140, 145, 150],
        base_capacity=5000,
        base_hub_height=130,
        base_rotor_diam=140,
        tcc_share=0.7,
        bos_share=0.15,
    )

    assert np.isclose(
        capex / 4200, [922.08068518, 940.59391215, 969.79729428, 1001.70863224]
    ).all()
