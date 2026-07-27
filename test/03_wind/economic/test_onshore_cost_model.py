import numpy as np

from reskit.wind.economic.onshore_cost_model import onshore_turbine_capex


def test_onshore_turbine_capex():
    capex = onshore_turbine_capex(capacity=4200, hub_height=120, rotor_diam=136)

    assert np.isclose(capex / 4200, 1100)

    capex = onshore_turbine_capex(
        capacity=4200,
        hub_height=120,
        rotor_diam=136,
        # base_capex=5000*1100,
        base_capacity=5000,
        base_hub_height=130,
        base_rotor_diam=140,
    )

    assert np.isclose(capex / 4200, 913.7163211549489)

    caps = np.array([4200, 4100, 4000, 3900])
    capex = onshore_turbine_capex(
        capacity=caps,
        hub_height=[120, 120, 120, 120],
        rotor_diam=[136, 140, 145, 150],
        # base_capex=5000*1100,
        base_capacity=5000,
        base_hub_height=130,
        base_rotor_diam=140,
        tcc_share=0.7,
        bos_share=0.15,
    )

    assert np.isclose(capex / caps, [931.38977592, 974.44510595, 1029.75686152, 1090.17668668]).all()
