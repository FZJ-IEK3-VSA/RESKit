import numpy as np
import pytest

from reskit.wind.economic.onshore_cost_model import _onshore_tcc_scalar, onshore_tcc, onshore_turbine_capex


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


@pytest.mark.parametrize(
    "kwargs",
    [
        # the defaults only
        {},
        # discrete inputs (int and bool), passed on without unit conversion
        {"turbine_class": 1, "blade_has_carbon": True, "crane": True},
        # the other branch of TowerCost2015's `if`, an input feeding a mass and a cost component,
        # and an int value for a continuous input
        {"tower_cost_external": 5e5, "gearbox_torque_density": 150.0, "max_tip_speed": 90},
        # one value per design, also for a discrete input
        {"turbine_class": np.array([1, 2, 1]), "tower_mass_coeff": np.array([15.0, 20.0, 25.0])},
    ],
)
def test_onshore_tcc_matches_openmdao_reference(kwargs):
    caps, hubs, rotors = np.array([2000, 4200, 6000]), np.array([98, 120, 150]), np.array([82, 136, 160])

    tcc = onshore_tcc(caps, hubs, rotors, **kwargs)

    # the OpenMDAO problem, set up and run once per design
    reference = [
        _onshore_tcc_scalar(
            caps[i], hubs[i], rotors[i], **{k: v[i].item() if np.ndim(v) else v for k, v in kwargs.items()}
        )
        for i in range(3)
    ]
    assert np.allclose(tcc, reference, rtol=1e-12, atol=0)
    # a single design still gives a float, as before
    assert isinstance(onshore_tcc(4200, 120, 136, **{k: v for k, v in kwargs.items() if np.ndim(v) == 0}), float)
