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


def test_onshore_turbine_capex_array_matches_scalar_calls():
    caps = np.array([[900, 2000, 3000], [4200, 5600, 7000]])
    hubs = np.array([[60, 98, 120], [120, 140, 166]])
    rotors = np.array([[44, 82, 115], [136, 150, 175]])
    with pytest.warns(UserWarning, match="negative spinner mass"):
        capex = onshore_turbine_capex(caps, hubs, rotors, base_capex=1000 * 4200)

    assert capex.shape == caps.shape
    with pytest.warns(UserWarning, match="negative spinner mass"):
        scalar = [
            onshore_turbine_capex(c, h, r, base_capex=1000 * 4200) for c, h, r in zip(caps.flat, hubs.flat, rotors.flat)
        ]
    assert np.allclose(capex.ravel(), scalar, rtol=1e-12, atol=0)
    assert onshore_turbine_capex(caps[:, :0], hubs[:, :0], rotors[:, :0]).shape == (2, 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"turbine_class": 1, "blade_has_carbon": True, "crane": True},
        {"tower_cost_external": 5e5, "gearbox_torque_density": 150.0, "max_tip_speed": 90},
        # one value per design, also for a discrete input
        {"turbine_class": np.array([1, 2, 1]), "tower_mass_coeff": np.array([15.0, 20.0, 25.0])},
    ],
)
def test_onshore_tcc_matches_openmdao_reference(kwargs):
    caps, hubs, rotors = np.array([2000, 4200, 6000]), np.array([98, 120, 150]), np.array([82, 136, 160])

    tcc = onshore_tcc(caps, hubs, rotors, **kwargs)

    reference = [
        _onshore_tcc_scalar(
            caps[i], hubs[i], rotors[i], **{k: v[i].item() if np.ndim(v) else v for k, v in kwargs.items()}
        )
        for i in range(3)
    ]
    assert np.allclose(tcc, reference, rtol=1e-12, atol=0)
    assert isinstance(onshore_tcc(4200, 120, 136, **{k: v for k, v in kwargs.items() if np.ndim(v) == 0}), float)
