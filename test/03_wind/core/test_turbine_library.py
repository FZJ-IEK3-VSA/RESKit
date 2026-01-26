import numpy as np

from reskit.wind.core.turbine_library import turbine_library


def test_turbine_library():
    turb = turbine_library().loc["E115_2500"]

    assert turb.Manufacturer == "Enercon"
    assert turb.Capacity == 2500
    assert turb.Usage == "Onshore"
    assert (turb.Hub_Height == [92.5, 149.0]).all()
    assert turb.Rotordiameter == 115
    assert np.isclose(turb.PowerCurve.capacity_factor.sum(), 18.2798)
