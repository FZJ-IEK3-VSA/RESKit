from reskit.core.wind.turbine_library import TurbineLibrary
import numpy as np


def test_TurbineLibrary():
    turb = TurbineLibrary().loc["E115_2500"]

    assert turb.Manufacturer == "Enercon"
    assert turb.Capacity == 2500
    assert turb.Usage == "Onshore"
    assert (turb.Hub_Height == [92.5, 149.0]).all()
    assert turb.Rotordiameter == 115
    assert np.isclose(turb.PowerCurve.capacity_factor.sum(), 18.2798)
