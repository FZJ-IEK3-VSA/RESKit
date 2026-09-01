import numpy as np
import pytest

from reskit.solar.core.system_design import location_to_tilt
from reskit.util import ResError

LOCS = [(6.0, 51.0), (7.0, 45.0), (0.0, 0.0)]


def test_location_to_tilt_ryberg2020():
    tilt = location_to_tilt(LOCS, convention="Ryberg2020")

    expected = 42.327719357601396 * np.arctan(1.5 * np.radians(np.abs([51.0, 45.0, 0.0])))
    assert np.allclose(tilt, expected)
    # the default convention gives the same result
    assert np.allclose(location_to_tilt(LOCS), expected)


def test_location_to_tilt_accepts_a_callable():
    tilt = location_to_tilt(LOCS, convention=lambda latitude: latitude * 0.76)

    assert np.allclose(tilt, np.array([51.0, 45.0, 0.0]) * 0.76)


def test_location_to_tilt_expression_is_deprecated():
    with pytest.warns(DeprecationWarning):
        tilt = location_to_tilt(LOCS, convention="latitude*0.76")

    assert np.allclose(tilt, np.array([51.0, 45.0, 0.0]) * 0.76)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_location_to_tilt_expression_allows_permitted_functions():
    tilt = location_to_tilt(LOCS, convention="degrees(arctan(abs(latitude) / 90))")

    assert np.allclose(tilt, np.degrees(np.arctan(np.abs([51.0, 45.0, 0.0]) / 90)))


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(
    "convention",
    [
        "__import__('os').system('echo unsafe')",
        "open('/etc/passwd').read()",
        "latitude.__class__",
        "[x for x in range(3)]",
        "latitude*",
    ],
)
def test_location_to_tilt_expression_rejects_unsafe_input(convention):
    # the expression must not reach the Python built-ins
    with pytest.raises(ResError):
        location_to_tilt(LOCS, convention=convention)


def test_location_to_tilt_rejects_a_wrong_convention_type():
    with pytest.raises(ResError):
        location_to_tilt(LOCS, convention=17)
