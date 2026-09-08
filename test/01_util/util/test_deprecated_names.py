"""
Tests for the names renamed for PEP 8 in v0.6.0 (issue #226).

Each old name must stay usable until v1.0.0, must raise a ``DeprecationWarning``,
and must return exactly what the new name returns. The new names must stay silent.
"""

import warnings

import pytest

import reskit as rk
from reskit.util.input_preparation import (
    DEPRECATED_WORKFLOW_NAMES,
    _merge_dependencies,
    depends_on,
)

# (deprecated name, current name) pairs, as exposed on the reskit namespace
RENAMED_FUNCTIONS = [
    (rk.csp.CSP_PTR_ERA5, rk.csp.csp_ptr_era5),
    (rk.geothermal.EGSworkflow, rk.geothermal.egs_workflow),
    (rk.wind.TurbineLibrary, rk.wind.turbine_library),
    (rk.wind.calculateSpecificOffshoreCapex, rk.wind.calculate_specific_offshore_capex),
]


@pytest.mark.parametrize("deprecated, current", RENAMED_FUNCTIONS)
def test_both_names_are_importable(deprecated, current):
    assert callable(deprecated)
    assert callable(current)
    assert deprecated is not current


def test_deprecated_function_warns_and_delegates():
    with pytest.warns(DeprecationWarning, match="TurbineLibrary"):
        old = rk.wind.TurbineLibrary()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        new = rk.wind.turbine_library()
    assert old.equals(new)


@pytest.mark.parametrize("deprecated_name, current_name", sorted(DEPRECATED_WORKFLOW_NAMES.items()))
def test_deprecated_workflow_string_still_resolves(deprecated_name, current_name):
    """A wrapper function cannot cover a workflow name that is passed as a string."""
    assert depends_on[deprecated_name] == depends_on[current_name]

    with pytest.warns(DeprecationWarning, match=deprecated_name):
        old = _merge_dependencies([deprecated_name])
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        new = _merge_dependencies([current_name])
    assert old == new
