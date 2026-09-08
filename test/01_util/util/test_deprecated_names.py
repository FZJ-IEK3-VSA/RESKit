"""
Tests for the names renamed for PEP 8 in v0.6.0 (issue #226).

Each old name must stay usable until v1.0.0, must raise a ``DeprecationWarning``,
and must return exactly what the new name returns. The new names must stay silent.
"""

import importlib
import re
import warnings
from pathlib import Path

import pytest

import reskit as rk
from reskit._deprecated_modules import MODULE_RENAMES, DeprecatedModuleAlias
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


# ---------------------------------------------------------------------------
# Module and package names renamed for PEP 8 (see reskit/_deprecated_modules.py)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("old_path, new_path", sorted(MODULE_RENAMES.items()))
def test_deprecated_module_path_delegates_to_the_renamed_module(old_path, new_path):
    """Each old path stays importable and hands back the renamed module's objects.

    Comparing with ``is`` also proves the alias holds no second copy of anything,
    so an isinstance check against the new path accepts an object from the old one.
    """
    new_module = importlib.import_module(new_path)
    old_module = importlib.import_module(old_path)

    assert isinstance(old_module, DeprecatedModuleAlias)
    assert old_module.target_name == new_path

    for name in [name for name in vars(new_module) if not name.startswith("_")]:
        with pytest.warns(DeprecationWarning, match=re.escape(old_path)):
            assert getattr(old_module, name) is getattr(new_module, name)


@pytest.mark.parametrize("old_path, new_path", sorted(MODULE_RENAMES.items()))
def test_renamed_module_path_is_silent(old_path, new_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        importlib.import_module(new_path)


def test_star_import_from_a_deprecated_module_path_binds_the_names():
    """``from <old> import *`` reads ``__all__`` first, which the alias has to supply."""
    namespace = {}
    with pytest.warns(DeprecationWarning, match="reskit.weather.NCSource"):
        exec("from reskit.weather.NCSource import *", namespace)
    assert namespace["NCSource"] is rk.weather.NCSource


def test_importing_a_deprecated_path_keeps_the_class_on_the_package():
    """``rk.weather.Era5Source`` is the class, and importing the old path must not replace it.

    A stub module on disk would be bound onto the parent package by the import
    system and would shadow the class. The alias lives in ``sys.modules``, which
    skips that binding.
    """
    importlib.import_module("reskit.weather.Era5Source")
    assert rk.weather.Era5Source is rk.weather.era5_source.Era5Source
    assert isinstance(rk.weather.Era5Source, type)

    importlib.import_module("reskit.util.create_LRA")
    assert rk.util.create_LRA is rk.util.long_run_average.create_LRA
    assert callable(rk.util.create_LRA)


def test_no_module_or_package_uses_a_camel_case_name():
    """PEP 8 keeps module and package names lower case; the classes inside stay CamelCase."""
    root = Path(rk.__file__).parent
    offenders = sorted(
        str(path.relative_to(root.parent))
        for path in root.rglob("*")
        if "__pycache__" not in path.parts
        and (path.suffix == ".py" or (path.is_dir() and (path / "__init__.py").exists()))
        and any(character.isupper() for character in (path.stem if path.suffix else path.name))
    )
    assert offenders == [], f"These modules or packages still use a CamelCase name: {offenders}"
