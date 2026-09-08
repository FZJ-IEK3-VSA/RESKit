"""Keep the pre-0.6.0 module paths importable after the PEP 8 module renaming.

RESKit used CamelCase module and package names, which PEP 8 reserves for classes.
Renaming them to ``lower_case`` changes import paths such as::

    from reskit.weather.NCSource import NCSource

Those paths keep working, and warn, until they are removed in RESKit 1.0.0.

The aliases are registered in :data:`sys.modules` rather than kept as stub files
on disk, for two reasons:

* A stub file would be imported by the import system, which then binds it onto
  the parent package. ``reskit.weather.Era5Source`` is the *class*, so importing
  a stub of the same name would silently replace the class with a module. Seeding
  :data:`sys.modules` skips that binding, so the attribute keeps its meaning --
  which is exactly how the real submodules behaved before the rename.
* Three of the renames differ only in case (``GSA_mean`` -> ``gsa_mean``), and a
  stub cannot coexist with its replacement on a case-insensitive filesystem.

Known limitation: ``python -m reskit.util.create_LRA`` no longer works, because an
alias has no loader for :mod:`runpy` to use. Use the new path instead.
"""

import importlib
import sys
import warnings
from types import ModuleType

# Old dotted path -> new dotted path. Removed in RESKit 1.0.0 together with the
# deprecated function and class aliases from #226.
MODULE_RENAMES = {
    "reskit.util.create_LRA": "reskit.util.long_run_average",
    "reskit.weather.NCSource": "reskit.weather.nc_source",
    "reskit.weather.SarahSource": "reskit.weather.sarah_source",
    "reskit.weather.CosmoSource": "reskit.weather.cosmo_source",
    "reskit.weather.CosmoSource.CosmoSource": "reskit.weather.cosmo_source.cosmo_source",
    "reskit.weather.Era5Source": "reskit.weather.era5_source",
    "reskit.weather.Era5Source.Era5Source": "reskit.weather.era5_source.era5_source",
    "reskit.weather.Era5Source.Era5ZarrSource": "reskit.weather.era5_source.era5_zarr_source",
    "reskit.weather.Era5Source.Era5Prepare": "reskit.weather.era5_source.era5_prepare",
    "reskit.weather.Era5Source.data": "reskit.weather.era5_source.data",
    "reskit.weather.MerraSource": "reskit.weather.merra_source",
    "reskit.weather.MerraSource.MerraSource": "reskit.weather.merra_source.merra_source",
    "reskit.weather.MerraSource.data": "reskit.weather.merra_source.data",
    "reskit.weather.IconlamSource": "reskit.weather.iconlam_source",
    "reskit.weather.IconlamSource.IconlamSource": "reskit.weather.iconlam_source.iconlam_source",
    "reskit.weather.GSA_mean": "reskit.weather.gsa_mean",
    "reskit.weather.GSA_mean.GSAmeanSource": "reskit.weather.gsa_mean.gsa_mean_source",
    "reskit.weather.GWA_mean": "reskit.weather.gwa_mean",
    "reskit.weather.GWA_mean.GWAmeanSource": "reskit.weather.gwa_mean.gwa_mean_source",
}

# Looked up by the import system itself, so they must not warn or be delegated.
_IMPORT_MACHINERY_ATTRIBUTES = frozenset(
    {
        "__path__",
        "__spec__",
        "__loader__",
        "__package__",
        "__file__",
        "__cached__",
        "__builtins__",
    }
)


class DeprecatedModuleAlias(ModuleType):
    """A stand-in for a renamed module which warns whenever it is used.

    Attribute access is forwarded to the renamed module, so the alias never holds
    a second copy of anything: ``old.Thing is new.Thing``.
    """

    def __init__(self, old_name, new_name):
        super().__init__(old_name)
        self.__old_name = old_name
        self.__new_name = new_name

    @property
    def __doc__(self):
        return f"Deprecated alias of {self.__new_name}."

    def _target(self):
        return importlib.import_module(self.__new_name)

    def __getattr__(self, name):
        # Raised rather than delegated, so that the import system treats this
        # alias as a plain module and does not walk into the renamed package.
        if name in _IMPORT_MACHINERY_ATTRIBUTES or name.startswith("_DeprecatedModuleAlias"):
            raise AttributeError(name)

        target = self._target()
        if name == "__all__":
            # "from <old> import *" reads __all__ before falling back to __dict__,
            # and the alias has an empty __dict__. Without this the star import
            # would quietly import nothing.
            return getattr(target, "__all__", [n for n in vars(target) if not n.startswith("_")])

        warnings.warn(
            f"{self.__old_name} is deprecated and will be removed in RESKit 1.0.0. "
            f"Use {self.__new_name} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(target, name)

    def __dir__(self):
        return dir(self._target())

    def __repr__(self):
        return f"<deprecated module alias {self.name!r} -> {self.__new_name!r}>"

    @property
    def name(self):
        """The deprecated path this alias answers to."""
        return self.__old_name

    @property
    def target_name(self):
        """The path of the renamed module this alias forwards to."""
        return self.__new_name


def install(renames=None):
    """Register every old module path in :data:`sys.modules`.

    Existing entries are left alone, so a real module always wins over an alias.
    """
    for old_name, new_name in (renames or MODULE_RENAMES).items():
        sys.modules.setdefault(old_name, DeprecatedModuleAlias(old_name, new_name))
