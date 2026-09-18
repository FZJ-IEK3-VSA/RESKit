"""Paths to the test fixtures, keyed for convenience.

The fixture tree under ``data/reskit-test-data/`` is organised **by provenance**
-- one directory per upstream product, each a member of the ``reskit-test-data``
family in the ETHOS.Data catalogue -- so that what a file is, and what licence it
carries, is visible from where it sits. This directory is an ETHOS.Data bundle:
``bundle.json`` beside ``data/`` records every file with the size and hash the
catalogue declares, and ``datasets/`` archives the licences. ``reskit.data``
reads the same files by catalogue key.

``TEST_DATA`` maps three kinds of key to an absolute path:

* the relative path, always::      TEST_DATA["era5/2m_temperature.nc"]
* the bare filename, when that filename is unique in the tree::
                                   TEST_DATA["gwa100-like.tif"]
* a provenance directory name::    TEST_DATA["era5"]

Three filenames appear in both ``era5/`` and ``era5-csp/``
(``2m_temperature.nc``, ``10m_wind_speed.processed.nc`` and
``total_sky_direct_solar_radiation_at_surface.processed.t_adjusted.nc``). They
are deliberately NOT registered under their bare filename -- ask for the
relative path instead, and you will be told so if you forget.
"""

from collections import OrderedDict
from os.path import abspath, basename, dirname, join, relpath
from os import walk

_ROOT = join(dirname(abspath(__file__)), "data", "reskit-test-data")

# Directory names used before the tree was grouped by provenance. Kept so that
# existing callers keep working; delete this map, and fix the call sites, when
# the fixtures are renamed to drop the misleading "-like" suffix.
_LEGACY_DIRECTORY_ALIASES = {
    "era5-like": "era5",
    "csp-era5-like": "era5-csp",
    "merra-like": "merra2",
    "sarah-like": "sarah",
    "iconlam-like": "icon-lam",
}

_SKIP_DIRS = {"__pycache__"}
_SKIP_FILES = {"__init__.py"}


class _TestData(OrderedDict):
    """A dict that explains itself when a key is missing."""

    def __missing__(self, key):
        if key in _ambiguous:
            raise KeyError(
                f"{key!r} is not unique in the fixture tree -- it exists in "
                + " and ".join(sorted(_ambiguous[key]))
                + ". Ask for the relative path instead, e.g. "
                f"TEST_DATA['{sorted(_ambiguous[key])[0]}/{key}']."
            )
        raise KeyError(
            f"{key!r} is not a test fixture. The tree is grouped by provenance; "
            f"available directories are {', '.join(sorted(_directories))}."
        )


_ambiguous: dict[str, set[str]] = {}
_directories: set[str] = set()
TEST_DATA = _TestData()

_by_basename: dict[str, list[str]] = {}
for _dirpath, _dirnames, _filenames in walk(_ROOT):
    _dirnames[:] = [d for d in _dirnames if d not in _SKIP_DIRS]
    for _name in _filenames:
        if _name in _SKIP_FILES:
            continue
        _full = join(_dirpath, _name)
        _rel = relpath(_full, _ROOT)
        TEST_DATA[_rel] = _full
        _by_basename.setdefault(_name, []).append(_rel)

for _name, _rels in _by_basename.items():
    if len(_rels) == 1:
        TEST_DATA[_name] = join(_ROOT, _rels[0])
    else:
        _ambiguous[_name] = {dirname(r) or "." for r in _rels}

for _entry in sorted(next(walk(_ROOT))[1]):
    if _entry in _SKIP_DIRS:
        continue
    _directories.add(_entry)
    TEST_DATA[_entry] = join(_ROOT, _entry)

for _old, _new in _LEGACY_DIRECTORY_ALIASES.items():
    if _new in TEST_DATA:
        TEST_DATA[_old] = TEST_DATA[_new]
