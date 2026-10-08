"""Access to the datasets RESKit needs: a thin configuration of ETHOS.Data.

Data is described by the shared ETHOS.Data catalogue and downloaded on demand
into a cache that every ETHOS tool shares, so a dataset used by more than one
tool is fetched once. RESKit contributes configuration only: its collections
file, ``collections.yaml`` beside this module, which names what each workflow
reads, and its bundle (:data:`BUNDLES`), the ``reskit-test-data`` fixtures it
ships. Fetching, verifying, bundles, staging and the command line are
ETHOS.Data's.

    from reskit import data

    inputs = data.paths("wind_era5_PenaSanchezDunkelWinklerEtAl2025", test=True)  # {handle: Path}
    files = data.fetch("offshore_siting", test=True)                                 # {key: Path}
    clc = data.catalog_path("corine-land-cover/CLC2018_CLC2018_V2018_20.tif")       # one catalogue key

The same from the shell, with the ``reskit-data`` command this module provides:

    reskit-data show                                     # the collections and their size
    reskit-data show wind_era5_PenaSanchezDunkelWinklerEtAl2025 --test
    reskit-data fetch wind_era5_PenaSanchezDunkelWinklerEtAl2025 --test --paths
    reskit-data bundle verify reskit/data/test_cache     # the shipped fixtures
    reskit-data staging add trial /path/to/data          # unpublished development data
    reskit-data config show                              # where the cache is, which catalogue

``paths`` is what a workflow wants. The collection names each input the workflow
takes (``era5``, ``gwa_100m``, ...) under ``paths:`` in ``collections.yaml``, so
the caller gets ``{handle: pathlib.Path}`` without knowing a single catalogue
key. Workflow collections use the corresponding Python function's name.
``test=True`` selects the small fixtures the collection pairs with the full
data; both variants offer the same handles, so the same code runs on either.
The full data is the default: a forgotten flag must never silently run a real
calculation on fixtures, while an accidental full download is visible and can
be interrupted. A collection whose full inputs the catalogue does not hold yet
has a test variant only, and asking for its full variant is refused.

The bundle. ``test_cache`` beside this module is an ETHOS.Data bundle of the
``reskit-test-data`` family: ``bundle.json`` records every file with its size
and SHA-256 and its alignment with the catalogue, the files lie under
``data/<dataset>/<path>``, and ``datasets/`` holds each dataset's description and
licence documents, which a repository that redistributes data owes its users.
ETHOS.Data reads the bundle before the catalogue, hash-checked once per process,
and opens no catalogue index for a collection the bundle holds whole: the test
suite and the examples' test variants run offline. A bundled file that is
missing, or changed without ``reskit-data bundle update`` recording it, is an
error and never a reason to download. ``ETHOS_DATA_DOWNLOAD=1`` reads a bundled
file the catalogue holds under the same key through the catalogue route instead,
to test the download. A bundle ahead of the catalogue -- one holding recorded
changes or data the catalogue does not describe yet -- is read all the same, with
an ``ethos_data.BundleAlignmentWarning`` once per process;
docs/how_to/get_input_data.md says how to realign it.

Configuration (all optional):
    ETHOS_DATA_CATALOG    the catalogue every ETHOS.Data-based tool reads -- point
                          it at the institute's internal one, or set it for good
                          with ``reskit-data config set-catalog <path-or-url>``
    RESKIT_DATA_CATALOG   a RESKit-only override; wins over the above when set
    ETHOS_DATA_DIR        where the shared public cache lives
    ETHOS_DATA_DOWNLOAD   1/true/yes/on: read bundled files through the catalogue
Run ``reskit-data config show`` to see what is in effect.
"""

from __future__ import annotations

import os
import warnings
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ethos_data import Collections, DataFiles, NamedPaths

__all__ = [
    "BUNDLES",
    "CATALOG_ENV",
    "COLLECTIONS_FILE",
    "TOOL",
    "catalog_path",
    "fetch",
    "handle",
    "main",
    "paths",
]

COLLECTIONS_FILE = Path(__file__).resolve().parent / "collections.yaml"
#: The bundles RESKit ships, read before the catalogue: the ``reskit-test-data``
#: fixtures the test suite and the examples' test variants run on.
BUNDLES = (Path(__file__).resolve().parent / "test_cache",)
#: RESKit's own catalogue override, below ``--catalog`` and above ``$ETHOS_DATA_CATALOG``.
CATALOG_ENV = "RESKIT_DATA_CATALOG"
TOOL = "reskit"


def _ethos_data():
    # Imported on first use, not at module load: ``import reskit`` loads this
    # package, and must work without ethos_data installed.
    try:
        import ethos_data
    except ImportError as error:  # pragma: no cover - import-time guidance
        raise ImportError(
            "reskit.data needs the 'ethos_data' package.\n"
            "    conda install -c conda-forge ethos_data     (or: pip install ethos_data)"
        ) from error
    return ethos_data


def _catalog_override() -> str | None:
    """``$RESKIT_DATA_CATALOG`` if set; None leaves the choice to ETHOS.Data."""
    return os.environ.get(CATALOG_ENV) or None


@lru_cache(maxsize=1)
def handle() -> Collections:
    """The ``ethos_data.Collections`` handle on RESKit's collections file and bundles, built once.

    The file, the settings and the bundles are read the first time anything
    here is called, and never again in this process. ``handle().catalog`` is
    the catalogue RESKit reads, opened only when something needs it.
    """
    return _ethos_data().collections(COLLECTIONS_FILE, tool=TOOL, catalog=_catalog_override(), bundles=BUNDLES)


def paths(
    collection: str,
    *,
    test: bool = False,
    fetch: bool = True,
    progressbar: bool = True,
) -> NamedPaths:
    """The inputs a collection names, as ``{handle: Path}``, fetched.

    The handles are the ones ``collections.yaml`` defines under ``paths:`` for
    the collection -- ``era5``, ``gwa_100m`` -- resolved to where the data is on
    this machine:

        inputs = data.paths("wind_era5_PenaSanchezDunkelWinklerEtAl2025", test=True)
        rk.wind.wind_era5_PenaSanchezDunkelWinklerEtAl2025(
            placements,
            era5_path=inputs["era5"],
            gwa_100m_path=inputs["gwa_100m"],
            height_scaling_data={50: inputs["gwa_50m"], 200: inputs["gwa_200m"]},
        )

    ``test=True`` selects the small ``test`` variant, which the bundle answers
    offline; omitting it selects the full variant. ``fetch=False`` downloads
    nothing and raises ``ethos_data.NotFetched`` for a file that is not here.
    See ``ethos_data.Collections.paths``.
    """
    return handle().paths(collection, test=test, fetch=fetch, progressbar=progressbar)


def fetch(collection: str, *, test: bool = False, progressbar: bool = True) -> DataFiles:
    """Every file a collection selects, as ``{"<dataset>/<path>": Path}``, fetched.

    The handles the collection names are on ``.named``. See
    ``ethos_data.Collections.fetch``.
    """
    return handle().fetch(collection, test=test, progressbar=progressbar)


def catalog_path(key: str, *, fetch: bool = True, progressbar: bool = False) -> Path:
    """One dataset, folder or file by catalogue key, in the catalogue RESKit reads.

    ``key`` is ``"<dataset>/<path>"`` for one file -- a shapefile brings its
    sidecars along -- or a folder, a dataset or a family for the directory
    holding their files. For trying a dataset before a collection names it:
    workflows, examples and tests take their inputs from :func:`paths`. The
    catalogue index is read even for a key the bundle holds.
    """
    return handle().catalog.path(key, fetch=fetch, progressbar=progressbar)


def main(argv: list[str] | None = None) -> int:
    """``reskit-data``: RESKit's collections, its bundle and development staging.

    ``show``, ``fetch``, ``verify``, ``bundle``, ``staging``, ``config``,
    ``propose`` and ``report``; ``reskit-data --help`` lists them. The
    catalogue is loaded only for the commands that need it, so ``--help``,
    ``config show``, ``staging`` and ``bundle verify`` work offline. Access by
    catalogue key (``ethos-data ls``, ``ethos-data fetch``), shared cache
    maintenance and catalogue publishing are ``ethos-data``'s.
    """
    return _ethos_data().tool_main(COLLECTIONS_FILE, tool=TOOL, catalog=_catalog_override(), bundles=BUNDLES, argv=argv)


#: Directory names the fixtures had before they were grouped by provenance; still
#: keys of the deprecated ``reskit.TEST_DATA``.
_LEGACY_DIRECTORY_ALIASES = {
    "era5-like": "era5",
    "csp-era5-like": "era5-csp",
    "merra-like": "merra2",
    "sarah-like": "sarah",
    "iconlam-like": "icon-lam",
}


class _LegacyTestData(dict):
    """``reskit.TEST_DATA``: a dict that explains itself when a key is missing."""

    def __init__(self, *args, ambiguous: dict[str, set[str]], directories: set[str]):
        super().__init__(*args)
        self.ambiguous = ambiguous
        self.directories = directories

    def __missing__(self, key):
        if key in self.ambiguous:
            places = sorted(self.ambiguous[key])
            raise KeyError(
                f"{key!r} is not unique in the fixture tree -- it exists in {' and '.join(places)}. "
                f"Ask for the relative path instead, e.g. TEST_DATA['{places[0]}/{key}']."
            )
        raise KeyError(
            f"{key!r} is not a test fixture. The tree is grouped by provenance; "
            f"available directories are {', '.join(sorted(self.directories))}."
        )


@lru_cache(maxsize=1)
def _legacy_test_data() -> _LegacyTestData:
    """The fixtures under the keys ``reskit.TEST_DATA`` used, read through ETHOS.Data.

    Every file of the ``test_suite`` collection under its path inside the
    ``reskit-test-data`` family (``"era5/2m_temperature.nc"``), under its bare
    file name where that is unique, and every member directory under its name
    (``"era5"``) and its name from before the provenance grouping
    (``"era5-like"``). The values are strings, as they were.
    """
    family = "reskit-test-data/"
    by_relative: dict[str, str] = {}
    directories: dict[str, str] = {}
    for key, local in fetch("test_suite", progressbar=False).items():
        relative = key.removeprefix(family)
        by_relative[relative] = str(local)
        member, _, inner = relative.partition("/")
        directory = Path(local)
        for _ in PurePosixPath(inner).parts:
            directory = directory.parent
        directories[member] = str(directory)

    names: dict[str, list[str]] = {}
    for relative in by_relative:
        names.setdefault(PurePosixPath(relative).name, []).append(relative)
    unique = {name: by_relative[found[0]] for name, found in names.items() if len(found) == 1}
    ambiguous = {
        name: {str(PurePosixPath(each).parent) for each in found} for name, found in names.items() if len(found) > 1
    }
    legacy = {old: directories[new] for old, new in _LEGACY_DIRECTORY_ALIASES.items() if new in directories}
    return _LegacyTestData(
        {**by_relative, **unique, **directories, **legacy},
        ambiguous=ambiguous,
        directories=set(directories),
    )


def legacy_test_data() -> _LegacyTestData:
    """``reskit.TEST_DATA``, deprecated: warns, then answers from the ``test_suite`` collection."""
    warnings.warn(
        "reskit.TEST_DATA is deprecated and will be removed in RESKit 1.0.0. Use the handles of the "
        "test_suite collection instead: reskit.data.paths('test_suite')['era5'].",
        DeprecationWarning,
        stacklevel=3,
    )
    return _legacy_test_data()
