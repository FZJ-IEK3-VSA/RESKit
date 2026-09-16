"""Access to the datasets RESKit needs.

Data is described by the shared ETHOS.Data catalogue and downloaded on demand
into a cache that every ETHOS tool shares, so a dataset used by more than one
tool is fetched once. Apart from the small test fixtures in ``test_cache`` (see
``reskit.TEST_DATA``), nothing is bundled with the package.

    from reskit import data

    inputs = data.paths("onshore_wind", test=True)   # {handle: Path}, small fixtures
    inputs = data.paths("onshore_wind")              # the same handles, full data
    files  = data.fetch("onshore_wind")              # a whole collection, by key
    clc    = data.path("landcover/C3S-LC-L4-LCCS-Map-300m-P1Y-2018-v2.1.1.tif")
    era5   = data.directory("reskit-test-data/era5")

The same from the shell, with the ``reskit-data`` command this module provides:

    reskit-data list                         # the collections and their size
    reskit-data paths onshore_wind --test    # fetch one, print handle<TAB>path
    reskit-data config show                  # where the cache is, which catalogue

``paths`` is what a workflow wants. The collection names each input the workflow
takes (``era5``, ``gwa_100m``, ...) under ``paths:`` in ``collections.yaml``, so
the caller gets ``{handle: pathlib.Path}`` without knowing a single resource
key. ``test=True`` selects the small fixtures the collection pairs with the full
data; both variants offer the same handles, so the same code runs on either.
The full data is the default: a forgotten flag must never silently run a real
calculation on fixtures, while an accidental full download is visible and can
be interrupted.

``fetch`` returns a mapping of ``"<dataset>/<path>" -> pathlib.Path``. Files are
verified against the checksums in the catalogue, and anything already present is
not downloaded again.

Which collections exist is defined in ``collections.yaml`` next to this module;
what each dataset contains is defined in the catalogue it pins. Everything here
is a thin layer over one ``ethos_data.Collections`` handle on that file, built
once per process by :func:`handle`; nothing is registered anywhere, and no
lookup happens beyond reading the file beside this module.

Configuration (all optional):
    ETHOS_DATA_CATALOG    the catalogue every ETHOS.Data-based tool reads -- point
                          it at the institute's internal one, or set it for good
                          with ``reskit-data config set-catalog <path-or-url>``
    RESKIT_DATA_CATALOG   a RESKit-specific override; wins over the above when set
    ETHOS_DATA_DIR        where the shared cache lives
Without either catalogue variable the one ``collections.yaml`` pins is used.
Every function in this module reads the same handle, so ``paths``, ``fetch``,
``path`` and ``directory`` never read different catalogues.
Run ``reskit-data config show`` to see what is in effect, or
``reskit-data config set-cache <dir> --scope environment`` to set it permanently.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

__all__ = [
    "COLLECTIONS_FILE",
    "cache_dir",
    "collections",
    "describe",
    "directory",
    "fetch",
    "handle",
    "main",
    "path",
    "paths",
    "plan",
]

COLLECTIONS_FILE = Path(__file__).resolve().parent / "collections.yaml"
CATALOG_ENV = "RESKIT_DATA_CATALOG"
TOOL = "reskit"
COMMAND = "reskit-data"


def _ethos_data():
    # Imported on first use, not at module load: ``import reskit`` loads this
    # package for the bundled fixtures in ``test_cache``, and must work without
    # ethos_data installed.
    try:
        import ethos_data
    except ImportError as error:  # pragma: no cover - import-time guidance
        raise ImportError(
            "reskit.data needs the 'ethos_data' package.\n"
            "    conda install -c conda-forge ethos_data     (or: pip install ethos_data)"
        ) from error
    return ethos_data


def _catalog_override() -> str | None:
    """``$RESKIT_DATA_CATALOG`` if set; None leaves the choice to ETHOS.Data.

    ETHOS.Data then reads ``$ETHOS_DATA_CATALOG`` or ``catalog:`` in an
    ``ethos-data`` config file, and after that the pin in ``collections.yaml``.
    """
    return os.environ.get(CATALOG_ENV) or None


@lru_cache(maxsize=1)
def handle():
    """The ``ethos_data.Collections`` handle on RESKit's collections file, built once.

    The file and the catalogue are read the first time anything here is called
    and never again in this process. ``handle().catalog`` is the catalogue
    RESKit reads, for access by key -- what :func:`path` and :func:`directory`
    use -- so every route through this module sees the same one.
    """
    return _ethos_data().collections(COLLECTIONS_FILE, tool=TOOL, catalog=_catalog_override())


def collections() -> list[str]:
    """Names of the collections RESKit defines."""
    return handle().names()


def describe(collection: str, test: bool = False) -> dict:
    """What a collection is for, and what it would fetch.

    ``variants`` lists the variants the collection defines -- ``("test",
    "full")``, or empty for a plain collection -- and ``paths`` is
    ``{handle: catalogue key}`` for the variant ``test`` selects: the handles
    :func:`paths` would return, not yet resolved to disk.
    """
    loaded = handle()
    resources = loaded.resolve(collection, test=test)
    return {
        "title": loaded.describe(collection).get("title", ""),
        "variants": loaded.variants(collection),
        "paths": loaded.named_keys(collection, test=test),
        "files": len(resources),
        "bytes": sum(r.bytes for r in resources),
        "keys": [r.key for r in resources],
    }


def plan(collection: str, test: bool = False) -> dict:
    """What fetching this collection would download, without downloading it."""
    return handle().plan(collection, test=test)


def fetch(
    collection: str,
    progressbar: bool = True,
    test: bool = False,
    skip_unavailable: bool | None = None,
):
    """Make a collection available locally.

    Returns a mapping of ``"<dataset>/<resource path>" -> Path``, in catalogue
    order. Also usable as a list via ``.paths``; the inputs the collection names
    under ``paths:`` are on ``.named``. ``test=True`` selects the collection's
    small ``test`` variant where it defines one; the default is the full data.
    ``skip_unavailable`` decides what happens to licensed data this machine
    cannot reach: ``True`` leaves it out with a warning, ``False`` raises,
    ``None`` takes the ``ethos-data`` configuration.
    """
    return handle().fetch(
        collection, progressbar=progressbar, test=test, skip_unavailable=skip_unavailable
    )


def paths(
    collection: str,
    test: bool = False,
    progressbar: bool = True,
    skip_unavailable: bool | None = None,
):
    """The inputs a collection names, as ``{handle: Path}``, fetched.

    The handles are the ones ``collections.yaml`` defines under ``paths:`` for
    the collection -- ``era5``, ``gwa_100m`` -- resolved to where the data is on
    this machine::

        inputs = data.paths("onshore_wind", test=True)
        rk.wind.wind_era5_PenaSanchezDunkelWinklerEtAl2025(
            placements, era5_path=inputs["era5"], gwa_100m_path=inputs["gwa_100m"],
            height_scaling_data={50: inputs["gwa_50m"], 200: inputs["gwa_200m"]},
        )

    ``test=True`` selects the small ``test`` variant; drop it to run the same
    code on the full data. Raises ``ethos_data.CollectionError`` if the
    collection declares no ``paths``. Under ``skip_unavailable`` a handle whose
    data this machine cannot reach is left out of the mapping, with a warning
    naming it (see :func:`fetch`).
    """
    return handle().paths(
        collection, test=test, progressbar=progressbar, skip_unavailable=skip_unavailable
    )


def path(key: str, progressbar: bool = False) -> Path:
    """Return the local path of one file, fetching it if necessary.

    ``key`` is ``"<dataset>/<resource path>"``, e.g.
    ``"reskit-test-data/era5/2m_temperature.nc"``; a shapefile brings its
    sidecars along. Looked up in the same catalogue :func:`paths` reads.
    """
    return handle().catalog.path(key, progressbar=progressbar)


def directory(key_prefix: str, progressbar: bool = False) -> Path:
    """Fetch every file under a prefix and return the directory holding them.

    For readers that want a folder rather than a file list -- RESKit's weather
    sources, for instance, are pointed at a directory of netCDF files.
    ``key_prefix`` is a dataset (``"landcover"``), a member of a family
    (``"reskit-test-data/era5"``), a folder inside a dataset
    (``"reskit-test-data/merra2/merged"``) or a whole family. Dataset names may
    themselves contain "/", so the split is by the longest catalogued name, not
    the first slash.
    """
    return handle().catalog.path(key_prefix, progressbar=progressbar)


def cache_dir() -> Path:
    """Where the shared ETHOS.Data cache lives on this machine."""
    return _ethos_data().cache_dir()


def main(argv: list[str] | None = None) -> int:
    """The ``reskit-data`` command: ETHOS.Data's collection commands on RESKit's file.

    ``list``, ``info``, ``plan``, ``fetch``, ``paths``, ``verify``, ``path``,
    ``ls``, ``bundle`` and ``config``; ``reskit-data --help`` lists them. The
    catalogue is loaded only for the commands that need it, so ``--help`` and
    ``config show`` work offline.
    """
    return _ethos_data().tool_main(
        COLLECTIONS_FILE, tool=TOOL, prog=COMMAND, catalog=_catalog_override(), argv=argv
    )
