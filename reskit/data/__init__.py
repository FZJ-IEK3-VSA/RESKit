"""Access to the datasets RESKit needs.

Data is described by the shared ETHOS.Data catalogue and downloaded on demand
into a cache that every ICE-2 tool shares, so a dataset used by more than one
tool is fetched once. Apart from the small test fixtures in ``test_cache`` (see
``reskit.TEST_DATA``), nothing is bundled with the package.

    from reskit import data

    files = data.fetch("onshore_wind")          # a whole collection
    clc   = data.path("landcover/C3S-LC-L4-LCCS-Map-300m-P1Y-2018-v2.1.1.tif")
    era5  = data.directory("reskit-test-data/era5")

``fetch`` returns a mapping of ``"<dataset>/<path>" -> pathlib.Path``. Files are
verified against the checksums in the catalogue, and anything already present is
not downloaded again.

Which collections exist is defined in ``collections.yaml`` next to this module;
what each dataset contains is defined in the catalogue it pins.

Configuration (all optional):
    RESKIT_DATA_CATALOG   use a different catalogue, e.g. the internal ETHOS.Data one
    ETHOS_DATA_DIR        where the shared cache lives
Run ``ethos-data config show`` to see what is in effect, or
``ethos-data config set-cache <dir> --scope environment`` to set it permanently.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

__all__ = ["cache_dir", "collections", "describe", "directory", "fetch", "path", "plan"]

COLLECTIONS_FILE = Path(__file__).resolve().parent / "collections.yaml"
CATALOG_ENV = "RESKIT_DATA_CATALOG"


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


@lru_cache(maxsize=1)
def _loaded():
    return _ethos_data().load_collections(COLLECTIONS_FILE, catalog=os.environ.get(CATALOG_ENV))


def collections() -> list[str]:
    """Names of the collections RESKit defines."""
    return _loaded().names()


def describe(collection: str) -> dict:
    """What a collection is for, and what it would fetch."""
    loaded = _loaded()
    resources = loaded.resolve(collection)
    return {
        "title": loaded.describe(collection).get("title", ""),
        "files": len(resources),
        "bytes": sum(r.bytes for r in resources),
        "keys": [r.key for r in resources],
    }


def plan(collection: str) -> dict:
    """What fetching this collection would download, without downloading it."""
    loaded = _loaded()
    return _ethos_data().plan(loaded.catalog, loaded.resolve(collection))


def fetch(collection: str, progressbar: bool = True):
    """Make a collection available locally.

    Returns a mapping of ``"<dataset>/<resource path>" -> Path``, in catalogue
    order. Also usable as a list via ``.paths``.
    """
    return _ethos_data().fetch(
        collection,
        collections=COLLECTIONS_FILE,
        catalog=os.environ.get(CATALOG_ENV),
        progressbar=progressbar,
    )


def path(key: str, progressbar: bool = False) -> Path:
    """Return the local path of one file, fetching it if necessary.

    ``key`` is ``"<dataset>/<resource path>"``, e.g.
    ``"reskit-test-data/era5/2m_temperature.nc"``.
    """
    return _ethos_data().fetch_one(
        key,
        catalog=_loaded().catalog,
        progressbar=progressbar,
    )


def directory(key_prefix: str, progressbar: bool = False) -> Path:
    """Fetch every file under a prefix and return the directory holding them.

    For readers that want a folder rather than a file list -- RESKit's weather
    sources, for instance, are pointed at a directory of netCDF files.
    """
    loaded = _loaded()
    dataset_name, _, sub = key_prefix.partition("/")
    dataset = loaded.catalog.dataset(dataset_name)
    # Match on a directory boundary, so that "merra-like" asks for the folder and
    # does not also pick up the sibling file "merra-like.nc4".
    sub = sub.rstrip("/") + "/" if sub else ""
    resources = [r for p, r in dataset.resources.items() if p.startswith(sub)]
    if not resources:
        raise KeyError(f"nothing in the catalogue under {key_prefix!r}")

    files = _ethos_data().download(loaded.catalog, resources, progressbar=progressbar)
    directories = files.directories
    if len(directories) > 1:
        raise ValueError(
            f"{key_prefix!r} spans {len(directories)} directories; use fetch() and pass the individual paths instead"
        )
    return directories[0]


def cache_dir() -> Path:
    """Where the shared ETHOS.Data cache lives on this machine."""
    return _ethos_data().cache_dir()
