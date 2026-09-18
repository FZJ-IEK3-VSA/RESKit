"""Access to the datasets RESKit needs.

Data is described by the shared ETHOS.Data catalogue and downloaded on demand
into a cache that every ETHOS tool shares, so a dataset used by more than one
tool is fetched once. The one dataset RESKit carries itself is the
``reskit-test-data`` fixture family: ``test_cache`` beside this module is a
verified copy of it (an ETHOS.Data *bundle*, see :data:`BUNDLE`), and by
default the fixtures are read from there, offline, rather than downloaded.

    from reskit import data

    inputs = data.paths("wind_era5_PenaSanchezDunkelWinklerEtAl2025", test=True)   # {handle: Path}, small fixtures
    files = data.fetch("wind_era5_PenaSanchezDunkelWinklerEtAl2025", test=True)
    clc    = data.path("corine-land-cover/CLC2018_CLC2018_V2018_20.tif")
    era5   = data.directory("reskit-test-data/era5")

The same from the shell, with the ``reskit-data`` command this module provides:

    reskit-data show                              # the collections and their size
    reskit-data show wind_era5_PenaSanchezDunkelWinklerEtAl2025 --test          # one of them, and its inputs
    reskit-data fetch wind_era5_PenaSanchezDunkelWinklerEtAl2025 --test --paths # fetch one, print handle<TAB>path
    reskit-data config show                       # where the cache is, which catalogue
    reskit-data staging add trial /path/to/data   # use unpublished development data
    reskit-data bundle verify <BUNDLE> test_suite # check the bundled fixtures

``show`` never downloads and ``fetch`` is the only command that does. Both work
in collections; a single catalogue key -- one dataset, folder or file -- is
``ethos-data ls`` and ``ethos-data fetch``, which read the same catalogue.

``paths`` is what a workflow wants. The collection names each input the workflow
takes (``era5``, ``gwa_100m``, ...) under ``paths:`` in ``collections.yaml``, so
the caller gets ``{handle: pathlib.Path}`` without knowing a single resource
key. Call it immediately before the workflow; no CLI setup step is required.
Workflow collections use the corresponding Python function's name.
``test=True`` selects the small fixtures the collection pairs with the full
data; both variants offer the same handles, so the same code runs on either.
The full data is the default: a forgotten flag must never silently run a real
calculation on fixtures, while an accidental full download is visible and can
be interrupted.

The pinned public catalogue currently has no full ERA5 dataset. The full variant
of ``wind_era5_PenaSanchezDunkelWinklerEtAl2025`` therefore raises
``UnknownDataset`` until a catalogue containing its full inputs is selected.

``fetch`` returns a mapping of ``"<dataset>/<path>" -> pathlib.Path``. Files are
verified against the checksums in the catalogue, and anything already present is
not downloaded again.

The bundled fixtures. ``test_cache/bundle.json`` lists every file of the
``reskit-test-data`` family with the size and SHA-256 the pinned catalogue
records, and the files sit beside it under ``data/<dataset>/<path>``.
``paths``, ``fetch``, ``path`` and ``directory`` answer from that copy whenever
it holds what was asked for -- checked against those hashes once per process,
never downloaded, never written to -- and go to the catalogue for everything
else. Pass ``download=True``, or set ``RESKIT_DATA_DOWNLOAD=1``, to skip the
bundle and fetch the fixtures from the catalogue's store into the shared cache
like any other dataset; the argument wins over the variable. A bundled file
that is missing or altered is an error, not a reason to download: the copy in
a checkout is what the tests are meant to run on. ``describe`` and ``plan``
always describe the catalogue route, and so does every ``reskit-data`` command
except ``bundle``. The bundle is re-exported with ``reskit-data bundle export``
whenever the catalogue pin moves; docs/how_to/get_input_data.md has the steps.

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
    RESKIT_DATA_DOWNLOAD  1/true/yes/on: fetch the fixtures from the catalogue's
                          store instead of reading the bundled copy
    ETHOS_DATA_DIR        where the shared cache lives
Without either catalogue variable the one ``collections.yaml`` pins is used.
Every function that goes to the catalogue reads the same handle, so ``paths``,
``fetch``, ``path`` and ``directory`` never read different catalogues.
Run ``reskit-data config show`` to see what is in effect, or
``reskit-data config set-cache <dir> --scope environment`` to set it permanently.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

__all__ = [
    "BUNDLE",
    "COLLECTIONS_FILE",
    "bundled",
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
#: RESKit's copy of the ``reskit-test-data`` family, as an ETHOS.Data bundle:
#: ``bundle.json`` (the catalogue's inventory, sizes and hashes) beside
#: ``data/<dataset>/<path>``. ``reskit.TEST_DATA`` reads the same files by name.
BUNDLE = Path(__file__).resolve().parent / "test_cache"
CATALOG_ENV = "RESKIT_DATA_CATALOG"
DOWNLOAD_ENV = "RESKIT_DATA_DOWNLOAD"
TOOL = "reskit"
COMMAND = "reskit-data"

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"", "0", "false", "no", "off"}


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


def _download_requested(download: bool | None) -> bool:
    """Whether to bypass the bundled fixtures: the argument if given, else ``$RESKIT_DATA_DOWNLOAD``."""
    if download is not None:
        return bool(download)
    value = os.environ.get(DOWNLOAD_ENV, "").strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    accepted = ", ".join(sorted(_TRUTHY | (_FALSY - {""})))
    raise ValueError(f"${DOWNLOAD_ENV}={value!r} is not a yes/no value; use one of {accepted}")


@lru_cache(maxsize=1)
def handle():
    """The ``ethos_data.Collections`` handle on RESKit's collections file, built once.

    The file and the catalogue are read the first time anything here is called
    and never again in this process. ``handle().catalog`` is the catalogue
    RESKit reads, for access by key -- what :func:`path` and :func:`directory`
    use -- so every route through this module sees the same one.
    """
    return _ethos_data().collections(COLLECTIONS_FILE, tool=TOOL, catalog=_catalog_override())


@lru_cache(maxsize=1)
def _definitions():
    """The collections file on its own: what each collection *names*, with no catalogue behind it.

    Enough for ``named_keys`` -- the ``paths:`` handles and their keys -- which
    is all the bundle needs to answer :func:`paths` offline. Anything that needs
    a dataset's inventory goes through :func:`handle`.
    """
    ethos_data = _ethos_data()
    return ethos_data.load_collections(
        COLLECTIONS_FILE,
        catalog=ethos_data.Catalog(location="", descriptor={}, datasets={}),
        include_staging=False,
        tool=TOOL,
    )


@lru_cache(maxsize=1)
def _bundle():
    """The bundle at :data:`BUNDLE`, loaded once; None when this installation ships none."""
    if not (BUNDLE / "bundle.json").is_file():
        return None
    return _ethos_data().load_bundle(BUNDLE)


@lru_cache(maxsize=1)
def bundled():
    """Every file in the shipped bundle, as ``{"<dataset>/<path>": Path}``, hash-checked once per process.

    None when this installation ships no ``bundle.json``; everything then comes
    from the catalogue. A bundled file that is missing or altered raises
    ``ethos_data.BundleError`` -- the copy is never repaired by downloading.
    """
    bundle = _bundle()
    if bundle is None:
        return None
    files = _ethos_data().DataFiles()
    for name in bundle.names():
        files.update(bundle.fetch(name))
    return files


def _from_bundle(key: str) -> Path | None:
    """The bundled file or folder ``key`` names, or None when the bundle does not hold it.

    The catalogue's vocabulary: ``"<dataset>/<path>"`` is a file; a key with
    bundled files beneath it -- a folder, a dataset or the whole family -- is
    the directory holding them, laid out as ``data/<key>``.
    """
    files = bundled()
    if files is None:
        return None
    key = key.strip("/")
    if key in files:
        return files[key]
    if any(other.startswith(key + "/") for other in files):
        return _bundle().path.joinpath("data", *key.split("/"))
    return None


def _named_from_bundle(collection: str, test: bool):
    """A collection's ``paths`` resolved into the bundle, or None if it names anything not bundled."""
    keys = _definitions().named_keys(collection, test=test)
    if not keys:
        return None  # nothing under paths: -- the catalogue route raises the usual explanation
    named = _ethos_data().NamedPaths(collection=collection)
    for handle_, key in keys.items():
        local = _from_bundle(key)
        if local is None:
            return None
        named[handle_] = local
    return named


def _files_from_bundle(collection: str, test: bool):
    """A collection's files from the bundle, or None when the bundle cannot answer it whole.

    A collection the bundle was exported for (``test_suite``) is read straight
    off the manifest. Any other is resolved against the catalogue first --
    metadata only, no bytes -- and served from the bundle when every file it
    selects is there.
    """
    bundle = _bundle()
    if bundle is None:
        return None
    files = bundled()
    if collection in bundle.collections and not (test and _definitions().variants(collection)):
        keys = list(bundle.collections[collection])
    else:
        keys = [resource.key for resource in handle().resolve(collection, test=test)]
        if not all(key in files for key in keys):
            return None
    ethos_data = _ethos_data()
    result = ethos_data.DataFiles((key, files[key]) for key in keys)
    result.named = _named_from_bundle(collection, test) or ethos_data.NamedPaths(collection=collection)
    return result


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
    """What fetching this collection from the catalogue would download, without downloading it.

    Describes the download route only: with the bundled fixtures in front of
    it, :func:`fetch` transfers nothing for what the bundle holds.
    """
    return handle().plan(collection, test=test)


def fetch(
    collection: str,
    progressbar: bool = True,
    test: bool = False,
    skip_unavailable: bool | None = None,
    download: bool | None = None,
):
    """Make a collection available locally.

    Returns a mapping of ``"<dataset>/<resource path>" -> Path``, in catalogue
    order. Also usable as a list via ``.paths``; the inputs the collection names
    under ``paths:`` are on ``.named``. ``test=True`` selects the collection's
    small ``test`` variant where it defines one; the default is the full data.
    Files the bundled fixtures hold are returned from there unless
    ``download`` asks for the catalogue's store (default: ``$RESKIT_DATA_DOWNLOAD``).
    ``skip_unavailable`` decides what happens to licensed data this machine
    cannot reach: ``True`` leaves it out with a warning, ``False`` raises,
    ``None`` takes the ``ethos-data`` configuration.
    """
    if not _download_requested(download):
        files = _files_from_bundle(collection, test)
        if files is not None:
            return files
    return handle().fetch(collection, progressbar=progressbar, test=test, skip_unavailable=skip_unavailable)


def paths(
    collection: str,
    test: bool = False,
    progressbar: bool = True,
    skip_unavailable: bool | None = None,
    download: bool | None = None,
):
    """The inputs a collection names, as ``{handle: Path}``, fetched.

    The handles are the ones ``collections.yaml`` defines under ``paths:`` for
    the collection -- ``era5``, ``gwa_100m`` -- resolved to where the data is on
    this machine::

        inputs = data.paths(
            "wind_era5_PenaSanchezDunkelWinklerEtAl2025",
            test=True,
        )
        rk.wind.wind_era5_PenaSanchezDunkelWinklerEtAl2025(
            placements,
            era5_path=inputs[
                "era5"
            ],
            gwa_100m_path=inputs[
                "gwa_100m"
            ],
            height_scaling_data={
                50: inputs[
                    "gwa_50m"
                ],
                200: inputs[
                    "gwa_200m"
                ],
            },
        )

    ``test=True`` selects the small ``test`` variant; omitting it selects the
    full variant, which requires a catalogue containing all its inputs.
    Handles the bundled fixtures can answer are answered
    from there, offline, unless ``download`` asks for the catalogue's store
    (default: ``$RESKIT_DATA_DOWNLOAD``). Raises ``ethos_data.CollectionError``
    if the collection declares no ``paths``. Under ``skip_unavailable`` a handle
    whose data this machine cannot reach is left out of the mapping, with a
    warning naming it (see :func:`fetch`).
    """
    if not _download_requested(download):
        named = _named_from_bundle(collection, test)
        if named is not None:
            return named
    return handle().paths(collection, test=test, progressbar=progressbar, skip_unavailable=skip_unavailable)


def path(key: str, progressbar: bool = False, download: bool | None = None) -> Path:
    """Return the local path of one file, fetching it if necessary.

    ``key`` is ``"<dataset>/<resource path>"``, e.g.
    ``"reskit-test-data/era5/2m_temperature.nc"``; a shapefile brings its
    sidecars along. A bundled fixture is returned from the bundle unless
    ``download`` asks for the catalogue's store (default:
    ``$RESKIT_DATA_DOWNLOAD``); anything else is looked up in the same
    catalogue :func:`paths` reads.
    """
    if not _download_requested(download):
        local = _from_bundle(key)
        if local is not None:
            return local
    return handle().catalog.path(key, progressbar=progressbar)


def directory(key_prefix: str, progressbar: bool = False, download: bool | None = None) -> Path:
    """Fetch every file under a prefix and return the directory holding them.

    For readers that want a folder rather than a file list -- RESKit's weather
    sources, for instance, are pointed at a directory of netCDF files.
    ``key_prefix`` is a dataset (``"corine-land-cover"``), a member of a family
    (``"reskit-test-data/era5"``), a folder inside a dataset
    (``"reskit-test-data/merra2/merged"``) or a whole family. Dataset names may
    themselves contain "/", so the split is by the longest catalogued name, not
    the first slash. A prefix the bundled fixtures hold is answered from the
    bundle unless ``download`` asks for the catalogue's store (default:
    ``$RESKIT_DATA_DOWNLOAD``).
    """
    if not _download_requested(download):
        local = _from_bundle(key_prefix)
        if local is not None:
            return local
    return handle().catalog.path(key_prefix, progressbar=progressbar)


def cache_dir() -> Path:
    """Where the shared ETHOS.Data cache lives on this machine."""
    return _ethos_data().cache_dir()


def main(argv: list[str] | None = None) -> int:
    """RESKit's collections, test bundles and development staging.

    ``show``, ``fetch``, ``verify``, ``bundle``, ``staging`` and ``config``;
    ``reskit-data --help`` lists them. The catalogue is loaded only for the
    commands that need it, so ``--help``, ``config show``, ``staging`` and
    ``bundle`` work offline. ``fetch`` always goes through the catalogue -- it
    is the download route; ``bundle verify <BUNDLE> test_suite`` checks the
    bundled fixtures. Access by catalogue key (``ethos-data ls``, ``ethos-data
    fetch``), shared cache maintenance (``link``, ``unlink``, ``materialize``)
    and catalogue publishing are ``ethos-data``'s.
    """
    return _ethos_data().tool_main(COLLECTIONS_FILE, tool=TOOL, prog=COMMAND, catalog=_catalog_override(), argv=argv)
