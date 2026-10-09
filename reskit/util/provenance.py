"""Provenance of RESKit simulation results.

Every dataset a workflow returns or writes carries attributes which describe how it was made:

    reskit_version          the RESKit version
    reskit_git_commit       the commit of the RESKit checkout, "-dirty" with local changes
                            (only when RESKit runs from a git checkout)
    reskit_workflow         the workflow function, e.g. "reskit.wind.workflows.workflows.wind_config"
    reskit_created          when the dataset was made, ISO 8601 in UTC
    reskit_crs              the coordinate reference system of the 'lon'/'lat' variables
    reskit_python           the Python version and platform
    reskit_dependencies     JSON: {package: version} of the main dependencies
    reskit_parameters       JSON: the arguments the workflow was called with, except the placements
                            (they are part of the dataset anyway)
    reskit_weather_sources  JSON: list of the weather sources read, with variables and time span
    reskit_input_files      JSON: list of every file or folder the workflow was given or read
                            correction data from, with size, modification time and SHA-256

Values which netCDF cannot store as attributes are JSON strings; :func:`read_provenance` decodes them.
SHA-256 is computed for files up to :data:`HASH_SIZE_LIMIT` bytes, which covers parameter files and
correction rasters but skips weather archives.
"""

import contextvars
import functools
import hashlib
import inspect
import json
import os
import platform
import subprocess
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

import numpy as np
import pandas as pd
import xarray as xr

P = ParamSpec("P")
R = TypeVar("R")
StrPath = str | os.PathLike[str]

ATTRIBUTE_PREFIX = "reskit_"
JSON_ATTRIBUTES = ("dependencies", "parameters", "weather_sources", "input_files")

#: Files larger than this (in bytes) are identified by size and modification time only.
HASH_SIZE_LIMIT = 64 * 1024**2

# The distribution names of the dependencies whose versions are recorded, if installed.
DEPENDENCIES = (
    "numpy",
    "pandas",
    "scipy",
    "xarray",
    "netCDF4",
    "zarr",
    "geokit",
    "GDAL",
    "pvlib",
    "ethos-data",
)

_current_workflow: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "reskit_current_workflow", default=None
)


def record_provenance(workflow: Callable[P, R]) -> Callable[P, R]:
    """Decorates a workflow function so that its results name the workflow and its arguments.

    The WorkflowManager the workflow creates reads them in ``to_xarray``. When workflows are
    nested, the outermost one, i.e. the one the user called, is recorded.

    Parameters
    ----------
    workflow : callable
        The workflow function to decorate

    Returns
    -------
    callable
        A function with the signature and docstring of ``workflow``
    """
    signature = inspect.signature(workflow)
    name = f"{workflow.__module__}.{workflow.__qualname__}"  # type: ignore[attr-defined]

    @functools.wraps(workflow)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if _current_workflow.get() is not None:
            return workflow(*args, **kwargs)

        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments: dict[str, Any] = {}
        for key, value in bound.arguments.items():
            parameter = signature.parameters[key]
            if key == "placements":
                continue
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                arguments.update(value)
            elif parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                if value:
                    arguments[key] = list(value)
            else:
                arguments[key] = value

        token = _current_workflow.set({"workflow": name, "arguments": arguments})
        try:
            return workflow(*args, **kwargs)
        finally:
            _current_workflow.reset(token)

    return wrapper


def provenance_attributes(
    weather_sources: Sequence[Mapping[str, Any]] = (),
    input_files: Iterable[tuple[str, StrPath]] = (),
) -> dict[str, str]:
    """Returns the provenance attributes for a dataset made now.

    Parameters
    ----------
    weather_sources : sequence of dict
        The weather sources read, as recorded by the WorkflowManager
    input_files : iterable of (role, path)
        Further files read, e.g. correction rasters

    Returns
    -------
    dict
        The attributes to store in the dataset, with the "reskit_" prefix and the JSON values encoded
    """
    attributes: dict[str, Any] = {"version": metadata.version("reskit")}
    commit = _git_commit()
    if commit is not None:
        attributes["git_commit"] = commit

    workflow = _current_workflow.get()
    if workflow is not None:
        attributes["workflow"] = workflow["workflow"]

    attributes["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # WorkflowManager converts all placements to lon/lat in WGS84
    attributes["crs"] = "EPSG:4326"
    attributes["python"] = f"{platform.python_version()} ({platform.platform()})"
    attributes["dependencies"] = _dependency_versions()

    files: dict[str, dict[str, Any]] = {}

    def add_file(role: str, path: StrPath) -> None:
        """Adds the file under the given role, or the role to the file if it is already listed."""
        path = os.fspath(path)
        if path not in files:
            files[path] = {"path": path, "roles": [], **_fingerprint(path)}
        if role not in files[path]["roles"]:
            files[path]["roles"].append(role)

    if workflow is not None:
        attributes["parameters"] = {k: _to_json(v) for k, v in workflow["arguments"].items()}
        for key, value in workflow["arguments"].items():
            if key.startswith("output"):
                # e.g. output_netcdf_path, which may still hold an earlier result
                continue
            for path in _paths_in(value):
                add_file(f"argument:{key}", path)

    attributes["weather_sources"] = list(weather_sources)
    for source in weather_sources:
        if "path" in source:
            add_file(f"weather:{source['source_type']}", source["path"])
    for role, path in input_files:
        add_file(role, path)
    attributes["input_files"] = list(files.values())

    return {
        ATTRIBUTE_PREFIX + key: json.dumps(value) if key in JSON_ATTRIBUTES else value
        for key, value in attributes.items()
    }


def read_provenance(dataset: xr.Dataset) -> dict[str, Any]:
    """Returns the provenance attributes of a RESKit result, with the JSON values decoded.

    Parameters
    ----------
    dataset : xarray.Dataset
        A dataset a RESKit workflow returned or wrote

    Returns
    -------
    dict
        The provenance, keyed without the "reskit_" prefix
    """
    provenance: dict[str, Any] = {}
    for key, value in dataset.attrs.items():
        if not key.startswith(ATTRIBUTE_PREFIX):
            continue
        key = key[len(ATTRIBUTE_PREFIX) :]
        provenance[key] = json.loads(value) if key in JSON_ATTRIBUTES else value
    return provenance


def _is_path(value: Any) -> bool:
    """Tells whether a value is a path or URL: a PathLike, an existing local path or a remote URL string."""
    if isinstance(value, os.PathLike):
        return True
    if not isinstance(value, str) or value == "":
        return False
    return value.startswith(("gs://", "s3://", "http://", "https://")) or os.path.exists(value)


def _paths_in(value: Any) -> Iterator[StrPath]:
    """Yields every path in a (nested) argument value."""
    if _is_path(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _paths_in(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _paths_in(item)


def _fingerprint(path: str) -> dict[str, Any]:
    """Describes a file, directory or URL by its kind and, for files, size, modification time and SHA-256."""
    if "://" in path:
        return {"kind": "url"}
    if os.path.isdir(path):
        return {"kind": "directory"}
    if not os.path.isfile(path):
        return {"kind": "missing"}
    stat = os.stat(path)
    fingerprint = {
        "kind": "file",
        "size": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
    }
    if stat.st_size <= HASH_SIZE_LIMIT:
        fingerprint["sha256"] = _sha256(os.path.abspath(path), stat.st_size, stat.st_mtime_ns)
    return fingerprint


@functools.lru_cache(maxsize=None)
def _sha256(path: str, size: int, mtime_ns: int) -> str:
    """Returns the SHA-256 of a file; size and mtime_ns are part of the cache key, so a changed file is rehashed."""
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _to_json(value: Any) -> Any:
    """Converts a workflow argument into something JSON can hold.

    Arrays and tables are replaced by their shape and a hash, other unknown objects by their type.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        # JSON has no NaN or infinity
        return value if np.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return _to_json(value.item())
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, dict):
        return {str(k): _to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json(v) for v in value]
    if isinstance(value, slice):
        return {"slice": [_to_json(value.start), _to_json(value.stop), _to_json(value.step)]}
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return {
            "ndarray": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
        }
    if isinstance(value, (pd.DataFrame, pd.Series)):
        hashes = pd.util.hash_pandas_object(value, index=True).to_numpy()
        return {
            type(value).__name__: list(value.shape),
            "sha256": hashlib.sha256(hashes.tobytes()).hexdigest(),
        }
    if callable(value):
        return f"{getattr(value, '__module__', None)}.{getattr(value, '__qualname__', repr(value))}"
    return f"<{type(value).__module__}.{type(value).__qualname__}>"


@functools.lru_cache(maxsize=None)
def _dependency_versions() -> dict[str, str]:
    """Returns the installed versions of the :data:`DEPENDENCIES`, skipping those which are not installed."""
    versions: dict[str, str] = {}
    for name in DEPENDENCIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return versions


@functools.lru_cache(maxsize=None)
def _git_commit() -> str | None:
    """Returns the commit of the RESKit checkout, or None if RESKit is not run from one.

    The commit has the suffix "-dirty" if tracked files have local changes.
    """
    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return None
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        changes = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return commit + "-dirty" if changes else commit
