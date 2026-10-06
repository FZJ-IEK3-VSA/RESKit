"""Check the inputs of a RESKit workflow before running it, see validate_inputs.

A simulation discovers a missing placement column, an unavailable weather variable, a
raster without values at the placements or a time span outside the weather data only
while it runs, possibly after hours. validate_inputs performs these checks up front,
reading only metadata and single raster values, and reports all problems at once.

The checks are derived from the workflow itself wherever possible: its signature gives
the arguments, and its calls of WorkflowManager.read() give the weather sources and
variables. Only the placement columns are listed per workflow family, below.
"""

import ast
import importlib
import inspect
import netrc
import os
import re
import textwrap
import warnings
from dataclasses import dataclass, field
from urllib.parse import urlparse

import geokit as gk
import numpy as np
import pandas as pd

from reskit.util.errors import ResError
from reskit.util.paths import as_path_string, is_path_like

WORKFLOW_FAMILIES = ("wind", "solar", "csp", "dac", "cooling_heating", "geothermal")

# The columns the placements need besides their location, per workflow family. Each
# requirement lists alternatives, of which one must be present completely.
REQUIRED_PLACEMENT_COLUMNS = {
    "wind": [[("capacity",)], [("hub_height",)], [("rotor_diam",), ("powerCurve",)]],
    "solar": [[("capacity",), ("modules_per_string", "strings_per_inverter")]],
    "csp": [[("land_area_m2",), ("aperture_area_m2",), ("area",), ("area_m2",)]],
    "dac": [[("capacity",)]],
    "cooling_heating": [[("capacity",)]],
    "geothermal": [],
}
NON_NUMERIC_COLUMNS = {"powerCurve"}

# Arguments naming a file the workflow writes rather than reads
OUTPUT_ARGUMENTS = {"output_netcdf_path", "savepath"}

# Arguments of WorkflowManager.read() which do not configure the weather source
_READ_ONLY_ARGUMENTS = {
    "variables",
    "source",
    "source_type",
    "set_time_index",
    "spatial_interpolation_mode",
    "temporal_reindex_method",
}

_FILE_EXTENSIONS = {".tif", ".tiff", ".nc", ".nc4", ".zarr", ".shp", ".gpkg", ".csv", ".xlsx", ".json", ".yaml"}


@dataclass(frozen=True)
class Finding:
    """One result of validate_inputs.

    level is "error" (the workflow would fail or give wrong results), "warning" (the
    workflow runs, but probably not as intended) or "info".
    """

    level: str
    check: str
    message: str

    def __str__(self):
        return f"{self.level.upper():7} [{self.check}] {self.message}"


@dataclass
class ValidationReport:
    """The findings of validate_inputs for one workflow call."""

    workflow: str
    findings: list = field(default_factory=list)

    @property
    def errors(self):
        return [f for f in self.findings if f.level == "error"]

    @property
    def warnings(self):
        return [f for f in self.findings if f.level == "warning"]

    @property
    def ok(self):
        """True if no error was found."""
        return not self.errors

    def raise_if_errors(self):
        """Raises a ResError listing all findings if an error was found."""
        if not self.ok:
            raise ResError(str(self))

    def _add(self, level, check, message):
        self.findings.append(Finding(level, check, message))

    def __str__(self):
        header = f"Inputs of {self.workflow}: {len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        return "\n".join([header] + [f"  {finding}" for finding in self.findings])


def validate_inputs(workflow, placements, **workflow_kwargs):
    """Checks the inputs of a workflow without running the simulation.

    The following is checked:
        * The arguments match the signature of the workflow
        * The placements have a valid location and the columns the workflow needs, with
          numeric values for every placement
        * Every weather source opens, provides the variables the workflow reads, covers
          all placements and has a regular time axis; a 'time_slice' lies within the
          available time span; a 'https://' store has credentials in ~/.netrc
        * Every input file exists, and every raster has values at the placements
        * The directory of every output file exists and is writable

    Only metadata and the raster values at the placements are read, so the check takes
    seconds also for large inputs.

    Parameters
    ----------
    workflow : str or callable
        The workflow, e.g. "wind_era5_PenaSanchezDunkelWinklerEtAl2025", or the
        workflow function itself
    placements : pandas.DataFrame
        The placements, as they would be passed to the workflow
    **workflow_kwargs
        The further arguments, as they would be passed to the workflow

    Returns
    -------
    ValidationReport
        Print it to see all findings; `.ok` is False if an error was found, and
        `.raise_if_errors()` turns errors into an exception

    Raises
    ------
    ValueError
        If the workflow is unknown

    Examples
    --------
    >>> report = rk.validate_inputs(
    ...     "wind_era5_PenaSanchezDunkelWinklerEtAl2025",
    ...     placements,
    ...     era5_path=era5_path,
    ...     gwa_100m_path=gwa_100m_path,
    ...     height_scaling_data=height_scaling_data,
    ... )
    >>> print(report)
    >>> report.raise_if_errors()
    """
    function = _resolve_workflow(workflow)
    report = ValidationReport(function.__name__)

    if "RESKitDeprecationError" in inspect.getsource(function):
        report._add("error", "workflow", f"'{function.__name__}' was removed from RESKit, see its docstring")
        return report

    signature = inspect.signature(function)
    try:
        bound = signature.bind(placements, **workflow_kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
    except TypeError as error:
        report._add("error", "arguments", str(error))
        arguments = {**_defaults(signature), **workflow_kwargs}

    manager = _check_placements(report, function.__module__.split(".")[1], placements)

    reads = _weather_reads(function)
    if manager is not None:
        for read in reads:
            _check_weather(report, manager, read, arguments)

    skip = {"placements"} | {read.argument for read in reads}
    _check_files(report, manager, {k: v for k, v in arguments.items() if k not in skip})
    return report


def _resolve_workflow(workflow):
    """The workflow function for a name, or the given function."""
    if callable(workflow):
        return workflow

    from reskit.util.input_preparation import DEPRECATED_WORKFLOW_NAMES

    name = DEPRECATED_WORKFLOW_NAMES.get(workflow, workflow)
    for family in WORKFLOW_FAMILIES:
        module = importlib.import_module(f"reskit.{family}.workflows.workflows")
        function = getattr(module, name, None)
        if inspect.isfunction(function) and function.__module__ == module.__name__:
            return function
    raise ValueError(f"Unknown RESKit workflow: {workflow!r}")


def _defaults(signature):
    return {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }


def _check_placements(report, family, placements):
    """Checks the placements; returns a WorkflowManager for them, or None if they are unusable."""
    from reskit.workflow_manager import WorkflowManager

    if not isinstance(placements, pd.DataFrame):
        report._add("error", "placements", f"must be a pandas DataFrame, not {type(placements).__name__}")
        return None
    if placements.empty:
        report._add("error", "placements", "contain no placements")
        return None

    for requirement in REQUIRED_PLACEMENT_COLUMNS.get(family, []):
        present = [columns for columns in requirement if all(c in placements.columns for c in columns)]
        if not present:
            alternatives = " or ".join(" and ".join(f"'{c}'" for c in columns) for columns in requirement)
            report._add("error", "placements", f"need the column {alternatives}")
            continue
        for column in present[0]:
            if column in NON_NUMERIC_COLUMNS:
                continue
            values = placements[column]
            if not pd.api.types.is_numeric_dtype(values):
                report._add("error", "placements", f"column '{column}' must be numeric, not {values.dtype}")
            elif values.isna().any():
                report._add("error", "placements", f"{values.isna().sum()} placements have no '{column}'")

    try:
        return WorkflowManager(placements)
    except Exception as error:  # every reason the locations are unusable is a finding
        report._add("error", "placements", f"have no valid location: {error}")
        return None


@dataclass
class _WeatherRead:
    """A call of WorkflowManager.read() in a workflow, see _weather_reads."""

    argument: str  # the workflow argument holding the path of the weather source
    source_type: object  # the source type, or the name of the argument holding it
    source_type_is_argument: bool
    variables: list
    options: dict  # further arguments: values, or names of workflow arguments (as ast.Name)


def _weather_reads(function):
    """The weather reads of a workflow, taken from its calls of WorkflowManager.read().

    A call of another workflow of the same module, e.g. by a wrapper, contributes the
    reads of that workflow.
    """
    parameters = inspect.signature(function).parameters
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    reads = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            callee = function.__globals__.get(node.func.id)
            if inspect.isfunction(callee) and callee is not function and callee.__module__ == function.__module__:
                reads.extend(read for read in _weather_reads(callee) if read.argument in parameters)
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "read"):
            continue
        keywords = {k.arg: k.value for k in node.keywords if k.arg is not None}
        source = keywords.get("source")
        if not (isinstance(source, ast.Name) and source.id in parameters):
            continue
        source_type = keywords["source_type"]
        options = {}
        for name, value in keywords.items():
            if name in _READ_ONLY_ARGUMENTS:
                continue
            if isinstance(value, ast.Name) and value.id in parameters:
                options[name] = value
            else:
                try:
                    options[name] = ast.literal_eval(value)
                except ValueError:
                    pass
        reads.append(
            _WeatherRead(
                argument=source.id,
                source_type=source_type.id if isinstance(source_type, ast.Name) else ast.literal_eval(source_type),
                source_type_is_argument=isinstance(source_type, ast.Name),
                variables=ast.literal_eval(keywords["variables"]),
                options=options,
            )
        )

    # a wrapper may call the same workflow more than once, e.g. once per dataset
    unique = {}
    for read in reads:
        unique.setdefault((read.argument, read.source_type, tuple(read.variables)), read)
    return list(unique.values())


def _check_weather(report, manager, read, arguments):
    """Opens a weather source for its metadata only and checks it against the workflow."""
    check = f"weather '{read.argument}'"
    path = arguments.get(read.argument)
    if path is None:
        report._add("error", check, "no weather source given")
        return
    if not is_path_like(path):
        report._add("info", check, "given as an initialized source, not checked")
        return
    path = as_path_string(path)
    source_type = arguments.get(read.source_type) if read.source_type_is_argument else read.source_type
    options = {k: arguments.get(v.id) if isinstance(v, ast.Name) else v for k, v in read.options.items()}

    if re.search(r"<[^>]*>", path):
        report._add(
            "error",
            check,
            f"contains tile placeholders: {path}. Workflows need the path of one tile; "
            "execute_workflow_iteratively resolves the placeholders per placement",
        )
        return
    remote = urlparse(path).scheme in ("http", "https", "gs", "s3")
    if not remote and not os.path.exists(path):
        report._add("error", check, f"does not exist: {path}")
        return
    if urlparse(path).scheme == "https" and not _has_netrc_entry(urlparse(path).hostname):
        report._add("warning", check, f"~/.netrc has no credentials for {urlparse(path).hostname}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            source = manager._open_source(source_type, path, **options)
        except Exception as error:  # every reason the source does not open is a finding
            report._add("error", check, f"cannot be opened as {source_type} source: {error}")
            return
        used_variables = _probe_variables(report, check, source, read.variables)

    _check_time_axis(report, check, source, used_variables, options.get("time_slice"))
    _check_coverage(report, check, source, manager)


def _has_netrc_entry(host):
    try:
        return netrc.netrc().authenticators(host) is not None
    except (FileNotFoundError, netrc.NetrcParseError):
        return False


def _probe_variables(report, check, source, variables):
    """Runs the source's standard loaders without reading data, reporting what is missing.

    The loaders decide which raw variables a standard variable needs, including their
    fallbacks. Replacing 'load' by a check of the source's variable table lets them run
    on placeholders. Returns the raw variables the loaders would read.
    """
    used = []

    def probe(variable, name=None, *args, **kwargs):
        if variable not in source.variables.index:
            raise ResError(f"the source has no variable '{variable}'")
        used.append(variable)
        source.data[name or variable] = np.zeros((1, 1, 1))

    source.load = probe
    for variable in variables:
        try:
            source.sload(variable)
        except Exception as error:  # every reason a loader fails is a finding
            report._add("error", check, f"cannot provide '{variable}': {error}")
    return used


def _check_time_axis(report, check, source, used_variables, time_slice):
    time_index = source.time_index
    if len(time_index) == 0:
        report._add("error", check, "has no time steps in the requested time span")
        return
    report._add("info", check, f"{len(time_index)} time steps from {time_index[0]} to {time_index[-1]}")

    steps = np.unique(np.diff(time_index.values))
    if steps.size > 1:
        report._add("warning", check, f"has an irregular time axis with steps of {', '.join(map(str, steps))}")

    if isinstance(time_slice, slice):
        for bound, outside, edge in (
            (time_slice.start, lambda t: t < time_index[0], "starts before"),
            (time_slice.stop, lambda t: t > time_index[-1], "ends after"),
        ):
            if bound is None:
                continue
            bound = pd.Timestamp(bound)
            if time_index.tz is not None and bound.tz is None:
                bound = bound.tz_localize(time_index.tz)
            if outside(bound):
                report._add(
                    "warning",
                    check,
                    f"'time_slice' {edge} the data; only {time_index[0]} to {time_index[-1]} is simulated",
                )

    # every file of a netCDF4 source must have the time steps of the time axis
    if "shape" in source.variables.columns:
        expected = len(getattr(source, "_timeindex_full", source._timeindex_raw))
        for variable in dict.fromkeys(used_variables):
            steps = source.variables.loc[variable, "shape"][0]
            if steps != expected:
                report._add(
                    "error", check, f"variable '{variable}' has {steps} time steps, the time axis has {expected}"
                )


def _check_coverage(report, check, source, manager):
    """Reports placements which are more than half a grid cell outside the source's grid."""
    lats, lons = np.asarray(source.lats), np.asarray(source.lons)
    half_lat = np.abs(np.diff(lats)).max() / 2 if lats.size > 1 else 0
    half_lon = np.abs(np.diff(lons)).max() / 2 if lons.size > 1 else 0
    lon = manager.locs.lons % 360 if getattr(source, "_longitude_360", False) else manager.locs.lons
    lat = manager.locs.lats
    outside = (
        (lat < lats.min() - half_lat)
        | (lat > lats.max() + half_lat)
        | (lon < lons.min() - half_lon)
        | (lon > lons.max() + half_lon)
    )
    if outside.any():
        first = np.flatnonzero(outside)[0]
        report._add(
            "error",
            check,
            f"does not cover {outside.sum()} of {outside.size} placements, "
            f"e.g. lon={manager.locs.lons[first]}, lat={manager.locs.lats[first]}",
        )


def _check_files(report, manager, arguments):
    """Checks every argument which names a file: input files exist, rasters have values at
    the placements, and output files can be written.
    """
    for name, value in arguments.items():
        for label, path in _paths_in(name, value):
            if name in OUTPUT_ARGUMENTS:
                directory = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
                if not os.path.isdir(directory):
                    report._add("error", label, f"the output directory does not exist: {directory}")
                elif not os.access(directory, os.W_OK):
                    report._add("error", label, f"the output directory is not writable: {directory}")
                continue
            if not os.path.exists(path):
                report._add("error", label, f"does not exist: {path}")
                continue
            if manager is None or not _is_raster(path):
                continue
            try:
                values = gk.raster.interpolateValues(path, points=manager.locs, mode="near")
            except Exception as error:  # every reason a raster cannot be read is a finding
                report._add("error", label, f"cannot be read as raster: {error}")
                continue
            missing = int(np.isnan(np.asarray(values, dtype=float)).sum())
            if missing:
                report._add(
                    "warning",
                    label,
                    f"has no value at {missing} of {manager.locs.count} placements (outside the raster or nodata)",
                )


def _is_raster(path):
    try:
        return gk.util.isRaster(path)
    except Exception:  # geokit fails on some files instead of returning False, e.g. on netCDF4 files
        return False


def _paths_in(name, value):
    """The (label, path) pairs of an argument value which names files, also inside a dict or list."""
    if isinstance(value, dict):
        items = [(f"{name}[{key!r}]", item) for key, item in value.items()]
    elif isinstance(value, (list, tuple)):
        items = [(f"{name}[{i}]", item) for i, item in enumerate(value)]
    else:
        items = [(name, value)]
    for label, item in items:
        if _looks_like_path(item):
            yield label, as_path_string(item)


def _looks_like_path(value):
    if isinstance(value, os.PathLike):
        return True
    if not isinstance(value, str) or urlparse(value).scheme in ("http", "https", "gs", "s3"):
        return False
    return "/" in value or os.sep in value or os.path.splitext(value)[1].lower() in _FILE_EXTENSIONS
