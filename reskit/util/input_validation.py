"""Check the inputs of a RESKit workflow before running it, see validate_inputs.

A simulation discovers a missing placement column, an unavailable weather variable, a
raster without values at the placements or a time span outside the weather data only
while it runs, possibly after hours. validate_inputs performs these checks up front,
reading only metadata and the raster values at the placements, and reports all problems
at once.

Every workflow declares its inputs with declare_inputs: its workflow manager, which
knows the placement columns it needs, the weather sources it reads, and the arguments
which name input or output files.
"""

import functools
import importlib
import inspect
import netrc
import os
import re
import warnings
from dataclasses import dataclass, field
from urllib.parse import urlparse

import geokit as gk
import numpy as np
import pandas as pd

from reskit.util.errors import ResError
from reskit.util.paths import as_path_string, is_path_like

WORKFLOW_FAMILIES = ("wind", "solar", "csp", "dac", "cooling_heating", "geothermal")

_REMOTE_SCHEMES = ("http", "https", "gs", "s3")


@dataclass(frozen=True)
class WeatherInput:
    """A weather source which a workflow reads.

    Parameters
    ----------
    source_type : str, optional
        The source type, e.g. "ERA5", as WorkflowManager.read() takes it
    variables : tuple of str
        The standard variables the workflow reads, e.g. "elevated_wind_speed"
    source_type_argument : str, optional
        The workflow argument which gives the source type, instead of `source_type`
    time_index_from : str, optional
        The variable whose time axis the workflow uses, see WorkflowManager.read()
    """

    source_type: str = None
    variables: tuple = ()
    source_type_argument: str = None
    time_index_from: str = None


@dataclass(frozen=True)
class WorkflowInputs:
    """The inputs a workflow declares with declare_inputs."""

    manager: type
    weather: dict
    files: tuple
    outputs: tuple


def declare_inputs(manager, weather=None, files=(), outputs=("output_netcdf_path",)):
    """Declares the inputs of a workflow, which the workflow then validates before it runs.

    The decorated workflow takes the additional keyword argument `validate`, by default
    True: the workflow first checks its inputs as validate_inputs does, raises a ResError
    listing all problems if it finds an error, and emits a warning for every warning
    found. Pass validate=False to skip the check.

    Parameters
    ----------
    manager : type
        The workflow manager class; its placement_problems() checks the placements
    weather : dict, optional
        The weather sources the workflow reads, as {argument: WeatherInput}, where the
        argument gives the path of the source
    files : tuple of str, optional
        The arguments which may name input files. An argument may also give a dict or a
        list of files, or a value which is not a path, which is then not checked.
    outputs : tuple of str, optional
        The arguments which may name an output file or directory
    """

    def decorate(function):
        inputs = WorkflowInputs(manager, dict(weather or {}), tuple(files), tuple(outputs))
        signature = inspect.signature(function)

        @functools.wraps(function)
        def workflow(*args, validate=True, **kwargs):
            if validate:
                _validate_call(function.__name__, signature, inputs, args, kwargs)
            return function(*args, **kwargs)

        workflow.inputs = inputs
        workflow.__signature__ = _with_validate_parameter(signature)
        workflow.__doc__ = (function.__doc__ or "") + _VALIDATE_DOC
        return workflow

    return decorate


_VALIDATE_DOC = """

    Input validation
    ----------------
    The workflow checks its inputs before it simulates and raises a ResError listing all
    errors, see reskit.validate_inputs. Pass validate=False to skip the check.
"""


def _validate_call(name, signature, inputs, args, kwargs):
    """Validates the arguments of a workflow call, see declare_inputs."""
    try:
        bound = signature.bind(*args, **kwargs)
    except TypeError:
        return  # the call of the workflow itself raises the clearer error
    bound.apply_defaults()
    report = ValidationReport(name)
    _validate(report, inputs, bound.arguments)
    report.raise_if_errors()
    for finding in report.warnings:
        warnings.warn(f"{name}: [{finding.check}] {finding.message}", stacklevel=3)


def _with_validate_parameter(signature):
    """The signature of a workflow with the keyword argument 'validate' added."""
    parameters = list(signature.parameters.values())
    validate = inspect.Parameter("validate", inspect.Parameter.KEYWORD_ONLY, default=True)
    has_var_keyword = bool(parameters) and parameters[-1].kind == inspect.Parameter.VAR_KEYWORD
    parameters.insert(len(parameters) - has_var_keyword, validate)
    return signature.replace(parameters=parameters)


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
        * The placements have valid locations and the columns the workflow needs
        * Every weather source opens, provides the variables the workflow reads with the
          time steps of its time axis, covers all placements and has a regular time axis;
          a 'time_slice' lies within the available time span; a 'https://' store has
          credentials in ~/.netrc
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
    signature = inspect.signature(function)
    try:
        bound = signature.bind(placements, **workflow_kwargs)
    except TypeError as error:
        report._add("error", "arguments", str(error))
        defaults = {
            name: parameter.default
            for name, parameter in signature.parameters.items()
            if parameter.default is not inspect.Parameter.empty
        }
        arguments = {**defaults, **workflow_kwargs, "placements": placements}
    else:
        bound.apply_defaults()
        arguments = bound.arguments
    _validate(report, function.inputs, arguments)
    return report


def _resolve_workflow(workflow):
    """The workflow function for a name, or the given function, if it declares its inputs."""
    if callable(workflow):
        function = workflow
    else:
        from reskit.util.input_preparation import DEPRECATED_WORKFLOW_NAMES

        name = DEPRECATED_WORKFLOW_NAMES.get(workflow, workflow)
        modules = (importlib.import_module(f"reskit.{family}.workflows.workflows") for family in WORKFLOW_FAMILIES)
        function = next((getattr(m, name) for m in modules if hasattr(getattr(m, name, None), "inputs")), None)
        if function is None:
            raise ValueError(f"Unknown or removed RESKit workflow: {workflow!r}")
    if not isinstance(getattr(function, "inputs", None), WorkflowInputs):
        raise ValueError(f"{function.__name__} does not declare its inputs, see reskit.util.input_validation")
    return function


def _validate(report, inputs, arguments):
    """Runs all checks of validate_inputs for the bound arguments of a workflow."""
    from reskit.workflow_manager import WorkflowManager

    placements = arguments["placements"]
    for problem in inputs.manager.placement_problems(placements):
        report._add("error", "placements", problem)

    # the locations of the placements, for the checks of the weather sources and rasters,
    # which a missing column does not prevent
    manager = None
    if not WorkflowManager.placement_problems(placements):
        if len(placements) == 0:
            report._add("error", "placements", "contain no placements")
        else:
            try:
                manager = WorkflowManager(placements)
            except ValueError as error:
                report._add("error", "placements", str(error))

    if manager is not None:
        for argument, weather in inputs.weather.items():
            _check_weather(report, manager, argument, weather, arguments)
    _check_files(report, manager, inputs, arguments)


def _check_weather(report, manager, argument, weather, arguments):
    """Opens a weather source for its metadata only and checks it against the workflow."""
    check = f"weather '{argument}'"
    path = arguments.get(argument)
    if path is None:
        report._add("error", check, "no weather source given")
        return
    if not is_path_like(path):
        report._add("info", check, "given as an initialized source, not checked")
        return
    path = as_path_string(path)
    if re.search(r"<[^>]*>", path):
        report._add(
            "error",
            check,
            f"contains tile placeholders: {path}. Workflows need the path of one tile; "
            "execute_workflow_iteratively resolves the placeholders per placement",
        )
        return
    url = urlparse(path)
    if url.scheme not in _REMOTE_SCHEMES and not os.path.exists(path):
        report._add("error", check, f"does not exist: {path}")
        return
    if url.scheme == "https" and not _has_netrc_entry(url.hostname):
        report._add("warning", check, f"~/.netrc has no credentials for {url.hostname}")

    source_type = arguments.get(weather.source_type_argument) if weather.source_type_argument else weather.source_type
    options = dict(time_index_from=weather.time_index_from, verbose=False)
    time_slice = arguments.get("time_slice")
    if time_slice is not None:
        options["time_slice"] = time_slice

    with warnings.catch_warnings():
        # e.g. the notes of Era5ZarrSource on variables it derives on the fly
        warnings.simplefilter("ignore")
        try:
            source = manager._open_source(source_type, path, **options)
        except (ResError, RuntimeError, OSError, ValueError, KeyError) as error:
            report._add("error", check, f"cannot be opened as {source_type} source: {error}")
            return
        unavailable = source.unavailable_variables(*weather.variables)
    for variable, reason in unavailable.items():
        report._add("error", check, f"cannot provide '{variable}': {reason}")

    _check_time_axis(report, check, source.time_index, time_slice)
    _check_coverage(report, check, source, manager)


def _has_netrc_entry(host):
    try:
        return netrc.netrc().authenticators(host) is not None
    except (FileNotFoundError, netrc.NetrcParseError):
        return False


def _check_time_axis(report, check, time_index, time_slice):
    if len(time_index) == 0:
        report._add("error", check, "has no time steps in the requested time span")
        return
    report._add("info", check, f"{len(time_index)} time steps from {time_index[0]} to {time_index[-1]}")

    steps = np.unique(np.diff(time_index.values))
    if steps.size > 1:
        report._add("warning", check, f"has an irregular time axis with steps of {', '.join(map(str, steps))}")

    if not isinstance(time_slice, slice):
        return
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
                "warning", check, f"'time_slice' {edge} the data; only {time_index[0]} to {time_index[-1]} is simulated"
            )


def _check_coverage(report, check, source, manager):
    """Reports placements which are more than half a grid cell outside the source's grid."""
    lats, lons = np.asarray(source.lats), np.asarray(source.lons)
    half_lat = np.abs(np.diff(lats)).max() / 2 if lats.size > 1 else 0
    half_lon = np.abs(np.diff(lons)).max() / 2 if lons.size > 1 else 0
    lat = manager.locs.lats
    # a source may give longitudes on a [0, 360) grid
    lon = manager.locs.lons % 360 if lons.max() > 180 else manager.locs.lons
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


def _check_files(report, manager, inputs, arguments):
    """Checks that input files exist and rasters have values at the placements, and that
    output files can be written.
    """
    for name in inputs.outputs:
        for label, path in _paths_in(name, arguments.get(name)):
            directory = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
            if not os.path.isdir(directory):
                report._add("error", label, f"the output directory does not exist: {directory}")
            elif not os.access(directory, os.W_OK):
                report._add("error", label, f"the output directory is not writable: {directory}")

    for name in inputs.files:
        for label, path in _paths_in(name, arguments.get(name)):
            if not os.path.exists(path):
                report._add("error", label, f"does not exist: {path}")
            elif manager is not None and _is_raster(path):
                _check_raster_values(report, label, path, manager)


def _check_raster_values(report, label, path, manager):
    try:
        values = gk.raster.interpolateValues(path, points=manager.locs, mode="near")
    except (gk.error.GeoKitError, RuntimeError, OSError, ValueError) as error:
        report._add("error", label, f"cannot be read as raster: {error}")
        return
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
    except (AttributeError, RuntimeError):  # geokit fails on some files, e.g. netCDF4, instead of returning False
        return False


def _paths_in(name, value):
    """The (label, path) pairs of an argument value, which may be a path, a dict or list of
    paths, or no path at all, e.g. a number.
    """
    if isinstance(value, dict):
        items = [(f"{name}[{key!r}]", item) for key, item in value.items()]
    elif isinstance(value, (list, tuple)):
        items = [(f"{name}[{i}]", item) for i, item in enumerate(value)]
    else:
        items = [(name, value)]
    for label, item in items:
        if is_path_like(item) and urlparse(as_path_string(item)).scheme not in _REMOTE_SCHEMES:
            yield label, as_path_string(item)
