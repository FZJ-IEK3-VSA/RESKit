import os
import warnings

import pandas as pd

import reskit as rk

#######################################################
############### DATA DEPENDENCIES #####################
#######################################################

depends_on = {
    "wind_era5_PenaSanchezDunkelWinklerEtAl2025": {
        "GWA4": ["wind-speed_50m", "wind-speed_100m", "wind-speed_200m"],
        "ERA5": [
            "100m_u_component_of_wind",
            "100m_v_component_of_wind",
            "2m_temperature",
            "surface_pressure",
            "boundary_layer_height",
        ],
    },
    "openfield_pv_era5": {
        "ERA5": [
            "surface_solar_radiation_downwards",
            "total_sky_direct_solar_radiation_at_surface",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "surface_pressure",
            "2m_temperature",
            "2m_dewpoint_temperature",
        ],
        # long-run-average rasters passed via global_solar_atlas_ghi_path /
        # global_solar_atlas_dni_path (used to bias-correct GHI and DNI)
        "GSA": ["GHI", "DNI"],
    },
    "csp_ptr_era5": {
        "ERA5": [
            "total_sky_direct_solar_radiation_at_surface",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
        ],
        # global_solar_atlas_dni_path (DNI long-run-average correction) plus
        # global_solar_atlas_tamb_path (TEMP), used to pick the optimal HTF per
        # placement in the multi-dataset case
        "GSA": ["DNI", "TEMP"],
    },
    # core implementation behind the csp_ptr_era5 wrapper; same ERA5 needs but
    # only the DNI raster (HTF selection happens in the wrapper)
    "csp_ptr_era5_specific_dataset": {
        "ERA5": [
            "total_sky_direct_solar_radiation_at_surface",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
        ],
        "GSA": ["DNI"],
    },
    "ht_dac_era5_wenzel2025": {
        "ERA5": [
            "2m_temperature",
            "2m_dewpoint_temperature",
        ],
    },
    "lt_dac_era5_wenzel2025": {
        "ERA5": [
            "2m_temperature",
            "2m_dewpoint_temperature",
        ],
    },
    "air_cooling_wenzel2025": {
        "ERA5": [
            "2m_temperature",
        ],
    },
    "air_source_heat_pump": {
        "ERA5": [
            "2m_temperature",
        ],
    },
    "evaporative_cooling_wortmann2025": {
        "ERA5": [
            "2m_temperature",
            "2m_dewpoint_temperature",
        ],
    },
}


# Workflow names renamed for PEP 8 in v0.6.0 (#226). A wrapper function cannot cover
# a name that is passed as a string, so the old keys stay resolvable here until v1.0.0.
DEPRECATED_WORKFLOW_NAMES = {
    "CSP_PTR_ERA5": "csp_ptr_era5",
    "CSP_PTR_ERA5_specific_dataset": "csp_ptr_era5_specific_dataset",
    # EGSworkflow is not a key of depends_on; its old name is covered by the
    # deprecated wrapper in reskit/geothermal/workflows/workflows.py instead.
}

for _deprecated_name, _current_name in DEPRECATED_WORKFLOW_NAMES.items():
    if _current_name in depends_on:
        depends_on[_deprecated_name] = depends_on[_current_name]


def _merge_dependencies(workflows):
    """
    Union the weather-data dependencies of one or more workflows.

    Merges the given workflows' ``depends_on`` entries into a single
    ``{source: [variables]}`` mapping, de-duplicating variables while preserving order.

    Parameters
    ----------
    workflows : iterable of str
        Names of registered workflows (keys of ``depends_on``).

    Returns
    -------
    dict
        ``{source: [variables]}`` — the deduplicated union.

    Raises
    ------
    ValueError
        If any workflow name is not registered in ``depends_on``.
    """
    merged = {}
    for workflow in workflows:
        if workflow in DEPRECATED_WORKFLOW_NAMES:
            warnings.warn(
                f"Workflow name {workflow!r} is deprecated and will be removed in RESKit v1.0.0. "
                f"Use {DEPRECATED_WORKFLOW_NAMES[workflow]!r} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if workflow not in depends_on:
            raise ValueError(f"Unknown RESKit workflow: {workflow!r}. Supported workflows: {sorted(depends_on)}.")
        for source, variables in depends_on[workflow].items():
            bucket = merged.setdefault(source, [])
            bucket.extend(var for var in variables if var not in bucket)
    return merged


#######################################################
############### SOURCE PREPARE FUNCTIONS ##############
#######################################################


def _prepare_era5(
    variables, *, start_date, end_date, boundary_box, output_dir, tiling, zoom_level, tile_output_dir, **_
):
    """
    Preparer for the ERA5 weather source: download, preprocess and (optionally) tile the
    given ERA5 CDS variables.

    Returns
    -------
    dict
        ``{"era5_path": <path>}`` — the ``processed/`` directory, or a tile-path template
        (``.../<ZOOM>/<X-TILE>/<Y-TILE>``) when ``tiling`` is True.
    """
    tile_out = tile_output_dir or os.path.join(output_dir, "tiles")
    era5_path = rk.prepare_era5(
        start_date=start_date,
        end_date=end_date,
        boundary_box=boundary_box,
        output_dir=output_dir,
        variables=variables,
        tiling=tiling,
        zoom_level=zoom_level,
        tile_output_dir=tile_out,
        raw_variables=rk.weather.Era5Source.raw_passthrough_variables(variables),
    )
    if tiling:
        # return a path template for weather_tile.get_tilepath()
        era5_path = os.path.join(era5_path, "<ZOOM>", "<X-TILE>", "<Y-TILE>")
    return {"era5_path": era5_path}


def _prepare_gwa4(variables, **_):
    """
    Placeholder preparer for the Global Wind Atlas (GWA4) source.

    Automated GWA4 download is not implemented yet. This does not download anything; it only
    notifies the user that the required rasters must be fetched manually and contributes
    nothing to the result (returns ``None``).
    """
    print(
        "NOTE: Automated Global Wind Atlas (GWA4) download is not implemented yet. Download the "
        "required rasters manually from https://globalwindatlas.info/en/download/gis-files "
        "and pass them to the workflow (e.g. gwa_100m_path / height_scaling_data)."
    )
    return None


def _prepare_gsa(variables, **_):
    """
    Placeholder preparer for the Global Solar Atlas (GSA) source.

    Automated GSA download is not implemented yet. This does not download anything; it only
    notifies the user that the required long-term-average rasters must be fetched manually and
    contributes nothing to the result (returns ``None``). The rasters are passed to the solar
    workflows via ``global_solar_atlas_ghi_path`` / ``global_solar_atlas_dni_path`` (and
    ``global_solar_atlas_tamb_path`` for CSP HTF selection).
    """
    print(
        "NOTE: Automated Global Solar Atlas (GSA) download is not implemented yet. Download the "
        f"required long-term-average rasters ({', '.join(variables)}) manually from "
        "https://globalsolaratlas.info/download and pass them to the workflow (e.g. "
        "global_solar_atlas_ghi_path / global_solar_atlas_dni_path / global_solar_atlas_tamb_path)."
    )
    return None


# Registry of per-source preparers. Each callable takes the workflow's variable list for
# that source plus the shared download context (start/end date, boundary box, output dir,
# tiling options) and returns a partial result dict, or ``None`` if it contributes nothing
# (e.g. a source whose automated download is not implemented yet and only prints guidance).
# To add a new weather source, implement a ``_prepare_<source>`` and register it here.
_SOURCE_PREPARERS = {
    "ERA5": _prepare_era5,
    "GWA4": _prepare_gwa4,  # manual download only for now — notifies and returns None
    "GSA": _prepare_gsa,  # manual download only for now — notifies and returns None
}

#######################################################
############### PRE-DOWNLOAD CHECKS ###################
#######################################################

# The first day of ERA5, and how far it lags behind real time
_ERA5_FIRST_DAY = pd.Timestamp("1940-01-01")
_ERA5_DELAY = pd.Timedelta(days=5)


def _cds_credentials_configured():
    """Whether cdsapi finds an API key: in its environment variables or in the file it reads."""
    if os.environ.get("CDSAPI_URL") and os.environ.get("CDSAPI_KEY"):
        return True
    return os.path.isfile(os.environ.get("CDSAPI_RC", os.path.expanduser("~/.cdsapirc")))


def _check_download_inputs(required_sources, start_date, end_date, boundary_box, output_dir):
    """
    Raise every problem which would make the download fail, together and before anything
    is downloaded. Otherwise, e.g. an invalid boundary box is rejected only once the CDS
    request leaves the queue, and missing credentials only after earlier sources are done.

    Raises
    ------
    ValueError
        Listing all problems found.
    """
    problems = []

    try:
        start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    except (TypeError, ValueError) as error:
        problems.append(f"the dates cannot be read: {error}")
    else:
        if end < start:
            problems.append(f"end_date {end_date} is before start_date {start_date}")
        if "ERA5" in required_sources:
            if start < _ERA5_FIRST_DAY:
                problems.append(f"ERA5 starts on {_ERA5_FIRST_DAY.date()}, but start_date is {start_date}")
            if end > pd.Timestamp.now().normalize() - _ERA5_DELAY:
                problems.append(
                    f"ERA5 is available until about {_ERA5_DELAY.days} days before today, but end_date is {end_date}"
                )

    sides = ("north", "south", "west", "east")
    missing = [side for side in sides if side not in boundary_box]
    if missing:
        problems.append(f"boundary_box lacks {', '.join(missing)}")
    else:
        try:
            north, south, west, east = (float(boundary_box[side]) for side in sides)
        except (TypeError, ValueError):
            problems.append(f"boundary_box values must be numbers, got {boundary_box}")
        else:
            if not -90 <= south < north <= 90:
                problems.append(f"boundary_box needs -90 <= south < north <= 90, got south={south}, north={north}")
            if not (-180 <= west <= 360 and -180 <= east <= 360) or west == east:
                problems.append(
                    f"boundary_box needs two different longitudes between -180 and 360, got west={west}, east={east}"
                )

    # output_dir is created by the preparers, so its nearest existing parent must be writable
    existing = os.path.abspath(output_dir)
    while not os.path.exists(existing):
        existing = os.path.dirname(existing)
    if not os.path.isdir(existing) or not os.access(existing, os.W_OK):
        problems.append(f"output_dir {output_dir} cannot be written: {existing} is not a writable directory")

    if "ERA5" in required_sources and not _cds_credentials_configured():
        problems.append(
            "no CDS API key found for the ERA5 download; save it in ~/.cdsapirc, "
            "see https://cds.climate.copernicus.eu/how-to-api"
        )

    if problems:
        raise ValueError("Cannot prepare the weather data:\n" + "\n".join(f"  - {p}" for p in problems))


def _print_plan(required_sources, start_date, end_date, boundary_box, output_dir):
    """Print what download_and_process would download, see its 'dry_run'."""
    print(f"Weather data from {start_date} to {end_date} in {boundary_box}, into {output_dir}:")
    for source, variables in required_sources.items():
        if source == "ERA5":
            requests = len(rk.weather.era5_source.era5_prepare._era5_download_jobs(start_date, end_date))
            how = f"download in {requests} CDS request(s)"
        else:
            how = "no automated download, provide it manually"
        print(f"  {source} ({how}): {', '.join(variables)}")


#######################################################
############### USER FUNCTIONS ########################
#######################################################


def download_and_process(
    workflows,
    start_date,
    end_date,
    boundary_box,
    output_dir,
    tiling=False,
    zoom_level=4,
    tile_output_dir=None,
    dry_run=False,
):
    """
    Download and process the weather data one or more RESKit workflows need.

    Before anything is downloaded, the dates, the boundary box, the output directory and,
    for ERA5, the CDS API key are checked, and all problems are raised together.

    A workflow may depend on several weather sources (see ``depends_on``); each is prepared
    by its own registered preparer (see ``_SOURCE_PREPARERS``) and contributes its outputs to
    the returned dict (e.g. the ERA5 preparer adds ``"era5_path"``). Sources whose automated
    download is not implemented yet (e.g. GWA4) do not fail the call — they just print a
    notice that the data must be downloaded manually.

    Parameters
    ----------
    workflows : str or list of str
        Name of a registered RESKit workflow (a key of ``depends_on``), or a list of such
        names. When a list is given, the union of their variable requirements is prepared in
        a single call.
    start_date, end_date : str
        Inclusive date range to download (``"YYYY-MM-DD"``).
    boundary_box : dict
        Bounding box ``{"north", "south", "west", "east"}`` in degrees.
    output_dir : str
        Directory to download/process into.
    tiling : bool, optional
        If True, tile the processed data into the ``<zoom>/<x>/<y>/<year>/`` structure.
    zoom_level : int, optional
        Web-Mercator tiling zoom level, by default 4.
    tile_output_dir : str, optional
        Override for the tile output directory (defaults to ``<output_dir>/tiles``).
    dry_run : bool, optional
        If True, only run the checks and print which variables of which source would be
        downloaded, without downloading or writing anything. By default False.

    Returns
    -------
    dict
        Merged outputs of the workflows' sources' preparers. With ``dry_run=True``, the
        ``{source: [variables]}`` which would be prepared instead.

    Raises
    ------
    ValueError
        If any given workflow name is unknown, or if the checks find a problem.
    """
    workflows = [workflows] if isinstance(workflows, str) else list(workflows)
    required_sources = _merge_dependencies(workflows)
    _check_download_inputs(required_sources, start_date, end_date, boundary_box, output_dir)
    if dry_run:
        _print_plan(required_sources, start_date, end_date, boundary_box, output_dir)
        return required_sources

    context = dict(
        start_date=start_date,
        end_date=end_date,
        boundary_box=boundary_box,
        output_dir=output_dir,
        tiling=tiling,
        zoom_level=zoom_level,
        tile_output_dir=tile_output_dir,
    )

    result = {}
    for source, variables in required_sources.items():
        preparer = _SOURCE_PREPARERS.get(source)
        if preparer is None:
            print(f"NOTE: weather source '{source}' has no preparer registered; skipping.")
            continue
        partial = preparer(variables, **context)
        if partial:
            result.update(partial)
    return result
