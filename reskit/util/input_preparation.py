import importlib
import inspect
import os
import pkgutil
import reskit as rk

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
    "CSP_PTR_ERA5": {
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
    # core implementation behind the CSP_PTR_ERA5 wrapper; same ERA5 needs but
    # only the DNI raster (HTF selection happens in the wrapper)
    "CSP_PTR_ERA5_specific_dataset": {
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

# Meta-workflows: prepare data for *many* workflows in a single call by downloading/
# processing the union of their per-source variable lists. ``all_workflows`` spans every
# registered source; each source additionally gets an ``all_<source>_workflows`` subset
# (e.g. ``all_era5_workflows``, ``all_gsa_workflows``), generated dynamically below so
# newly added workflows *and* newly added weather sources are picked up automatically
# without touching this block.
# Name of the everything-meta-workflow (union across all sources).
ALL_WORKFLOWS = "all_workflows"
# Populated as meta-workflows are registered below so they are excluded from the union
# (a meta-workflow must not fold its own aggregated deps back in). Never hardcoded.
_META_WORKFLOWS = set()


def source_meta_workflow(source):
    """Name of the per-source meta-workflow that unions only ``source``'s variables
    across every real workflow, e.g. ``"ERA5"`` -> ``"all_era5_workflows"``."""
    return f"all_{source.lower()}_workflows"


def all_workflow_dependencies(sources=None):
    """Union the weather-data dependencies of every real (non-meta) workflow.

    Merges the ``depends_on`` entries of all registered workflows into a single
    ``{source: [variables]}`` mapping, de-duplicating variables while preserving order.
    Because it reads ``depends_on`` at call time, both newly added workflows and newly
    added weather sources are included automatically.

    Parameters
    ----------
    sources : iterable of str, optional
        Restrict the union to these source keys (e.g. ``{"ERA5"}``). By default all
        sources found across the workflows are included.

    Returns
    -------
    dict
        ``{source: [variables]}`` — the deduplicated union, suitable as a ``depends_on``
        entry for ``download_and_process``.
    """
    merged = {}
    for workflow, deps in depends_on.items():
        if workflow in _META_WORKFLOWS:
            continue
        for source, variables in deps.items():
            if sources is not None and source not in sources:
                continue
            bucket = merged.setdefault(source, [])
            bucket.extend(var for var in variables if var not in bucket)
    return merged


def _register_meta_workflow(name, *, sources=None):
    """Register a meta-workflow whose deps are the union across the real workflows, and
    record its name so it is excluded from any subsequent union."""
    depends_on[name] = all_workflow_dependencies(sources=sources)
    _META_WORKFLOWS.add(name)


# Everything-meta first, then one per-source meta for each source that actually appears.
_register_meta_workflow(ALL_WORKFLOWS)
for _source in depends_on[ALL_WORKFLOWS]:
    _register_meta_workflow(source_meta_workflow(_source), sources={_source})


def _known_reskit_workflows() -> set:
    """Return the names of all workflow functions defined across RESKit's
    technology packages (solar, wind, csp, dac, geothermal, ...). Used to tell a
    real RESKit workflow that download_and_process does not support yet from an
    unknown/misspelled name. Discovered dynamically so new technologies are
    picked up without changes here.
    """
    names = set()
    for submodule in pkgutil.iter_modules(rk.__path__):
        if not submodule.ispkg:
            continue
        try:
            module = importlib.import_module(f"reskit.{submodule.name}.workflows.workflows")
        except ModuleNotFoundError:
            continue
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if obj.__module__ == module.__name__:
                names.add(name)
    return names


def _prepare_era5(
    variables, *, start_date, end_date, boundary_box, output_dir, tiling, zoom_level, tile_output_dir, **_
):
    """Preparer for the ERA5 weather source: download, preprocess and (optionally) tile the
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
        # return a path template for get_dataframe_with_weather_tilepaths()
        era5_path = os.path.join(era5_path, "<ZOOM>", "<X-TILE>", "<Y-TILE>")
    return {"era5_path": era5_path}


def _prepare_gwa4(variables, **_):
    """Placeholder preparer for the Global Wind Atlas (GWA4) source.

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
    """Placeholder preparer for the Global Solar Atlas (GSA) source.

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


def download_and_process(
    workflow,
    start_date,
    end_date,
    boundary_box,
    output_dir,
    tiling=False,
    zoom_level=4,
    tile_output_dir=None,
):
    """Download and process the weather data a RESKit workflow needs.

    A workflow may depend on several weather sources (see ``depends_on``); each is prepared
    by its own registered preparer (see ``_SOURCE_PREPARERS``) and contributes its outputs to
    the returned dict (e.g. the ERA5 preparer adds ``"era5_path"``). Sources whose automated
    download is not implemented yet (e.g. GWA4) do not fail the call — they just print a
    notice that the data must be downloaded manually.

    Parameters
    ----------
    workflow : str
        Name of a registered RESKit workflow (a key of ``depends_on``).
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

    Returns
    -------
    dict
        Merged outputs of the workflow's sources' preparers.

    Raises
    ------
    ValueError
        If ``workflow`` is unknown.
    NotImplementedError
        If ``workflow`` is a real RESKit workflow not registered here.
    """
    if workflow not in depends_on:
        if workflow in _known_reskit_workflows():
            raise NotImplementedError(
                f"Workflow '{workflow}' is a known RESKit workflow but is not yet implemented in "
                f"download_and_process. Supported workflows: {sorted(depends_on)}."
            )
        raise ValueError(f"Unknown RESKit workflow: {workflow}")

    required_sources = depends_on[workflow]
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
