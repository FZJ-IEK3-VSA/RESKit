import importlib
import inspect
import os
import pkgutil
import reskit as rk

depends_on = {
    "wind_era5_PenaSanchezDunkelWinklerEtAl2025": {
        "GWA4": ["50m", "100m", "200m"],
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
    },
    "CSP_PTR_ERA5": {
        "ERA5": [
            "total_sky_direct_solar_radiation_at_surface",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
        ],
    },
    "retile_DEBUG": {
        "ERA5": [
            "2m_temperature",
        ],
    },
}

# Maps CDS API variable names to their short names inside the downloaded NetCDF file.
_era5_cds_to_nc_name = {
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "100m_u_component_of_wind": "u100",
    "100m_v_component_of_wind": "v100",
    "2m_dewpoint_temperature": "d2m",
    "2m_temperature": "t2m",
    "surface_pressure": "sp",
    "boundary_layer_height": "blh",
    "forecast_surface_roughness": "fsr",
    "surface_solar_radiation_downwards": "ssrd",
    "total_sky_direct_solar_radiation_at_surface": "fdir",
}

# Variables whose NC short names are consumed by preprocess_era5_data and replaced
# by derived outputs (e.g. u100+v100 → ws100). These should not be re-tiled as raw.
_era5_preprocessed_nc_names = {"u10", "v10", "u100", "v100", "ssrd", "fdir"}


def _raw_variables_for_workflow(workflow: str) -> list:
    """Return the NC short names that need to be tiled from the raw download file
    for a given workflow — i.e. the variables that are not replaced by preprocessing.
    """
    era5_cds_names = depends_on[workflow].get("ERA5", [])
    return [
        _era5_cds_to_nc_name[cds_name]
        for cds_name in era5_cds_names
        if cds_name in _era5_cds_to_nc_name and _era5_cds_to_nc_name[cds_name] not in _era5_preprocessed_nc_names
    ]


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
    if workflow not in depends_on:
        if workflow in _known_reskit_workflows():
            raise NotImplementedError(
                f"Workflow '{workflow}' is a known RESKit workflow but is not yet supported by "
                f"download_and_process. Supported workflows: {sorted(depends_on)}."
            )
        raise ValueError(f"Unknown workflow: {workflow}")

    _tile_out = tile_output_dir or os.path.join(output_dir, "tiles")
    era5_path = rk.prepare_era5(
        start_date=start_date,
        end_date=end_date,
        boundary_box=boundary_box,
        output_dir=output_dir,
        variables=depends_on[workflow]["ERA5"],
        tiling=tiling,
        zoom_level=zoom_level,
        tile_output_dir=_tile_out,
        raw_variables=_raw_variables_for_workflow(workflow),
    )
    if tiling:
        # return a path template for get_dataframe_with_weather_tilepaths()
        era5_path = os.path.join(era5_path, "<ZOOM>", "<X-TILE>", "<Y-TILE>")

    result = {"era5_path": era5_path}

    # the wind workflow additionally needs Global Wind Atlas data for height scaling
    if workflow == "wind_era5_PenaSanchezDunkelWinklerEtAl2025":
        print(
            f"ERA5 data prepared and tiled at: {era5_path}. Please download data for height scaling from the Global Wind Atlas: https://globalwindatlas.info/en/download/gis-files"
        )
        result["height_scaling_data"] = {
            50: "Please download directly from https://globalwindatlas.info/en/download/gis-files",
            100: "Please download directly from https://globalwindatlas.info/en/download/gis-files",
            200: "Please download directly from https://globalwindatlas.info/en/download/gis-files",
        }

    return result
